import os
import re
import json
import argparse
import warnings
from pathlib import Path

# 1. Force vLLM multiprocessing behavior (Must be before vllm/torch imports)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from ..neo4j_client import AsyncNeo4jFleetClient
from ..prompts import CYPHER_GENERATION_CONTENT_PROMPT, CYPHER_GENERATION_CONTENT_PROMPT

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# PARSING & UTILITIES (FROM TRAINING PIPELINE)
# ============================================================================
def print_header(text: str):
    print(f"\n{'=' * 95}\n {text.center(93)}\n{'=' * 95}")


def parse_qwen_output(raw_text: str) -> tuple[str, str]:
    """
    Parses Qwen 2.5 Coder output.
    Separates the reasoning (think) from the final channel (Cypher query).
    Handles markdown blocks and stray backticks intelligently.
    """
    # Split by the last occurrence of Cypher:
    parts = re.split(r"(?i)\nCypher\s*:", raw_text)
    if len(parts) == 1:
        # Fallback if no newline precedes the marker
        parts = re.split(r"(?i)Cypher\s*:", raw_text)

    if len(parts) > 1:
        analysis_text = "".join(parts[:-1]).strip()
        final_text = parts[-1].strip()
    else:
        analysis_text = ""
        final_text = raw_text.strip()

    # Smart extraction: If the final text contains markdown code blocks, extract contents
    match = re.search(r"```(?:cypher|sql)?\s*(.*?)\s*```", final_text, re.DOTALL | re.IGNORECASE)

    if match:
        final_text = match.group(1).strip()
    else:
        # Fallback: Strip dangling backticks and whitespaces
        final_text = final_text.strip("` \n")

    return analysis_text, final_text


# ============================================================================
# EVALUATION ENGINE
# ============================================================================
class CypherPipelineEvaluator:
    def __init__(self, base_model: str, checkpoints_dir: str, dataset_path: str, output_dir: str, registry_dir: str):
        self.base_model = base_model
        self.checkpoints_dir = Path(checkpoints_dir)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Initializing Neo4j Fleet Client...")
        self.neo4j_client = AsyncNeo4jFleetClient(registry_dir=registry_dir)

        # Load JSON Dataset
        print(f"[INFO] Loading Cypher Test dataset: {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.df_cypher_test = pd.DataFrame(data)

        # Directly use 'graph' as 'db_name'
        if 'graph' in self.df_cypher_test.columns:
            self.df_cypher_test['db_name'] = self.df_cypher_test['graph']

        # Inject Deterministic Schemas
        print("[INFO] Injecting deterministic schemas into Cypher Test dataset...")
        self.df_cypher_test = self._inject_deterministic_schemas(self.df_cypher_test)

        # Sampling Parameters exactly mapped from the Iterative Self-Rewarding Pipeline
        self.cypher_greedy_sampling = SamplingParams(temperature=0.0, max_tokens=1024, n=1)
        self.cypher_stochastic_sampling = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=1024, n=10)

    def _inject_deterministic_schemas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetches the deterministic (unshuffled) schema from the neo4j client."""
        new_schemas = []
        for _, row in df.iterrows():
            db_name = row.get("db_name", "")
            if pd.isna(db_name) or not db_name:
                new_schemas.append(str(row.get("schema", "")))
                continue
            try:
                # shuffle=False ensures we get the deterministic version
                schema = self.neo4j_client.get_schema(db_name, shuffle=False)
                new_schemas.append(schema)
            except Exception as e:
                print(f"[WARNING] Failed to fetch deterministic schema for '{db_name}': {e}. Using dataset fallback.")
                new_schemas.append(str(row.get("schema", "")))

        df["schema"] = new_schemas
        return df

    def _chat_prompt(self, tokenizer, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def get_lora_checkpoints(self) -> list[tuple[int, str]]:
        checkpoints = [(0, None)]  # 0 indicates base model
        if self.checkpoints_dir.exists():
            for d in self.checkpoints_dir.iterdir():
                if d.is_dir() and d.name.startswith("iter_") and d.name.endswith("_adapter"):
                    match = re.search(r"iter_(\d+)_adapter", d.name)
                    if match:
                        checkpoints.append((int(match.group(1)), str(d)))
        return sorted(checkpoints, key=lambda x: x[0])

    def format_cypher_prompts(self, tokenizer, df: pd.DataFrame) -> list[str]:
        prompts = []
        for _, row in df.iterrows():
            user_content = CYPHER_GENERATION_CONTENT_PROMPT.format(
                question=str(row.get("nl_question", "")),
                schema=str(row.get("schema", ""))
            )
            prompts.append(self._chat_prompt(tokenizer, CYPHER_GENERATION_CONTENT_PROMPT, user_content))
        return prompts

    def process_cypher_split(self, llm, prompts: list[str], df: pd.DataFrame, split_name: str, lora_req) -> list[dict]:
        print(f"[INFO] Running Cypher Greedy Generation ({split_name})...")
        greedy_outs = llm.generate(prompts, self.cypher_greedy_sampling, lora_request=lora_req, use_tqdm=True)

        print(f"[INFO] Running Cypher Stochastic Generation ({split_name})...")
        stoch_outs = llm.generate(prompts, self.cypher_stochastic_sampling, lora_request=lora_req, use_tqdm=True)

        records = []
        for i in range(len(prompts)):
            row = df.iloc[i]
            g_raw = greedy_outs[i].outputs[0].text
            _, cypher_g = parse_qwen_output(g_raw)

            stoch_raws = [choice.text for choice in stoch_outs[i].outputs]
            stoch_parsed = [parse_qwen_output(r)[1] for r in stoch_raws]

            inst_id = str(row.get("qid", f"{split_name}_{i}"))

            records.append({
                "qid": inst_id,
                "split": split_name,
                "nl_question": row.get("nl_question", ""),
                "schema": row.get("schema", ""),
                "graph": row.get("graph", ""),
                "gold_cypher": row.get("gold_cypher", ""),
                "greedy_raw": g_raw,
                "greedy_parsed": cypher_g,
                "ensemble_raws": json.dumps(stoch_raws),
                "ensemble_parsed": json.dumps(stoch_parsed)
            })
        return records

    def run(self):
        checkpoints = self.get_lora_checkpoints()
        print(f"[INFO] Found {len(checkpoints)} iterations to evaluate (including base model).")

        llm = LLM(
            model=self.base_model,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            enable_lora=True,
            max_lora_rank=64,
            max_loras=1
        )

        tokenizer = llm.get_tokenizer()
        prompts_test = self.format_cypher_prompts(tokenizer, self.df_cypher_test)

        print_header("SAMPLE CYPHER GENERATION PROMPT")
        print(prompts_test[0])
        print("=" * 95 + "\n")

        for iteration, lora_path in checkpoints:
            ext_file_path = self.output_dir / f"iter_{iteration}_cypher_results.csv"

            # --- Automatic Resume ---
            if ext_file_path.exists():
                print(f"[RESUME] Output file for iteration {iteration} already exists at {ext_file_path}. Skipping generation.")
                continue

            adapter_label = f"iter_{iteration}" if lora_path else "base_model"
            print_header(f"STARTING EVALUATION - {adapter_label.upper()}")
            lora_req = LoRARequest("adapter", 1, lora_path) if lora_path else None

            # 1. Cypher Test Set
            records_test = self.process_cypher_split(llm, prompts_test, self.df_cypher_test, "cypher_test", lora_req)

            # Combine all records into one unified DataFrame per iteration
            df_unified = pd.DataFrame(records_test)

            # Save iteration unified CSV
            df_unified.to_csv(ext_file_path, index=False)
            print(f"\n[INFO] Saved iteration results ({len(df_unified)} rows) to: {ext_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cypher Evaluation Pipeline Evaluator")
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace path to base model.")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Directory containing iter_X_adapter folders.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test.json file.")
    parser.add_argument("--registry_dir", type=str, required=True, help="Directory containing the database registry JSON files.")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs", help="Where to save outputs.")

    args = parser.parse_args()

    evaluator = CypherPipelineEvaluator(
        base_model=args.base_model,
        checkpoints_dir=args.checkpoints_dir,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        registry_dir=args.registry_dir
    )

    evaluator.run()
