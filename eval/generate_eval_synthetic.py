import os
import sys
import re
import json
import argparse
import statistics
import warnings
from pathlib import Path
from collections import Counter

# 1. Force vLLM multiprocessing behavior (Must be before vllm/torch imports)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from sklearn.metrics import cohen_kappa_score

SCRIPT_DIR = Path(__file__).resolve().parent

from ..neo4j_client import AsyncNeo4jFleetClient
from ..prompts import (CYPHER_JUDGE_SYSTEM_PROMPT, CYPHER_JUDGE_CONTENT_PROMPT,
                       CYPHER_GENERATION_CONTENT_PROMPT, CYPHER_GENERATION_SYSTEM_PROMPT)

warnings.filterwarnings('ignore', category=RuntimeWarning)

DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "datasets"

# ============================================================================
# PARSING & UTILITIES (FROM TRAINING PIPELINE)
# ============================================================================
def print_header(text: str):
    print(f"\n{'=' * 95}\n {text.center(93)}\n{'=' * 95}")


def parse_qwen_output(raw_text: str) -> tuple[str, str]:
    """
    Parses Qwen 2.5 Coder output.
    Separates the reasoning (think) from the final channel (Cypher query or Score).
    Handles markdown blocks and stray backticks intelligently.
    """
    # Split by the last occurrence of Cypher: or Score:
    parts = re.split(r"(?i)\n(?:Cypher|Score)\s*:", raw_text)
    if len(parts) == 1:
        # Fallback if no newline precedes the marker
        parts = re.split(r"(?i)(?:Cypher|Score)\s*:", raw_text)

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


def extract_score(final_judgement: str) -> int:
    """Exact score extraction logic from the self-rewarding pipeline."""
    fallback_match = re.search(r"\b([1-5])\b", final_judgement)
    if fallback_match and len(final_judgement.strip()) < 10:
        return int(fallback_match.group(1))
    return -1


def calculate_mode(scores: list[int]) -> int:
    valid_scores = [s for s in scores if s != -1]
    if not valid_scores:
        return -1
    return Counter(valid_scores).most_common(1)[0][0]


def calculate_std(scores: list[int]) -> float:
    valid_scores = [s for s in scores if s != -1]
    if len(valid_scores) < 2:
        return 0.0
    return statistics.stdev(valid_scores)


# ============================================================================
# UNIFIED EVALUATION ENGINE
# ============================================================================
class UnifiedPipelineEvaluator:
    def __init__(self, base_model: str, checkpoints_dir: str, data_dir: str, output_dir: str, registry_dir: str):
        self.base_model = base_model
        self.checkpoints_dir = Path(checkpoints_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Initializing Neo4j Fleet Client...")
        self.neo4j_client = AsyncNeo4jFleetClient(registry_dir=registry_dir)

        # Dataset Paths matching the training pipeline relative structure
        self.test_path = self.data_dir / "test.csv"
        self.judge_path = self.data_dir / "synthetic_judge_dataset.csv"

        # Load Datasets
        print(f"[INFO] Loading Cypher Test dataset: {self.test_path}")
        self.df_cypher_test = pd.read_csv(self.test_path)
        if 'database_reference_alias' in self.df_cypher_test.columns:
            self.df_cypher_test['db_name'] = self.df_cypher_test['database_reference_alias'].str.replace('neo4jlabs_demo_db_', '', regex=False)

        print(f"[INFO] Loading Judge dataset: {self.judge_path}")
        self.df_judge = pd.read_csv(self.judge_path)
        if 'database_reference_alias' in self.df_judge.columns:
            self.df_judge['db_name'] = self.df_judge['database_reference_alias'].str.replace('neo4jlabs_demo_db_', '', regex=False)
        self.df_judge['score'] = pd.to_numeric(self.df_judge['score'], errors='coerce')
        self.df_judge = self.df_judge.dropna(subset=['score'])
        self.df_judge['score'] = self.df_judge['score'].astype(int)

        # Inject Deterministic Schemas
        print("[INFO] Injecting deterministic schemas into Cypher Test dataset...")
        self.df_cypher_test = self._inject_deterministic_schemas(self.df_cypher_test)

        print("[INFO] Injecting deterministic schemas into Judge dataset...")
        self.df_judge = self._inject_deterministic_schemas(self.df_judge)

        # Sampling Parameters exactly mapped from the Iterative Self-Rewarding Pipeline
        self.cypher_greedy_sampling = SamplingParams(temperature=0.0, max_tokens=1024, n=1)
        self.cypher_stochastic_sampling = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=1024, n=10)

        self.judge_greedy_sampling = SamplingParams(temperature=0.0, max_tokens=2048, n=1)
        # UPDATED: Generating 99 judgements to evaluate nine full sets of 11 against each other
        self.judge_ensemble_sampling = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048, n=99)

        self.cross_iteration_stats = []

    def _inject_deterministic_schemas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetches the deterministic (unshuffled) schema from the neo4j client."""
        new_schemas = []
        for _, row in df.iterrows():
            db_name = row.get("db_name", "")
            schema = self.neo4j_client.get_schema(db_name, shuffle=False)
            new_schemas.append(schema)

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
                question=str(row.get("question", "")),
                schema=str(row.get("schema", ""))
            )
            prompts.append(self._chat_prompt(tokenizer, CYPHER_GENERATION_SYSTEM_PROMPT, user_content))
        return prompts

    def format_judge_prompts(self, tokenizer, df: pd.DataFrame) -> list[str]:
        prompts = []
        for _, row in df.iterrows():
            user_content = CYPHER_JUDGE_CONTENT_PROMPT.replace("{schema}", str(row.get('schema', '')))
            user_content = user_content.replace("{question}", str(row.get('question', '')))
            user_content = user_content.replace("{generated_cypher}", str(row.get('generated_cyphers', '')))
            user_content = user_content.replace("{db_output}", str(row.get('long_execution_result', '')))
            prompts.append(self._chat_prompt(tokenizer, CYPHER_JUDGE_SYSTEM_PROMPT, user_content))
        return prompts

    def compute_judge_metrics(self, true_scores: np.ndarray, pred_scores: np.ndarray, prefix: str = "") -> dict:
        total = len(true_scores)
        valid_mask = pred_scores != -1
        valid_preds = pred_scores[valid_mask]
        valid_trues = true_scores[valid_mask]

        total_valid = len(valid_preds)
        format_errors = total - total_valid
        fail_pct = (format_errors / total * 100) if total > 0 else 0
        safe_valid = total_valid if total_valid > 0 else 1

        if total_valid > 0:
            deltas = valid_preds - valid_trues
            abs_deltas = np.abs(deltas)
            acc = (len(abs_deltas[abs_deltas == 0]) / safe_valid) * 100
            off_1 = (len(abs_deltas[abs_deltas == 1]) / safe_valid) * 100
            crit = (len(abs_deltas[abs_deltas >= 3]) / safe_valid) * 100
            mae = abs_deltas.mean()
            mean_bias = deltas.mean()
        else:
            acc, off_1, crit, mae, mean_bias = 0.0, 0.0, 0.0, 0.0, 0.0

        qwk = float('nan')
        if total_valid > 1:
            qwk = cohen_kappa_score(valid_trues, valid_preds, weights='quadratic', labels=[1, 2, 3, 4, 5])

        return {
            f"{prefix}Total": total,
            f"{prefix}Valid": total_valid,
            f"{prefix}Fails": format_errors,
            f"{prefix}Fail(%)": round(fail_pct, 2),
            f"{prefix}Acc(%)": round(acc, 2),
            f"{prefix}Off-1(%)": round(off_1, 2),
            f"{prefix}Crit(%)": round(crit, 2),
            f"{prefix}MAE": round(mae, 3),
            f"{prefix}MeanBias": round(mean_bias, 3),
            f"{prefix}QWK": round(qwk, 3) if not pd.isna(qwk) else "N/A"
        }

    def print_judge_report(self, df_judge_subset: pd.DataFrame, iteration: int) -> dict:
        true_scores = df_judge_subset['ground_truth_score'].values
        greedy_scores = df_judge_subset['greedy_parsed'].values
        ensemble_scores = df_judge_subset['ensemble_mode'].values

        greedy_metrics = self.compute_judge_metrics(true_scores, greedy_scores, prefix="Greedy_")
        ens_metrics = self.compute_judge_metrics(true_scores, ensemble_scores, prefix="Ens_")

        def safe_qwk(y_pred):
            mask = y_pred != -1
            if not mask.any():
                return float('nan')
            return cohen_kappa_score(true_scores[mask], y_pred[mask], weights='quadratic', labels=[1,2,3,4,5])

        # Dynamically calculate QWKs for Sets A through I
        set_qwks = {}
        for letter in "abcdefghi":
            val = safe_qwk(df_judge_subset[f'set_{letter}_mode'].values)
            set_qwks[f"Set{letter.upper()}_QWK"] = round(val, 3) if not pd.isna(val) else "N/A"

        # Calculate inter-set agreement across all 36 combinations of the 9 sets
        set_cols = [f"set_{letter}_mode" for letter in "abcdefghi"]
        agreements = []
        for i in range(len(set_cols)):
            for j in range(i + 1, len(set_cols)):
                agree = (df_judge_subset[set_cols[i]] == df_judge_subset[set_cols[j]]).mean() * 100
                agreements.append(agree)

        avg_inter_set_agreement = sum(agreements) / len(agreements) if agreements else 0.0

        stats = {
            "Iter": iteration,
            **greedy_metrics,
            **ens_metrics,
            **set_qwks,
            "Ensemble_StdDev": round(df_judge_subset['ensemble_variance'].mean(), 4),
            "Set_Agreement(%)": round(avg_inter_set_agreement, 2)
        }

        print("\n" + "="*95)
        print(f"JUDGE EVALUATION RESULTS: ITERATION {iteration}".center(95))
        print("="*95)
        print(f"{'Metric':<18} | {'Greedy (T=0.0)':<15} | {'Ensemble Mode (T=0.6, N=99)':<15}")
        print("-" * 65)
        print(f"{'Total Samples':<18} | {stats['Greedy_Total']:<15} | {stats['Ens_Total']:<15}")
        print(f"{'Format Fails':<18} | {stats['Greedy_Fails']} ({stats['Greedy_Fail(%)']}%)  | {stats['Ens_Fails']} ({stats['Ens_Fail(%)']}%)")
        print(f"{'Accuracy (E.M.)':<18} | {stats['Greedy_Acc(%)']}%{'':<6} | {stats['Ens_Acc(%)']}%")
        print(f"{'Mean Abs. Error':<18} | {stats['Greedy_MAE']:<15.3f} | {stats['Ens_MAE']:<15.3f}")
        print(f"{'QWK Score':<18} | {stats['Greedy_QWK']:<15} | {stats['Ens_QWK']:<15}")
        print("-" * 65)
        qwks_str = ", ".join([str(stats[f"Set{L}_QWK"]) for L in "ABCDEFGHI"])
        print(f"Set QWKs (A-I)     : {qwks_str}")
        print(f"Inter-Set Agreement: {stats['Set_Agreement(%)']}% (average across 36 pairs)")

        return stats

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

            inst_id = str(row.get("instance_id", f"{split_name}_{i}"))

            records.append({
                "record_key": inst_id,
                "instance_id": inst_id,
                "sample_index": np.nan,
                "evaluated_cypher": np.nan,
                "split": split_name,
                "question": row.get("question", ""),
                "schema": row.get("schema", ""),
                "database_reference_alias": row.get("database_reference_alias", ""),
                "cypher": row.get("cypher", ""),
                "db_name": row.get("db_name", ""),
                "greedy_raw": g_raw,
                "greedy_parsed": cypher_g,
                "ensemble_raws": json.dumps(stoch_raws),
                "ensemble_parsed": json.dumps(stoch_parsed),
                "ground_truth_score": np.nan,
                "set_a_mode": np.nan,
                "set_b_mode": np.nan,
                "set_c_mode": np.nan,
                "set_d_mode": np.nan,
                "set_e_mode": np.nan,
                "set_f_mode": np.nan,
                "set_g_mode": np.nan,
                "set_h_mode": np.nan,
                "set_i_mode": np.nan,
                "ensemble_mode": np.nan,
                "ensemble_variance": np.nan
            })
        return records

    def process_judge_split(self, llm, prompts: list[str], df: pd.DataFrame, lora_req) -> list[dict]:
        print("[INFO] Running Judge Greedy Generation (T=0.0)...")
        greedy_outs = llm.generate(prompts, self.judge_greedy_sampling, lora_request=lora_req, use_tqdm=True)

        # UPDATED: N=99
        print("[INFO] Running Judge Ensemble Generation (T=0.6, N=99)...")
        ensemble_outs = llm.generate(prompts, self.judge_ensemble_sampling, lora_request=lora_req, use_tqdm=True)

        records = []
        for i in range(len(prompts)):
            row = df.iloc[i]
            g_raw = greedy_outs[i].outputs[0].text
            _, g_final = parse_qwen_output(g_raw)
            g_score = extract_score(g_final)

            ens_raws = [choice.text for choice in ensemble_outs[i].outputs]
            ens_scores = [extract_score(parse_qwen_output(r)[1]) for r in ens_raws]

            # UPDATED: Sub-split into 9 equal chunks of 11
            set_a_mode = calculate_mode(ens_scores[0:11])
            set_b_mode = calculate_mode(ens_scores[11:22])
            set_c_mode = calculate_mode(ens_scores[22:33])
            set_d_mode = calculate_mode(ens_scores[33:44])
            set_e_mode = calculate_mode(ens_scores[44:55])
            set_f_mode = calculate_mode(ens_scores[55:66])
            set_g_mode = calculate_mode(ens_scores[66:77])
            set_h_mode = calculate_mode(ens_scores[77:88])
            set_i_mode = calculate_mode(ens_scores[88:99])

            ens_mode = calculate_mode(ens_scores)
            ens_std = calculate_std(ens_scores)

            inst_id = str(row.get("instance_id", f"judge_{i}"))
            samp_idx = str(row.get("sample_index", "0"))
            eval_cypher = str(row.get("generated_cyphers", ""))

            # Create a unique composite key combining the requested identifiers
            record_key = f"{inst_id}|{samp_idx}|{eval_cypher}"

            records.append({
                "record_key": record_key,
                "instance_id": inst_id,
                "sample_index": samp_idx,
                "evaluated_cypher": eval_cypher,
                "split": "judge_val",
                "question": row.get("question", ""),
                "schema": row.get("schema", ""),
                "database_reference_alias": row.get("database_reference_alias", ""),
                "db_name": row.get("db_name", ""),
                "greedy_raw": g_raw,
                "greedy_parsed": g_score,
                "ensemble_raws": json.dumps(ens_raws),
                "ensemble_parsed": json.dumps(ens_scores),
                "ground_truth_score": int(row['score']),
                "set_a_mode": set_a_mode,
                "set_b_mode": set_b_mode,
                "set_c_mode": set_c_mode,
                "set_d_mode": set_d_mode,
                "set_e_mode": set_e_mode,
                "set_f_mode": set_f_mode,
                "set_g_mode": set_g_mode,
                "set_h_mode": set_h_mode,
                "set_i_mode": set_i_mode,
                "ensemble_mode": ens_mode,
                "ensemble_variance": ens_std
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
        prompts_judge = self.format_judge_prompts(tokenizer, self.df_judge)

        for iteration, lora_path in checkpoints:
            ext_file_path = self.output_dir / f"iter_{iteration}_unified_results.csv"

            # --- Automatic Resume ---
            if ext_file_path.exists():
                print(f"[RESUME] Output file for iteration {iteration} already exists at {ext_file_path}. Skipping generation.")

                # Try to load it to retain cross-iteration stats in summary (if available)
                try:
                    df_existing = pd.read_csv(ext_file_path)
                    df_judge_sub = df_existing[df_existing['split'] == 'judge_val'].copy()
                    if not df_judge_sub.empty:
                        stats = self.print_judge_report(df_judge_sub, iteration)
                        self.cross_iteration_stats.append(stats)
                except Exception:
                    pass
                continue

            adapter_label = f"iter_{iteration}" if lora_path else "base_model"
            print_header(f"STARTING UNIFIED EVALUATION - {adapter_label.upper()}")
            lora_req = LoRARequest("adapter", 1, lora_path) if lora_path else None

            # 1. Cypher Test Set
            records_test = self.process_cypher_split(llm, prompts_test, self.df_cypher_test, "cypher_test", lora_req)

            # 2. Judge Validation Set
            records_judge = self.process_judge_split(llm, prompts_judge, self.df_judge, lora_req)

            # Combine all datasets into one unified DataFrame per iteration
            all_records = records_test + records_judge
            df_unified = pd.DataFrame(all_records)

            # Save iteration unified CSV
            df_unified.to_csv(ext_file_path, index=False)
            print(f"\n[INFO] Saved unified iteration results ({len(df_unified)} rows) to: {ext_file_path}")

            # Compute and log Judge metrics for this iteration
            df_judge_sub = df_unified[df_unified['split'] == 'judge_val'].copy()
            stats = self.print_judge_report(df_judge_sub, iteration)
            self.cross_iteration_stats.append(stats)

        # Save cross-iteration summary if any new evaluations occurred
        if self.cross_iteration_stats:
            df_summary = pd.DataFrame(self.cross_iteration_stats)
            summary_path = self.output_dir / "judge_evolution_metrics_summary.csv"
            df_summary.to_csv(summary_path, index=False)
            print_header("FULL CROSS-ITERATION EVALUATION COMPLETE")
            print(f"[INFO] Evolution summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Cypher (Test) & Judge Pipeline Evaluator")
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace path to base model.")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Directory containing iter_X_adapter folders.")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Directory containing test_2025.csv, and judge_dataset_v1.csv.")
    parser.add_argument("--registry_dir", type=str, default=str(DEFAULT_DATA_DIR / "cluster_registry"), help="Directory containing the database registry JSON files.")
    parser.add_argument("--output_dir", type=str, default="./unified_eval_outputs", help="Where to save unified CSVs.")

    args = parser.parse_args()

    evaluator = UnifiedPipelineEvaluator(
        base_model=args.base_model,
        checkpoints_dir=args.checkpoints_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        registry_dir=args.registry_dir
    )

    evaluator.run()
