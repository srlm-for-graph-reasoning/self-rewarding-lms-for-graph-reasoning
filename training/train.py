import os
import multiprocessing as mp
import pandas as pd
import re

# 1. Force vLLM to use spawn for its EngineCore background process
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# 2. Force Python's multiprocessing to use spawn BEFORE importing torch/vllm
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import argparse
import gc
import json
import math
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import statistics
import random
import torch
import torch.distributed as dist
import xgrammar
import time
from datasets import load_dataset, Dataset, load_from_disk
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from trl import DPOConfig, DPOTrainer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from vllm.lora.request import LoRARequest

from sklearn.metrics import cohen_kappa_score

from ..neo4j_client import AsyncNeo4jFleetClient
from ..prompts import (
    CYPHER_GENERATION_CONTENT_PROMPT, CYPHER_GENERATION_SYSTEM_PROMPT, CypherResponse,
    CYPHER_JUDGE_SYSTEM_PROMPT, CYPHER_JUDGE_CONTENT_PROMPT, JudgeResponse
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent.parent / "datasets"

train_path = DATA_PATH /  "train.csv"
val_path = DATA_PATH /  "test.csv"

def print_info(msg: str):
    print(f"[INFO] {msg}")

def print_step(msg: str):
    print(f"\n>>> {msg}")

def print_header(iteration: int):
    print(f"\n{'=' * 85}\n  STARTING ITERATION {iteration}\n{'=' * 85}\n")


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
    match = re.search(r"```(?:cypher)?\s*(.*?)\s*```", final_text, re.DOTALL | re.IGNORECASE)

    if match:
        final_text = match.group(1).strip()
    else:
        # Fallback: Strip dangling backticks and whitespaces
        final_text = final_text.strip("` \n")

    return analysis_text, final_text


def _isolated_training_process(base_model_path: str, current_adapter_path: str | None, dataset_path: str, adapter_dir: str):
    """
    Runs completely isolated in a separate OS process to train the LoRA.
    Upon completion, exiting this process forcefully returns 100% of the VRAM back to the OS.
    """
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig
    from trl import DPOConfig, DPOTrainer

    print(f"[Training Process] Loading dataset from {dataset_path}")
    dpo_dataset = load_from_disk(dataset_path)

    print(f"[Training Process] Loading model {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------------------------------------------------
    # METHOD 1: AUTO-CLONE METHOD SETUP
    # ----------------------------------------------------------------------
    if current_adapter_path is not None:
        print(f"[Training Process] Resuming from adapter (Auto-Clone Method): {current_adapter_path}")
        model = PeftModel.from_pretrained(model, current_adapter_path, is_trainable=True)
        active_peft_config = None
    else:
        print("[Training Process] Initializing new standard LoRA adapter (Iteration 1)")
        active_peft_config = LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.025, target_modules="all-linear",
            task_type="CAUSAL_LM", bias="none",
        )

    training_args = DPOConfig(
        output_dir=adapter_dir, per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=2e-5, num_train_epochs=1, beta=0.075, max_length=None,
        logging_steps=10, report_to="none"
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # Explicitly None. TRL manages reference models automatically.
        args=training_args,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
        peft_config=active_peft_config, # Passing None here triggers the Auto-Clone logic
    )

    trainer.train()
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("[Training Process] Training complete. Exiting clean to free memory.")


class PipelineMemoryManager:
    @staticmethod
    def flush() -> None:
        try:
            destroy_model_parallel()
            destroy_distributed_environment()
            if dist.is_initialized():
                dist.destroy_process_group()
            if hasattr(xgrammar, 'clear_cache'):
                xgrammar.clear_cache()
        except Exception:
            pass

        # Force kill any lingering worker background processes
        for child in mp.active_children():
            child.kill()
            child.join(timeout=2)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

@dataclass
class GenerationCandidate:
    raw_cypher_text: str
    cypher_response: CypherResponse
    db_output_str: str = ""
    status: str = "PENDING"
    judgements: list[tuple[JudgeResponse, str]] = field(default_factory=list)

    @property
    def valid_judgements(self) -> list[JudgeResponse]:
        return [j for j, raw in self.judgements if j is not None and j.score is not None]

    @property
    def score_std_dev(self) -> float:
        valid = self.valid_judgements
        if not valid or len(valid) < 2:
            return 0.0
        return statistics.stdev([j.score for j in valid])

    @property
    def is_consensus_reached(self) -> bool:
        MAX_ALLOWED_STD_DEV = 0.8
        return self.valid_judgements and (self.score_std_dev <= MAX_ALLOWED_STD_DEV)

    @property
    def mean_score(self) -> float:
        valid = self.valid_judgements
        return sum(j.score for j in valid) / len(valid) if valid else 0.0

    @property
    def effective_score(self) -> float:
        if self.status == "WRONG_FORMAT":
            return 0.5 # automatic reject for format errors
        if self.is_consensus_reached and self.status in ["SYNTAX_ERROR", "EXECUTION_ERROR", "SUCCESS"]:
            return self.mean_score
        return -1.0 # remove unexpected behavior


@dataclass
class TrainingSample:
    question: str
    schema: str
    db_name: str
    candidates: list[GenerationCandidate] = field(default_factory=list)

    def get_actor_dpo_pair(self, margin_threshold: float = 0.8) -> tuple[GenerationCandidate, GenerationCandidate] | None:
        valid_candidates = [c for c in self.candidates if c.effective_score >= 0.0]
        if len(valid_candidates) < 2:
            return None

        sorted_candidates = sorted(valid_candidates, key=lambda c: c.effective_score)
        rejected = sorted_candidates[0]

        successful_candidates = [c for c in valid_candidates if c.status == "SUCCESS"]
        if not successful_candidates:
            return None

        chosen = successful_candidates[-1]

        if (chosen.effective_score - rejected.effective_score) >= margin_threshold:
            if chosen.effective_score >= 4:
                return chosen, rejected
        return None


# ----------------------------------------------------------------------
# Core Pipeline
# ----------------------------------------------------------------------
class IterativeSelfRewardingPipeline:
    def __init__(self, base_model: str | Path, db_client: AsyncNeo4jFleetClient,
                 num_generations: int = 16, num_judge_generations: int = 11, output_dir: str | Path = "./checkpoints", static_judge: bool = False):
        self.current_model_path = str(base_model)
        self.current_adapter_path = None
        self.db_client = db_client
        self.num_generations = num_generations
        self.num_judge_generations = num_judge_generations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.static_judge = static_judge

        self.judge_evolution_stats = []

        self.actor_sampling = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=1024, n=num_generations,
        )
        self.judge_sampling = SamplingParams(
            temperature=0.6, top_p=0.95, max_tokens=2048, n=num_judge_generations,
        )

    def get_resume_state(self) -> int:
        """
        Scans the output directory to find the highest completed iteration.
        Updates `current_adapter_path` and returns the iteration number.
        """
        max_iter = 0
        if not self.output_dir.exists():
            return 0

        # Scan the output directory for adapter folders
        for item in self.output_dir.iterdir():
            if item.is_dir():
                match = re.match(r"iter_(\d+)_adapter", item.name)
                if match:
                    iter_num = int(match.group(1))
                    if (item / "adapter_config.json").exists():
                        max_iter = max(max_iter, iter_num)

        if max_iter > 0:
            self.current_adapter_path = str(self.output_dir / f"iter_{max_iter}_adapter")
            print_info(f"Resume state found! Latest valid adapter is Iteration {max_iter}")
            print_info(f"Adapter path set to: {self.current_adapter_path}")
        else:
            print_info("No previous valid adapters found. Starting from scratch.")

        return max_iter

    @staticmethod
    def _chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _execute_candidates_batch(self, samples: list[TrainingSample]):
        async def evaluate_candidate(cand: GenerationCandidate, db_name: str, sem: asyncio.Semaphore):
            async with sem:
                if not cand.cypher_response.cypher_query:
                    cand.status = "WRONG_FORMAT"
                    cand.db_output_str = "No valid Cypher generated. Assign this query a score of 1."
                    return

                max_attempts = 2
                for attempt in range(max_attempts):
                    try:
                        result = await self.db_client.query(db_name=db_name, cypher=cand.cypher_response.cypher_query)
                        cand.db_output_str = result or "[Empty Result Set]"
                        cand.status = "SUCCESS"
                        break  # Successful execution, break out of retry loop

                    except Exception as e:
                        error_msg = str(e)
                        if len(error_msg) > 1000:
                            error_msg = error_msg[:1000] + "\n... [ERROR MESSAGE TRUNCATED]"

                        cand.db_output_str = f"[Execution Failed]: {error_msg}"

                        if "SyntaxError" in error_msg:
                            cand.status = "SYNTAX_ERROR"
                            break  # Syntax errors won't resolve on retry, break immediately

                        cand.status = "EXECUTION_ERROR"

                        # If we have attempts left, wait briefly before retrying
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(2)

        async def run_all():
            sem = asyncio.Semaphore(16)
            tasks = []
            for sample in samples:
                for cand in sample.candidates:
                    tasks.append(evaluate_candidate(cand, sample.db_name, sem))
            if tasks:
                await asyncio.gather(*tasks)

        self.db_client.loop.run_until_complete(run_all())


    def construct_cypher_dpo_dataset(self, samples: list[TrainingSample], margin_threshold: float = 1.0) -> Dataset:
        dpo_rows = []
        for sample in samples:
            pair = sample.get_actor_dpo_pair(margin_threshold=margin_threshold)
            if not pair:
                continue

            chosen_cand, rejected_cand = pair
            user_prompt = CYPHER_GENERATION_CONTENT_PROMPT.format(question=sample.question, schema=sample.schema)

            dpo_rows.append({
                "prompt": [
                    {"role": "system", "content": CYPHER_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "chosen": [{"role": "assistant", "content": chosen_cand.raw_cypher_text}],
                "rejected": [{"role": "assistant", "content": rejected_cand.raw_cypher_text}],
            })

        print_info(f"Constructed {len(dpo_rows)} Actor DPO preference pairs (Margin >= {margin_threshold}).")
        return Dataset.from_list(dpo_rows).shuffle(seed=42) if dpo_rows else Dataset.from_dict({})

    def evaluate_validation_set(self, val_dataset: Dataset, judge_df: pd.DataFrame, iteration: int, llm: LLM, actor_save_path: str):
        print_header(f"VALIDATION EVALUATION - ITERATION {iteration}")
        tokenizer = llm.get_tokenizer()

        actor_lora_request = LoRARequest("adapter", 1, self.current_adapter_path) if self.current_adapter_path else None
        judge_lora_request = actor_lora_request
        if self.static_judge:
            print_info("Using static judge (base model) for validation.")
            judge_lora_request = None

        print_step("Evaluating Actor Performance...")
        eval_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            n=1
        )

        actor_prompts = []
        for ex in val_dataset:
            db_n = ex.get("db_name")
            schema_str = self.db_client.get_schema(db_n, shuffle=False)

            prompt = self._chat_prompt(
                tokenizer,
                CYPHER_GENERATION_SYSTEM_PROMPT,
                CYPHER_GENERATION_CONTENT_PROMPT.format(question=ex["question"], schema=schema_str)
            )
            actor_prompts.append(prompt)

        actor_outputs = llm.generate(actor_prompts, eval_sampling, lora_request=actor_lora_request, use_tqdm=True)

        generations = []
        raw_actor_responses = []
        for output in actor_outputs:
            raw_text = output.outputs[0].text
            raw_actor_responses.append(raw_text)
            _, final_cypher = parse_qwen_output(raw_text)
            generations.append(final_cypher)

        val_with_generations = val_dataset.add_column("generated_cypher", generations)
        val_with_generations = val_with_generations.add_column("raw_model_response", raw_actor_responses)
        val_with_generations.to_pandas().to_csv(actor_save_path, index=False)
        print_info(f"Saved validation generations and raw responses to {actor_save_path}")

        if judge_df is not None and not judge_df.empty:
            print_step("Evaluating Judge Performance...")
            judge_prompts = []

            for _, row in judge_df.iterrows():
                # Attempt to retrieve deterministic schema from live DB
                db_n = row.get('db_name')
                if pd.isna(db_n) and 'database_reference_alias' in row:
                    db_n = str(row['database_reference_alias']).replace('neo4jlabs_demo_db_', '')

                schema_str = self.db_client.get_schema(db_n, shuffle=False)

                user_content = CYPHER_JUDGE_CONTENT_PROMPT.replace("{schema}", schema_str)
                user_content = user_content.replace("{question}", str(row['question']))
                user_content = user_content.replace("{generated_cypher}", str(row['generated_cyphers']))
                user_content = user_content.replace("{db_output}", str(row['long_execution_result']))

                judge_prompts.append(self._chat_prompt(
                    tokenizer,
                    CYPHER_JUDGE_SYSTEM_PROMPT,
                    user_content
                ))

            judge_outputs = llm.generate(judge_prompts, eval_sampling, lora_request=judge_lora_request, use_tqdm=True)

            parsed_scores = []
            raw_judge_responses = []
            parsed_judgements = []

            for out in judge_outputs:
                raw_text = out.outputs[0].text
                raw_judge_responses.append(raw_text)
                _, final_judgement = parse_qwen_output(raw_text)
                parsed_judgements.append(final_judgement)

                fallback_match = re.search(r"\b([1-5])\b", final_judgement)
                if fallback_match and len(final_judgement.strip()) < 10:
                    parsed_scores.append(int(fallback_match.group(1)))
                else:
                    parsed_scores.append(-1)

            df_eval = judge_df.copy()
            df_eval['raw_model_response'] = raw_judge_responses
            df_eval['parsed_judgement'] = parsed_judgements
            df_eval['llm_judge_score'] = parsed_scores
            df_eval['score'] = pd.to_numeric(df_eval['score'], errors='coerce')

            bad_parsing_df = df_eval[df_eval['llm_judge_score'] == -1]
            if not bad_parsing_df.empty:
                bad_example = bad_parsing_df.iloc[0]
                print("\n" + "#"*85)
                print("EXAMPLE OF BAD PARSING (SCORE EXTRACTION FAILED)".center(85))
                print("#"*85)
                print(f"Expected Score: {bad_example.get('score', 'N/A')}")
                print("-" * 85)
                print("RAW MODEL RESPONSE:")
                print(bad_example['raw_model_response'])
                print("-" * 85)
                print("PARSED JUDGEMENT EXTRACT:")
                print(bad_example['parsed_judgement'])
                print("#"*85 + "\n")

            valid_df = df_eval[df_eval['llm_judge_score'] != -1].dropna(subset=['score']).copy()
            valid_df['score_delta'] = valid_df['llm_judge_score'] - valid_df['score']
            valid_df['abs_delta'] = valid_df['score_delta'].abs()

            total_dataset = len(df_eval)
            total_valid = len(valid_df)
            format_errors = total_dataset - total_valid

            safe_valid = total_valid if total_valid > 0 else 1

            qwk = float('nan')
            if total_valid > 1:
                try:
                    qwk = cohen_kappa_score(
                        valid_df['score'].astype(int),
                        valid_df['llm_judge_score'].astype(int),
                        weights='quadratic', labels=[1, 2, 3, 4, 5]
                    )
                except Exception:
                    pass

            stats = {
                "Iter": iteration,
                "Total": total_dataset,
                "Valid": total_valid,
                "Fails": format_errors,
                "Fail(%)": round((format_errors / total_dataset) * 100, 2) if total_dataset > 0 else 0,
                "Acc(%)": round((len(valid_df[valid_df['abs_delta'] == 0]) / safe_valid) * 100, 2),
                "Off-1(%)": round((len(valid_df[valid_df['abs_delta'] == 1]) / safe_valid) * 100, 2),
                "Crit(%)": round((len(valid_df[valid_df['abs_delta'] >= 3]) / safe_valid) * 100, 2),
                "MAE": round(valid_df['abs_delta'].mean(), 3) if total_valid > 0 else 0.0,
                "MeanBias": round(valid_df['score_delta'].mean(), 3) if total_valid > 0 else 0.0,
                "QWK": round(qwk, 3) if not pd.isna(qwk) else "N/A"
            }

            self.judge_evolution_stats.append(stats)

            df_eval.to_csv(self.output_dir / f"iter_{iteration}_judge_predictions.csv", index=False)
            stats_df = pd.DataFrame(self.judge_evolution_stats)
            stats_df.to_csv(self.output_dir / "evolution_statistics_summary.csv", index=False)

            print("\n" + "="*85)
            print(f"JUDGE PERFORMANCE: ITERATION {iteration}".center(85))
            print("="*85)

            print("--- ENTIRE DATASET STATS ---")
            print(f"Total Samples  : {total_dataset}")
            print(f"Parsing Fails  : {format_errors} ({stats['Fail(%)']}%)")
            mean_true = df_eval['score'].mean()
            print(f"Mean True Score: {mean_true:.3f}" if not pd.isna(mean_true) else "Mean True Score: N/A")

            print("\n--- VALID SUBSET STATS (Working Queries Only) ---")
            if total_valid > 0:
                print(f"Valid Samples  : {total_valid}")
                print(f"Accuracy (E.M.): {stats['Acc(%)']}%")
                print(f"Off-by-1       : {stats['Off-1(%)']}%")
                print(f"Critical Error : {stats['Crit(%)']}%")
                print(f"MAE            : {stats['MAE']:.3f}")
                print(f"Mean Bias      : {stats['MeanBias']:+.3f}")
                print(f"QWK            : {stats['QWK']}")
            else:
                print("No valid queries to evaluate.")

            print("\n--- EVOLUTION SUMMARY DATAFRAME ---")
            formatters = {
                'Fail(%)': '{:.1f}'.format, 'Acc(%)': '{:.1f}'.format, 'Off-1(%)': '{:.1f}'.format,
                'Crit(%)': '{:.1f}'.format, 'MAE': '{:.3f}'.format, 'MeanBias': '{:+.3f}'.format
            }
            print(stats_df.tail(1).to_string(index=False, formatters=formatters))
            print("="*85 + "\n")

    def run_generation_phase(self, dataset: Dataset, iteration: int, llm: LLM) -> list[TrainingSample]:
        tokenizer = llm.get_tokenizer()
        actor_lora_request = LoRARequest("adapter", 1, self.current_adapter_path) if self.current_adapter_path else None

        judge_lora_request = actor_lora_request
        if self.static_judge:
            print_info("Using static judge (base model). No LoRA adapter will be used for judging.")
            judge_lora_request = None

        samples = []
        for ex in dataset:
            db_n = ex.get("db_name")
            schema_str = self.db_client.get_schema(db_n, shuffle=False)

            samples.append(
                TrainingSample(
                    question=ex["question"], schema=schema_str, db_name=db_n
                )
            )

        # 1. Actor Generation
        print_step("1. Generating Actor Candidates...")
        actor_prompts = [
            self._chat_prompt(
                tokenizer,
                CYPHER_GENERATION_SYSTEM_PROMPT,
                CYPHER_GENERATION_CONTENT_PROMPT.format(question=s.question, schema=s.schema)
            )
            for s in samples
        ]

        actor_outputs = llm.generate(actor_prompts, self.actor_sampling, lora_request=actor_lora_request, use_tqdm=True)

        for sample, output in zip(samples, actor_outputs):
            for choice in output.outputs:
                raw_text = choice.text.strip()
                analysis_text, final_cypher = parse_qwen_output(raw_text)

                try:
                    parsed = CypherResponse(reasoning=analysis_text, cypher_query=final_cypher)
                    sample.candidates.append(GenerationCandidate(raw_cypher_text=raw_text, cypher_response=parsed))
                except Exception as e:
                    print(f"Failed to populate CypherResponse: {e}")

        print_step("2. Executing Queries...")
        self._execute_candidates_batch(samples)

        print_step("2.1 Query Execution Statistics:")
        status_counts = Counter()
        total_cands = 0
        for sample in samples:
            for cand in sample.candidates:
                status_counts[cand.status] += 1
                total_cands += 1

        print_info(f"Total Candidates Generated: {total_cands}")
        for status, count in status_counts.most_common():
            pct = (count / total_cands * 100) if total_cands else 0
            print_info(f"  - {status}: {count} ({pct:.1f}%)")

        print_step("3. Generating Diverse Judge Evaluations...")
        judge_prompts, prompt_mapping = [], []
        for s_idx, sample in enumerate(samples):
            for c_idx, candidate in enumerate(sample.candidates):

                user_content = CYPHER_JUDGE_CONTENT_PROMPT.replace("{schema}", str(sample.schema))
                user_content = user_content.replace("{question}", str(sample.question))
                user_content = user_content.replace("{generated_cypher}", str(candidate.cypher_response.cypher_query or "None"))
                user_content = user_content.replace("{db_output}", str(candidate.db_output_str))

                prompt = self._chat_prompt(
                    tokenizer,
                    CYPHER_JUDGE_SYSTEM_PROMPT,
                    user_content
                )
                judge_prompts.append(prompt)
                prompt_mapping.append((s_idx, c_idx))

        chunk_size = 2_000 # chunking necessary, RAM restrictions sometimes crashed the pipeline when generating 30k+ judgments
        judge_outputs = []
        total_chunks = (len(judge_prompts) + chunk_size - 1) // chunk_size

        for i in range(0, len(judge_prompts), chunk_size):
            chunk_prompts = judge_prompts[i : i + chunk_size]
            print(f"Processing chunk {i // chunk_size + 1}/{total_chunks} ({len(chunk_prompts)} prompts)...")

            chunk_outputs_vllm = llm.generate(chunk_prompts, self.judge_sampling, lora_request=judge_lora_request, use_tqdm=True)
            judge_outputs.extend(chunk_outputs_vllm)

        for (s_idx, c_idx), output in zip(prompt_mapping, judge_outputs):
            candidate = samples[s_idx].candidates[c_idx]
            for choice in output.outputs:
                raw_judge_text = choice.text.strip()
                _, final_judgement = parse_qwen_output(raw_judge_text)

                fallback_match = re.search(r"\b([1-5])\b", final_judgement)
                if fallback_match and len(final_judgement.strip()) < 10:
                    score = int(fallback_match.group(1))
                else:
                    score = -1

                if score != -1:
                    parsed_judge = JudgeResponse(rationale=final_judgement, score=score)
                    candidate.judgements.append((parsed_judge, raw_judge_text))

        print_step("3.1 Judge Evaluation Statistics:")
        total_judgements = 0
        valid_judgements = 0
        score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        status_scores = defaultdict(list)

        for sample in samples:
            for cand in sample.candidates:
                total_judgements += len(cand.judgements)
                valid_judgements += len(cand.valid_judgements)

                for j in cand.valid_judgements:
                    if j.score in score_distribution:
                        score_distribution[j.score] += 1

                if cand.valid_judgements:
                    status_scores[cand.status].append(cand.mean_score)

        print_info(f"Total Judgements Generated: {total_judgements}")
        if total_judgements > 0:
            valid_pct = (valid_judgements / total_judgements) * 100
            print_info(f"Valid (Parsable) Judgements: {valid_judgements} ({valid_pct:.1f}%)")

            print_info("Score Distribution:")
            for score in range(1, 6):
                count = score_distribution[score]
                pct = (count / valid_judgements * 100) if valid_judgements else 0
                print_info(f"  - Score {score}: {count} ({pct:.1f}%)")

        success_count = sum(1 for s in samples for c in s.candidates if c.status == "SUCCESS")
        print_info(f"Iteration {iteration} Generation Done. Total Executable Candidates: {success_count}")
        return samples

    def run_training_phase(self, dataset_path: str, iteration: int) -> str:
        """Spawns an isolated process to run the DPO Trainer and completely bypass VRAM leaks."""
        print_step(f"Starting DPO Training with standard LoRA (Iteration {iteration})")
        adapter_dir = str(self.output_dir / f"iter_{iteration}_adapter")

        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=_isolated_training_process,
            args=(
                self.current_model_path,
                self.current_adapter_path,
                dataset_path,
                adapter_dir
            )
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(f"Training process failed with exit code {p.exitcode}")

        return adapter_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative Self-Rewarding Pipeline")
    parser.add_argument("--base_model", type=str, required=True, help="Path to initial model.")
    parser.add_argument("--registry_dir", type=str, default="./cluster_registry", help="Path to Neo4j fleet registry dir.")
    parser.add_argument("--output_path", type=str, default="./out/checkpoints_base_self_rewarding")
    parser.add_argument("--judge_val_path", type=str, default=str(DATA_PATH / "synthetic_judge_dataset.csv"))
    parser.add_argument("--chunk_size", type=int, default=3700, help="Batch size per iteration (automatically rounded for even distribution across iterations).")
    parser.add_argument("--static-judge", action="store_true", help="Use base model as judge instead of dynamic LoRA judge.")
    args = parser.parse_args()

    print_info("Connecting to Async Neo4j Fleet...")
    db_client = AsyncNeo4jFleetClient(registry_dir=args.registry_dir)

    try:
        pipeline = IterativeSelfRewardingPipeline(
            base_model=args.base_model,
            db_client=db_client,
            output_dir=args.output_path,
            static_judge=args.static_judge
        )

        print_info("Loading datasets...")
        dataset = load_dataset('csv', data_files={'train': str(train_path), 'test': str(val_path)})
        dataset = dataset.map(lambda x: {'db_name': x['database_reference_alias'].replace('neo4jlabs_demo_db_', '')})
        train_dataset, val_dataset = dataset['train'], dataset['test']

        judge_df = pd.read_csv(args.judge_val_path)
        print_info(f"Loaded Judge Evaluation dataset with {len(judge_df)} records.")

        print_info("Stratifying training dataset chunks by db_name...")
        db_to_indices = defaultdict(list)
        for i, row in enumerate(train_dataset):
            db_to_indices[row['db_name']].append(i)

        random.seed(42)
        for db in db_to_indices:
            random.shuffle(db_to_indices[db])

        num_chunks = math.ceil(len(train_dataset) / args.chunk_size)
        chunk_buckets = [[] for _ in range(num_chunks)]
        current_bucket = 0
        for db, indices in db_to_indices.items():
            for idx in indices:
                chunk_buckets[current_bucket].append(idx)
                current_bucket = (current_bucket + 1) % num_chunks
        for bucket in chunk_buckets:
            random.shuffle(bucket)

        iteration = 0
        resume_iter = pipeline.get_resume_state()

        if resume_iter == 0:
            print_info(f"Loading vLLM Engine ({pipeline.current_model_path}) for Initial Evaluation (Base Model)")
            llm = LLM(
                model=pipeline.current_model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.90,
                enable_lora=True,
                max_lora_rank=64,
                max_loras=1
            )
            actor_save_path_0 = str(pipeline.output_dir / "iter_0_actor_predictions.csv")
            pipeline.evaluate_validation_set(val_dataset, judge_df, 0, llm, actor_save_path_0)

            del llm
            llm = None
            PipelineMemoryManager.flush()
        else:
            print_info(f"Skipping base model evaluation. Resuming loop from Iteration {resume_iter + 1}")
            llm = None

        for bucket_indices in chunk_buckets:
            chunk = train_dataset.select(bucket_indices)
            iteration += 1

            # Fast-forward past already completed iterations
            if iteration <= resume_iter:
                print_info(f"Skipping Iteration {iteration} (already trained).")
                continue

            print_header(iteration)

            dataset_path = str(pipeline.output_dir / f"iter_{iteration}_dataset")
            if os.path.exists(dataset_path) and os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
                print_info(f"Dataset for iteration {iteration} already exists at {dataset_path}. Skipping generation.")
                dpo_data = load_from_disk(dataset_path)

                if llm is not None:
                    if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "model_executor"):
                        if hasattr(llm.llm_engine.model_executor, "shutdown"):
                            llm.llm_engine.model_executor.shutdown()
                    del llm
                    llm = None
                    PipelineMemoryManager.flush()
            else:
                if llm is None:
                    print_info(f"Loading vLLM Engine ({pipeline.current_model_path}) with LoRA Support Enabled")
                    llm = LLM(
                        model=pipeline.current_model_path,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=0.90,
                        enable_lora=True,
                        max_lora_rank=64,
                        max_loras=1
                    )

                samples = pipeline.run_generation_phase(chunk, iteration, llm)

                dpo_data = pipeline.construct_cypher_dpo_dataset(samples)
                dpo_data.save_to_disk(dataset_path)

                if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "model_executor"):
                    if hasattr(llm.llm_engine.model_executor, "shutdown"):
                        llm.llm_engine.model_executor.shutdown()

                del llm
                llm = None
                PipelineMemoryManager.flush()

            if len(dpo_data) < 16:
                print_info(f"Skipping training for iter {iteration} (insufficient DPO pairs: {len(dpo_data)}).")
                continue

            pipeline.current_adapter_path = pipeline.run_training_phase(dataset_path, iteration)
            PipelineMemoryManager.flush()

            if llm is None:
                print_info(f"Loading vLLM Engine ({pipeline.current_model_path}) for Post-Training Evaluation")
                llm = LLM(
                    model=pipeline.current_model_path,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.90,
                    enable_lora=True,
                    max_lora_rank=64,
                    max_loras=1
                )

            actor_save_path = str(pipeline.output_dir / f"iter_{iteration}_actor_predictions.csv")
            pipeline.evaluate_validation_set(val_dataset, judge_df, iteration, llm, actor_save_path)

        print_info("Pipeline exhausted training dataset and finished successfully.")

    finally:
        print_info("Shutting down Neo4j connections...")
        db_client.loop.run_until_complete(db_client.close())
        db_client.loop.close()
