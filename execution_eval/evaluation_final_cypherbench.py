import argparse
import asyncio
import pandas as pd
import re
import json
from pathlib import Path

from .cached_db_connector import AsyncNeo4jFleetConnector
from .cached_metrics import (
    executable,
    execution_accuracy,
    provenance_subgraph_jaccard_similarity,
    compute_target_ea,
    compute_target_psjs
)


def extract_cyphers(raw_val) -> list:
    """Safely extracts a list of Cypher queries from a JSON string, array, or raw string."""
    if pd.isna(raw_val) or not raw_val:
        return []

    if isinstance(raw_val, list):
        data = raw_val
    else:
        raw_str = str(raw_val)
        try:
            data = json.loads(raw_str)
        except Exception:
            # Fallback to single string
            return [raw_str]

    extracted = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                extracted.append(str(item.get("cypher_query", item.get("cypher", ""))))
            else:
                extracted.append(str(item))
    elif isinstance(data, dict):
        extracted.append(str(data.get("cypher_query", data.get("cypher", ""))))
    else:
        extracted.append(str(data))

    return extracted


def get_top_n_unique_queries(queries, n):
    """Deduplicates queries and returns the first n valid ones."""
    unique = []
    seen = set()
    for q in queries:
        if not isinstance(q, str):
            continue
        val = q.replace("\\n", "\n").replace("\\t", " ").strip()
        if val and val not in seen:
            seen.add(val)
            unique.append(val)
        if len(unique) == n:
            break
    return unique


async def evaluate_single_candidate(
    connector: AsyncNeo4jFleetConnector,
    instance_id,
    cand_idx,
    pred_cypher,
    target_cypher,
    db_alias,
    timeout,
    target_ea_cache=None,
    target_psjs_cache=None,
    golden_is_empty=None
):
    """Evaluates a single prediction candidate against the target Cypher query."""
    ex, ea, psjs = 0.0, 0.0, 0.0

    # If the candidate is empty, default metrics to 0 immediately
    if not pred_cypher.strip():
        return {
            "instance_id": instance_id,
            "candidate_idx": cand_idx,
            "db_alias": db_alias,
            "EX": 0.0, "EA": 0.0, "PSJS": 0.0,
            "pred_cypher": pred_cypher,
            "golden_is_empty": golden_is_empty
        }

    try:
        ex = await executable(pred_cypher, target_cypher, connector, db_alias, timeout)
    except TimeoutError as e:
        print(f"[Timeout] EX metric timed out for DB {db_alias} at instance {instance_id}: {e}")
    except Exception:
        pass

    try:
        ea = await execution_accuracy(
            pred_cypher, target_cypher, connector, db_alias, timeout, target_cache=target_ea_cache
        )
    except TimeoutError as e:
        print(f"[Timeout] EA metric timed out for DB {db_alias} at instance {instance_id}: {e}")
    except Exception:
        pass

    try:
        psjs = await provenance_subgraph_jaccard_similarity(
            pred_cypher, target_cypher, connector, db_alias, timeout, target_cache=target_psjs_cache
        )
    except TimeoutError as e:
        print(f"[Timeout] PSJS metric timed out for DB {db_alias} at instance {instance_id}: {e}")
    except Exception:
        pass

    return {
        "instance_id": instance_id,
        "candidate_idx": cand_idx,
        "db_alias": db_alias,
        "EX": ex,
        "EA": ea,
        "PSJS": psjs,
        "pred_cypher": pred_cypher,
        "golden_is_empty": golden_is_empty
    }


async def run_evaluation(df_subset: pd.DataFrame, connector: AsyncNeo4jFleetConnector, pass_n: int, timeout: float, concurrency: int, global_golden_cache: dict):
    sem = asyncio.Semaphore(concurrency)

    async def sem_task(*args):
        async with sem:
            return await evaluate_single_candidate(*args)

    group_key = "instance_id"
    instances = df_subset.groupby(group_key)
    total_instances = len(instances)

    print(f"Preparing to evaluate {total_instances} unique instances...")

    # ---------------------------------------------------------
    # PHASE 1: Pre-compute missing Golden Targets asynchronously
    # ---------------------------------------------------------
    async def fetch_golden(inst_id, t_cypher, db):
        ea_cache = await compute_target_ea(t_cypher, connector, db, timeout)
        ps_cache = await compute_target_psjs(t_cypher, connector, db, timeout)
        global_golden_cache[inst_id] = {"ea": ea_cache, "psjs": ps_cache}

    golden_tasks = []
    for name, group in instances:
        if name not in global_golden_cache:
            target_col = "cypher"
            target_cypher = str(group[target_col].iloc[0]).replace("\\n", "\n").replace("\\t", " ")
            
            db_col = "database_reference_alias"
            db_alias = str(group[db_col].iloc[0]).replace("neo4jlabs_demo_db_", "")

            async def sem_fetch(n=name, tc=target_cypher, dba=db_alias):
                async with sem:
                    await fetch_golden(n, tc, dba)

            golden_tasks.append(sem_fetch())

    if golden_tasks:
        print("Pre-computing golden targets...")
        for i, coro in enumerate(asyncio.as_completed(golden_tasks), 1):
            await coro
            if i % 100 == 0 or i == len(golden_tasks):
                print(f"[{i}/{len(golden_tasks)}] Golden queries pre-computed...")
    else:
        print("All golden targets are already cached for this subset! Skipping pre-computation.")

    # ---------------------------------------------------------
    # PHASE 2: Evaluate Candidates using Cached Targets
    # ---------------------------------------------------------
    tasks = []
    for name, group in instances:
        target_col = "cypher"
        target_cypher = str(group[target_col].iloc[0]).replace("\\n", "\n").replace("\\t", " ")
        
        db_col = "database_reference_alias"
        db_alias = str(group[db_col].iloc[0]).replace("neo4jlabs_demo_db_", "")

        # Extract unique list of predicted cyphers
        all_queries = []
        for raw in group["generated_cypher"]:
            all_queries.extend(extract_cyphers(raw))

        unique_queries = get_top_n_unique_queries(all_queries, pass_n)

        # Enforce at least 1 empty string query so we register a score (0) instead of dropping the instance entirely
        if not unique_queries:
            unique_queries = [""]

        ea_cache = global_golden_cache[name]["ea"]
        psjs_cache = global_golden_cache[name]["psjs"]

        golden_empty = None
        if ea_cache and ea_cache.get("success", False):
            golden_empty = len(ea_cache.get("data", [])) == 0

        for c_idx, q in enumerate(unique_queries):
            tasks.append(sem_task(connector, name, c_idx, q, target_cypher, db_alias, timeout, ea_cache, psjs_cache, golden_empty))

    results = []
    print(f"Evaluating {len(tasks)} candidate queries...")

    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        results.append(result)
        if i % 100 == 0 or i == len(tasks):
            print(f"[{i}/{len(tasks)}] Candidate queries processed...")

    return pd.DataFrame(results)


def print_single_report(df_res: pd.DataFrame, title: str):
    """Aggregates and formats the evaluation metrics for a single pass configuration."""
    global_ex = df_res['EX'].mean()
    global_ea = df_res['EA'].mean()
    global_psjs = df_res['PSJS'].mean()

    print(f"\n{'='*60}")
    print(f" METRICS: {title}")
    print(f"{'='*60}")
    print(f"Executable (EX):                       {global_ex:.4f}")
    print(f"Execution Accuracy (EA):               {global_ea:.4f}")
    print(f"Provenance Subgraph Jaccard Sim (PSJS): {global_psjs:.4f}")

    if 'golden_is_empty' in df_res.columns:
        df_empty = df_res[df_res['golden_is_empty'] == True]
        df_non = df_res[df_res['golden_is_empty'] == False]

        print(f"\n--- Output Subgroups ---")
        print(f"Golden returned NO output ({len(df_empty)} instances):")
        print(f"  EX: {df_empty['EX'].mean() if not df_empty.empty else 0:.4f} | EA: {df_empty['EA'].mean() if not df_empty.empty else 0:.4f} | PSJS: {df_empty['PSJS'].mean() if not df_empty.empty else 0:.4f}")
        print(f"Golden returned SOME output ({len(df_non)} instances):")
        print(f"  EX: {df_non['EX'].mean() if not df_non.empty else 0:.4f} | EA: {df_non['EA'].mean() if not df_non.empty else 0:.4f} | PSJS: {df_non['PSJS'].mean() if not df_non.empty else 0:.4f}")

    print(f"\n--- Metrics Per Database ---")
    db_metrics = df_res.groupby('db_alias')[['EX', 'EA', 'PSJS']].mean().reset_index()
    db_metrics.rename(columns={'db_alias': 'Database'}, inplace=True)
    print(db_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_iteration_comparison(summary_results: list):
    """Prints a comparison table of all evaluated parameters."""
    print("\n\n" + "="*95)
    print(" PASS@N EVALUATION SUMMARY ")
    print("="*95)

    comp_df = pd.DataFrame(summary_results)
    print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*95 + "\n")


async def main(args):
    input_path = Path(args.predictions)

    if not input_path.exists():
        print(f"Error: Path not found at {input_path}")
        return

    dfs = []
    if input_path.is_file():
        dfs.append((input_path.name, pd.read_csv(input_path)))
    else:
        for p in input_path.glob("*.csv"):
            # Skip previously exported eval results to avoid recursive reprocessing
            if "_all_cands" in p.name or "pass@" in p.name or "_eval_results" in p.name or "summary" in p.name:
                continue
            dfs.append((p.name, pd.read_csv(p)))

    if not dfs:
        print("No valid CSV prediction files found for evaluation.")
        return

    # 1. Enforce specific evaluation properties based on arguments
    target_splits = ['val', 'test'] if args.split == 'both' else [args.split]
    target_temps = [0.0, 0.6]

    connector = AsyncNeo4jFleetConnector(
        registry_dir=args.registry,
        max_connection_pool_size=args.concurrency,
        debug=False
    )

    summary_results = []
    global_golden_cache = {}  # In-memory dictionary cache for precomputed target hashes

    for fname, df in dfs:
        print(f"\nProcessing file: {fname}")

        # --- Split Enforcement (Substring Match) ---
        df_split = None
        if 'split' in df.columns:
            mask = df['split'].astype(str).str.contains('|'.join(target_splits), case=False, na=False)
            df_split = df[mask]
        else:
            found_split = next((sp for sp in target_splits if sp in fname.lower()), None)
            if found_split:
                df_split = df
            else:
                print(f"  Warning: No 'split' column and couldn't match {target_splits} in filename {fname}. Processing anyway.")
                df_split = df

        if df_split.empty:
            print("  No data matching the requested split(s).")
            continue

        if 'cypher' not in df_split.columns:
            raise RuntimeError("Warning: Target 'cypher' column is missing!")

        # Setup expected base output path
        out_base = input_path.parent if input_path.is_file() else input_path
        split_name = target_splits[0] if len(target_splits) == 1 else 'both'

        # --- Map Unified Columns to Temperatures & Evaluate ---
        for temp in target_temps:
            df_temp = df_split.copy()

            # ---> DYNAMICALLY SET PASS_N <---
            # temp 0.0 = singular evaluation (Pass@1)
            # temp 0.6 = pass@K evaluation (Pass@args.pass_n)
            current_pass_n = 1 if temp == 0.0 else args.pass_n

            if 'greedy_parsed' in df_temp.columns and 'ensemble_parsed' in df_temp.columns:
                if temp == 0.0:
                    df_temp['generated_cypher'] = df_temp['greedy_parsed']
                elif temp == 0.6:
                    df_temp['generated_cypher'] = df_temp['ensemble_parsed']
            else:
                # Fallback to older format handling
                if 'temperature' in df_temp.columns:
                    df_temp = df_temp[df_temp['temperature'].astype(float) == temp]
                else:
                    if not any(ts in fname.lower() for ts in [f"t{temp}", f"_{temp}", f"temp_{temp}", f"temp{temp}"]):
                        df_temp = pd.DataFrame()

            if df_temp is None or df_temp.empty:
                continue

            # --- Define Output Paths ---
            raw_out = out_base / f"{Path(fname).stem}_split_{split_name}_t{temp}_all_cands.csv"
            agg_out = out_base / f"{Path(fname).stem}_split_{split_name}_t{temp}_pass@{current_pass_n}.csv"

            # -------------------------------------------------------------
            # CACHE CHECK - If output already exists, load and append!
            # -------------------------------------------------------------
            if agg_out.exists():
                print(f"\n--- Loading CACHED Evaluation: {fname} | Temp: {temp} | Pass@{current_pass_n} ---")
                pass_n_df = pd.read_csv(agg_out)

                ea_empty, ea_non, psjs_non = float('nan'), float('nan'), float('nan')
                if 'golden_is_empty' in pass_n_df.columns:
                    df_empty = pass_n_df[pass_n_df['golden_is_empty'] == True]
                    df_non = pass_n_df[pass_n_df['golden_is_empty'] == False]
                    ea_empty = df_empty['EA'].mean() if not df_empty.empty else 0.0
                    ea_non = df_non['EA'].mean() if not df_non.empty else 0.0
                    psjs_non = df_non['PSJS'].mean() if not df_non.empty else 0.0

                summary_results.append({
                    'File': fname,
                    'Temp': temp,
                    'Pass@N': current_pass_n,
                    'EX': pass_n_df['EX'].mean() if not pass_n_df.empty else 0.0,
                    'EA': pass_n_df['EA'].mean() if not pass_n_df.empty else 0.0,
                    'PSJS': pass_n_df['PSJS'].mean() if not pass_n_df.empty else 0.0,
                    'EA (Empty)': ea_empty,
                    'EA (Non-Empty)': ea_non,
                    'PSJS (Non-Empty)': psjs_non
                })

                print_single_report(pass_n_df, f"{fname} (Temp {temp}, Pass@{current_pass_n}) [CACHED]")
                continue

            print(f"\n--- Evaluating File: {fname} | Temp: {temp} | Pass@{current_pass_n} ---")

            raw_results_df = await run_evaluation(
                df_temp,
                connector,
                current_pass_n,
                args.timeout,
                args.concurrency,
                global_golden_cache
            )

            # Save Raw Results
            raw_results_df.to_csv(raw_out, index=False)
            print(f"  -> Saved all {len(raw_results_df)} raw candidates to {raw_out.name}")

            # Compute Pass@N Max Score Aggregation (evaluate_single_candidate always sets 'instance_id')
            group_key = "instance_id"
            pass_n_df = raw_results_df.groupby(group_key)[['EX', 'EA', 'PSJS']].max().reset_index()

            # Merge back DB alias & golden_is_empty flag
            meta_cols = [group_key, 'db_alias']
            if 'golden_is_empty' in raw_results_df.columns:
                meta_cols.append('golden_is_empty')

            meta_df = raw_results_df[meta_cols].drop_duplicates(subset=[group_key])
            pass_n_df = pass_n_df.merge(meta_df, on=group_key, how='left')

            pass_n_df.to_csv(agg_out, index=False)
            print(f"  -> Saved Pass@{current_pass_n} aggregated max results to {agg_out.name}")

            ea_empty, ea_non, psjs_non = float('nan'), float('nan'), float('nan')
            if 'golden_is_empty' in pass_n_df.columns:
                df_empty = pass_n_df[pass_n_df['golden_is_empty'] == True]
                df_non = pass_n_df[pass_n_df['golden_is_empty'] == False]
                ea_empty = df_empty['EA'].mean() if not df_empty.empty else 0.0
                ea_non = df_non['EA'].mean() if not df_non.empty else 0.0
                psjs_non = df_non['PSJS'].mean() if not df_non.empty else 0.0

            summary_results.append({
                'File': fname,
                'Temp': temp,
                'Pass@N': current_pass_n,
                'EX': pass_n_df['EX'].mean() if not pass_n_df.empty else 0.0,
                'EA': pass_n_df['EA'].mean() if not pass_n_df.empty else 0.0,
                'PSJS': pass_n_df['PSJS'].mean() if not pass_n_df.empty else 0.0,
                'EA (Empty)': ea_empty,
                'EA (Non-Empty)': ea_non,
                'PSJS (Non-Empty)': psjs_non
            })

            print_single_report(pass_n_df, f"{fname} (Temp {temp}, Pass@{current_pass_n})")

    await connector.close()

    # 4. Global table summary across parameters
    if summary_results:
        print_iteration_comparison(summary_results)
    else:
        print("\nNo matching temperature 0.0 or 0.6 data found in the provided files to summarize.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted Cypher queries with Pass@N logic.")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to a single CSV file, or a directory containing prediction CSVs.")
    parser.add_argument("--split", type=str, choices=['test', 'val', 'both'], required=True,
                        help="Split to compute over (test, val, both).")
    parser.add_argument("--pass_n", type=int, required=True,
                        help="The number of deduplicated queries to evaluate per instance (e.g., 5).")
    parser.add_argument("--registry", type=str, default="cluster_registry",
                        help="Path to the cluster registry directory (default: cluster_registry).")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent connections / semaphore limit (default: 20).")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Timeout in seconds for query execution (default: 30.0).")

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
