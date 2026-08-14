import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter
import warnings

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, kendalltau

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# UTILITIES & CATEGORIZATION
# ============================================================================
def print_header(text: str, width: int = 145):
    print(f"\n{'=' * width}\n {text.center(width - 2)}\n{'=' * width}")


def categorize_mistake(true_label, pred_label) -> str:
    """Categorizes the mistake based on distance groups and severity directions."""
    if pd.isna(true_label) or true_label == -1:
        return "Invalid GT"
    if pd.isna(pred_label) or pred_label == -1:
        return "Format Fail"
    
    try:
        t_val = int(float(true_label))
        p_val = int(float(pred_label))
    except ValueError:
        return "Format Fail"

    if t_val == p_val:
        return "Correct"
        
    direction = "Under" if p_val > t_val else "Over"
    small_sets = {1: {1, 2}, 2: {1, 2, 3}, 3: {2, 3}, 4: {4, 5}, 5: {4, 5}}
    medium_sets = {1: {1, 2, 3}, 2: {1, 2, 3}, 3: {1, 2, 3}, 4: {3, 4, 5}, 5: {3, 4, 5}}
    
    if p_val in small_sets.get(t_val, set()):
        size = "Small"
    elif p_val in medium_sets.get(t_val, set()):
        size = "Medium"
    else:
        size = "Large"
        
    return f"{size}-{direction}"


def calculate_ensemble_mean(parsed_json_str) -> float:
    """Extracts the continuous mean score from the JSON-stringified ensemble results."""
    try:
        if pd.isna(parsed_json_str):
            return -1.0
        scores = json.loads(parsed_json_str)
        valid_scores = [s for s in scores if s != -1 and s is not None]
        if not valid_scores:
            return -1.0
        return sum(valid_scores) / len(valid_scores)
    except Exception:
        return -1.0


# ============================================================================
# METRICS ENGINE
# ============================================================================
class MetricsEngine:
    @staticmethod
    def compute_judge_metrics(true_scores: np.ndarray, pred_scores: np.ndarray, prefix: str = "") -> dict:
        total = len(true_scores)
        
        valid_mask = (pred_scores != -1) & (~pd.isna(pred_scores)) & (~pd.isna(true_scores))
        
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

        y_t = np.round(valid_trues).astype(int)
        y_p = np.round(valid_preds).astype(int)
        qwk = cohen_kappa_score(y_t, y_p, weights='quadratic', labels=[1, 2, 3, 4, 5])

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

    @staticmethod
    def compute_binary_metrics(true_scores: np.ndarray, pred_scores: np.ndarray, prefix: str = "") -> dict:
        """Computes binary classification metrics (1-3: Fail/0, 4-5: Pass/1)"""
        valid_mask = (pred_scores != -1) & (~pd.isna(pred_scores)) & (~pd.isna(true_scores))
        valid_preds = pred_scores[valid_mask]
        valid_trues = true_scores[valid_mask]

        if len(valid_preds) == 0:
            return {f"{prefix}Bin_Acc(%)": float('nan'), f"{prefix}Bin_Prec(%)": float('nan'), 
                    f"{prefix}Bin_Rec(%)": float('nan'), f"{prefix}Bin_F1(%)": float('nan')}

        y_t = (valid_trues >= 4).astype(int)
        y_p = (valid_preds >= 4).astype(int)

        tp = np.sum((y_t == 1) & (y_p == 1))
        fp = np.sum((y_t == 0) & (y_p == 1))
        fn = np.sum((y_t == 1) & (y_p == 0))
        tn = np.sum((y_t == 0) & (y_p == 0))

        acc = (tp + tn) / len(y_t) * 100
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            f"{prefix}Bin_Acc(%)": acc,
            f"{prefix}Bin_Prec(%)": prec * 100,
            f"{prefix}Bin_Rec(%)": rec * 100,
            f"{prefix}Bin_F1(%)": f1 * 100
        }

    @staticmethod
    def print_judge_report(df_judge_subset: pd.DataFrame, iteration: int, errors_ens: Counter, errors_greedy: Counter) -> dict:
        true_scores = pd.to_numeric(df_judge_subset['ground_truth_score'], errors='coerce').astype(float).values
        greedy_scores = pd.to_numeric(df_judge_subset['greedy_parsed'], errors='coerce').astype(float).values
        ensemble_scores = pd.to_numeric(df_judge_subset['ensemble_mode'], errors='coerce').astype(float).values
        ensemble_mean = pd.to_numeric(df_judge_subset['ensemble_mean'], errors='coerce').astype(float).values

        # Standard Metrics
        greedy_metrics = MetricsEngine.compute_judge_metrics(true_scores, greedy_scores, prefix="Greedy_")
        ens_metrics = MetricsEngine.compute_judge_metrics(true_scores, ensemble_scores, prefix="Ens_")
        
        # Binary Metrics
        greedy_bin = MetricsEngine.compute_binary_metrics(true_scores, greedy_scores, prefix="Greedy_")
        ens_bin = MetricsEngine.compute_binary_metrics(true_scores, ensemble_scores, prefix="Ens_")

        def safe_qwk(y_true, y_pred):
            mask = (y_true != -1) & (y_pred != -1) & (~pd.isna(y_true)) & (~pd.isna(y_pred))
            if not mask.any() or sum(mask) < 2:
                return float('nan')
            y_t = np.round(y_true[mask]).astype(int)
            y_p = np.round(y_pred[mask]).astype(int)
            return cohen_kappa_score(y_t, y_p, weights='quadratic', labels=[1, 2, 3, 4, 5])

        def safe_corr(y_true, y_pred, corr_type='spearman'):
            mask = (y_true != -1) & (y_pred != -1) & (~pd.isna(y_true)) & (~pd.isna(y_pred))
            if not mask.any() or sum(mask) < 2:
                return float('nan')
            if corr_type == 'spearman':
                res = spearmanr(y_true[mask], y_pred[mask])
                return res.correlation if not pd.isna(res.correlation) else float('nan')
            elif corr_type == 'kendall':
                res = kendalltau(y_true[mask], y_pred[mask])
                return res.correlation if not pd.isna(res.correlation) else float('nan')
            return float('nan')

        # Baseline Correlations
        gt_mean_qwk = safe_qwk(true_scores, ensemble_mean)
        gt_mean_spr = safe_corr(true_scores, ensemble_mean, corr_type='spearman')
        gt_mean_ken = safe_corr(true_scores, ensemble_mean, corr_type='kendall')

        greedy_spr = safe_corr(true_scores, greedy_scores, corr_type='spearman')
        greedy_ken = safe_corr(true_scores, greedy_scores, corr_type='kendall')

        ens_var = pd.to_numeric(df_judge_subset['ensemble_variance'], errors='coerce').mean()

        stats = {
            "Iter": iteration,
            **greedy_metrics,
            **ens_metrics,
            **greedy_bin,
            **ens_bin,
            "Ensemble_StdDev": round(ens_var, 4) if not pd.isna(ens_var) else 0.0,
            "GTMean_QWK": round(gt_mean_qwk, 3) if not pd.isna(gt_mean_qwk) else "N/A",
            "GTMean_Spr": round(gt_mean_spr, 3) if not pd.isna(gt_mean_spr) else "N/A",
            "GTMean_Ken": round(gt_mean_ken, 3) if not pd.isna(gt_mean_ken) else "N/A",
            "Greedy_Spr": round(greedy_spr, 3) if not pd.isna(greedy_spr) else "N/A",
            "Greedy_Ken": round(greedy_ken, 3) if not pd.isna(greedy_ken) else "N/A",
        }
        
        for k, v in errors_ens.items():
            stats[f"Ens_{k}"] = v
        for k, v in errors_greedy.items():
            stats[f"Greedy_{k}"] = v

        # --- Inter-Ensemble Subsets Analysis ---
        set_letters = [chr(97+i) for i in range(9)]
        set_metrics = []
        set_bin_metrics = []
        sets_arrays = []
        
        for letter in set_letters:
            col = f'set_{letter}_mode'
            if col in df_judge_subset.columns:
                s_arr = pd.to_numeric(df_judge_subset[col], errors='coerce').astype(float).values
                sets_arrays.append(s_arr)
                
                # Standard
                m = MetricsEngine.compute_judge_metrics(true_scores, s_arr, prefix="")
                m['QWK'] = safe_qwk(true_scores, s_arr)
                m['Spr'] = safe_corr(true_scores, s_arr, 'spearman')
                m['Ken'] = safe_corr(true_scores, s_arr, 'kendall')
                set_metrics.append(m)
                
                # Binary
                b_m = MetricsEngine.compute_binary_metrics(true_scores, s_arr, prefix="")
                set_bin_metrics.append(b_m)

        # 1. Aggregates for Standard Metrics
        metrics_to_agg = ['Acc(%)', 'MAE', 'MeanBias', 'QWK', 'Spr', 'Ken']
        for k in metrics_to_agg:
            vals = [x.get(k, float('nan')) for x in set_metrics]
            vals = [v for v in vals if v is not None and not pd.isna(v) and v != "N/A"]
            if vals:
                stats[f"Set_{k}_Mean"] = np.mean(vals)
                stats[f"Set_{k}_Min"] = np.min(vals)
                stats[f"Set_{k}_Max"] = np.max(vals)
                stats[f"Set_{k}_Std"] = np.std(vals)
            else:
                stats[f"Set_{k}_Mean"] = stats[f"Set_{k}_Min"] = stats[f"Set_{k}_Max"] = stats[f"Set_{k}_Std"] = "N/A"
                
        # 2. Aggregates for Binary Metrics
        bin_metrics_to_agg = ['Bin_Acc(%)', 'Bin_F1(%)', 'Bin_Prec(%)', 'Bin_Rec(%)']
        for k in bin_metrics_to_agg:
            vals = [x.get(k, float('nan')) for x in set_bin_metrics]
            vals = [v for v in vals if v is not None and not pd.isna(v)]
            if vals:
                stats[f"Set_{k}_Mean"] = np.mean(vals)
                stats[f"Set_{k}_Min"] = np.min(vals)
                stats[f"Set_{k}_Max"] = np.max(vals)
                stats[f"Set_{k}_Std"] = np.std(vals)
            else:
                stats[f"Set_{k}_Mean"] = stats[f"Set_{k}_Min"] = stats[f"Set_{k}_Max"] = stats[f"Set_{k}_Std"] = "N/A"

        # 3. Calculate Pairwise Agreement/Correlations between Sets (The 36 Pairs)
        p_agree, p_qwk, p_spr, p_ken = [], [], [], []
        p_bin_agree, p_bin_f1 = [], []
        
        for i in range(len(sets_arrays)):
            for j in range(i + 1, len(sets_arrays)):
                s1, s2 = sets_arrays[i], sets_arrays[j]
                v_m = (s1 != -1) & (~pd.isna(s1)) & (s2 != -1) & (~pd.isna(s2))
                if v_m.any():
                    # Standard Agreement
                    p_agree.append((s1[v_m] == s2[v_m]).mean() * 100)
                    
                    # Binary Agreement & F1
                    b1 = (s1[v_m] >= 4).astype(int)
                    b2 = (s2[v_m] >= 4).astype(int)
                    p_bin_agree.append((b1 == b2).mean() * 100)
                    
                    # Pairwise F1 is completely symmetric. 2*TP / (2*TP + FP + FN)
                    tp = np.sum((b1 == 1) & (b2 == 1))
                    fp = np.sum((b1 == 0) & (b2 == 1))
                    fn = np.sum((b1 == 1) & (b2 == 0))
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                    p_bin_f1.append(f1 * 100)
                
                q = safe_qwk(s1, s2)
                if not pd.isna(q): p_qwk.append(q)
                s = safe_corr(s1, s2, 'spearman')
                if not pd.isna(s): p_spr.append(s)
                k = safe_corr(s1, s2, 'kendall')
                if not pd.isna(k): p_ken.append(k)

        stats["Pairwise_Agree"] = np.mean(p_agree) if p_agree else "N/A"
        stats["Pairwise_QWK"] = np.mean(p_qwk) if p_qwk else "N/A"
        stats["Pairwise_Spr"] = np.mean(p_spr) if p_spr else "N/A"
        stats["Pairwise_Ken"] = np.mean(p_ken) if p_ken else "N/A"
        
        stats["Pairwise_Bin_Agree"] = np.mean(p_bin_agree) if p_bin_agree else "N/A"
        stats["Pairwise_Bin_F1"] = np.mean(p_bin_f1) if p_bin_f1 else "N/A"

        def f_s(val): return f"{val:.3f}" if isinstance(val, float) else val

        print_header(f"JUDGE EVALUATION RESULTS: ITERATION {iteration}", 95)
        print(f"{'Metric':<25} | {'Greedy (T=0.0)':<25} | {'Ensemble Mode (T=0.6)':<25}")
        print("-" * 85)
        print(f"{'Total Samples':<25} | {stats['Greedy_Total']:<25} | {stats['Ens_Total']:<25}")
        print(f"{'Format Fails':<25} | {stats['Greedy_Fails']} ({stats['Greedy_Fail(%)']}%)  {'':<10}| {stats['Ens_Fails']} ({stats['Ens_Fail(%)']}%)")
        print(f"{'Accuracy (E.M.)':<25} | {stats['Greedy_Acc(%)']}%{'':<15} | {stats['Ens_Acc(%)']}%")
        print(f"{'Mean Abs. Error':<25} | {stats['Greedy_MAE']:<25.3f} | {stats['Ens_MAE']:<25.3f}")
        print(f"{'Mean Bias':<25} | {stats['Greedy_MeanBias']:<25.3f} | {stats['Ens_MeanBias']:<25.3f}")
        print(f"{'QWK (Mode/Greedy)':<25} | {stats['Greedy_QWK']:<25} | {stats['Ens_QWK']:<25}")
        print("-" * 85)
        print("   --- Binary Classification Metrics (1-3=Fail, 4-5=Pass) ---")
        print(f"{'Binary Accuracy':<25} | {stats['Greedy_Bin_Acc(%)']:<25.1f} | {stats['Ens_Bin_Acc(%)']:<25.1f}")
        print(f"{'Binary Precision':<25} | {stats['Greedy_Bin_Prec(%)']:<25.1f} | {stats['Ens_Bin_Prec(%)']:<25.1f}")
        print(f"{'Binary Recall':<25} | {stats['Greedy_Bin_Rec(%)']:<25.1f} | {stats['Ens_Bin_Rec(%)']:<25.1f}")
        print(f"{'Binary F1 Score':<25} | {stats['Greedy_Bin_F1(%)']:<25.1f} | {stats['Ens_Bin_F1(%)']:<25.1f}")
        print("-" * 85)
        print("   --- Advanced Correlational Metrics ---")
        print(f"{'GT vs Ens Mean QWK':<25} | {'N/A':<25} | {stats['GTMean_QWK']:<25}")
        print(f"{'GT vs Mode/Greedy Spr':<25} | {stats['Greedy_Spr']:<25} | {stats['GTMean_Spr']:<25}")
        print(f"{'GT vs Mode/Greedy Ken':<25} | {stats['Greedy_Ken']:<25} | {stats['GTMean_Ken']:<25}")
        print("-" * 85)
        print("   --- Inter-Ensemble Subsets Analysis (9 Chunks) ---")
        print(f"   Pairwise Standard Agree: {f_s(stats['Pairwise_Agree'])}%   | Pairwise Standard QWK: {f_s(stats['Pairwise_QWK'])}")
        print(f"   Pairwise Binary Agree  : {f_s(stats['Pairwise_Bin_Agree'])}%   | Pairwise Binary F1   : {f_s(stats['Pairwise_Bin_F1'])}")
        print(f"   Set Acc(%) Mean±Std    : {f_s(stats['Set_Acc(%)_Mean'])} ± {f_s(stats['Set_Acc(%)_Std'])} (Min: {f_s(stats['Set_Acc(%)_Min'])}, Max: {f_s(stats['Set_Acc(%)_Max'])})")
        print(f"   Set QWK Mean±Std       : {f_s(stats['Set_QWK_Mean'])} ± {f_s(stats['Set_QWK_Std'])} (Min: {f_s(stats['Set_QWK_Min'])}, Max: {f_s(stats['Set_QWK_Max'])})")
        print(f"   Set Bin F1 Mean±Std    : {f_s(stats['Set_Bin_F1(%)_Mean'])} ± {f_s(stats['Set_Bin_F1(%)_Std'])} (Min: {f_s(stats['Set_Bin_F1(%)_Min'])}, Max: {f_s(stats['Set_Bin_F1(%)_Max'])})")
        print("-" * 85)
        print(f"Ensemble Mean StdDev      : {stats['Ensemble_StdDev']}")
        
        print("\n   --- Detailed Error Distributions ---")
        print(f"   [ENSEMBLE] Small Over:  {errors_ens.get('Small-Over', 0):<5} | Small Under:  {errors_ens.get('Small-Under', 0):<5}")
        print(f"   [ENSEMBLE] Medium Over: {errors_ens.get('Medium-Over', 0):<5} | Medium Under: {errors_ens.get('Medium-Under', 0):<5}")
        print(f"   [ENSEMBLE] Large Over:  {errors_ens.get('Large-Over', 0):<5} | Large Under:  {errors_ens.get('Large-Under', 0):<5}")
        print("   " + "-"*50)
        print(f"   [GREEDY]   Small Over:  {errors_greedy.get('Small-Over', 0):<5} | Small Under:  {errors_greedy.get('Small-Under', 0):<5}")
        print(f"   [GREEDY]   Medium Over: {errors_greedy.get('Medium-Over', 0):<5} | Medium Under: {errors_greedy.get('Medium-Under', 0):<5}")
        print(f"   [GREEDY]   Large Over:  {errors_greedy.get('Large-Over', 0):<5} | Large Under:  {errors_greedy.get('Large-Under', 0):<5}")

        return stats


def main():
    parser = argparse.ArgumentParser(description="Standalone Metrics Calculator for Unified Evaluator Output")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing iter_X_unified_results.csv files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Directory {results_dir} does not exist.")
        return

    csv_files = []
    for f in results_dir.iterdir():
        if f.is_file() and f.name.endswith("_unified_results.csv"):
            match = re.search(r"iter_(\d+)_", f.name)
            if match:
                csv_files.append((int(match.group(1)), f))
            elif f.name.startswith("base_model_"): 
                csv_files.append((0, f))
    
    if not csv_files:
        print(f"[WARNING] No iter_X_unified_results.csv files found in {results_dir}")
        return

    csv_files.sort(key=lambda x: x[0])
    cross_iteration_stats = []

    print(f"[INFO] Found {len(csv_files)} iteration files. Computing advanced metrics...")

    for iteration, filepath in csv_files:
        df = pd.read_csv(filepath)
        
        df_judge = df[df['split'] == 'judge_val'].copy()
        if df_judge.empty:
            print(f"[WARNING] No judge_val split found in {filepath}. Skipping.")
            continue
            
        df_judge['ensemble_mean'] = df_judge['ensemble_parsed'].apply(calculate_ensemble_mean)

        errors_ens = Counter()
        errors_greedy = Counter()
        
        for _, row in df_judge.iterrows():
            gt = row.get('ground_truth_score', -1)
            mode = row.get('ensemble_mode', -1)
            greedy = row.get('greedy_parsed', -1)
            
            if pd.notnull(gt) and gt != -1:
                errors_ens[categorize_mistake(gt, mode)] += 1
                errors_greedy[categorize_mistake(gt, greedy)] += 1

        stats = MetricsEngine.print_judge_report(df_judge, iteration, errors_ens, errors_greedy)
        cross_iteration_stats.append(stats)

    # -----------------------------------------------------------------------------------------------------------------
    # GLOBAL COMPARISON SUMMARIES
    # -----------------------------------------------------------------------------------------------------------------
    def fmt(val):
        if pd.isna(val) or val == "N/A": return "N/A"
        return f"{float(val):.3f}"

    def fmt_agg(mn, sd, mi, mx, is_pct=False):
        if pd.isna(mn) or mn == "N/A": return "N/A"
        if is_pct:
            return f"{mn:5.1f}±{sd:.1f} [{mi:.1f}-{mx:.1f}]"
        return f"{mn:5.3f}±{sd:.3f} [{mi:.3f}-{mx:.3f}]"

    # Global Comparison Summary - ENSEMBLE STANDARD
    print_header("ENSEMBLE ITERATION PERFORMANCE COMPARISON SUMMARY (STANDARD)", 120)
    header_ens = (
        f"| {'Iter':<4} | {'Acc(%)':<6} | {'MAE':<5} | {'Bias':<5} | "
        f"{'QWK(Md)':<7} | {'QWK(Mn)':<7} | {'Spr(Mn)':<7} | {'Ken(Mn)':<7} | "
        f"{'S-Ovr':<5} | {'S-Und':<5} | {'M-Ovr':<5} | {'M-Und':<5} | {'L-Ovr':<5} | {'L-Und':<5} |"
    )
    print(header_ens)
    print("-" * 120)
    for res in cross_iteration_stats:
        it = res["Iter"]
        acc = res["Ens_Acc(%)"]
        mae = res["Ens_MAE"]
        bias = res["Ens_MeanBias"]
        
        qwk_md = res.get("Ens_QWK", "N/A")
        qwk_mn = res.get("GTMean_QWK", "N/A")
        spr_mn = res.get("GTMean_Spr", "N/A")
        ken_mn = res.get("GTMean_Ken", "N/A")
        
        row_str = (
            f"| {it:<4} | {acc:>5.1f}% | {mae:>5.3f} | {bias:>5.2f} | "
            f"{fmt(qwk_md):>7} | {fmt(qwk_mn):>7} | {fmt(spr_mn):>7} | {fmt(ken_mn):>7} | "
            f"{res.get('Ens_Small-Over', 0):>5} | {res.get('Ens_Small-Under', 0):>5} | "
            f"{res.get('Ens_Medium-Over', 0):>5} | {res.get('Ens_Medium-Under', 0):>5} | "
            f"{res.get('Ens_Large-Over', 0):>5} | {res.get('Ens_Large-Under', 0):>5} |"
        )
        print(row_str)

    # Global Comparison Summary - GREEDY STANDARD
    print_header("GREEDY ITERATION PERFORMANCE COMPARISON SUMMARY (STANDARD)", 108)
    header_greedy = (
        f"| {'Iter':<4} | {'Acc(%)':<6} | {'MAE':<5} | {'Bias':<5} | "
        f"{'QWK':<7} | {'Spr':<7} | {'Ken':<7} | "
        f"{'S-Ovr':<5} | {'S-Und':<5} | {'M-Ovr':<5} | {'M-Und':<5} | {'L-Ovr':<5} | {'L-Und':<5} |"
    )
    print(header_greedy)
    print("-" * 108)
    for res in cross_iteration_stats:
        it = res["Iter"]
        acc = res["Greedy_Acc(%)"]
        mae = res["Greedy_MAE"]
        bias = res["Greedy_MeanBias"]
        qwk = res.get("Greedy_QWK", "N/A")
        spr = res.get("Greedy_Spr", "N/A")
        ken = res.get("Greedy_Ken", "N/A")
        
        row_str = (
            f"| {it:<4} | {acc:>5.1f}% | {mae:>5.3f} | {bias:>5.2f} | "
            f"{fmt(qwk):>7} | {fmt(spr):>7} | {fmt(ken):>7} | "
            f"{res.get('Greedy_Small-Over', 0):>5} | {res.get('Greedy_Small-Under', 0):>5} | "
            f"{res.get('Greedy_Medium-Over', 0):>5} | {res.get('Greedy_Medium-Under', 0):>5} | "
            f"{res.get('Greedy_Large-Over', 0):>5} | {res.get('Greedy_Large-Under', 0):>5} |"
        )
        print(row_str)

    # Global Comparison Summary - INTER-ENSEMBLE STANDARD
    print_header("INTER-ENSEMBLE METRICS SUMMARY (STANDARD Pairwise & Sets Aggregates)", 145)
    header_ie = (
        f"| {'Iter':<4} | {'PW-Agree(%)':<11} | {'PW-QWK':<6} | {'PW-Spr':<6} | {'PW-Ken':<6} | "
        f"{'Set Acc(%) Mean±Std [Min-Max]':<31} | {'Set QWK Mean±Std [Min-Max]':<29} | {'Set MAE Mean±Std [Min-Max]':<29} |"
    )
    print(header_ie)
    print("-" * 145)
    for res in cross_iteration_stats:
        it = res["Iter"]
        p_agr = res.get("Pairwise_Agree", "N/A")
        p_qwk = res.get("Pairwise_QWK", "N/A")
        p_spr = res.get("Pairwise_Spr", "N/A")
        p_ken = res.get("Pairwise_Ken", "N/A")

        s_acc = fmt_agg(res.get("Set_Acc(%)_Mean"), res.get("Set_Acc(%)_Std"), res.get("Set_Acc(%)_Min"), res.get("Set_Acc(%)_Max"), True)
        s_qwk = fmt_agg(res.get("Set_QWK_Mean"), res.get("Set_QWK_Std"), res.get("Set_QWK_Min"), res.get("Set_QWK_Max"))
        s_mae = fmt_agg(res.get("Set_MAE_Mean"), res.get("Set_MAE_Std"), res.get("Set_MAE_Min"), res.get("Set_MAE_Max"))

        p_agr_str = f"{p_agr:>5.1f}%" if p_agr != "N/A" else "N/A"

        row_str = (
            f"| {it:<4} | {p_agr_str:<11} | {fmt(p_qwk):<6} | {fmt(p_spr):<6} | {fmt(p_ken):<6} | "
            f"{s_acc:<31} | {s_qwk:<29} | {s_mae:<29} |"
        )
        print(row_str)
        
    print("\n\n")

    # =================================================================================================================
    # BINARY CLASSIFICATION GLOBAL SUMMARIES
    # =================================================================================================================
    
    # ENSEMBLE BINARY
    print_header("ENSEMBLE BINARY CLASSIFICATION SUMMARY (1-3: Fail, 4-5: Pass)", 80)
    header_ens_bin = f"| {'Iter':<4} | {'Bin-Acc(%)':<12} | {'Bin-F1(%)':<12} | {'Bin-Prec(%)':<12} | {'Bin-Rec(%)':<12} |"
    print(header_ens_bin)
    print("-" * 80)
    for res in cross_iteration_stats:
        it = res["Iter"]
        b_acc = res.get("Ens_Bin_Acc(%)", "N/A")
        b_f1 = res.get("Ens_Bin_F1(%)", "N/A")
        b_pr = res.get("Ens_Bin_Prec(%)", "N/A")
        b_re = res.get("Ens_Bin_Rec(%)", "N/A")
        print(f"| {it:<4} | {fmt(b_acc):>11}% | {fmt(b_f1):>11}% | {fmt(b_pr):>11}% | {fmt(b_re):>11}% |")

    # GREEDY BINARY
    print_header("GREEDY BINARY CLASSIFICATION SUMMARY (1-3: Fail, 4-5: Pass)", 80)
    header_greedy_bin = f"| {'Iter':<4} | {'Bin-Acc(%)':<12} | {'Bin-F1(%)':<12} | {'Bin-Prec(%)':<12} | {'Bin-Rec(%)':<12} |"
    print(header_greedy_bin)
    print("-" * 80)
    for res in cross_iteration_stats:
        it = res["Iter"]
        b_acc = res.get("Greedy_Bin_Acc(%)", "N/A")
        b_f1 = res.get("Greedy_Bin_F1(%)", "N/A")
        b_pr = res.get("Greedy_Bin_Prec(%)", "N/A")
        b_re = res.get("Greedy_Bin_Rec(%)", "N/A")
        print(f"| {it:<4} | {fmt(b_acc):>11}% | {fmt(b_f1):>11}% | {fmt(b_pr):>11}% | {fmt(b_re):>11}% |")

    # INTER-ENSEMBLE BINARY
    print_header("INTER-ENSEMBLE BINARY SUMMARY (Pairwise & Sets Aggregates)", 125)
    header_ie_bin = (
        f"| {'Iter':<4} | {'PW-Bin-Agree':<14} | {'PW-Bin-F1':<11} | "
        f"{'Set Bin-Acc(%) Mean±Std [Min-Max]':<34} | {'Set Bin-F1(%) Mean±Std [Min-Max]':<34} |"
    )
    print(header_ie_bin)
    print("-" * 125)
    for res in cross_iteration_stats:
        it = res["Iter"]
        p_b_agr = res.get("Pairwise_Bin_Agree", "N/A")
        p_b_f1 = res.get("Pairwise_Bin_F1", "N/A")
        
        s_b_acc = fmt_agg(res.get("Set_Bin_Acc(%)_Mean"), res.get("Set_Bin_Acc(%)_Std"), res.get("Set_Bin_Acc(%)_Min"), res.get("Set_Bin_Acc(%)_Max"), True)
        s_b_f1 = fmt_agg(res.get("Set_Bin_F1(%)_Mean"), res.get("Set_Bin_F1(%)_Std"), res.get("Set_Bin_F1(%)_Min"), res.get("Set_Bin_F1(%)_Max"), True)

        p_b_agr_str = f"{p_b_agr:>5.1f}%" if p_b_agr != "N/A" else "N/A"
        p_b_f1_str = f"{p_b_f1:>5.1f}%" if p_b_f1 != "N/A" else "N/A"

        row_str = (
            f"| {it:<4} | {p_b_agr_str:<14} | {p_b_f1_str:<11} | "
            f"{s_b_acc:<34} | {s_b_f1:<34} |"
        )
        print(row_str)

    print("-" * 145)
    print("LEGEND: Md = Ensemble Mode. Mn = Ensemble Mean. PW = Pairwise (average across 36 permutations of the 9 sets).")
    print("S = Small, M = Medium, L = Large Mistake. Ovr = Over-moderation, Und = Under-moderation.")
    print("Over-moderation = Model gave a lower score than Truth. Under-moderation = Model gave a higher score than Truth.")

if __name__ == "__main__":
    main()
