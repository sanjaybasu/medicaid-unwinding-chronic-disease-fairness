"""Aim 2 — Signal stress-test under unwinding-induced distribution shift.

For each window (pre-unwinding 2018-2022, unwinding 2023-Q2 to 2024-Q2,
post-unwinding 2024-Q3+) and each model in {Signal Stage 2, Signal Stage 1,
HHS-HCC v07, CDPS+Rx v7, standard cost-based}, compute calibration,
discrimination, fairness metrics by subgroup with bootstrap 95% CIs.

Pre-specified equity-persistence test: difference in (sensitivity-Black -
sensitivity-White) between pre-unwinding and post-unwinding windows.
Patel-Baum-Basu 2024 reported this difference reversed sign relative to a
standard cost-based model; this audit asks whether the reversal persists
under distribution shift.

Output: tables/aim2_distribution_shift.csv, tables/aim2_equity_persistence.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from config import DATA_CLEAN, TABLES, SEED, BOOTSTRAP_ITERATIONS, PRIMARY_SUBGROUPS_AIM2
from utils import calibration_intercept_slope, expected_calibration_error, predicted_to_actual_ratio

WINDOWS = ["pre_unwinding", "unwinding", "post_unwinding"]
MODELS = ["signal_non_emerg", "signal_all_cause", "hhs_hcc_v07", "cdps_rx_v7", "standard_cost_based"]


def load_scores_and_outcomes() -> pd.DataFrame | None:
    score_path = DATA_CLEAN / "waymark_signal_scores.parquet"
    outcome_path = DATA_CLEAN / "waymark_outcomes.parquet"
    if not (score_path.exists() and outcome_path.exists()):
        return None
    scores = pd.read_parquet(score_path)
    outcomes = pd.read_parquet(outcome_path)
    return scores.merge(outcomes, on=["member_id"], how="inner")


def metric_block(
    df: pd.DataFrame,
    model_col: str,
    outcome_col: str,
    subgroup_col: str,
    threshold_quantile: float = 0.90,
    seed: int = SEED,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for sg, sub in df.groupby(subgroup_col):
        if len(sub) < 30:
            continue
        a = sub[outcome_col].astype(float).values
        p = sub[model_col].astype(float).values
        if np.any(np.isnan(a)) or np.any(np.isnan(p)):
            continue
        threshold = np.quantile(p, threshold_quantile)
        pos = (p >= threshold).astype(int)
        true_pos = (a >= np.quantile(a, threshold_quantile)).astype(int)

        boot_idx = [rng.integers(0, len(sub), len(sub)) for _ in range(BOOTSTRAP_ITERATIONS)]
        sens = []
        spec = []
        ratios = []
        ints = []
        slopes = []
        eces = []
        aurocs = []
        auprcs = []
        for idx in boot_idx:
            tp = ((pos[idx] == 1) & (true_pos[idx] == 1)).sum()
            fn = ((pos[idx] == 0) & (true_pos[idx] == 1)).sum()
            tn = ((pos[idx] == 0) & (true_pos[idx] == 0)).sum()
            fp = ((pos[idx] == 1) & (true_pos[idx] == 0)).sum()
            sens.append(tp / max(tp + fn, 1))
            spec.append(tn / max(tn + fp, 1))
            ratios.append(predicted_to_actual_ratio(a[idx], p[idx]))
            i, s = calibration_intercept_slope(a[idx], p[idx])
            ints.append(i)
            slopes.append(s)
            eces.append(expected_calibration_error(a[idx], p[idx]))
            try:
                aurocs.append(roc_auc_score(true_pos[idx], p[idx]))
            except Exception:
                aurocs.append(np.nan)
            try:
                auprcs.append(average_precision_score(true_pos[idx], p[idx]))
            except Exception:
                auprcs.append(np.nan)

        rows.append({
            "subgroup": sg,
            "n": int(len(sub)),
            "sensitivity": float(np.nanmean(sens)),
            "sensitivity_lower": float(np.nanquantile(sens, 0.025)),
            "sensitivity_upper": float(np.nanquantile(sens, 0.975)),
            "specificity": float(np.nanmean(spec)),
            "specificity_lower": float(np.nanquantile(spec, 0.025)),
            "specificity_upper": float(np.nanquantile(spec, 0.975)),
            "predicted_to_actual_ratio": float(np.nanmean(ratios)),
            "calibration_intercept": float(np.nanmean(ints)),
            "calibration_slope": float(np.nanmean(slopes)),
            "ece": float(np.nanmean(eces)),
            "auroc": float(np.nanmean(aurocs)),
            "auprc": float(np.nanmean(auprcs)),
        })
    return pd.DataFrame(rows)


def equity_persistence_test(metrics_df: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    sub = metrics_df[metrics_df["model"] == model]
    pre = sub[sub["window"] == "pre_unwinding"]
    post = sub[sub["window"] == "post_unwinding"]
    if pre.empty or post.empty:
        return pd.DataFrame()
    for race_pair in [("nh_black", "nh_white"), ("hispanic", "nh_white"), ("nh_aian", "nh_white")]:
        a, b = race_pair
        try:
            pre_a = pre[pre["subgroup"] == a]["sensitivity"].iloc[0]
            pre_b = pre[pre["subgroup"] == b]["sensitivity"].iloc[0]
            post_a = post[post["subgroup"] == a]["sensitivity"].iloc[0]
            post_b = post[post["subgroup"] == b]["sensitivity"].iloc[0]
        except (IndexError, KeyError):
            continue
        diff_pre = pre_a - pre_b
        diff_post = post_a - post_b
        rows.append({
            "model": model,
            "race_pair": f"{a}_vs_{b}",
            "sensitivity_diff_pre": float(diff_pre),
            "sensitivity_diff_post": float(diff_post),
            "delta": float(diff_post - diff_pre),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_scores_and_outcomes()
    if df is None:
        print("Aim 2: Waymark scores or outcomes missing; skipping.", file=sys.stderr)
        return

    metric_rows = []
    for window in WINDOWS:
        sub_w = df[df["window"] == window]
        if sub_w.empty:
            continue
        for model in MODELS:
            if model not in sub_w.columns:
                continue
            for sg_col, outcome_col in [
                ("race_ethnicity", "ed_non_emergent_flag"),
                ("primary_language", "ed_non_emergent_flag"),
                ("urbanicity", "ed_non_emergent_flag"),
            ]:
                if sg_col not in sub_w.columns or outcome_col not in sub_w.columns:
                    continue
                block = metric_block(sub_w, model, outcome_col, sg_col)
                block["window"] = window
                block["model"] = model
                block["subgroup_var"] = sg_col
                metric_rows.append(block)
    if not metric_rows:
        print("Aim 2: no metric blocks computed.", file=sys.stderr)
        return
    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(TABLES / "aim2_distribution_shift.csv", index=False)

    persistence_frames = []
    for model in MODELS:
        df_eq = equity_persistence_test(metrics, model)
        if not df_eq.empty:
            persistence_frames.append(df_eq)
    if persistence_frames:
        pd.concat(persistence_frames, ignore_index=True).to_csv(
            TABLES / "aim2_equity_persistence.csv", index=False
        )
    print(f"Wrote {TABLES / 'aim2_distribution_shift.csv'}")


if __name__ == "__main__":
    main()
