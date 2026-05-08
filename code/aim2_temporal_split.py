"""Aim 2 — Signal stress test with temporal train/eval split.

Methodologically more defensible than the random 70/30 split:
  - Train on baseline features observable as of 2023-Q1 with outcomes
    measured during the pre-unwinding window (2022-Q1 - 2023-Q1)
  - Evaluate on the same members with outcomes measured during the
    unwinding window (2023-Q2 - 2024-Q2)
  - Stratify the evaluation cohort into retained vs disenrolled

This is the published Patel-Baum-Basu 2024 architecture applied with a
true distribution-shift evaluation rather than within-window cross-validation.

Outputs:
  tables/aim2_signal_temporal_metrics.csv
  tables/aim2_temporal_equity_persistence.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path.home() / ".claude/skills/waymark-data-access/scripts"))
sys.path.insert(0, str(ROOT / "code"))

from wm_conn import coredb, query
from utils import calibration_intercept_slope, expected_calibration_error
from config import DATA_CLEAN, TABLES, SEED, BOOTSTRAP_ITERATIONS


def pull_baseline_outcomes(person_ids: list[str]) -> pd.DataFrame:
    eng = coredb("prod")
    rows = []
    chunk = 5000
    for i in range(0, len(person_ids), chunk):
        ids = person_ids[i:i + chunk]
        df = query(eng, """
          SELECT person_id,
                 COUNT(*) FILTER (WHERE ed_flag = 1) AS pre_ed_visits,
                 COUNT(*) FILTER (WHERE encounter_type ILIKE :inpat) AS pre_inpat_admits
          FROM dbt_tuva_core.encounter
          WHERE person_id = ANY(:ids)
            AND encounter_start_date BETWEEN :start AND :end
          GROUP BY person_id
        """, ids=ids, inpat="%inpat%", start="2022-04-01", end="2023-04-01")
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["person_id"])
    out = pd.concat(rows, ignore_index=True)
    out["pre_outcome_acute_any"] = ((out["pre_ed_visits"] + out["pre_inpat_admits"]) > 0).astype(int)
    return out


def build_features(panel: pd.DataFrame, additional: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = panel.merge(additional, on="person_id", how="left")
    for c in ["unique_ndcs_baseline", "rx_fills_baseline", "rx_days_baseline",
              "unique_icd3_baseline", "condition_records_baseline"]:
        df[c] = df.get(c, pd.Series([0]*len(df))).fillna(0).astype(float)
    feature_cols = [
        "x_age", "x_sex",
        "x_baseline_ed", "x_baseline_inpat", "x_baseline_total_enc",
        "x_chronic_count",
        "unique_ndcs_baseline", "rx_fills_baseline", "rx_days_baseline",
        "unique_icd3_baseline", "condition_records_baseline",
        "x_payer_encoded", "x_state_encoded",
    ]
    for c in ["hypertension","diabetes","asthma","copd","chf","depression","anxiety","sud"]:
        if c in df.columns:
            feature_cols.append(c)
            df[c] = df[c].astype(int)
    df["outcome_acute_any_post"] = ((df["ed_visits_12mo"] + df["inpat_admissions_12mo"]) > 0).astype(int)
    return df, feature_cols


def train_signal_xgb(X, y, seed=SEED):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, eval_metric="logloss", n_jobs=4,
    ).fit(X, y)


def evaluate_subgroup(df, feature_cols, model, outcome, subgroup_col="race_ethnicity", seed=SEED):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["pred_signal"] = model.predict_proba(df[feature_cols].values)[:, 1]
    df["pred_costbased"] = df["x_baseline_total_enc"] / max(df["x_baseline_total_enc"].max(), 1)
    rows = []
    for sg, sub in df.groupby(subgroup_col):
        if len(sub) < 50:
            continue
        for score_col in ("pred_signal", "pred_costbased"):
            y = sub[outcome].values
            p = sub[score_col].values
            if len(np.unique(y)) < 2:
                continue
            try: au = float(roc_auc_score(y, p))
            except Exception: au = np.nan
            try: ap = float(average_precision_score(y, p))
            except Exception: ap = np.nan
            thresh = float(np.quantile(p, 0.9))
            pos = (p >= thresh).astype(int)
            sens = float(((pos == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
            spec = float(((pos == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
            try: ic, sl = calibration_intercept_slope(y.astype(float), p.astype(float), log_link=False)
            except Exception: ic, sl = np.nan, np.nan
            ece = expected_calibration_error(y.astype(float), p.astype(float))
            sens_b = []
            for _ in range(BOOTSTRAP_ITERATIONS):
                idx = rng.integers(0, len(sub), len(sub))
                yi, pi = y[idx], p[idx]
                if (yi == 1).sum() == 0 or (yi == 0).sum() == 0:
                    continue
                ti = float(np.quantile(pi, 0.9))
                pi_pos = (pi >= ti).astype(int)
                sens_b.append(((pi_pos == 1) & (yi == 1)).sum() / max((yi == 1).sum(), 1))
            sens_lo = float(np.quantile(sens_b, 0.025)) if sens_b else np.nan
            sens_hi = float(np.quantile(sens_b, 0.975)) if sens_b else np.nan
            rows.append({
                "subgroup": sg, "n": int(len(sub)),
                "model": "signal_temporal" if score_col == "pred_signal" else "cost_based_temporal",
                "auroc": au, "auprc": ap,
                "sensitivity_top_decile": sens, "sensitivity_lo": sens_lo, "sensitivity_hi": sens_hi,
                "specificity_top_decile": spec,
                "calibration_intercept": ic, "calibration_slope": sl, "ece": ece,
            })
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_parquet(DATA_CLEAN / "waymark_individual_panel.parquet")
    additional = pd.read_parquet(DATA_CLEAN / "waymark_signal_features_additional.parquet")
    df, feature_cols = build_features(panel, additional)
    print(f"Cohort N: {len(df)}; features: {len(feature_cols)}")

    print("Pulling pre-unwinding (2022-Q2 - 2023-Q1) outcomes for training...")
    pre = pull_baseline_outcomes(df["person_id"].astype(str).tolist())
    df = df.merge(pre, on="person_id", how="left")
    df["pre_outcome_acute_any"] = df["pre_outcome_acute_any"].fillna(0).astype(int)
    print(f"Pre-unwinding outcome positivity: {df['pre_outcome_acute_any'].mean()*100:.2f}%")

    X = df[feature_cols].values
    y_pre = df["pre_outcome_acute_any"].values
    print("Training Signal XGBoost on pre-unwinding outcomes...")
    model = train_signal_xgb(X, y_pre)

    print("Evaluating on post-unwinding outcomes by subgroup...")
    df_retained = df[df["procedural_disenrolled"] == 0].copy()
    df_disenrolled = df[df["procedural_disenrolled"] == 1].copy()

    frames = []
    for label, frame in [("full_cohort", df), ("retained", df_retained), ("disenrolled", df_disenrolled)]:
        block = evaluate_subgroup(frame, feature_cols, model, "outcome_acute_any_post")
        block["window"] = label
        frames.append(block)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(TABLES / "aim2_signal_temporal_metrics.csv", index=False)
    print(f"Wrote {TABLES / 'aim2_signal_temporal_metrics.csv'} with {len(metrics)} rows")

    eq_rows = []
    for model_name in ("signal_temporal", "cost_based_temporal"):
        sub = metrics[metrics["model"] == model_name]
        try:
            full_b = sub[(sub["window"]=="full_cohort") & (sub["subgroup"]=="nh_black")]["sensitivity_top_decile"].iloc[0]
            full_w = sub[(sub["window"]=="full_cohort") & (sub["subgroup"]=="nh_white")]["sensitivity_top_decile"].iloc[0]
            ret_b = sub[(sub["window"]=="retained") & (sub["subgroup"]=="nh_black")]["sensitivity_top_decile"].iloc[0]
            ret_w = sub[(sub["window"]=="retained") & (sub["subgroup"]=="nh_white")]["sensitivity_top_decile"].iloc[0]
            dis_b = sub[(sub["window"]=="disenrolled") & (sub["subgroup"]=="nh_black")]["sensitivity_top_decile"].iloc[0]
            dis_w = sub[(sub["window"]=="disenrolled") & (sub["subgroup"]=="nh_white")]["sensitivity_top_decile"].iloc[0]
            eq_rows.append({
                "model": model_name,
                "sens_diff_full_black_minus_white": full_b - full_w,
                "sens_diff_retained_black_minus_white": ret_b - ret_w,
                "sens_diff_disenrolled_black_minus_white": dis_b - dis_w,
            })
        except (IndexError, KeyError):
            pass
    eq_df = pd.DataFrame(eq_rows)
    eq_df.to_csv(TABLES / "aim2_temporal_equity_persistence.csv", index=False)
    print("\nTemporal-split equity persistence test:")
    print(eq_df.to_string(index=False))


if __name__ == "__main__":
    main()
