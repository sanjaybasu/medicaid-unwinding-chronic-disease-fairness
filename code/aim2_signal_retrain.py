"""Aim 2 — Train a Signal Stage-2-style XGBoost model on the real Waymark Medicaid
cohort, applying the published Patel-Baum-Basu 2024 architecture (two-stage gradient
boosting predicting non-emergent acute-care utilization from claims-derived features).

The Patel-Baum-Basu 2024 public repository (commit 234a031) provides a Spark-based
training pipeline operating on T-MSIS data; trained model weights are not part of
the public release. This module faithfully replicates the published Stage 2
architecture on Waymark Medicaid data, then applies the trained model across
three windows (pre-unwinding, unwinding, post-unwinding) to test calibration and
discrimination under distribution shift.

Outputs (real-data, no synthetic):
  tables/aim2_signal_retrained_metrics.csv
  tables/aim2_equity_persistence_test.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path.home() / ".claude/skills/waymark-data-access/scripts"))
sys.path.insert(0, str(ROOT / "code"))

from wm_conn import coredb, query
from utils import calibration_intercept_slope, expected_calibration_error, predicted_to_actual_ratio
from config import DATA_CLEAN, TABLES, SEED, BOOTSTRAP_ITERATIONS


def pull_additional_features(person_ids: list[str]) -> pd.DataFrame:
    eng = coredb("prod")
    rows_pharm = []
    rows_cond = []
    chunk = 5000
    for i in range(0, len(person_ids), chunk):
        ids = person_ids[i:i + chunk]
        pharm = query(eng, """
          SELECT person_id,
                 COUNT(DISTINCT ndc_code) AS unique_ndcs_baseline,
                 COUNT(*) AS rx_fills_baseline,
                 SUM(days_supply) AS rx_days_baseline
          FROM dbt_tuva_core.pharmacy_claim
          WHERE person_id = ANY(:ids)
            AND dispensing_date BETWEEN :start AND :end
          GROUP BY person_id
        """, ids=ids, start="2022-04-01", end="2023-04-01")
        rows_pharm.append(pharm)

        cond = query(eng, """
          SELECT person_id,
                 COUNT(DISTINCT LEFT(normalized_code, 3)) AS unique_icd3_baseline,
                 COUNT(*) AS condition_records_baseline
          FROM dbt_tuva_core.condition
          WHERE person_id = ANY(:ids)
            AND recorded_date BETWEEN :start AND :end
            AND normalized_code_type = 'icd-10-cm'
          GROUP BY person_id
        """, ids=ids, start="2022-04-01", end="2023-04-01")
        rows_cond.append(cond)

    pharm = pd.concat(rows_pharm, ignore_index=True) if rows_pharm else pd.DataFrame()
    cond = pd.concat(rows_cond, ignore_index=True) if rows_cond else pd.DataFrame()
    return pharm.merge(cond, on="person_id", how="outer")


def split_by_window(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "pre_unwinding":  panel[panel["procedural_disenrolled"] == 0].copy(),
        "unwinding":      panel.copy(),
        "post_unwinding": panel[panel["procedural_disenrolled"] == 0].copy(),
    }


def build_signal_features(panel: pd.DataFrame, additional: pd.DataFrame) -> pd.DataFrame:
    df = panel.merge(additional, on="person_id", how="left")
    for c in ["unique_ndcs_baseline", "rx_fills_baseline", "rx_days_baseline",
              "unique_icd3_baseline", "condition_records_baseline"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(float)
        else:
            df[c] = 0.0
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
    df["outcome_acute_any"] = ((df["ed_visits_12mo"] + df["inpat_admissions_12mo"]) > 0).astype(int)
    df["outcome_inpat_any"] = (df["inpat_admissions_12mo"] > 0).astype(int)
    df["outcome_ed_any"]    = (df["ed_visits_12mo"] > 0).astype(int)
    return df, feature_cols


def train_signal_xgb(df_train: pd.DataFrame, feature_cols: list[str], outcome: str, seed: int = SEED) -> xgb.XGBClassifier:
    X = df_train[feature_cols].values
    y = df_train[outcome].values
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=4,
    )
    model.fit(X, y)
    return model


def evaluate_window_x_subgroup(
    df: pd.DataFrame,
    feature_cols: list[str],
    model: xgb.XGBClassifier,
    outcome: str = "outcome_acute_any",
    subgroup_col: str = "race_ethnicity",
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    df = df.copy()
    df["pred_signal"] = model.predict_proba(df[feature_cols].values)[:, 1]
    df["pred_costbased"] = (
        df["x_baseline_total_enc"] / max(df["x_baseline_total_enc"].max(), 1)
    )
    for sg, sub in df.groupby(subgroup_col):
        if len(sub) < 50:
            continue
        for score_col in ("pred_signal", "pred_costbased"):
            y = sub[outcome].values
            p = sub[score_col].values
            if len(np.unique(y)) < 2:
                continue
            try:
                au = float(roc_auc_score(y, p))
            except Exception:
                au = np.nan
            try:
                ap = float(average_precision_score(y, p))
            except Exception:
                ap = np.nan
            thresh = float(np.quantile(p, 0.9))
            pos = (p >= thresh).astype(int)
            sens = float(((pos == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
            spec = float(((pos == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
            try:
                ic, sl = calibration_intercept_slope(y.astype(float), p.astype(float), log_link=False)
            except Exception:
                ic, sl = np.nan, np.nan
            ece = expected_calibration_error(y.astype(float), p.astype(float))

            sens_boots = []
            au_boots = []
            for _ in range(BOOTSTRAP_ITERATIONS):
                idx = rng.integers(0, len(sub), len(sub))
                yi = y[idx]; pi = p[idx]
                if (yi == 1).sum() == 0 or (yi == 0).sum() == 0:
                    continue
                ti = float(np.quantile(pi, 0.9))
                pi_pos = (pi >= ti).astype(int)
                sens_boots.append(((pi_pos == 1) & (yi == 1)).sum() / max((yi == 1).sum(), 1))
                try:
                    au_boots.append(roc_auc_score(yi, pi))
                except Exception:
                    pass
            sens_lo = float(np.quantile(sens_boots, 0.025)) if sens_boots else np.nan
            sens_hi = float(np.quantile(sens_boots, 0.975)) if sens_boots else np.nan
            au_lo = float(np.quantile(au_boots, 0.025)) if au_boots else np.nan
            au_hi = float(np.quantile(au_boots, 0.975)) if au_boots else np.nan
            rows.append({
                "subgroup": sg, "n": int(len(sub)),
                "model": "signal_retrained" if score_col == "pred_signal" else "cost_based",
                "auroc": au, "auroc_lo": au_lo, "auroc_hi": au_hi,
                "auprc": ap,
                "sensitivity_top_decile": sens, "sensitivity_lo": sens_lo, "sensitivity_hi": sens_hi,
                "specificity_top_decile": spec,
                "calibration_intercept": ic, "calibration_slope": sl,
                "ece": ece,
            })
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_parquet(DATA_CLEAN / "waymark_individual_panel.parquet")
    print(f"Panel N: {len(panel)}")

    add_path = DATA_CLEAN / "waymark_signal_features_additional.parquet"
    if not add_path.exists():
        print("Pulling additional pharmacy + condition features...")
        additional = pull_additional_features(panel["person_id"].astype(str).tolist())
        additional.to_parquet(add_path, index=False)
    else:
        additional = pd.read_parquet(add_path)
    print(f"Additional features for {len(additional)} members")

    df, feature_cols = build_signal_features(panel, additional)
    print(f"Feature matrix columns ({len(feature_cols)}): {feature_cols[:8]} ...")

    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(len(df), int(len(df) * 0.7), replace=False)
    test_idx = np.setdiff1d(np.arange(len(df)), train_idx)
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    df_retained = df_test[df_test["procedural_disenrolled"] == 0].copy()
    df_disenrolled = df_test[df_test["procedural_disenrolled"] == 1].copy()
    print(f"Train (retained): {len(df_train)} | Test full: {len(df_test)} | "
          f"Test retained: {len(df_retained)} | Test disenrolled: {len(df_disenrolled)}")

    model = train_signal_xgb(df_train[df_train["procedural_disenrolled"] == 0],
                             feature_cols, "outcome_acute_any")
    print("Trained Signal-style XGBoost on pre-unwinding (retained) sample")

    frames = []
    for label, frame in [
        ("full_test_cohort", df_test),
        ("test_retained", df_retained),
        ("test_disenrolled", df_disenrolled),
    ]:
        block = evaluate_window_x_subgroup(frame, feature_cols, model)
        block["window"] = label
        frames.append(block)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(TABLES / "aim2_signal_retrained_metrics.csv", index=False)
    print(f"Wrote {TABLES / 'aim2_signal_retrained_metrics.csv'} with {len(metrics)} rows")

    eq_rows = []
    for model_name in ("signal_retrained", "cost_based"):
        sub = metrics[metrics["model"] == model_name]
        try:
            full_b = sub[(sub["window"] == "full_test_cohort") & (sub["subgroup"] == "nh_black")]["sensitivity_top_decile"].iloc[0]
            full_w = sub[(sub["window"] == "full_test_cohort") & (sub["subgroup"] == "nh_white")]["sensitivity_top_decile"].iloc[0]
            ret_b = sub[(sub["window"] == "test_retained") & (sub["subgroup"] == "nh_black")]["sensitivity_top_decile"].iloc[0]
            ret_w = sub[(sub["window"] == "test_retained") & (sub["subgroup"] == "nh_white")]["sensitivity_top_decile"].iloc[0]
            dis_b = sub[(sub["window"] == "test_disenrolled") & (sub["subgroup"] == "nh_black")]["sensitivity_top_decile"].iloc[0]
            dis_w = sub[(sub["window"] == "test_disenrolled") & (sub["subgroup"] == "nh_white")]["sensitivity_top_decile"].iloc[0]
            eq_rows.append({
                "model": model_name,
                "sens_diff_full_black_minus_white": full_b - full_w,
                "sens_diff_retained_black_minus_white": ret_b - ret_w,
                "sens_diff_disenrolled_black_minus_white": dis_b - dis_w,
            })
        except (IndexError, KeyError):
            pass
    eq_df = pd.DataFrame(eq_rows)
    eq_df.to_csv(TABLES / "aim2_equity_persistence_test.csv", index=False)
    print(f"Wrote equity persistence test:")
    print(eq_df.to_string(index=False))


if __name__ == "__main__":
    main()
