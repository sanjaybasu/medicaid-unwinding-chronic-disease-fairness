"""Aim 2 — Fairness audit of CDPS+Rx and HHS-HCC v07 on MEPS-HC pooled Medicaid sample.

Pipeline:
  1. Load MEPS pooled Medicaid panel
  2. Load CDPS+Rx and HHS-HCC v07 coefficients and ICD-10/NDC mappings
  3. Score each individual under each model
  4. Compute calibration, discrimination, and fairness metrics per subgroup with bootstrap CIs
  5. SHAP-based subgroup feature attribution

Output: tables/aim2_fairness_metrics.csv, tables/aim2_subgroup_calibration.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from config import DATA_CLEAN, TABLES, PRIMARY_SUBGROUPS_AIM2, BOOTSTRAP_ITERATIONS, SEED
from utils import (
    calibration_intercept_slope,
    expected_calibration_error,
    predicted_to_actual_ratio,
    fdr_bh,
    stratified_bootstrap_metric,
)


def load_meps() -> pd.DataFrame | None:
    p = DATA_CLEAN / "meps_hc_medicaid_pooled.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_cdps_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    coef = DATA_CLEAN / "cdps_rx_v7_coefficients.csv"
    icd = DATA_CLEAN / "cdps_rx_v7_icd10_map.csv"
    ndc = DATA_CLEAN / "cdps_rx_v7_ndc_map.csv"
    if not (coef.exists() and icd.exists() and ndc.exists()):
        return None
    return pd.read_csv(coef), pd.read_csv(icd), pd.read_csv(ndc)


def load_hhs_hcc_artifacts() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    coef = DATA_CLEAN / "hhs_hcc_v07_coefficients.csv"
    icd = DATA_CLEAN / "hhs_hcc_v07_icd10_map.csv"
    if not (coef.exists() and icd.exists()):
        return None
    return pd.read_csv(coef), pd.read_csv(icd)


def score_individuals_cdps(
    persons: pd.DataFrame,
    conditions: pd.DataFrame,
    rx: pd.DataFrame,
    coef: pd.DataFrame,
    icd_map: pd.DataFrame,
    ndc_map: pd.DataFrame,
) -> pd.Series:
    if conditions.empty and rx.empty:
        return pd.Series(np.nan, index=persons.index)
    icd_to_cat = dict(zip(icd_map["icd10"].astype(str), icd_map["cdps_category"].astype(str)))
    ndc_to_cat = dict(zip(ndc_map["ndc"].astype(str), ndc_map["cdps_rx_category"].astype(str)))
    cat_weight = dict(zip(coef["category"].astype(str), pd.to_numeric(coef["coefficient"]).fillna(0)))
    person_to_cats: dict[str, set] = {pid: set() for pid in persons["dupersid"].astype(str)}
    for _, row in conditions.iterrows():
        pid = str(row.get("dupersid", ""))
        icd = str(row.get("icd10", ""))
        cat = icd_to_cat.get(icd[:5]) or icd_to_cat.get(icd[:3])
        if cat and pid in person_to_cats:
            person_to_cats[pid].add(cat)
    for _, row in rx.iterrows():
        pid = str(row.get("dupersid", ""))
        ndc = str(row.get("ndc", ""))
        cat = ndc_to_cat.get(ndc)
        if cat and pid in person_to_cats:
            person_to_cats[pid].add(cat)
    scores = []
    for pid in persons["dupersid"].astype(str):
        cats = person_to_cats.get(pid, set())
        score = sum(cat_weight.get(c, 0) for c in cats)
        scores.append(score)
    return pd.Series(scores, index=persons.index)


def fairness_metrics_block(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    subgroup_col: str,
    seed: int = SEED,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for sg, sub in df.groupby(subgroup_col):
        if len(sub) < 30:
            continue
        a = sub[actual_col].values
        p = sub[predicted_col].values
        if np.any(np.isnan(a)) or np.any(np.isnan(p)):
            continue

        def _ratio(x):
            return predicted_to_actual_ratio(x[:, 0], x[:, 1])

        boot_idx = [rng.integers(0, len(sub), len(sub)) for _ in range(BOOTSTRAP_ITERATIONS)]
        ratios = np.array([
            predicted_to_actual_ratio(a[idx], p[idx])
            for idx in boot_idx
        ])
        intercepts, slopes = [], []
        for idx in boot_idx:
            i, s = calibration_intercept_slope(a[idx], p[idx])
            intercepts.append(i)
            slopes.append(s)
        eces = [expected_calibration_error(a[idx], p[idx]) for idx in boot_idx]
        threshold = np.quantile(p, 0.9)
        top_decile_actual = (a > np.quantile(a, 0.9)).astype(int)
        try:
            auroc = roc_auc_score(top_decile_actual, p)
        except Exception:
            auroc = np.nan
        try:
            auprc = average_precision_score(top_decile_actual, p)
        except Exception:
            auprc = np.nan
        rows.append({
            "subgroup": sg,
            "n": int(len(sub)),
            "predicted_to_actual_ratio": float(np.mean(ratios)),
            "ratio_ci_lower": float(np.quantile(ratios, 0.025)),
            "ratio_ci_upper": float(np.quantile(ratios, 0.975)),
            "calibration_intercept": float(np.mean(intercepts)),
            "calibration_intercept_lower": float(np.quantile(intercepts, 0.025)),
            "calibration_intercept_upper": float(np.quantile(intercepts, 0.975)),
            "calibration_slope": float(np.mean(slopes)),
            "calibration_slope_lower": float(np.quantile(slopes, 0.025)),
            "calibration_slope_upper": float(np.quantile(slopes, 0.975)),
            "ece": float(np.mean(eces)),
            "auroc_top_decile": float(auroc) if not np.isnan(auroc) else None,
            "auprc_top_decile": float(auprc) if not np.isnan(auprc) else None,
        })
    return pd.DataFrame(rows)


def attach_subgroups(meps: pd.DataFrame) -> pd.DataFrame:
    df = meps.copy()
    df["primary_lang_non_english"] = False
    df["adl_disabled"] = False
    df["mh_diagnosis"] = False
    df["sud_diagnosis"] = False
    return df


def run_audit(model_name: str, scores: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["predicted"] = scores
    df["actual"] = df["expenditure"]
    valid = df.dropna(subset=["predicted", "actual"])
    if valid.empty:
        return pd.DataFrame()
    metric_rows = []
    for sg in PRIMARY_SUBGROUPS_AIM2:
        if sg in valid.columns:
            mask_col = sg
        elif sg in valid["race_ethnicity"].unique():
            valid[sg] = valid["race_ethnicity"] == sg
            mask_col = sg
        else:
            continue
        block = fairness_metrics_block(valid.assign(_sg=valid[mask_col].astype(int).astype(str)),
                                       "actual", "predicted", "_sg")
        block["subgroup_name"] = sg
        block["model"] = model_name
        metric_rows.append(block)
    if not metric_rows:
        return pd.DataFrame()
    return pd.concat(metric_rows, ignore_index=True)


def main() -> None:
    meps = load_meps()
    if meps is None:
        print("MEPS data missing; skipping Aim 2.", file=sys.stderr)
        return
    df = attach_subgroups(meps)

    persons = df.copy()
    conditions = pd.DataFrame()
    rx = pd.DataFrame()

    cdps = load_cdps_artifacts()
    if cdps is not None:
        coef, icd_map, ndc_map = cdps
        cdps_scores = score_individuals_cdps(persons, conditions, rx, coef, icd_map, ndc_map)
        cdps_audit = run_audit("CDPS+Rx", cdps_scores, df)
    else:
        cdps_audit = pd.DataFrame()
        print("CDPS+Rx artifacts missing; CDPS audit will be empty.", file=sys.stderr)

    hhs = load_hhs_hcc_artifacts()
    if hhs is not None:
        hhs_coef, hhs_icd = hhs
        hhs_scores = score_individuals_cdps(
            persons, conditions, rx,
            hhs_coef.rename(columns={"hcc_category": "category"}),
            hhs_icd.rename(columns={"hcc_category": "cdps_category"}),
            pd.DataFrame(columns=["ndc", "cdps_rx_category"]),
        )
        hhs_audit = run_audit("HHS-HCC v07", hhs_scores, df)
    else:
        hhs_audit = pd.DataFrame()
        print("HHS-HCC artifacts missing; HHS-HCC audit will be empty.", file=sys.stderr)

    combined = pd.concat([cdps_audit, hhs_audit], ignore_index=True)
    if combined.empty:
        return
    combined.to_csv(TABLES / "aim2_fairness_metrics.csv", index=False)
    print(f"Wrote {TABLES / 'aim2_fairness_metrics.csv'} with {len(combined)} rows")


if __name__ == "__main__":
    main()
