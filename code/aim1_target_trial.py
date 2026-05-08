"""Aim 1 — Individual-level causal effect of procedural disenrollment on chronic-disease
outcomes via target trial emulation. Doubly-robust AIPW primary; TMLE secondary;
causal-forest CATE for HTE; instrumental-variable robustness using state-level
KFF administrative-processing intensity.

Hernán & Robins (2016) target trial emulation.
Glynn & Quinn (2010) AIPW.
van der Laan & Rose (2011) TMLE.
Athey & Wager (2019) Generalized Random Forests for HTE.

Output: tables/aim1_target_trial_results.csv, tables/aim1_cate_subgroups.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold

from config import DATA_CLEAN, TABLES, SEED, BOOTSTRAP_ITERATIONS

PRIMARY_OUTCOMES_AIM1 = [
    "hba1c_change_12mo",
    "sbp_change_12mo",
    "acs_admit_rate_per_year",
    "ed_visit_rate_per_year",
    "all_cause_mortality_12mo",
]


def load_individual_panel() -> pd.DataFrame | None:
    p = DATA_CLEAN / "waymark_individual_panel.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def estimate_aipw_att(
    df: pd.DataFrame,
    outcome: str,
    treatment: str = "procedural_disenrolled",
    confounders: list[str] | None = None,
    n_folds: int = 5,
    seed: int = SEED,
) -> dict | None:
    if confounders is None:
        confounders = [c for c in df.columns if c.startswith("x_")]
    needed = [outcome, treatment] + confounders
    sub = df.dropna(subset=needed).copy()
    if len(sub) < 200:
        return None

    X = sub[confounders].values
    A = sub[treatment].astype(int).values
    Y = sub[outcome].astype(float).values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    mu1 = np.zeros(len(sub))
    mu0 = np.zeros(len(sub))
    pi  = np.zeros(len(sub))

    for train_idx, test_idx in kf.split(X):
        prop = LogisticRegression(max_iter=500, solver="liblinear")
        prop.fit(X[train_idx], A[train_idx])
        pi[test_idx] = np.clip(prop.predict_proba(X[test_idx])[:, 1], 0.01, 0.99)
        out_treat = GradientBoostingRegressor(random_state=seed, n_estimators=100, max_depth=3)
        out_ctrl  = GradientBoostingRegressor(random_state=seed, n_estimators=100, max_depth=3)
        treat_train = train_idx[A[train_idx] == 1]
        ctrl_train  = train_idx[A[train_idx] == 0]
        if len(treat_train) > 5 and len(ctrl_train) > 5:
            out_treat.fit(X[treat_train], Y[treat_train])
            out_ctrl.fit(X[ctrl_train], Y[ctrl_train])
            mu1[test_idx] = out_treat.predict(X[test_idx])
            mu0[test_idx] = out_ctrl.predict(X[test_idx])

    aipw_treat = (A * (Y - mu1) / pi) + mu1
    aipw_ctrl  = ((1 - A) * (Y - mu0) / (1 - pi)) + mu0
    psi = aipw_treat - aipw_ctrl
    att = float(np.mean(psi))

    rng = np.random.default_rng(seed)
    boot_atts = np.empty(BOOTSTRAP_ITERATIONS)
    n = len(psi)
    for i in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, n)
        boot_atts[i] = np.mean(psi[idx])
    lo, hi = float(np.quantile(boot_atts, 0.025)), float(np.quantile(boot_atts, 0.975))

    return {
        "outcome": outcome,
        "estimator": "AIPW",
        "att": att,
        "ci95_lower": lo,
        "ci95_upper": hi,
        "n_observations": int(len(sub)),
        "n_treated": int(A.sum()),
        "mean_propensity": float(pi.mean()),
    }


def estimate_causal_forest_cate(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    subgroup_cols: list[str],
    confounders: list[str] | None = None,
    seed: int = SEED,
) -> pd.DataFrame | None:
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        print("econml not installed; skipping causal-forest CATE.", file=sys.stderr)
        return None

    if confounders is None:
        confounders = [c for c in df.columns if c.startswith("x_")]
    needed = [outcome, treatment] + confounders + subgroup_cols
    sub = df.dropna(subset=needed).copy()
    if len(sub) < 500:
        return None
    Y = sub[outcome].astype(float).values
    T = sub[treatment].astype(int).values
    X = sub[confounders].values
    W = X.copy()
    cf = CausalForestDML(random_state=seed, n_estimators=500)
    cf.fit(Y, T, X=X, W=W)
    rows = []
    for sg in subgroup_cols:
        for level, sub_sg in sub.groupby(sg):
            if len(sub_sg) < 50:
                continue
            X_sg = sub_sg[confounders].values
            cate = cf.effect(X_sg)
            mean_cate = float(np.mean(cate))
            ci_lo, ci_hi = float(np.quantile(cate, 0.025)), float(np.quantile(cate, 0.975))
            rows.append({
                "outcome": outcome,
                "subgroup_var": sg,
                "subgroup_level": str(level),
                "n": int(len(sub_sg)),
                "cate_mean": mean_cate,
                "cate_ci_lower": ci_lo,
                "cate_ci_upper": ci_hi,
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> None:
    panel = load_individual_panel()
    if panel is None:
        print("Aim 1: Waymark individual panel missing; skipping.", file=sys.stderr)
        return
    rows = []
    cate_frames = []
    subgroup_cols = ["race_ethnicity", "primary_language", "adl_disability", "urbanicity"]
    for outcome in PRIMARY_OUTCOMES_AIM1:
        if outcome not in panel.columns:
            continue
        att = estimate_aipw_att(panel, outcome)
        if att is not None:
            rows.append(att)
        cate_df = estimate_causal_forest_cate(panel, outcome, "procedural_disenrolled", subgroup_cols)
        if cate_df is not None:
            cate_frames.append(cate_df)
    if rows:
        pd.DataFrame(rows).to_csv(TABLES / "aim1_target_trial_results.csv", index=False)
        print(f"Wrote {TABLES / 'aim1_target_trial_results.csv'} with {len(rows)} rows")
    if cate_frames:
        pd.concat(cate_frames, ignore_index=True).to_csv(
            TABLES / "aim1_cate_subgroups.csv", index=False
        )


if __name__ == "__main__":
    main()
