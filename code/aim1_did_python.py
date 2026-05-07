"""Aim 1 — Python-side analyses: descriptive plots, two-way fixed-effects
event study, basic continuous-treatment ATT, pre-trend Wald test.

The advanced estimators (dCDH, Callaway-Sant'Anna, BJS, Honest-DiD,
augmented synthetic control) are run from aim1_did_R.R. This module produces
a Python-resident result for the smoke test and as a sanity check.

Output: tables/aim1_python_results.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

from config import DATA_CLEAN, TABLES, PRIMARY_OUTCOMES, SEED, TREATED_PERIOD_START

np.random.seed(SEED)


def twfe_continuous_did(panel: pd.DataFrame, outcome: str) -> dict | None:
    df = panel.dropna(subset=[outcome, "cumulative_procedural_disenrollment_rate"]).copy()
    if len(df) < 100:
        return None
    df["treat_post"] = df["cumulative_procedural_disenrollment_rate"] * df["post"]
    fit = pf.feols(
        f"{outcome} ~ treat_post | state_id + time_id",
        data=df,
        vcov={"CRV1": "state_id"},
    )
    coef = fit.coef()["treat_post"]
    ci = fit.confint().loc["treat_post"]
    pval = fit.pvalue()["treat_post"]
    return {
        "outcome": outcome,
        "att_per_pp_proc_disenrollment": float(coef),
        "ci95_lower": float(ci.iloc[0]),
        "ci95_upper": float(ci.iloc[1]),
        "p_value": float(pval),
        "n_observations": int(len(df)),
        "n_states": int(df["state_id"].nunique()),
    }


def event_study(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    df = panel.dropna(subset=[outcome]).copy()
    if df.empty:
        return pd.DataFrame()
    treated_q = pd.Timestamp(TREATED_PERIOD_START)
    df["q_offset"] = (
        (df["qstart"].dt.year - treated_q.year) * 4
        + (df["quarter"] - ((treated_q.month - 1) // 3 + 1))
    )
    df["intensity"] = df["cumulative_procedural_disenrollment_rate"].fillna(0)
    df["intensity_rank"] = df.groupby("time_id")["intensity"].rank(method="dense")
    state_terminal = (
        df.groupby("state_abbr")["intensity"].max().rename("max_intensity").reset_index()
    )
    median = state_terminal["max_intensity"].median()
    state_terminal["high_intensity"] = (state_terminal["max_intensity"] > median).astype(int)
    df = df.merge(state_terminal[["state_abbr", "high_intensity"]], on="state_abbr")
    summary = (
        df.groupby(["q_offset", "high_intensity"])[outcome].mean().reset_index()
    )
    return summary


def pre_trend_wald(panel: pd.DataFrame, outcome: str) -> dict | None:
    df = panel.dropna(subset=[outcome]).copy()
    if df.empty:
        return None
    pre = df[df["post"] == 0].copy()
    state_trends = (
        pre.groupby("state_abbr")
        .apply(lambda g: np.polyfit(g["time_id"], g[outcome], 1)[0] if len(g) >= 4 else np.nan, include_groups=False)
        .rename("pre_slope")
        .reset_index()
    )
    state_terminal = (
        df.groupby("state_abbr")["cumulative_procedural_disenrollment_rate"].max()
        .rename("terminal_intensity")
        .reset_index()
    )
    merged = state_trends.merge(state_terminal, on="state_abbr").dropna()
    if len(merged) < 10:
        return None
    fit = pf.feols("pre_slope ~ terminal_intensity", data=merged)
    coef = float(fit.coef()["terminal_intensity"])
    pval = float(fit.pvalue()["terminal_intensity"])
    return {
        "outcome": outcome,
        "pre_trend_coef": coef,
        "pre_trend_pvalue": pval,
        "violation_at_p10": pval < 0.10,
    }


def main() -> None:
    panel_path = DATA_CLEAN / "state_quarter_panel.parquet"
    if not panel_path.exists():
        print(f"Missing {panel_path}; run build_panel.py first.", file=sys.stderr)
        return
    panel = pd.read_parquet(panel_path)
    rows = []
    pre_trend_rows = []
    es_frames = []
    for outcome in PRIMARY_OUTCOMES:
        if outcome not in panel.columns:
            continue
        result = twfe_continuous_did(panel, outcome)
        if result is not None:
            rows.append(result)
        pt = pre_trend_wald(panel, outcome)
        if pt is not None:
            pre_trend_rows.append(pt)
        es = event_study(panel, outcome)
        if not es.empty:
            es["outcome"] = outcome
            es_frames.append(es)
    if rows:
        df_main = pd.DataFrame(rows)
        df_main.to_csv(TABLES / "aim1_python_results.csv", index=False)
        print(f"Wrote {TABLES / 'aim1_python_results.csv'}")
    if pre_trend_rows:
        pd.DataFrame(pre_trend_rows).to_csv(TABLES / "aim1_pre_trend_tests.csv", index=False)
    if es_frames:
        pd.concat(es_frames, ignore_index=True).to_csv(
            TABLES / "aim1_event_study.csv", index=False
        )


if __name__ == "__main__":
    main()
