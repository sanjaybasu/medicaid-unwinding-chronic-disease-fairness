"""Aim 3 — synthesis between subgroup miscalibration and subgroup disenrollment exposure.

Primary specification: state x subgroup regression with state and subgroup fixed effects,
clustering at state level. Predictor: subgroup miscalibration magnitude (Aim 2 output).
Dependent: subgroup procedural-disenrollment rate from KFF state-by-demographic data.

Fallback (when KFF state-by-demographic data covers <20 states): subgroup-only Spearman.

Output: tables/aim3_synthesis.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import pyfixest as pf

from config import DATA_CLEAN, TABLES, BOOTSTRAP_ITERATIONS, SEED


def load_aim2() -> pd.DataFrame | None:
    p = TABLES / "aim2_fairness_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["miscalibration_magnitude"] = (df["predicted_to_actual_ratio"] - 1.0).abs()
    return df


def load_kff_demographic() -> pd.DataFrame | None:
    p = DATA_CLEAN / "kff_unwinding_state_demographic.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def primary_specification(aim2: pd.DataFrame, kff_demo: pd.DataFrame) -> pd.DataFrame | None:
    if aim2 is None or kff_demo is None:
        return None
    n_states = kff_demo["state_abbr"].nunique() if "state_abbr" in kff_demo else 0
    if n_states < 20:
        return None
    merged = kff_demo.merge(
        aim2[["model", "subgroup_name", "miscalibration_magnitude"]],
        left_on="subgroup",
        right_on="subgroup_name",
        how="inner",
    )
    if merged.empty:
        return None
    merged["state_id"] = merged["state_abbr"]
    merged["subgroup_id"] = merged["subgroup_name"]
    rows = []
    for model_name, sub in merged.groupby("model"):
        try:
            fit = pf.feols(
                "procedural_disenrollment_rate ~ miscalibration_magnitude | state_id + subgroup_id",
                data=sub,
                vcov={"CRV1": "state_id"},
            )
            coef = float(fit.coef()["miscalibration_magnitude"])
            ci = fit.confint().loc["miscalibration_magnitude"]
            rows.append({
                "specification": "state_x_subgroup_regression",
                "model": model_name,
                "coefficient": coef,
                "ci95_lower": float(ci.iloc[0]),
                "ci95_upper": float(ci.iloc[1]),
                "n_observations": int(len(sub)),
            })
        except Exception as e:
            print(f"  primary spec failed for {model_name}: {e}", file=sys.stderr)
    if not rows:
        return None
    return pd.DataFrame(rows)


def secondary_spearman(aim2: pd.DataFrame, kff_state_total: pd.DataFrame | None) -> pd.DataFrame | None:
    if aim2 is None:
        return None
    if kff_state_total is None or kff_state_total.empty:
        return None
    state_avg = (
        kff_state_total.groupby("state_abbr")["cumulative_procedural_disenrollment_rate"]
        .max()
        .mean()
    )
    rows = []
    rng = np.random.default_rng(SEED)
    for model_name, sub in aim2.groupby("model"):
        sub = sub.copy()
        sub["disenrollment_proxy"] = state_avg
        if "subgroup_name" not in sub:
            continue
        if sub["miscalibration_magnitude"].std() == 0:
            continue
        sub["disenrollment_proxy"] = sub["miscalibration_magnitude"] * 0
        try:
            rho, p = stats.spearmanr(sub["miscalibration_magnitude"], sub["disenrollment_proxy"])
        except Exception:
            continue
        if np.isnan(rho):
            continue
        boot_rhos = []
        for _ in range(BOOTSTRAP_ITERATIONS):
            idx = rng.integers(0, len(sub), len(sub))
            try:
                br, _ = stats.spearmanr(
                    sub["miscalibration_magnitude"].values[idx],
                    sub["disenrollment_proxy"].values[idx],
                )
                if not np.isnan(br):
                    boot_rhos.append(br)
            except Exception:
                pass
        rows.append({
            "specification": "subgroup_only_spearman",
            "model": model_name,
            "rho": float(rho),
            "ci95_lower": float(np.quantile(boot_rhos, 0.025)) if boot_rhos else np.nan,
            "ci95_upper": float(np.quantile(boot_rhos, 0.975)) if boot_rhos else np.nan,
            "n_subgroups": int(len(sub)),
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> None:
    aim2 = load_aim2()
    kff_demo = load_kff_demographic()
    kff_total = None
    p = DATA_CLEAN / "state_quarter_panel.parquet"
    if p.exists():
        kff_total = pd.read_parquet(p)

    primary = primary_specification(aim2, kff_demo)
    secondary = secondary_spearman(aim2, kff_total)

    out_frames = []
    if primary is not None:
        out_frames.append(primary)
    if secondary is not None:
        out_frames.append(secondary)
    if out_frames:
        pd.concat(out_frames, ignore_index=True).to_csv(
            TABLES / "aim3_synthesis.csv", index=False
        )
        print(f"Wrote {TABLES / 'aim3_synthesis.csv'}")
    else:
        print("Aim 3: no specifications could be evaluated (need Aim 2 + KFF demographic data).")


if __name__ == "__main__":
    main()
