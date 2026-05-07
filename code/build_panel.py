"""Assemble the state-quarter analytic panel for Aim 1.

Inputs (clean/):
  - kff_unwinding_state_month.parquet
  - cdc_wonder_mortality_state_quarter.parquet
  - hcupnet_outcomes_state_quarter.parquet
  - state_year_covariates.parquet

Output:
  - state_quarter_panel.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_CLEAN, PANEL_START, PANEL_END, TREATED_PERIOD_START
from utils import US_STATES


def state_quarter_skeleton() -> pd.DataFrame:
    quarters = pd.date_range(PANEL_START, PANEL_END, freq="QS")
    rows = [
        {"state_abbr": s, "year": q.year, "quarter": (q.month - 1) // 3 + 1, "qstart": q}
        for s in US_STATES
        for q in quarters
    ]
    return pd.DataFrame(rows)


def attach_treatment(panel: pd.DataFrame) -> pd.DataFrame:
    kff_path = DATA_CLEAN / "kff_unwinding_state_month.parquet"
    if not kff_path.exists():
        panel["cumulative_procedural_disenrollment_rate"] = np.nan
        panel["procedural_fraction"] = np.nan
        return panel
    kff = pd.read_parquet(kff_path)
    cum_col = next(
        (c for c in kff.columns if "cumulative" in c and "procedural" in c),
        None,
    )
    frac_col = next(
        (c for c in kff.columns if "procedural_fraction" in c or "procedural_share" in c),
        None,
    )
    if cum_col is None:
        panel["cumulative_procedural_disenrollment_rate"] = np.nan
        panel["procedural_fraction"] = np.nan
        return panel
    state_terminal = (
        kff.groupby("state_abbr")[cum_col].last().rename("terminal_proc_rate").reset_index()
    )
    panel = panel.merge(state_terminal, on="state_abbr", how="left")
    treated_start = pd.Timestamp(TREATED_PERIOD_START)
    panel["months_into_unwinding"] = (
        ((panel["qstart"] - treated_start).dt.days // 30).clip(lower=0)
    )
    panel["cumulative_procedural_disenrollment_rate"] = np.where(
        panel["qstart"] >= treated_start,
        panel["terminal_proc_rate"]
        * np.minimum(1.0, panel["months_into_unwinding"] / 18.0),
        0.0,
    )
    if frac_col is not None:
        state_frac = (
            kff.groupby("state_abbr")[frac_col].last().rename("procedural_fraction").reset_index()
        )
        panel = panel.merge(state_frac, on="state_abbr", how="left")
    else:
        panel["procedural_fraction"] = np.nan
    return panel


def attach_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    wonder_path = DATA_CLEAN / "cdc_wonder_mortality_state_quarter.parquet"
    if wonder_path.exists():
        wonder = pd.read_parquet(wonder_path)
        wonder_pivot = (
            wonder.groupby(["state_abbr", "year", "quarter", "cause_category"])
            .agg(deaths=("deaths", "sum"), population=("population", "sum"))
            .reset_index()
        )
        wonder_pivot["rate_per_100k"] = (
            wonder_pivot["deaths"] / wonder_pivot["population"] * 1e5
        )
        for cause in ("all_cause", "amenable"):
            sub = wonder_pivot[wonder_pivot["cause_category"] == cause][[
                "state_abbr", "year", "quarter", "rate_per_100k",
            ]].rename(columns={"rate_per_100k": f"{cause}_mortality_per_100k_35_64"})
            panel = panel.merge(sub, on=["state_abbr", "year", "quarter"], how="left")
    else:
        panel["all_cause_mortality_per_100k_35_64"] = np.nan
        panel["amenable_mortality_per_100k_35_64"] = np.nan

    hcup_path = DATA_CLEAN / "hcupnet_outcomes_state_quarter.parquet"
    if hcup_path.exists():
        hcup = pd.read_parquet(hcup_path)
        for outcome_label, target_col in [
            ("acs_admit", "acs_admit_rate_per_1000"),
            ("bh_ed", "bh_ed_rate_per_1000"),
        ]:
            sub = hcup[
                (hcup["outcome"] == outcome_label)
                & (hcup["race_ethnicity"].isin(["all", ""]))
                & (hcup["payer"].isin(["all", "medicaid", "all primary payers"]))
            ]
            agg = (
                sub.groupby(["state_abbr", "year", "quarter"])["rate_per_1000"]
                .mean()
                .reset_index()
                .rename(columns={"rate_per_1000": target_col})
            )
            panel = panel.merge(agg, on=["state_abbr", "year", "quarter"], how="left")
    else:
        panel["acs_admit_rate_per_1000"] = np.nan
        panel["bh_ed_rate_per_1000"] = np.nan
    return panel


def attach_covariates(panel: pd.DataFrame) -> pd.DataFrame:
    cov_path = DATA_CLEAN / "state_year_covariates.parquet"
    if not cov_path.exists():
        return panel
    cov = pd.read_parquet(cov_path)
    return panel.merge(cov, on=["state_abbr", "year"], how="left")


def build() -> pd.DataFrame:
    panel = state_quarter_skeleton()
    panel = attach_treatment(panel)
    panel = attach_outcomes(panel)
    panel = attach_covariates(panel)
    panel["post"] = (panel["qstart"] >= pd.Timestamp(TREATED_PERIOD_START)).astype(int)
    panel["state_id"] = panel["state_abbr"]
    panel["time_id"] = panel["qstart"].dt.year * 4 + (panel["quarter"] - 1)
    return panel


def main() -> None:
    panel = build()
    out = DATA_CLEAN / "state_quarter_panel.parquet"
    panel.to_parquet(out, index=False)
    print(f"Wrote {out} with {len(panel)} rows; states = {panel['state_abbr'].nunique()}")


if __name__ == "__main__":
    main()
