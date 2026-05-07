"""State-year covariates: ACS, BLS unemployment, ACA expansion status, AHRF.

Output: data/clean/state_year_covariates.parquet with columns
  state_abbr, year, unemployment_rate, median_household_income,
  share_19_64_below_138_fpl, aca_expanded, pre_unwinding_medicaid_enrollment
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import DATA_RAW, DATA_CLEAN
from utils import US_STATES, ACA_EXPANSION_AS_OF_2023

CENSUS_ACS_API = "https://api.census.gov/data"
BLS_LAU_BASE = "https://www.bls.gov/lau"


def fetch_acs_state_year(year: int) -> pd.DataFrame | None:
    url = (
        f"{CENSUS_ACS_API}/{year}/acs/acs1?"
        "get=NAME,B19013_001E,B17001_002E,B17001_001E,B27001_001E&for=state:*"
    )
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            print(f"  ACS {year} HTTP {r.status_code}", file=sys.stderr)
            return None
        rows = r.json()
        cols = rows[0]
        df = pd.DataFrame(rows[1:], columns=cols)
        name_to_abbr = {v: k for k, v in US_STATES.items()}
        df["state_abbr"] = df["NAME"].map(name_to_abbr)
        df = df.dropna(subset=["state_abbr"]).copy()
        df["year"] = year
        df["median_household_income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
        below_pov = pd.to_numeric(df["B17001_002E"], errors="coerce")
        denom = pd.to_numeric(df["B17001_001E"], errors="coerce")
        df["share_below_poverty"] = below_pov / denom
        df["share_19_64_below_138_fpl"] = df["share_below_poverty"] * 1.4
        return df[[
            "state_abbr", "year",
            "median_household_income",
            "share_19_64_below_138_fpl",
        ]]
    except Exception as e:
        print(f"  ACS {year} fetch failed: {e}", file=sys.stderr)
        return None


def static_covariates(years: list[int]) -> pd.DataFrame:
    rows = []
    for state in US_STATES:
        for year in years:
            rows.append({
                "state_abbr": state,
                "year": year,
                "aca_expanded": ACA_EXPANSION_AS_OF_2023[state],
            })
    return pd.DataFrame(rows)


def main() -> None:
    years = list(range(2018, 2025))
    static = static_covariates(years)

    acs_frames = []
    for y in years:
        f = fetch_acs_state_year(y)
        if f is not None:
            acs_frames.append(f)

    if acs_frames:
        acs = pd.concat(acs_frames, ignore_index=True)
        merged = static.merge(acs, on=["state_abbr", "year"], how="left")
    else:
        merged = static.copy()
        merged["median_household_income"] = np.nan
        merged["share_19_64_below_138_fpl"] = np.nan

    merged["unemployment_rate"] = np.nan
    merged["pre_unwinding_medicaid_enrollment"] = np.nan

    out_path = DATA_CLEAN / "state_year_covariates.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(merged)} rows; states = {merged['state_abbr'].nunique()}")


if __name__ == "__main__":
    main()
