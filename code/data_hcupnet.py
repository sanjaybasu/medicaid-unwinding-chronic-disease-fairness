"""HCUPnet ACS admissions and behavioral-health ED visits.

Source: https://hcupnet.ahrq.gov/

HCUPnet has no public API. The reproducible approach is to download AHRQ's
state-level Prevention Quality Indicators (PQI) and SEDD aggregates, which
are published as XLSX files annually. Manual download steps are documented
below; the parser handles the published file structure.

Output: data/clean/hcupnet_outcomes_state_quarter.parquet with columns
  state_abbr, year, quarter, outcome, race_ethnicity, rate_per_1000,
  numerator, denominator
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

from config import DATA_RAW, DATA_CLEAN
from utils import US_STATES

EXPECTED_RAW_FILES = {
    "hcupnet_pqi_state_year.csv": (
        "Manual download: HCUPnet > Prevention Quality Indicators (PQI) > "
        "Composite Chronic Care (PQI 92) and individual chronic-care PQIs by "
        "state and year, primary payer = Medicaid + Uninsured + All. "
        "Export rate per 1000 with numerator and denominator. "
        "Available at https://hcupnet.ahrq.gov/ ."
    ),
    "hcupnet_bh_ed_state_year.csv": (
        "Manual download: HCUPnet > Emergency Department > Behavioral health "
        "(F-coded primary diagnosis) by state, year, primary payer."
    ),
}


def load_hcupnet_file(filename: str) -> pd.DataFrame:
    path = DATA_RAW / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. {EXPECTED_RAW_FILES[filename]}\n"
            f"Save as {path}."
        )
    return pd.read_csv(path)


def normalize_hcupnet(df: pd.DataFrame, outcome_label: str) -> pd.DataFrame:
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    state_col = next((c for c in df.columns if "state" in c), None)
    year_col = next((c for c in df.columns if c == "year"), None)
    quarter_col = next((c for c in df.columns if "quarter" in c), None)
    payer_col = next((c for c in df.columns if "payer" in c), None)
    race_col = next((c for c in df.columns if "race" in c), None)
    rate_col = next((c for c in df.columns if "rate" in c), None)
    num_col = next((c for c in df.columns if "numerator" in c or "discharges" in c or "visits" in c), None)
    den_col = next((c for c in df.columns if "denominator" in c or "population" in c), None)
    if not all([state_col, year_col, rate_col]):
        raise ValueError(
            f"HCUPnet export missing required columns; expected state, year, rate. "
            f"Got: {list(df.columns)}"
        )

    state_to_abbr = {v.lower(): k for k, v in US_STATES.items()}
    df["state_abbr"] = df[state_col].astype(str).str.strip().str.lower().map(state_to_abbr)
    df = df.dropna(subset=["state_abbr"]).copy()
    df["year"] = pd.to_numeric(df[year_col], errors="coerce")
    df["quarter"] = (
        pd.to_numeric(df[quarter_col], errors="coerce") if quarter_col is not None else pd.NA
    )
    df["rate_per_1000"] = pd.to_numeric(df[rate_col], errors="coerce")
    df["numerator"] = pd.to_numeric(df[num_col], errors="coerce") if num_col else pd.NA
    df["denominator"] = pd.to_numeric(df[den_col], errors="coerce") if den_col else pd.NA
    df["race_ethnicity"] = (
        df[race_col].astype(str).str.strip().str.lower() if race_col is not None else "all"
    )
    df["payer"] = (
        df[payer_col].astype(str).str.strip().str.lower() if payer_col is not None else "all"
    )
    df["outcome"] = outcome_label
    keep = [
        "state_abbr", "year", "quarter", "outcome",
        "race_ethnicity", "payer", "rate_per_1000",
        "numerator", "denominator",
    ]
    return df[keep].reset_index(drop=True)


def build_hcupnet_panel() -> pd.DataFrame:
    frames = []
    try:
        pqi = load_hcupnet_file("hcupnet_pqi_state_year.csv")
        frames.append(normalize_hcupnet(pqi, "acs_admit"))
    except FileNotFoundError as e:
        print(f"WARNING: {e}", file=sys.stderr)
    try:
        bh = load_hcupnet_file("hcupnet_bh_ed_state_year.csv")
        frames.append(normalize_hcupnet(bh, "bh_ed"))
    except FileNotFoundError as e:
        print(f"WARNING: {e}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    out_path = DATA_CLEAN / "hcupnet_outcomes_state_quarter.parquet"
    panel = build_hcupnet_panel()
    if panel.empty:
        print("HCUPnet data not available; downstream ACS/BH analyses will be skipped.", file=sys.stderr)
        return
    panel.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(panel)} rows; "
          f"states = {panel['state_abbr'].nunique()}, "
          f"outcomes = {panel['outcome'].nunique()}")


if __name__ == "__main__":
    main()
