"""CDC WONDER mortality data acquisition.

Source: https://wonder.cdc.gov/ucd-icd10-expanded.html

WONDER's API requires session-based XML POSTs. The reproducible approach taken here:
direct CSV pulls from WONDER's pre-aggregated public files where available, and
documented manual-download fallback for the underlying-cause-of-death detail tables.

Output: data/clean/cdc_wonder_mortality_state_quarter.parquet with columns
  state_abbr, year, quarter, age_band, deaths, population, race_ethnicity,
  cause_category in {"all_cause", "amenable"}
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import requests

from config import DATA_RAW, DATA_CLEAN
from utils import US_STATES

WONDER_API_ENDPOINT = "https://wonder.cdc.gov/controller/datarequest/D77"

WONDER_AGE_BAND = "35-64"

EXPECTED_RAW_FILES = {
    "all_cause_state_quarter_35_64.csv": (
        "Manual download: WONDER UCD-ICD10 Expanded; Group by State, Year, Quarter, "
        "Race/Ethnicity (Single Race); Filter Age Group 35-44, 45-54, 55-64; All causes."
    ),
    "amenable_state_quarter_35_64.csv": (
        "Manual download: WONDER UCD-ICD10 Expanded; same group-by; "
        "Filter ICD-10 cause list to Nolte-McKee amenable codes."
    ),
}


def load_wonder_csv(filename: str) -> pd.DataFrame:
    path = DATA_RAW / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. {EXPECTED_RAW_FILES[filename]}\n"
            f"Save as {path} (tab-delimited TXT export from WONDER works)."
        )
    sep = "\t" if path.suffix.lower() in (".txt",) else ","
    df = pd.read_csv(path, sep=sep, dtype=str)
    return df


def normalize_wonder(df: pd.DataFrame, cause_category: str) -> pd.DataFrame:
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    state_col = next((c for c in df.columns if "state" in c), None)
    year_col = next((c for c in df.columns if c == "year"), None)
    quarter_col = next((c for c in df.columns if "quarter" in c), None)
    age_col = next((c for c in df.columns if "age" in c), None)
    race_col = next((c for c in df.columns if "race" in c or "ethnicity" in c), None)
    deaths_col = next((c for c in df.columns if c == "deaths"), None)
    pop_col = next((c for c in df.columns if "population" in c), None)
    if not all([state_col, year_col, quarter_col, deaths_col]):
        raise ValueError(
            "WONDER export missing required columns; expected state, year, quarter, deaths"
        )

    state_to_abbr = {v.lower(): k for k, v in US_STATES.items()}
    df["state_abbr"] = df[state_col].astype(str).str.strip().str.lower().map(state_to_abbr)
    df = df.dropna(subset=["state_abbr"]).copy()
    df["year"] = pd.to_numeric(df[year_col], errors="coerce")
    df["quarter"] = pd.to_numeric(df[quarter_col].astype(str).str.extract(r"(\d)").iloc[:, 0], errors="coerce")
    df["deaths"] = pd.to_numeric(df[deaths_col], errors="coerce")
    if pop_col is not None:
        df["population"] = pd.to_numeric(df[pop_col], errors="coerce")
    else:
        df["population"] = pd.NA
    df["age_band"] = df[age_col].astype(str) if age_col else WONDER_AGE_BAND
    df["race_ethnicity"] = df[race_col].astype(str).str.strip().str.lower() if race_col else "all"
    df["cause_category"] = cause_category
    keep = [
        "state_abbr", "year", "quarter", "age_band",
        "race_ethnicity", "deaths", "population", "cause_category",
    ]
    return df[keep].dropna(subset=["year", "quarter", "deaths"]).reset_index(drop=True)


def build_wonder_state_quarter() -> pd.DataFrame:
    frames = []
    for fname, cause in [
        ("all_cause_state_quarter_35_64.csv", "all_cause"),
        ("amenable_state_quarter_35_64.csv", "amenable"),
    ]:
        try:
            raw = load_wonder_csv(fname)
            frames.append(normalize_wonder(raw, cause))
        except FileNotFoundError as e:
            print(f"WARNING: {e}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    out_path = DATA_CLEAN / "cdc_wonder_mortality_state_quarter.parquet"
    panel = build_wonder_state_quarter()
    if panel.empty:
        print("CDC WONDER data not available; downstream mortality analyses will be skipped.", file=sys.stderr)
        return
    panel.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(panel)} rows; "
          f"states = {panel['state_abbr'].nunique()}, "
          f"causes = {panel['cause_category'].nunique()}")


if __name__ == "__main__":
    main()
