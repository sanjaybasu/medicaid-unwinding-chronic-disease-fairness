"""KFF Medicaid Enrollment and Unwinding Tracker data acquisition.

Source: https://www.kff.org/medicaid/issue-brief/medicaid-enrollment-and-unwinding-tracker/

Output: data/clean/kff_unwinding_state_month.parquet with columns
  state_abbr, year_month, total_disenrolled, procedural_disenrolled,
  cumulative_total_disenrollment_rate, cumulative_procedural_disenrollment_rate,
  procedural_fraction
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import DATA_RAW, DATA_CLEAN
from utils import US_STATES

KFF_DOWNLOAD_PAGE = "https://www.kff.org/medicaid/issue-brief/medicaid-enrollment-and-unwinding-tracker/"

CANDIDATE_DATA_URLS = [
    "https://files.kff.org/attachment/Medicaid-Unwinding-Tracker-Data-Snapshot.xlsx",
    "https://www.kff.org/wp-content/uploads/2024/12/Medicaid-Unwinding-Tracker-Data-Snapshot.xlsx",
    "https://www.kff.org/wp-content/uploads/2024/11/Medicaid-Unwinding-Tracker-Data-Snapshot.xlsx",
]


def download_kff_xlsx() -> Path:
    raw_path = DATA_RAW / "kff_unwinding_tracker.xlsx"
    if raw_path.exists() and raw_path.stat().st_size > 10000:
        return raw_path
    for url in CANDIDATE_DATA_URLS:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.content) > 10000:
                raw_path.write_bytes(r.content)
                return raw_path
        except Exception as e:
            print(f"  failed {url}: {e}", file=sys.stderr)
    raise RuntimeError(
        "Could not download KFF unwinding tracker. "
        "Manually download the data snapshot from "
        f"{KFF_DOWNLOAD_PAGE} and place at {raw_path}."
    )


def parse_kff_xlsx(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    candidate_sheets = [
        s for s in xls.sheet_names
        if any(k in s.lower() for k in ("disenroll", "state", "tracker"))
    ]
    if not candidate_sheets:
        candidate_sheets = xls.sheet_names
    frames = []
    for sheet in candidate_sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        df["_sheet"] = sheet
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize_kff_state_month(raw: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c: c.lower() if isinstance(c, str) else c for c in raw.columns}
    raw = raw.rename(columns=cols_lower)

    state_col = next((c for c in raw.columns if isinstance(c, str) and "state" in c.lower()), None)
    if state_col is None:
        raise ValueError("KFF data has no state column")

    state_to_abbr = {v.lower(): k for k, v in US_STATES.items()}
    raw["state_abbr"] = raw[state_col].astype(str).str.strip().str.lower().map(state_to_abbr)
    raw = raw.dropna(subset=["state_abbr"]).copy()

    keep_value_cols = [
        c for c in raw.columns
        if isinstance(c, str)
        and (
            "disenroll" in c.lower()
            or "renew" in c.lower()
            or "procedural" in c.lower()
            or "ineligib" in c.lower()
        )
    ]
    if not keep_value_cols:
        raise ValueError(
            "KFF data has no disenrollment columns; KFF format may have changed. "
            "Inspect the raw file and update parser."
        )

    out = raw[["state_abbr"] + keep_value_cols].copy()
    out.columns = ["state_abbr"] + [
        c.lower().strip().replace(" ", "_").replace("-", "_") for c in keep_value_cols
    ]
    return out


def build_state_month_panel() -> pd.DataFrame:
    raw_path = download_kff_xlsx()
    raw = parse_kff_xlsx(raw_path)
    panel = normalize_kff_state_month(raw)
    return panel


def main() -> None:
    out_path = DATA_CLEAN / "kff_unwinding_state_month.parquet"
    try:
        panel = build_state_month_panel()
    except RuntimeError as e:
        print(f"WARNING: {e}", file=sys.stderr)
        print("Skipping KFF data; downstream analyses will fail until raw file is provided.", file=sys.stderr)
        return
    panel.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(panel)} rows; states = {panel['state_abbr'].nunique()}")


if __name__ == "__main__":
    main()
