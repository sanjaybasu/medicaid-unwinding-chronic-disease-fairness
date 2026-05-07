"""BRFSS 2018-2024 download and cleaning.

Source: https://www.cdc.gov/brfss/annual_data/annual_data.htm

CDC publishes BRFSS as SAS XPT files per year. This module downloads each
year and computes state-year secondary outcomes:
  - htn_treated_pct: % of adults with diagnosed HTN reporting BP medication
  - dm_a1c_past_year_pct: % of adults with diagnosed DM reporting A1c in past year
  - dental_visit_past_year_pct: % of adults reporting any dental visit in past year
  - cost_related_skipped_care_pct: % reporting could not see doctor due to cost

Output: data/clean/brfss_state_year.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import pyreadstat
import requests

from config import DATA_RAW, DATA_CLEAN
from utils import US_STATES

BRFSS_BASE = "https://www.cdc.gov/brfss/annual_data/"
BRFSS_FILES = {
    2018: "2018/files/LLCP2018XPT.zip",
    2019: "2019/files/LLCP2019XPT.zip",
    2020: "2020/files/LLCP2020XPT.zip",
    2021: "2021/files/LLCP2021XPT.zip",
    2022: "2022/files/LLCP2022XPT.zip",
    2023: "2023/files/LLCP2023XPT.zip",
    2024: "2024/files/LLCP2024XPT.zip",
}

FIPS_TO_ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}


def download_brfss_year(year: int) -> Path | None:
    rel = BRFSS_FILES[year]
    url = urljoin(BRFSS_BASE, rel)
    out_zip = DATA_RAW / f"brfss_{year}.zip"
    if out_zip.exists() and out_zip.stat().st_size > 1_000_000:
        return out_zip
    try:
        r = requests.get(url, timeout=600, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            print(f"  BRFSS {year} HTTP {r.status_code}", file=sys.stderr)
            return None
        with open(out_zip, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
        return out_zip
    except Exception as e:
        print(f"  BRFSS {year} download failed: {e}", file=sys.stderr)
        return None


def extract_xpt_from_zip(zip_path: Path, year: int) -> Path | None:
    import zipfile
    out_dir = DATA_RAW / f"brfss_{year}_unzip"
    out_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            xpt_names = [n for n in zf.namelist() if n.lower().endswith(".xpt")]
            if not xpt_names:
                return None
            zf.extractall(out_dir)
            return out_dir / xpt_names[0]
    except Exception as e:
        print(f"  BRFSS {year} unzip failed: {e}", file=sys.stderr)
        return None


def code_outcomes(df: pd.DataFrame, year: int) -> pd.DataFrame:
    cols = {c.upper(): c for c in df.columns}

    state_col = cols.get("_STATE")
    weight_col = cols.get("_LLCPWT") or cols.get("LLCPWT")
    if state_col is None or weight_col is None:
        raise ValueError(f"BRFSS {year} missing _STATE or _LLCPWT")

    df["_state_fips"] = pd.to_numeric(df[state_col], errors="coerce").astype("Int64")
    df["state_abbr"] = df["_state_fips"].map(FIPS_TO_ABBR)
    df = df.dropna(subset=["state_abbr"]).copy()

    htn_diag = cols.get("BPHIGH4") or cols.get("BPHIGH6")
    htn_med = cols.get("BPMEDS")
    dm_diag = cols.get("DIABETE3") or cols.get("DIABETE4")
    a1c_freq = cols.get("CHKHEMO3") or cols.get("HEMOTEST")
    dental = cols.get("LASTDEN4") or cols.get("LASTDEN3")
    cost_skip = cols.get("MEDCOST")

    df["weight"] = pd.to_numeric(df[weight_col], errors="coerce")

    if htn_diag is not None and htn_med is not None:
        diag = pd.to_numeric(df[htn_diag], errors="coerce")
        med = pd.to_numeric(df[htn_med], errors="coerce")
        df["htn_diagnosed"] = (diag == 1).astype(float)
        df["htn_treated"] = ((diag == 1) & (med == 1)).astype(float)
    else:
        df["htn_diagnosed"] = np.nan
        df["htn_treated"] = np.nan

    if dm_diag is not None and a1c_freq is not None:
        dm = pd.to_numeric(df[dm_diag], errors="coerce")
        a1c = pd.to_numeric(df[a1c_freq], errors="coerce")
        df["dm_diagnosed"] = (dm == 1).astype(float)
        df["dm_a1c_past_year"] = ((dm == 1) & (a1c.between(1, 9))).astype(float)
    else:
        df["dm_diagnosed"] = np.nan
        df["dm_a1c_past_year"] = np.nan

    if dental is not None:
        d = pd.to_numeric(df[dental], errors="coerce")
        df["dental_past_year"] = (d == 1).astype(float)
    else:
        df["dental_past_year"] = np.nan

    if cost_skip is not None:
        c = pd.to_numeric(df[cost_skip], errors="coerce")
        df["cost_skip"] = (c == 1).astype(float)
    else:
        df["cost_skip"] = np.nan

    return df[[
        "state_abbr", "weight",
        "htn_diagnosed", "htn_treated",
        "dm_diagnosed", "dm_a1c_past_year",
        "dental_past_year", "cost_skip",
    ]]


def aggregate_state_year(coded: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    for state, sub in coded.groupby("state_abbr"):
        w = sub["weight"].fillna(0).values
        wsum = w.sum()
        if wsum == 0:
            continue
        d = {"state_abbr": state, "year": year}

        diag_mask = sub["htn_diagnosed"] == 1
        wd = sub.loc[diag_mask, "weight"].sum()
        d["htn_treated_pct"] = (
            float((sub.loc[diag_mask, "htn_treated"] * sub.loc[diag_mask, "weight"]).sum() / wd * 100)
            if wd > 0 else np.nan
        )

        dm_mask = sub["dm_diagnosed"] == 1
        wd = sub.loc[dm_mask, "weight"].sum()
        d["dm_a1c_past_year_pct"] = (
            float((sub.loc[dm_mask, "dm_a1c_past_year"] * sub.loc[dm_mask, "weight"]).sum() / wd * 100)
            if wd > 0 else np.nan
        )

        d["dental_visit_past_year_pct"] = float(
            (sub["dental_past_year"].fillna(0) * w).sum() / wsum * 100
        )
        d["cost_related_skipped_care_pct"] = float(
            (sub["cost_skip"].fillna(0) * w).sum() / wsum * 100
        )
        rows.append(d)
    return pd.DataFrame(rows)


def build_brfss_state_year() -> pd.DataFrame:
    frames = []
    for year in BRFSS_FILES:
        zip_path = download_brfss_year(year)
        if zip_path is None:
            continue
        xpt_path = extract_xpt_from_zip(zip_path, year)
        if xpt_path is None:
            continue
        try:
            df, _ = pyreadstat.read_xport(str(xpt_path))
        except Exception as e:
            print(f"  BRFSS {year} parse failed: {e}", file=sys.stderr)
            continue
        coded = code_outcomes(df, year)
        agg = aggregate_state_year(coded, year)
        frames.append(agg)
        print(f"  BRFSS {year}: {len(agg)} state-year rows")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    out_path = DATA_CLEAN / "brfss_state_year.parquet"
    panel = build_brfss_state_year()
    if panel.empty:
        print("BRFSS data not available; secondary BRFSS analyses will be skipped.", file=sys.stderr)
        return
    panel.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(panel)} rows; "
          f"states = {panel['state_abbr'].nunique()}, "
          f"years = {sorted(panel['year'].unique().tolist())}")


if __name__ == "__main__":
    main()
