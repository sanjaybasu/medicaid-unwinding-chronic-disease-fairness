"""MEPS-HC 2018-2022 download and Medicaid-cohort assembly.

Source: https://meps.ahrq.gov/data_stats/download_data_files.jsp

Each MEPS panel year has multiple files; the relevant ones:
  - HC-XXX (Full-Year Consolidated): demographics, expenditures, weights
  - HC-XXX (Medical Conditions): ICD-10 / CCS codes per condition
  - HC-XXX (Prescribed Medicines): NDC and therapeutic class

This module pulls the SAS XPT files for 2018-2022 and assembles a person-year
panel restricted to adults aged 19-64 with any-month Medicaid coverage.

Output: data/clean/meps_hc_medicaid_pooled.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import requests

from config import DATA_RAW, DATA_CLEAN

MEPS_FYC_URLS = {
    2018: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h209/h209xpt.zip",
    2019: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h216/h216xpt.zip",
    2020: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h224/h224xpt.zip",
    2021: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h233/h233xpt.zip",
    2022: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h243/h243xpt.zip",
}

MEPS_MEDCONDITIONS_URLS = {
    2018: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h207/h207xpt.zip",
    2019: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h214/h214xpt.zip",
    2020: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h222/h222xpt.zip",
    2021: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h231/h231xpt.zip",
    2022: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h241/h241xpt.zip",
}

MEPS_RX_URLS = {
    2018: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h206a/h206axpt.zip",
    2019: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h213a/h213axpt.zip",
    2020: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h220a/h220axpt.zip",
    2021: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h229a/h229axpt.zip",
    2022: "https://meps.ahrq.gov/mepsweb/data_files/pufs/h239a/h239axpt.zip",
}


def download(url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 100_000:
        return True
    try:
        r = requests.get(url, timeout=600, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        if r.status_code != 200:
            return False
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  download failed {url}: {e}", file=sys.stderr)
        return False


def unzip_to_xpt(zip_path: Path, year: int, kind: str) -> Path | None:
    import zipfile
    out_dir = DATA_RAW / f"meps_{kind}_{year}"
    out_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            xpt = [n for n in zf.namelist() if n.lower().endswith(".xpt")]
            if not xpt:
                xpt = [n for n in zf.namelist() if n.lower().endswith(".dat")]
            if not xpt:
                return None
            zf.extractall(out_dir)
            return out_dir / xpt[0]
    except Exception as e:
        print(f"  unzip failed: {e}", file=sys.stderr)
        return None


def load_xpt(year: int, kind: str, urls: dict[int, str]) -> pd.DataFrame | None:
    if year not in urls:
        return None
    zip_path = DATA_RAW / f"meps_{kind}_{year}.zip"
    if not download(urls[year], zip_path):
        return None
    xpt_path = unzip_to_xpt(zip_path, year, kind)
    if xpt_path is None or not xpt_path.exists():
        return None
    try:
        df, _ = pyreadstat.read_xport(str(xpt_path))
        return df
    except Exception as e:
        print(f"  parse failed {xpt_path}: {e}", file=sys.stderr)
        return None


def code_demographics(fyc: pd.DataFrame, year: int) -> pd.DataFrame:
    cols = {c.upper(): c for c in fyc.columns}
    suffix = str(year)[-2:]

    age_col = cols.get(f"AGE{suffix}X") or cols.get("AGELAST")
    sex_col = cols.get("SEX")
    race_col = cols.get(f"RACETHX") or cols.get("RACETHX")
    racev1 = cols.get("RACEV1X")
    hisp = cols.get("HISPANX")
    medicaid_months = [cols.get(f"MCD{m}{suffix}") for m in [
        "JA","FE","MA","AP","MY","JU","JL","AU","SE","OC","NO","DE",
    ]]
    expenditure = cols.get(f"TOTEXP{suffix}")
    weight = cols.get(f"PERWT{suffix}F")
    region = cols.get(f"REGION{suffix}")
    msa = cols.get(f"MSA{suffix}")

    if age_col is None or weight is None or expenditure is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["dupersid"] = fyc[cols.get("DUPERSID")] if "DUPERSID" in cols else fyc.index.astype(str)
    out["panel_year"] = year
    out["age"] = pd.to_numeric(fyc[age_col], errors="coerce")
    out["sex"] = pd.to_numeric(fyc[sex_col], errors="coerce") if sex_col else np.nan
    out["weight"] = pd.to_numeric(fyc[weight], errors="coerce")
    out["expenditure"] = pd.to_numeric(fyc[expenditure], errors="coerce")
    out["region"] = pd.to_numeric(fyc[region], errors="coerce") if region else np.nan
    out["msa"] = pd.to_numeric(fyc[msa], errors="coerce") if msa else np.nan

    if race_col:
        rt = pd.to_numeric(fyc[race_col], errors="coerce")
        race_map = {1: "hispanic", 2: "nh_white", 3: "nh_black", 4: "nh_asian", 5: "multiracial"}
        out["race_ethnicity"] = rt.map(race_map).fillna("unknown")
    elif racev1 and hisp:
        out["race_ethnicity"] = "unknown"
    else:
        out["race_ethnicity"] = "unknown"

    medicaid_any = pd.Series(False, index=fyc.index)
    for c in medicaid_months:
        if c is None:
            continue
        medicaid_any = medicaid_any | (pd.to_numeric(fyc[c], errors="coerce") == 1)
    out["medicaid_any_month"] = medicaid_any.values

    out["non_metro"] = (out["msa"] == 2)
    return out


def assemble_pooled() -> pd.DataFrame:
    frames = []
    for year in MEPS_FYC_URLS:
        fyc = load_xpt(year, "fyc", MEPS_FYC_URLS)
        if fyc is None:
            print(f"  MEPS {year} FYC unavailable, skipping panel", file=sys.stderr)
            continue
        demo = code_demographics(fyc, year)
        if demo.empty:
            continue
        sample = demo[
            (demo["age"].between(19, 64)) & demo["medicaid_any_month"]
        ].copy()
        sample["dual_eligible"] = False
        frames.append(sample)
        print(f"  MEPS {year}: {len(sample)} adult Medicaid person-years")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    out_path = DATA_CLEAN / "meps_hc_medicaid_pooled.parquet"
    pooled = assemble_pooled()
    if pooled.empty:
        print("MEPS-HC data not available; Aim 2 fairness audit will be skipped.", file=sys.stderr)
        return
    pooled.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(pooled)} person-years; "
          f"panels = {sorted(pooled['panel_year'].unique().tolist())}, "
          f"race groups = {pooled['race_ethnicity'].nunique()}")


if __name__ == "__main__":
    main()
