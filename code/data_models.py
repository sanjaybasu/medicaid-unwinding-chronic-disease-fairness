"""Risk-adjustment model artifacts: CDPS+Rx and HHS-HCC.

CDPS+Rx is distributed by UCSD's Health Workforce Studies Program. The model
package includes coefficient files and ICD-10 / NDC -> category crosswalks.
A registration form is required for full download; documented manual step below.

HHS-HCC is published by CMS in annual benefit-year ZIPs. Public download.

Output:
  data/clean/cdps_rx_coefficients.csv
  data/clean/cdps_rx_icd10_map.csv
  data/clean/cdps_rx_ndc_map.csv
  data/clean/hhs_hcc_v07_coefficients.csv
  data/clean/hhs_hcc_v07_icd10_map.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import requests

from config import DATA_RAW, DATA_CLEAN, CDPS_MODEL_VERSION, HHS_HCC_BENEFIT_YEAR

CDPS_LANDING_PAGE = "https://hwsph.ucsd.edu/research/cdps-medicaid-rx/"
HHS_HCC_LANDING = (
    "https://www.cms.gov/marketplace/resources/regulations-guidance/risk-adjustment"
)

EXPECTED_RAW_FILES = {
    "cdps_rx_v7_coefficients.csv": (
        f"Manual download from UCSD CDPS+Rx page ({CDPS_LANDING_PAGE}). "
        "Registration required. Place CDPS+Rx v7 coefficient table here."
    ),
    "cdps_rx_v7_icd10_map.csv": (
        f"From the same UCSD CDPS+Rx package: ICD-10 -> CDPS-category mapping CSV."
    ),
    "cdps_rx_v7_ndc_map.csv": (
        f"From the same UCSD CDPS+Rx package: NDC -> CDPS-Rx-category mapping CSV."
    ),
    "hhs_hcc_v07_coefficients.csv": (
        f"From CMS Risk Adjustment landing page ({HHS_HCC_LANDING}): "
        f"Benefit Year {HHS_HCC_BENEFIT_YEAR} HHS-HCC v07 coefficients."
    ),
    "hhs_hcc_v07_icd10_map.csv": (
        f"From CMS Risk Adjustment landing page: HHS-HCC v07 ICD-10 -> HCC mapping."
    ),
}


def copy_if_exists(filename: str) -> bool:
    src = DATA_RAW / filename
    dst = DATA_CLEAN / filename
    if not src.exists():
        return False
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return True
    dst.write_bytes(src.read_bytes())
    return True


def main() -> None:
    available = []
    missing = []
    for name in EXPECTED_RAW_FILES:
        if copy_if_exists(name):
            available.append(name)
        else:
            missing.append((name, EXPECTED_RAW_FILES[name]))
    if missing:
        print("Missing model artifacts (Aim 2 will be skipped until provided):", file=sys.stderr)
        for name, msg in missing:
            print(f"  - {name}: {msg}", file=sys.stderr)
    print(f"Available: {len(available)}; missing: {len(missing)}")


if __name__ == "__main__":
    main()
