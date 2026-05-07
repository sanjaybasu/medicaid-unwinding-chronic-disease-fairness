# Medicaid Unwinding, Chronic-Disease Outcomes, and Risk-Adjustment Fairness

Code repository for the analysis behind the manuscript at `notebooks/medicaid-unwinding-chronic-disease-fairness/main_text.md`.

This repository contains code only. Raw and cleaned data, manuscript artifacts, and the analytical protocol live alongside the code under their respective directories but are excluded from version control via `.gitignore` where they contain or reproduce restricted data.

## Layout

```
code/        Python and R analysis modules
data/raw/    Downloaded public-data files (excluded from git)
data/clean/  Analytic datasets (excluded from git)
figures/    Generated figures (PNG, SVG)
tables/     Generated tables (CSV, Markdown)
tests/      Unit tests and synthetic fixtures
env/        Pinned environment artifacts (requirements.txt, R DESCRIPTION)
```

## Reproduce

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
make data       # downloads public data
make analyze    # runs the analysis pipeline
make figures
make tables
```

## Data sources (all public)

- KFF Medicaid Enrollment & Unwinding Tracker
- HCUPnet (AHRQ)
- CDC WONDER mortality
- BRFSS 2018-2024
- MEPS-HC 2018-2022 (AHRQ)
- CDPS+Rx model artifacts (UCSD)
- HHS-HCC v07 model artifacts (CMS)
- Census ACS, BLS, HRSA AHRF for covariates

## License

MIT.
