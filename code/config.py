"""Project-wide configuration: paths, constants, seeds."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
TESTS = ROOT / "tests"

for d in (DATA_RAW, DATA_CLEAN, FIGURES, TABLES):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
BOOTSTRAP_ITERATIONS = 1000

PRE_PERIOD_END = "2023-03-31"
TREATED_PERIOD_START = "2023-04-01"
PANEL_START = "2018-01-01"
PANEL_END = "2024-12-31"

PRIMARY_OUTCOMES = [
    "acs_admit_rate_per_1000",
    "bh_ed_rate_per_1000",
    "all_cause_mortality_per_100k_35_64",
    "amenable_mortality_per_100k_35_64",
]

SECONDARY_OUTCOMES_BRFSS = [
    "htn_treated_pct",
    "dm_a1c_past_year_pct",
    "dental_visit_past_year_pct",
    "cost_related_skipped_care_pct",
]

PRIMARY_RACE_ETHNICITY = [
    "nh_white",
    "nh_black",
    "hispanic",
    "nh_asian",
    "nh_aian",
    "multiracial",
]

PRIMARY_SUBGROUPS_AIM2 = [
    "nh_white",
    "nh_black",
    "hispanic",
    "nh_asian",
    "nh_aian",
    "primary_lang_non_english",
    "adl_disabled",
    "non_metro",
    "mh_diagnosis",
    "sud_diagnosis",
]

FDR_ALPHA = 0.05

HONEST_DID_M_VALUES = [0.5, 1.0]

PQI_CHRONIC_CARE_SET = [1, 3, 5, 7, 8, 11, 13, 14, 15, 16, 92]

NOLTE_MCKEE_AMENABLE_ICD10 = [
    "A15-A19", "A35-A37", "A39", "B05", "B16-B19",
    "C18-C21", "C50", "C53-C55", "C73", "C81",
    "D55-D89", "E10-E14", "E40-E46",
    "I00-I09", "I10-I15", "I20-I25", "I26-I28", "I60-I69",
    "J09-J18", "J40-J47",
    "K25-K27", "K70-K77",
    "L00-L08",
    "M86",
    "N00-N08", "N10-N12", "N17-N19", "N20-N23",
    "O00-O99",
    "P00-P96",
    "Q00-Q99",
]

CDPS_MODEL_VERSION = 7
HHS_HCC_BENEFIT_YEAR = 2024

HYPOTHESIS_FAMILIES = {
    "F1": "Aim 1 primary outcomes",
    "F2": "Aim 1 subgroup interactions",
    "F3": "Aim 2 subgroup miscalibration",
    "F4": "Aim 3 synthesis",
}
