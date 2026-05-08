"""Synthetic fixtures for unit tests. No real public data required."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from config import DATA_CLEAN, TABLES, FIGURES, PANEL_START, PANEL_END
from utils import US_STATES


@pytest.fixture(scope="session")
def synthetic_state_quarter_panel(tmp_path_factory) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    states = list(US_STATES.keys())
    quarters = pd.date_range(PANEL_START, PANEL_END, freq="QS")
    rows = []
    state_baseline_admit = {s: rng.normal(50, 5) for s in states}
    state_baseline_mort = {s: rng.normal(180, 20) for s in states}
    state_terminal_intensity = {s: rng.beta(2, 5) * 0.6 for s in states}
    for state in states:
        for q in quarters:
            year = q.year
            quarter = (q.month - 1) // 3 + 1
            post = int(q >= pd.Timestamp("2023-04-01"))
            time_id = year * 4 + (quarter - 1)
            qs_pct = state_terminal_intensity[state]
            months_into = max(0, (q - pd.Timestamp("2023-04-01")).days // 30)
            cum_intensity = qs_pct * min(1.0, months_into / 18.0) if post else 0.0
            effect = 1.5 * cum_intensity * 100
            rows.append({
                "state_abbr": state,
                "state_id": state,
                "year": year,
                "quarter": quarter,
                "qstart": q,
                "time_id": time_id,
                "post": post,
                "cumulative_procedural_disenrollment_rate": cum_intensity,
                "procedural_fraction": 0.7,
                "acs_admit_rate_per_1000": state_baseline_admit[state] + 0.1 * effect + rng.normal(0, 1),
                "bh_ed_rate_per_1000": 25 + 0.05 * effect + rng.normal(0, 0.5),
                "all_cause_mortality_per_100k_35_64": state_baseline_mort[state] + 0.4 * effect + rng.normal(0, 4),
                "amenable_mortality_per_100k_35_64": 90 + 0.3 * effect + rng.normal(0, 3),
                "median_household_income": 60000,
                "share_19_64_below_138_fpl": 0.15,
                "aca_expanded": True,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_meps() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 5000
    races = ["nh_white", "nh_black", "hispanic", "nh_asian", "nh_aian", "multiracial"]
    df = pd.DataFrame({
        "dupersid": [f"P{i:06d}" for i in range(n)],
        "panel_year": rng.choice([2018, 2019, 2020, 2021, 2022], n),
        "age": rng.integers(19, 65, n),
        "sex": rng.choice([1, 2], n),
        "weight": rng.uniform(500, 2000, n),
        "expenditure": rng.lognormal(8.0, 1.2, n),
        "race_ethnicity": rng.choice(races, n, p=[0.4, 0.2, 0.25, 0.08, 0.02, 0.05]),
        "non_metro": rng.random(n) < 0.15,
        "medicaid_any_month": True,
        "dual_eligible": False,
    })
    return df


@pytest.fixture(scope="session")
def synthetic_individual_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 8000
    races = ["nh_white", "nh_black", "hispanic", "nh_asian", "nh_aian", "multiracial"]
    languages = ["english", "spanish", "other"]
    df = pd.DataFrame({
        "member_id": [f"M{i:06d}" for i in range(n)],
        "x_age": rng.integers(19, 65, n),
        "x_sex": rng.choice([0, 1], n),
        "x_baseline_visits": rng.poisson(3, n),
        "x_baseline_admits": rng.poisson(0.4, n),
        "x_baseline_signal": rng.beta(2, 8, n),
        "x_chronic_count": rng.poisson(1.2, n),
        "x_adi_decile": rng.integers(1, 11, n),
        "race_ethnicity": rng.choice(races, n, p=[0.4, 0.2, 0.25, 0.08, 0.02, 0.05]),
        "primary_language": rng.choice(languages, n, p=[0.75, 0.18, 0.07]),
        "adl_disability": rng.random(n) < 0.12,
        "urbanicity": rng.choice(["metro", "non_metro"], n, p=[0.85, 0.15]),
    })
    propensity = 1 / (1 + np.exp(-(0.5 * df["x_chronic_count"] - 0.3 * df["x_age"] / 50 + rng.normal(0, 0.5, n))))
    df["procedural_disenrolled"] = (rng.random(n) < propensity).astype(int)
    treatment_effect = 0.4
    df["all_cause_mortality_12mo"] = (
        rng.random(n) < (0.02 + treatment_effect * 0.05 * df["procedural_disenrolled"])
    ).astype(int)
    df["acs_admit_rate_per_year"] = (
        0.3 * df["x_chronic_count"]
        + treatment_effect * 0.5 * df["procedural_disenrolled"]
        + rng.normal(0, 0.2, n)
    )
    df["ed_visit_rate_per_year"] = (
        0.5 + 0.3 * df["x_chronic_count"]
        + treatment_effect * 0.6 * df["procedural_disenrolled"]
        + rng.normal(0, 0.3, n)
    )
    df["hba1c_change_12mo"] = (
        treatment_effect * 0.4 * df["procedural_disenrolled"]
        + rng.normal(0, 0.5, n)
    )
    df["sbp_change_12mo"] = (
        treatment_effect * 2.0 * df["procedural_disenrolled"]
        + rng.normal(0, 3.0, n)
    )
    df["signal_score_baseline"] = df["x_baseline_signal"]
    return df


@pytest.fixture(autouse=True)
def write_synthetic_panel_to_clean(synthetic_state_quarter_panel, synthetic_meps, synthetic_individual_panel):
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    synthetic_state_quarter_panel.to_parquet(DATA_CLEAN / "state_quarter_panel.parquet", index=False)
    synthetic_meps.to_parquet(DATA_CLEAN / "meps_hc_medicaid_pooled.parquet", index=False)
    synthetic_individual_panel.to_parquet(DATA_CLEAN / "waymark_individual_panel.parquet", index=False)
    yield
