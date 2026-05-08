"""Smoke test: Aim 1 target trial emulation (AIPW + causal forest) on synthetic
individual-level Medicaid panel.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from config import TABLES, DATA_CLEAN
from aim1_target_trial import (
    estimate_aipw_att,
    estimate_causal_forest_cate,
    main as aim1_target_trial_main,
)


def test_aipw_recovers_synthetic_effect_direction(synthetic_individual_panel):
    panel = synthetic_individual_panel
    result = estimate_aipw_att(panel, "ed_visit_rate_per_year")
    assert result is not None
    assert result["att"] > 0
    assert result["ci95_lower"] < result["att"] < result["ci95_upper"]
    assert result["estimator"] == "AIPW"
    assert result["mean_propensity"] > 0


def test_aipw_handles_binary_outcome(synthetic_individual_panel):
    panel = synthetic_individual_panel
    result = estimate_aipw_att(panel, "all_cause_mortality_12mo")
    assert result is not None


def test_causal_forest_cate_runs(synthetic_individual_panel):
    panel = synthetic_individual_panel
    out = estimate_causal_forest_cate(
        panel,
        outcome="ed_visit_rate_per_year",
        treatment="procedural_disenrolled",
        subgroup_cols=["race_ethnicity", "urbanicity"],
    )
    if out is None:
        pytest.skip("econml not installed in test env")
    assert "cate_mean" in out.columns
    assert len(out) > 0


def test_aim1_main_writes_outputs(synthetic_individual_panel):
    aim1_target_trial_main()
    p = TABLES / "aim1_target_trial_results.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert "att" in df.columns
    assert len(df) > 0
