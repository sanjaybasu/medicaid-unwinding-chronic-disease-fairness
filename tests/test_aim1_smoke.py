"""Smoke test: aim1_did_python end-to-end on synthetic panel."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from config import TABLES, DATA_CLEAN
from aim1_did_python import twfe_continuous_did, pre_trend_wald, event_study, main as aim1_main


def test_twfe_recovers_synthetic_effect(synthetic_state_quarter_panel):
    panel = synthetic_state_quarter_panel
    result = twfe_continuous_did(panel, "all_cause_mortality_per_100k_35_64")
    assert result is not None
    assert result["att_per_pp_proc_disenrollment"] > 0
    assert result["ci95_lower"] < result["att_per_pp_proc_disenrollment"] < result["ci95_upper"]


def test_pre_trend_test_runs(synthetic_state_quarter_panel):
    panel = synthetic_state_quarter_panel
    result = pre_trend_wald(panel, "all_cause_mortality_per_100k_35_64")
    assert result is not None
    assert "pre_trend_pvalue" in result


def test_event_study_runs(synthetic_state_quarter_panel):
    panel = synthetic_state_quarter_panel
    es = event_study(panel, "all_cause_mortality_per_100k_35_64")
    assert not es.empty
    assert {"q_offset", "high_intensity"} <= set(es.columns)


def test_aim1_main_writes_outputs(synthetic_state_quarter_panel):
    aim1_main()
    assert (TABLES / "aim1_python_results.csv").exists()
    df = pd.read_csv(TABLES / "aim1_python_results.csv")
    assert len(df) > 0
    assert "att_per_pp_proc_disenrollment" in df.columns
