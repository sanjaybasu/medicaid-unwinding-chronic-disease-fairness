"""Smoke test: Aim 3 mediation decomposition on synthetic individual-level panel."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from config import TABLES
from aim3_synthesis import estimate_nde_nie, evalue, main as aim3_main


def test_nde_nie_runs(synthetic_individual_panel):
    panel = synthetic_individual_panel
    confounders = [c for c in panel.columns if c.startswith("x_")]
    result = estimate_nde_nie(
        panel,
        outcome="ed_visit_rate_per_year",
        treatment="procedural_disenrolled",
        mediator="signal_score_baseline",
        confounders=confounders,
    )
    assert result is not None
    assert "nde" in result and "nie" in result


def test_evalue_monotone():
    assert evalue(2.0) > evalue(1.5)
    assert evalue(1.0) == 1.0


def test_aim3_main_writes_output(synthetic_individual_panel):
    aim3_main()
    p = TABLES / "aim3_mediation.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) > 0
