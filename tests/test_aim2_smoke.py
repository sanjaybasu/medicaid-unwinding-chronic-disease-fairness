"""Smoke test: Aim 2 microsim runs end-to-end. Aim 2 fairness audit requires
real CDPS+Rx and HHS-HCC artifacts so its main() exits gracefully without them;
we test only that the synthetic-data path through the metrics functions works.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from config import TABLES
from aim2_microsim import simulate, main as microsim_main
from aim2_fairness_audit import fairness_metrics_block


def test_microsim_returns_dataframe():
    df = simulate(n_per_group=200, n_replications=20, detection_shift=0, cert_shift=0)
    assert not df.empty
    assert "predicted_to_actual_ratio_mean" in df.columns


def test_microsim_main_writes_output():
    microsim_main()
    p = TABLES / "aim2_microsim_sensitivity.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) > 0


def test_fairness_metrics_block_runs_on_synthetic():
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "actual": rng.lognormal(8.0, 1.0, n),
        "predicted": rng.lognormal(8.0, 1.0, n),
        "_sg": rng.choice(["0", "1"], n).astype(str),
    })
    out = fairness_metrics_block(df, "actual", "predicted", "_sg")
    assert not out.empty
    assert "predicted_to_actual_ratio" in out.columns
