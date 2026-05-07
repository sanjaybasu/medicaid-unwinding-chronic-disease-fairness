"""Tests for shared utilities."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from utils import (
    bootstrap_ci,
    fdr_bh,
    calibration_intercept_slope,
    expected_calibration_error,
    predicted_to_actual_ratio,
    code_race_ethnicity,
    US_STATES,
    ACA_EXPANSION_AS_OF_2023,
)


def test_us_states_count():
    assert len(US_STATES) == 51
    assert "DC" in US_STATES


def test_aca_expansion_count():
    assert len(ACA_EXPANSION_AS_OF_2023) == 51


def test_bootstrap_ci_recovers_mean():
    rng = np.random.default_rng(42)
    data = rng.normal(10.0, 2.0, 500)
    point, lo, hi = bootstrap_ci(data, np.mean, n_iterations=200, alpha=0.05, rng=rng)
    assert lo < point < hi
    assert abs(point - 10.0) < 0.5


def test_fdr_bh_basic():
    p = [0.001, 0.01, 0.04, 0.5, 0.8]
    rejected = fdr_bh(p, alpha=0.05)
    assert rejected[0] and rejected[1]
    assert not rejected[3]
    assert not rejected[4]


def test_calibration_recovers_identity():
    rng = np.random.default_rng(42)
    pred = rng.uniform(100, 10000, 1000)
    actual = pred * 1.0 + rng.normal(0, 50, 1000)
    intercept, slope = calibration_intercept_slope(actual, pred, log_link=True)
    assert abs(slope - 1.0) < 0.1


def test_predicted_to_actual_ratio():
    actual = np.array([100, 200, 300])
    predicted = np.array([100, 200, 300])
    assert abs(predicted_to_actual_ratio(actual, predicted) - 1.0) < 1e-6


def test_ece_zero_for_perfect_calibration():
    pred = np.array([0.1] * 100 + [0.9] * 100)
    actual = np.array([0.1] * 100 + [0.9] * 100)
    ece = expected_calibration_error(actual, pred, n_bins=2)
    assert ece < 1e-6


def test_code_race_ethnicity():
    race = pd.Series(["white", "black", "asian", "white"])
    hispanic = pd.Series([False, False, False, True])
    out = code_race_ethnicity(race, hispanic)
    assert out.tolist() == ["nh_white", "nh_black", "nh_asian", "hispanic"]
