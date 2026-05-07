"""Shared utilities: state codes, race/ethnicity coding, bootstrap, FDR."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable, Sequence

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

ACA_EXPANSION_AS_OF_2023 = {
    "AL": False, "AK": True, "AZ": True, "AR": True, "CA": True, "CO": True,
    "CT": True, "DE": True, "DC": True, "FL": False, "GA": False, "HI": True,
    "ID": True, "IL": True, "IN": True, "IA": True, "KS": False, "KY": True,
    "LA": True, "ME": True, "MD": True, "MA": True, "MI": True, "MN": True,
    "MS": False, "MO": True, "MT": True, "NE": True, "NV": True, "NH": True,
    "NJ": True, "NM": True, "NY": True, "NC": True, "ND": True, "OH": True,
    "OK": True, "OR": True, "PA": True, "RI": True, "SC": False, "SD": True,
    "TN": False, "TX": False, "UT": True, "VT": True, "VA": True, "WA": True,
    "WV": True, "WI": False, "WY": False,
}


def code_race_ethnicity(race_code: pd.Series, hispanic_flag: pd.Series) -> pd.Series:
    out = pd.Series(index=race_code.index, dtype="object")
    h = hispanic_flag.astype(bool)
    out[h] = "hispanic"
    out[(~h) & (race_code == "white")] = "nh_white"
    out[(~h) & (race_code == "black")] = "nh_black"
    out[(~h) & (race_code == "asian")] = "nh_asian"
    out[(~h) & (race_code == "aian")] = "nh_aian"
    out[(~h) & (race_code == "nhpi")] = "nh_nhpi"
    out[(~h) & (race_code == "multiracial")] = "multiracial"
    out[out.isna()] = "unknown"
    return out


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_iterations: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    stratify: np.ndarray | None = None,
) -> tuple[float, float, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(data)
    estimates = np.empty(n_iterations)
    if stratify is None:
        for i in range(n_iterations):
            idx = rng.integers(0, n, n)
            estimates[i] = statistic(data[idx])
    else:
        groups = np.unique(stratify)
        for i in range(n_iterations):
            idx_parts = []
            for g in groups:
                gi = np.where(stratify == g)[0]
                idx_parts.append(rng.choice(gi, len(gi), replace=True))
            idx = np.concatenate(idx_parts)
            estimates[i] = statistic(data[idx])
    point = statistic(data)
    lo = float(np.quantile(estimates, alpha / 2))
    hi = float(np.quantile(estimates, 1 - alpha / 2))
    return float(point), lo, hi


def fdr_bh(pvalues: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = np.arange(1, n + 1) / n * alpha
    passing = ranked <= thresholds
    if not passing.any():
        return np.zeros(n, dtype=bool)
    k_max = np.max(np.where(passing)[0])
    rejected_in_rank_order = np.zeros(n, dtype=bool)
    rejected_in_rank_order[: k_max + 1] = True
    out = np.empty(n, dtype=bool)
    out[order] = rejected_in_rank_order
    return out


def calibration_intercept_slope(
    actual: np.ndarray,
    predicted: np.ndarray,
    log_link: bool = True,
    eps: float = 1.0,
) -> tuple[float, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if log_link:
        a = np.log(a + eps)
        p = np.log(p + eps)
    slope, intercept, _, _, _ = stats.linregress(p, a)
    return float(intercept), float(slope)


def expected_calibration_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    n_bins: int = 10,
) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_idx = np.digitize(p, edges) - 1
    n = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        ece += abs(p[mask].mean() - a[mask].mean()) * mask.sum() / n
    return float(ece)


def predicted_to_actual_ratio(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = float(np.mean(actual))
    p = float(np.mean(predicted))
    if a == 0:
        return float("nan")
    return p / a


def stratified_bootstrap_metric(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    stratify_col: str,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_iterations)
    groups = df[stratify_col].unique()
    for i in range(n_iterations):
        parts = []
        for g in groups:
            sub = df[df[stratify_col] == g]
            parts.append(sub.sample(n=len(sub), replace=True, random_state=int(rng.integers(0, 2**31))))
        boot = pd.concat(parts, ignore_index=True)
        estimates[i] = metric_fn(boot)
    point = metric_fn(df)
    lo, hi = float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2))
    return float(point), lo, hi
