"""Aim 2 appendix sensitivity — three-channel Monte Carlo microsim of risk-adjustment
fairness following the frailty-paper template (Basu and Berkowitz, 2026).

Channels:
  A — Algorithm design: ICD-10/NDC -> CDPS or HHS-HCC category mapping coverage
  B — Claims visibility: race-/rurality-differential utilization probability
  C — Documentation burden: specialist-access probability affecting category specificity

Outputs predicted-to-actual cost ratios and bootstrap CIs by subgroup, varying
detection and documentation parameters by +/- 1 SD of published estimates.

Output: tables/aim2_microsim_sensitivity.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_CLEAN, TABLES, SEED, BOOTSTRAP_ITERATIONS

DETECTION_BY_RACE = {
    "nh_white":   (0.72, 0.05),
    "nh_black":   (0.58, 0.05),
    "hispanic":   (0.61, 0.05),
    "nh_aian":    (0.52, 0.06),
    "nh_asian":   (0.69, 0.05),
    "multiracial": (0.65, 0.06),
}

CERT_BY_RACE = {
    "nh_white":   (0.81, 0.05),
    "nh_black":   (0.64, 0.06),
    "hispanic":   (0.67, 0.06),
    "nh_aian":    (0.55, 0.07),
    "nh_asian":   (0.78, 0.05),
    "multiracial": (0.72, 0.06),
}

NON_METRO_DETECTION_PENALTY = 0.08
NON_METRO_CERT_PENALTY = 0.06


def simulate(
    n_per_group: int = 2000,
    n_replications: int = 300,
    detection_shift: float = 0.0,
    cert_shift: float = 0.0,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for race, (det_mu, det_sd) in DETECTION_BY_RACE.items():
        cert_mu, cert_sd = CERT_BY_RACE[race]
        for non_metro in (False, True):
            ratios = np.empty(n_replications)
            for r in range(n_replications):
                det = float(np.clip(det_mu + detection_shift * det_sd + (rng.normal() * det_sd / 5), 0.05, 0.99))
                cert = float(np.clip(cert_mu + cert_shift * cert_sd + (rng.normal() * cert_sd / 5), 0.05, 0.99))
                if non_metro:
                    det = max(0.05, det - NON_METRO_DETECTION_PENALTY)
                    cert = max(0.05, cert - NON_METRO_CERT_PENALTY)
                true_need = rng.gamma(2.0, 5000.0, size=n_per_group)
                detected = rng.binomial(1, det, size=n_per_group).astype(bool)
                certified = rng.binomial(1, cert, size=n_per_group).astype(bool)
                captured = detected & certified
                predicted_score = np.where(captured, true_need * 0.9, true_need * 0.4)
                actual = true_need
                ratios[r] = float(predicted_score.sum() / actual.sum())
            rows.append({
                "race_ethnicity": race,
                "non_metro": bool(non_metro),
                "detection_shift_sd": detection_shift,
                "certification_shift_sd": cert_shift,
                "predicted_to_actual_ratio_mean": float(ratios.mean()),
                "predicted_to_actual_ratio_lower": float(np.quantile(ratios, 0.025)),
                "predicted_to_actual_ratio_upper": float(np.quantile(ratios, 0.975)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    frames = []
    for det_shift in (-1.0, 0.0, 1.0):
        for cert_shift in (-1.0, 0.0, 1.0):
            df = simulate(detection_shift=det_shift, cert_shift=cert_shift)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(TABLES / "aim2_microsim_sensitivity.csv", index=False)
    print(f"Wrote {TABLES / 'aim2_microsim_sensitivity.csv'} with {len(out)} rows")


if __name__ == "__main__":
    main()
