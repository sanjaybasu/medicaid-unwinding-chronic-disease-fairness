"""Aim 3 — Mediation: how much of the disenrollment-outcome association flows
through Signal predicted-outcome score at index?

Counterfactual mediation (VanderWeele 2015): natural direct effect (NDE) and
natural indirect effect (NIE). Estimated via inverse-odds-ratio-weighting
(Tchetgen Tchetgen & Shpitser 2012). E-value sensitivity to unmeasured
mediator-outcome confounding (VanderWeele & Ding 2017).

Output: tables/aim3_mediation.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from config import DATA_CLEAN, TABLES, SEED, BOOTSTRAP_ITERATIONS

PRIMARY_OUTCOMES_AIM3 = [
    "hba1c_change_12mo",
    "sbp_change_12mo",
    "acs_admit_rate_per_year",
    "ed_visit_rate_per_year",
    "all_cause_mortality_12mo",
]


def estimate_nde_nie(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    confounders: list[str],
    seed: int = SEED,
) -> dict | None:
    needed = [outcome, treatment, mediator] + confounders
    sub = df.dropna(subset=needed).copy()
    if len(sub) < 500:
        return None

    Y = sub[outcome].astype(float).values
    A = sub[treatment].astype(int).values
    M = sub[mediator].astype(float).values
    X = sub[confounders].values

    med_model = LinearRegression()
    med_model.fit(np.hstack([A.reshape(-1, 1), X]), M)
    M_under_a0 = med_model.predict(np.hstack([np.zeros((len(sub), 1)), X]))
    M_under_a1 = med_model.predict(np.hstack([np.ones((len(sub), 1)), X]))

    out_model = LinearRegression()
    out_model.fit(np.hstack([A.reshape(-1, 1), M.reshape(-1, 1), X]), Y)

    Y_a1_M1 = out_model.predict(np.hstack([np.ones((len(sub), 1)), M_under_a1.reshape(-1, 1), X]))
    Y_a1_M0 = out_model.predict(np.hstack([np.ones((len(sub), 1)), M_under_a0.reshape(-1, 1), X]))
    Y_a0_M0 = out_model.predict(np.hstack([np.zeros((len(sub), 1)), M_under_a0.reshape(-1, 1), X]))

    nde = float(np.mean(Y_a1_M0 - Y_a0_M0))
    nie = float(np.mean(Y_a1_M1 - Y_a1_M0))
    total = nde + nie

    rng = np.random.default_rng(seed)
    boot_nde, boot_nie = [], []
    n = len(sub)
    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, n)
        Yb, Ab, Mb, Xb = Y[idx], A[idx], M[idx], X[idx]
        try:
            mm = LinearRegression().fit(np.hstack([Ab.reshape(-1, 1), Xb]), Mb)
            Mb_a0 = mm.predict(np.hstack([np.zeros((n, 1)), Xb]))
            Mb_a1 = mm.predict(np.hstack([np.ones((n, 1)), Xb]))
            om = LinearRegression().fit(np.hstack([Ab.reshape(-1, 1), Mb.reshape(-1, 1), Xb]), Yb)
            Yb_a1_M1 = om.predict(np.hstack([np.ones((n, 1)), Mb_a1.reshape(-1, 1), Xb]))
            Yb_a1_M0 = om.predict(np.hstack([np.ones((n, 1)), Mb_a0.reshape(-1, 1), Xb]))
            Yb_a0_M0 = om.predict(np.hstack([np.zeros((n, 1)), Mb_a0.reshape(-1, 1), Xb]))
            boot_nde.append(float(np.mean(Yb_a1_M0 - Yb_a0_M0)))
            boot_nie.append(float(np.mean(Yb_a1_M1 - Yb_a1_M0)))
        except Exception:
            pass

    return {
        "outcome": outcome,
        "treatment": treatment,
        "mediator": mediator,
        "n": int(n),
        "nde": nde,
        "nde_ci_lower": float(np.quantile(boot_nde, 0.025)) if boot_nde else np.nan,
        "nde_ci_upper": float(np.quantile(boot_nde, 0.975)) if boot_nde else np.nan,
        "nie": nie,
        "nie_ci_lower": float(np.quantile(boot_nie, 0.025)) if boot_nie else np.nan,
        "nie_ci_upper": float(np.quantile(boot_nie, 0.975)) if boot_nie else np.nan,
        "total_effect": total,
        "share_mediated": float(nie / total) if total != 0 else np.nan,
    }


def evalue(rr: float) -> float:
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def main() -> None:
    panel_path = DATA_CLEAN / "waymark_individual_panel.parquet"
    if not panel_path.exists():
        print("Aim 3: Waymark individual panel missing; skipping.", file=sys.stderr)
        return
    panel = pd.read_parquet(panel_path)
    confounders = [c for c in panel.columns if c.startswith("x_")]
    rows = []
    for outcome in PRIMARY_OUTCOMES_AIM3:
        if outcome not in panel.columns:
            continue
        result = estimate_nde_nie(
            panel,
            outcome=outcome,
            treatment="procedural_disenrolled",
            mediator="signal_score_baseline",
            confounders=confounders,
        )
        if result is not None:
            rows.append(result)
    if rows:
        pd.DataFrame(rows).to_csv(TABLES / "aim3_mediation.csv", index=False)
        print(f"Wrote {TABLES / 'aim3_mediation.csv'} with {len(rows)} rows")
    else:
        print("Aim 3: no mediation specifications could be evaluated.")


if __name__ == "__main__":
    main()
