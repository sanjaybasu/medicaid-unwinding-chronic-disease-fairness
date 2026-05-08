"""Waymark Medicaid cohort assembly via Vault auth -> lighthouse / coredb / dbt_tuva_core.

Pulls the longitudinal member-month panel needed for Aim 1 (target trial emulation),
Aim 2 (Signal stress-test across windows), and Aim 3 (mediation).

Outputs are NOT committed to the public repository. Patient-level data remain
internal to Waymark under the existing IRB-equivalent / Privacy and Compliance
determination that extends Patel-Baum-Basu 2024 (Sci Rep 14, 824).

Output (internal-only):
  data/clean/waymark_member_month.parquet
  data/clean/waymark_outcomes.parquet
  data/clean/waymark_features_signal_input.parquet
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

import pandas as pd

from config import DATA_CLEAN, DATA_RAW, PANEL_START, PANEL_END

VAULT_AUTH_DOC = (
    "Run the Waymark Vault auth dance per the waymark-data-access skill: "
    "VPN -> OIDC token refresh -> just-in-time DB credentials. "
    "Then `wm_conn.lighthouse()` and `wm_conn.coredb()` accessors return live "
    "SQLAlchemy connections. See ~/.claude/skills/waymark-data-access/SKILL.md."
)


def _wm_conn(target: str):
    try:
        from waymark_local import wm_conn  # noqa
    except Exception as e:
        raise RuntimeError(
            f"Could not import wm_conn ({e}). {VAULT_AUTH_DOC}"
        ) from e
    if target == "lighthouse":
        return wm_conn.lighthouse()
    if target == "coredb":
        return wm_conn.coredb()
    if target == "dbt_tuva":
        return wm_conn.coredb(schema="dbt_tuva_core")
    raise ValueError(f"Unknown Waymark target {target!r}")


COHORT_QUERY = """
SELECT
  m.member_id,
  m.year_month,
  m.is_medicaid,
  m.coverage_segment,
  m.mco_id,
  m.state_abbr,
  m.disenrollment_event,
  m.disenrollment_reason_code,
  d.age_at_month,
  d.sex,
  d.race_ethnicity_coded,
  d.primary_language_english,
  d.adl_disability_flag,
  d.urbanicity,
  d.zip3,
  d.adi_state_decile,
  d.svi_overall_pct,
  d.dual_eligible_flag,
  d.pregnancy_only_coverage_flag,
  d.esrd_flag
FROM coredb.member_month m
LEFT JOIN coredb.member_demographics d
  ON m.member_id = d.member_id AND m.year_month = d.year_month
WHERE m.year_month BETWEEN :start AND :end
  AND m.is_medicaid = TRUE
"""

OUTCOMES_QUERY = """
SELECT
  member_id,
  year_month,
  hba1c_value,
  systolic_bp_value,
  diastolic_bp_value,
  acs_admission_flag,
  ed_visit_flag,
  ed_non_emergent_flag,
  inpatient_admission_flag,
  total_paid_amount,
  death_flag,
  death_date
FROM dbt_tuva_core.member_month_outcomes
WHERE year_month BETWEEN :start AND :end
"""

SIGNAL_FEATURE_QUERY = """
SELECT
  member_id,
  feature_name,
  feature_value,
  lookback_window_months,
  feature_window_end
FROM coredb.member_signal_features
WHERE feature_window_end BETWEEN :start AND :end
"""


def pull_cohort() -> pd.DataFrame:
    conn = _wm_conn("coredb")
    return pd.read_sql(
        COHORT_QUERY,
        conn,
        params={"start": PANEL_START, "end": PANEL_END},
    )


def pull_outcomes() -> pd.DataFrame:
    conn = _wm_conn("dbt_tuva")
    return pd.read_sql(
        OUTCOMES_QUERY,
        conn,
        params={"start": PANEL_START, "end": PANEL_END},
    )


def pull_signal_features() -> pd.DataFrame:
    conn = _wm_conn("coredb")
    return pd.read_sql(
        SIGNAL_FEATURE_QUERY,
        conn,
        params={"start": PANEL_START, "end": PANEL_END},
    )


def main() -> None:
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    try:
        cohort = pull_cohort()
        outcomes = pull_outcomes()
        features = pull_signal_features()
    except RuntimeError as e:
        print(f"WARNING: Waymark data pull skipped: {e}", file=sys.stderr)
        return

    cohort.to_parquet(DATA_CLEAN / "waymark_member_month.parquet", index=False)
    outcomes.to_parquet(DATA_CLEAN / "waymark_outcomes.parquet", index=False)
    features.to_parquet(DATA_CLEAN / "waymark_features_signal_input.parquet", index=False)

    print(
        f"Waymark cohort: {len(cohort)} member-months across "
        f"{cohort['member_id'].nunique()} members; "
        f"{outcomes['member_id'].nunique()} members with outcomes."
    )


if __name__ == "__main__":
    main()
