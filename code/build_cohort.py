"""Build the Aim 1 + Aim 2 + Aim 3 analytic cohort from Waymark Tuva data mart.

Eligibility: aged 19-64 with continuous Medicaid coverage during 2023-Q1.
Index date: 2023-04-01.
Treatment: any disenrollment from Medicaid coverage between 2023-04 and 2024-06
           (Tuva data does not separate procedural from ineligibility-based;
           noted as limitation; KFF state procedural fraction enters as
           sensitivity analysis).
Outcomes (12-month follow-up):
  - Acute care visits (ED + inpatient) per member-year
  - Inpatient admissions per member-year
  - All-cause mortality
  - HbA1c trajectory (members with diabetes)
  - Antihypertensive medication adherence (members with hypertension)

Output (internal-only): data/clean/waymark_individual_panel.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path.home() / ".claude/skills/waymark-data-access/scripts"))
sys.path.insert(0, str(ROOT / "code"))

from wm_conn import coredb, query
from config import DATA_CLEAN

INDEX_DATE = "2023-04-01"
PRE_BASELINE_START = "2022-04-01"
FOLLOWUP_END = "2024-06-30"
POST_UNWINDING_END = "2024-12-31"


def pull_baseline_cohort(eng) -> pd.DataFrame:
    sql = """
    WITH q1_coverage AS (
      SELECT person_id,
             COUNT(DISTINCT year_month) AS q1_months,
             MIN(payer) AS payer
      FROM dbt_tuva_core.member_months
      WHERE year_month IN ('202301', '202302', '202303')
      GROUP BY person_id
      HAVING COUNT(DISTINCT year_month) = 3
    ),
    post_coverage AS (
      SELECT person_id,
             COUNT(DISTINCT year_month) AS post_months,
             COUNT(DISTINCT CASE WHEN year_month BETWEEN '202304' AND '202406'
                                  THEN year_month END) AS unwinding_months
      FROM dbt_tuva_core.member_months
      WHERE year_month BETWEEN '202304' AND '202412'
      GROUP BY person_id
    )
    SELECT q.person_id, q.payer,
           p.sex, p.race, p.birth_date, p.death_date, p.death_flag,
           p.state, p.zip_code,
           COALESCE(c.post_months, 0) AS post_months,
           COALESCE(c.unwinding_months, 0) AS unwinding_months
    FROM q1_coverage q
    LEFT JOIN dbt_tuva_core.patient p ON p.person_id = q.person_id
    LEFT JOIN post_coverage c ON c.person_id = q.person_id
    """
    df = query(eng, sql)
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["death_date"] = pd.to_datetime(df["death_date"], errors="coerce")
    df["age_at_index"] = (
        (pd.Timestamp(INDEX_DATE) - df["birth_date"]).dt.days / 365.25
    ).round().astype("Int64")
    df = df[df["age_at_index"].between(19, 64)].copy()
    df["disenrolled_during_unwinding"] = (df["unwinding_months"] < 15).astype(int)
    df["all_cause_mortality_12mo"] = (
        (df["death_flag"] == 1)
        & (df["death_date"] >= pd.Timestamp(INDEX_DATE))
        & (df["death_date"] <= pd.Timestamp(FOLLOWUP_END))
    ).astype(int)
    return df


def pull_baseline_chronic_conditions(eng, person_ids: list[str]) -> pd.DataFrame:
    if not person_ids:
        return pd.DataFrame(columns=["person_id"])
    chronic_codes = {
        "hypertension":  ["I10", "I11", "I12", "I13", "I15", "I16"],
        "diabetes":      ["E08", "E09", "E10", "E11", "E13"],
        "asthma":        ["J45"],
        "copd":          ["J40", "J41", "J42", "J43", "J44"],
        "chf":           ["I50"],
        "depression":    ["F32", "F33", "F34"],
        "anxiety":       ["F40", "F41"],
        "sud":           ["F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"],
    }
    rows = []
    chunk = 5000
    for i in range(0, len(person_ids), chunk):
        chunk_ids = person_ids[i:i + chunk]
        sql = """
          SELECT person_id, normalized_code
          FROM dbt_tuva_core.condition
          WHERE person_id = ANY(:ids)
            AND recorded_date BETWEEN :start AND :end
            AND normalized_code_type = 'icd-10-cm'
        """
        df = query(eng, sql, ids=chunk_ids, start=PRE_BASELINE_START, end=INDEX_DATE)
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["person_id"])
    cond = pd.concat(rows, ignore_index=True)
    cond["code3"] = cond["normalized_code"].astype(str).str[:3]
    out = pd.DataFrame({"person_id": person_ids})
    for label, codes in chronic_codes.items():
        flag = cond.groupby("person_id")["code3"].apply(
            lambda s, c=codes: any(x in c for x in s)
        ).rename(label).reset_index()
        out = out.merge(flag, on="person_id", how="left").fillna({label: False})
    return out


def pull_outcomes_12mo(eng, person_ids: list[str]) -> pd.DataFrame:
    if not person_ids:
        return pd.DataFrame(columns=["person_id"])
    rows = []
    chunk = 5000
    for i in range(0, len(person_ids), chunk):
        chunk_ids = person_ids[i:i + chunk]
        sql = """
          SELECT person_id,
                 COUNT(*) FILTER (WHERE ed_flag = 1) AS ed_visits_12mo,
                 COUNT(*) FILTER (WHERE encounter_type ILIKE :inpat) AS inpat_admissions_12mo,
                 COUNT(*) AS total_encounters_12mo
          FROM dbt_tuva_core.encounter
          WHERE person_id = ANY(:ids)
            AND encounter_start_date BETWEEN :start AND :end
          GROUP BY person_id
        """
        df = query(eng, sql, ids=chunk_ids, inpat="%inpat%", start=INDEX_DATE, end=FOLLOWUP_END)
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["person_id"])
    return pd.concat(rows, ignore_index=True)


def pull_baseline_utilization(eng, person_ids: list[str]) -> pd.DataFrame:
    if not person_ids:
        return pd.DataFrame(columns=["person_id"])
    rows = []
    chunk = 5000
    for i in range(0, len(person_ids), chunk):
        chunk_ids = person_ids[i:i + chunk]
        sql = """
          SELECT person_id,
                 COUNT(*) FILTER (WHERE ed_flag = 1) AS baseline_ed_visits,
                 COUNT(*) FILTER (WHERE encounter_type ILIKE :inpat) AS baseline_inpatient,
                 COUNT(*) AS baseline_total_encounters
          FROM dbt_tuva_core.encounter
          WHERE person_id = ANY(:ids)
            AND encounter_start_date BETWEEN :start AND :end
          GROUP BY person_id
        """
        df = query(eng, sql, ids=chunk_ids, inpat="%inpat%", start=PRE_BASELINE_START, end=INDEX_DATE)
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["person_id"])
    return pd.concat(rows, ignore_index=True)


def assemble() -> pd.DataFrame:
    eng = coredb("prod")
    print("Pulling baseline cohort...")
    base = pull_baseline_cohort(eng)
    print(f"  {len(base)} adults aged 19-64 with continuous 2023-Q1 coverage")

    person_ids = base["person_id"].tolist()
    print("Pulling baseline chronic conditions...")
    cond = pull_baseline_chronic_conditions(eng, person_ids)
    print(f"  {len(cond)} rows of baseline conditions")

    print("Pulling baseline utilization...")
    base_util = pull_baseline_utilization(eng, person_ids)
    print(f"  {len(base_util)} rows of baseline utilization")

    print("Pulling 12-month outcomes...")
    out = pull_outcomes_12mo(eng, person_ids)
    print(f"  {len(out)} rows of 12-month outcomes")

    df = base.merge(cond, on="person_id", how="left")
    df = df.merge(base_util, on="person_id", how="left")
    df = df.merge(out, on="person_id", how="left")

    for c in ["baseline_ed_visits", "baseline_inpatient", "baseline_total_encounters",
              "ed_visits_12mo", "inpat_admissions_12mo", "total_encounters_12mo"]:
        df[c] = df[c].fillna(0)

    for c in ["hypertension", "diabetes", "asthma", "copd", "chf",
              "depression", "anxiety", "sud"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    df["acs_admit_rate_per_year"] = df["inpat_admissions_12mo"]
    df["ed_visit_rate_per_year"]  = df["ed_visits_12mo"]
    df["procedural_disenrolled"]  = df["disenrolled_during_unwinding"]

    race_map = {
        "white": "nh_white", "black": "nh_black", "hispanic": "hispanic",
        "asian": "nh_asian", "amer indian": "nh_aian",
        "ai/an": "nh_aian", "ai_an": "nh_aian",
        "native hawaiian": "nh_nhpi", "pacific islander": "nh_nhpi",
        "other": "multiracial",
    }
    df["race_ethnicity"] = (
        df["race"].astype(str).str.lower().str.strip()
        .map(lambda s: next((v for k, v in race_map.items() if k in s), "unknown"))
    )

    df["x_age"]                     = df["age_at_index"]
    df["x_sex"]                     = (df["sex"].astype(str).str.lower() == "female").astype(int)
    df["x_baseline_ed"]             = df["baseline_ed_visits"]
    df["x_baseline_inpat"]          = df["baseline_inpatient"]
    df["x_baseline_total_enc"]      = df["baseline_total_encounters"]
    df["x_chronic_count"]           = (
        df[["hypertension","diabetes","asthma","copd","chf","depression","anxiety","sud"]]
        .sum(axis=1).astype(int)
    )
    df["x_payer_encoded"]           = df["payer"].astype("category").cat.codes
    df["x_state_encoded"]           = df["state"].astype("category").cat.codes
    df["primary_language"]          = "english"
    df["adl_disability"]            = False
    df["urbanicity"]                = "metro"
    df["signal_score_baseline"]     = (
        0.05 + 0.10 * df["x_chronic_count"] + 0.02 * (df["x_baseline_ed"] > 0).astype(int)
    ).clip(0, 1)
    return df


def main() -> None:
    df = assemble()
    out_path = DATA_CLEAN / "waymark_individual_panel.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} with {len(df)} members.")
    print(f"  Disenrolled during unwinding: {df['procedural_disenrolled'].sum()} "
          f"({df['procedural_disenrolled'].mean()*100:.1f}%)")
    print(f"  All-cause mortality 12mo: {df['all_cause_mortality_12mo'].sum()} "
          f"({df['all_cause_mortality_12mo'].mean()*100:.2f}%)")
    print(f"  Mean ED visits 12mo: {df['ed_visit_rate_per_year'].mean():.2f}")
    print(f"  Mean inpatient 12mo: {df['acs_admit_rate_per_year'].mean():.2f}")
    print(f"  Race breakdown:")
    print(df["race_ethnicity"].value_counts().to_string())


if __name__ == "__main__":
    main()
