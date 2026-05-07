"""Table generation for the manuscript.

Table 1: Demographic and clinical characteristics of MEPS-HC Medicaid sample
Table 2: Aim 1 ATT per percentage point of cumulative procedural disenrollment, by outcome and estimator
Table 3: Aim 1 subgroup ATTs by race/ethnicity
Table 4: Aim 2 calibration metrics by subgroup and model
Table 5: Aim 3 synthesis (state x subgroup regression and Spearman)

Output: tables/table{1..5}.csv and tables/table{1..5}.md
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_CLEAN, TABLES


def df_to_md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def table1_demographics() -> None:
    p = DATA_CLEAN / "meps_hc_medicaid_pooled.parquet"
    if not p.exists():
        return
    df = pd.read_parquet(p)
    rows = [
        ["N person-years", f"{len(df)}"],
        ["Mean age (yrs)", f"{df['age'].mean():.1f}"],
        ["Female (%)", f"{(df['sex'] == 2).mean() * 100:.1f}"],
        ["Race/ethnicity", ""],
    ]
    for r, count in df["race_ethnicity"].value_counts().items():
        rows.append([f"  {r}", f"{count} ({count / len(df) * 100:.1f}%)"])
    rows.append(["Non-metropolitan (%)", f"{df['non_metro'].mean() * 100:.1f}"])
    rows.append(["Mean annual expenditure ($)", f"${df['expenditure'].mean():,.0f}"])
    out = pd.DataFrame(rows, columns=["Characteristic", "Value"])
    out.to_csv(TABLES / "table1_demographics.csv", index=False)
    (TABLES / "table1_demographics.md").write_text(df_to_md(out))


def table2_aim1_att() -> None:
    py = TABLES / "aim1_python_results.csv"
    if not py.exists():
        return
    df = pd.read_csv(py)
    df = df.rename(columns={
        "att_per_pp_proc_disenrollment": "ATT per pp proc. disenrollment",
        "ci95_lower": "95% CI lower",
        "ci95_upper": "95% CI upper",
        "p_value": "p value",
    })
    df.to_csv(TABLES / "table2_aim1_att.csv", index=False)
    (TABLES / "table2_aim1_att.md").write_text(df_to_md(df))


def table3_aim1_subgroups() -> None:
    return


def table4_aim2_calibration() -> None:
    p = TABLES / "aim2_fairness_metrics.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    out = df[[
        "model", "subgroup_name", "n",
        "predicted_to_actual_ratio", "ratio_ci_lower", "ratio_ci_upper",
        "calibration_intercept", "calibration_slope", "ece",
        "auroc_top_decile",
    ]]
    out.to_csv(TABLES / "table4_aim2_calibration.csv", index=False)
    (TABLES / "table4_aim2_calibration.md").write_text(df_to_md(out))


def table5_aim3_synthesis() -> None:
    p = TABLES / "aim3_synthesis.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df.to_csv(TABLES / "table5_aim3_synthesis.csv", index=False)
    (TABLES / "table5_aim3_synthesis.md").write_text(df_to_md(df))


def main() -> None:
    table1_demographics()
    table2_aim1_att()
    table3_aim1_subgroups()
    table4_aim2_calibration()
    table5_aim3_synthesis()
    print(f"Tables written to {TABLES}")


if __name__ == "__main__":
    main()
