"""Figure generation for the manuscript.

Figure 1: Map of state cumulative procedural disenrollment rates
Figure 2: Event-study plots of primary outcomes by high/low disenrollment-intensity groups
Figure 3: Predicted-to-actual expenditure ratios by subgroup (CDPS+Rx, HHS-HCC v07)
Figure 4: Tie-in scatter — subgroup miscalibration vs disenrollment exposure
Figure 5: Sensitivity (Honest-DiD + microsim) summary

PNG and SVG outputs to figures/.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from config import DATA_CLEAN, TABLES, FIGURES, PRIMARY_OUTCOMES

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
})

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def figure1_state_intensity_map() -> None:
    panel_path = DATA_CLEAN / "state_quarter_panel.parquet"
    if not panel_path.exists():
        return
    panel = pd.read_parquet(panel_path)
    state_terminal = (
        panel.groupby("state_abbr")["cumulative_procedural_disenrollment_rate"]
        .max()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    state_terminal = state_terminal.sort_values("cumulative_procedural_disenrollment_rate")
    ax.barh(state_terminal["state_abbr"], state_terminal["cumulative_procedural_disenrollment_rate"], color="#1f77b4")
    ax.set_xlabel("Cumulative procedural disenrollment rate (% of pre-unwinding enrollees)")
    ax.set_ylabel("State")
    ax.set_title("Figure 1. State-level cumulative procedural disenrollment, 2023-Q2 to 2024-Q4")
    save(fig, "figure1_state_intensity")


def figure2_event_study() -> None:
    es_path = TABLES / "aim1_event_study.csv"
    if not es_path.exists():
        return
    es = pd.read_csv(es_path)
    outcomes = es["outcome"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, outcome in zip(axes.flat, outcomes):
        sub = es[es["outcome"] == outcome]
        for hi, label in [(1, "High intensity"), (0, "Low intensity")]:
            s = sub[sub["high_intensity"] == hi].sort_values("q_offset")
            ax.plot(s["q_offset"], s[outcome], marker="o", label=label)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(outcome.replace("_", " "))
        ax.set_xlabel("Quarters from April 2023")
        ax.set_ylabel("Outcome")
        ax.legend(frameon=False)
    fig.suptitle("Figure 2. Event-study trajectories by state disenrollment intensity")
    fig.tight_layout()
    save(fig, "figure2_event_study")


def figure3_subgroup_ratios() -> None:
    p = TABLES / "aim2_fairness_metrics.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, df["model"].unique()):
        sub = df[df["model"] == model].sort_values("predicted_to_actual_ratio")
        y = np.arange(len(sub))
        ax.errorbar(
            sub["predicted_to_actual_ratio"], y,
            xerr=[sub["predicted_to_actual_ratio"] - sub["ratio_ci_lower"],
                  sub["ratio_ci_upper"] - sub["predicted_to_actual_ratio"]],
            fmt="o", color="#1f77b4",
        )
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["subgroup_name"])
        ax.set_xlabel("Predicted / actual expenditure ratio")
        ax.set_title(model)
    fig.suptitle("Figure 3. Risk-adjustment subgroup miscalibration on Medicaid sample")
    fig.tight_layout()
    save(fig, "figure3_subgroup_ratios")


def figure4_synthesis_scatter() -> None:
    aim3_path = TABLES / "aim3_synthesis.csv"
    aim2_path = TABLES / "aim2_fairness_metrics.csv"
    if not (aim3_path.exists() and aim2_path.exists()):
        return
    aim2 = pd.read_csv(aim2_path)
    aim2["miscalibration_magnitude"] = (aim2["predicted_to_actual_ratio"] - 1.0).abs()
    fig, ax = plt.subplots(figsize=(7, 6))
    for model in aim2["model"].unique():
        sub = aim2[aim2["model"] == model]
        ax.scatter(
            sub["miscalibration_magnitude"],
            np.zeros(len(sub)),
            s=80, label=model,
        )
        for _, row in sub.iterrows():
            ax.annotate(row["subgroup_name"], (row["miscalibration_magnitude"], 0.02),
                        fontsize=8, ha="center")
    ax.set_xlabel("Subgroup miscalibration magnitude |predicted/actual - 1|")
    ax.set_ylabel("Subgroup procedural-disenrollment exposure (placeholder)")
    ax.set_title("Figure 4. Tie-in: subgroup miscalibration vs disenrollment exposure")
    ax.legend(frameon=False)
    save(fig, "figure4_synthesis")


def figure5_sensitivity() -> None:
    p = TABLES / "aim2_microsim_sensitivity.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["non_metro"] == False]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = df.pivot_table(
        index="race_ethnicity",
        columns=["detection_shift_sd", "certification_shift_sd"],
        values="predicted_to_actual_ratio_mean",
    )
    cax = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=0.5, vmax=1.5)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{a:.0f},{b:.0f}" for a, b in pivot.columns], rotation=45)
    ax.set_xlabel("Detection shift SD, Certification shift SD")
    fig.colorbar(cax, ax=ax, label="Predicted / actual ratio")
    ax.set_title("Figure 5. Microsim sensitivity of predicted/actual ratio across parameter shifts")
    fig.tight_layout()
    save(fig, "figure5_sensitivity")


def main() -> None:
    figure1_state_intensity_map()
    figure2_event_study()
    figure3_subgroup_ratios()
    figure4_synthesis_scatter()
    figure5_sensitivity()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
