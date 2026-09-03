#!/usr/bin/env python3
"""Plot per-condition configuration comparisons from the cleaned position study table."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis_outputs/position_energy_method_study/method_rates_run_drone.csv"
OUTPUT = ROOT / "analysis_outputs/configuration_condition_bar_charts"

METRIC = "forward_event_drop_Bideal_pp"
FORMATIONS = ["column", "diamond", "echalon", "front", "vee"]
SPACINGS = [50, 75]
CONDITIONS = [
    ("head", 1),
    ("head", 2),
    ("side", 1),
    ("side", 2),
    ("tail", 1),
    ("tail", 2),
]
DRONES = [f"drone_{i}" for i in range(1, 6)]
DRONE_LABELS = [f"Drone {i}" for i in range(1, 6)]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#6F63A8"]


def prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "wind_direction",
        "wind_level",
        "formation",
        "inter_drone_spacing_cm",
        "drone_name",
        "battery_id",
    ]
    summary = (
        df.groupby(keys, dropna=False)[METRIC]
        .agg(n_runs="count", median="median", minimum="min", maximum="max")
        .reset_index()
    )
    return summary.sort_values(keys)


def draw_condition(df: pd.DataFrame, wind: str, level: int, y_max: float) -> Path:
    condition = df[(df["wind_direction"] == wind) & (df["wind_level"] == level)]
    fig, axes = plt.subplots(2, 5, figsize=(17.2, 8.2), sharex=True, sharey=True)

    for row, spacing in enumerate(SPACINGS):
        for col, formation in enumerate(FORMATIONS):
            ax = axes[row, col]
            subset = condition[
                (condition["formation"] == formation)
                & (condition["inter_drone_spacing_cm"] == spacing)
            ]

            ax.set_title(f"{formation.title()} · {spacing} cm", fontsize=11.5, pad=8)
            ax.set_ylim(0, y_max)
            ax.set_xlim(-0.65, 4.65)
            ax.grid(axis="y", color="#D7DDE3", linewidth=0.8, alpha=0.7)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if subset.empty:
                ax.text(
                    0.5,
                    0.51,
                    "No eligible data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#6B7280",
                    fontsize=11,
                )
                ax.set_xticks(range(5), DRONE_LABELS, rotation=35, ha="right", fontsize=8.5)
                continue

            medians = []
            for drone in DRONES:
                values = subset.loc[subset["drone_name"] == drone, METRIC].dropna().to_numpy()
                medians.append(np.median(values) if len(values) else np.nan)

            x = np.arange(5)
            ax.bar(
                x,
                medians,
                width=0.68,
                color=COLORS,
                edgecolor="#263238",
                linewidth=0.55,
                alpha=0.88,
                zorder=2,
            )

            # Deterministic offsets expose every selected run without suggesting extra precision.
            for i, drone in enumerate(DRONES):
                values = subset.loc[subset["drone_name"] == drone, METRIC].dropna().to_numpy()
                offsets = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
                ax.scatter(
                    i + offsets,
                    values,
                    s=25,
                    facecolor="white",
                    edgecolor="#111827",
                    linewidth=0.8,
                    zorder=3,
                )

            n_runs = subset["run_id"].nunique()
            ax.text(
                0.97,
                0.94,
                f"n={n_runs} run{'s' if n_runs != 1 else ''}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#4B5563",
            )
            ax.set_xticks(x, DRONE_LABELS, rotation=35, ha="right", fontsize=8.5)

    for row in range(2):
        axes[row, 0].set_ylabel(
            "Battery drop over 250 cm\n(Bideal-normalized percentage points)",
            fontsize=10.5,
        )

    fig.suptitle(f"{wind.title()} wind · Level {level}", fontsize=17, fontweight="semibold", y=0.985)
    fig.text(
        0.5,
        0.945,
        "Bars: median forward-only battery drop; dots: individual selected runs. All panels and figures use the same y-axis.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#374151",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.895, wspace=0.16, hspace=0.34)

    out = OUTPUT / f"{wind}_lv{level}_configuration_drone_battery_drop.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    return out


def make_contact_sheet(paths: list[Path]) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(16, 11.5))
    for ax, path in zip(axes.flat, paths):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.015, hspace=0.04)
    out = OUTPUT / "all_six_conditions_preview.png"
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    required = {
        "wind_direction",
        "wind_level",
        "formation",
        "inter_drone_spacing_cm",
        "drone_name",
        "battery_id",
        "run_id",
        METRIC,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df[METRIC].notna()].copy()
    df["wind_direction"] = df["wind_direction"].str.lower()
    df["formation"] = df["formation"].str.lower()
    df["wind_level"] = pd.to_numeric(df["wind_level"], errors="raise").astype(int)
    df["inter_drone_spacing_cm"] = pd.to_numeric(
        df["inter_drone_spacing_cm"], errors="raise"
    ).astype(int)

    # A shared rounded ceiling makes visual comparisons between conditions valid.
    observed_max = float(df[METRIC].max())
    y_max = max(12.0, np.ceil((observed_max + 0.5) / 2.0) * 2.0)

    prepare_summary(df).to_csv(
        OUTPUT / "configuration_drone_battery_drop_summary.csv", index=False
    )
    paths = [draw_condition(df, wind, level, y_max) for wind, level in CONDITIONS]
    make_contact_sheet(paths)

    notes = (
        "Chart definition\n"
        "================\n"
        "Question: Within each wind condition, how does forward-only battery use differ "
        "among formation + spacing configurations and among the five drones?\n"
        "Metric: forward_event_drop_Bideal_pp, measured over the first 250 cm of "
        "forward movement after hover removal.\n"
        "Bars: median across selected runs. Dots: individual selected runs.\n"
        f"Shared y-axis: 0 to {y_max:.0f} Bideal-normalized percentage points.\n"
        "Missing panels: no eligible selected run in the current cleaned table.\n"
    )
    (OUTPUT / "chart_notes.txt").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
