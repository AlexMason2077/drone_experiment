#!/usr/bin/env python3
"""Plot Bideal-normalized forward discharge rates for every condition/configuration."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis_outputs/position_energy_method_study/method_study_samples.csv.gz"
TIMING_INPUT = ROOT / "analysis_outputs/position_energy_method_study/method_rates_run_drone.csv"
OUTPUT = ROOT / "analysis_outputs/configuration_condition_rate_bar_charts"

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
DRONES = [f"drone_{number}" for number in range(1, 6)]
DRONE_LABELS = [f"Drone {number}" for number in range(1, 6)]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#6F63A8"]
CELL_KEYS = [
    "formation",
    "inter_drone_spacing_cm",
    "wind_direction",
    "wind_level",
    "drone_name",
]


def forward_island_fixed_effect_slope(group: pd.DataFrame) -> float:
    """Fit one slope while allowing every run/movement island its own intercept.

    Only samples classified as forward movement enter the fit. Each selected run
    receives equal total weight so telemetry sampling density cannot dominate the
    joint slope. The response is already standardized to Bideal.
    """
    moving = group[
        group["moving_forward"].astype(bool)
        & pd.to_numeric(group["movement_island_id"], errors="coerce").ge(0)
    ].copy()
    if moving.empty:
        return math.nan

    run_sizes = moving.groupby(["experiment_directory", "run_id"]).size().to_dict()
    numerator = 0.0
    denominator = 0.0

    island_keys = ["experiment_directory", "run_id", "movement_island_id"]
    for (directory, run_id, _), island in moving.groupby(island_keys, sort=False):
        time_minutes = island["forward_clock_sec"].to_numpy(float) / 60.0
        bideal_drop = island["standardized_drop_from_window_start_pp"].to_numpy(float)
        if len(time_minutes) < 3 or np.ptp(time_minutes) <= 0:
            continue

        time_centered = time_minutes - np.mean(time_minutes)
        drop_centered = bideal_drop - np.mean(bideal_drop)
        run_weight = 1.0 / max(run_sizes[(directory, run_id)], 1)
        numerator += run_weight * float(np.sum(time_centered * drop_centered))
        denominator += run_weight * float(np.sum(time_centered**2))

    if denominator <= 0:
        return math.nan
    return max(0.0, numerator / denominator)


def build_rate_tables(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict] = []
    for keys, indices in samples.groupby(CELL_KEYS, sort=True).groups.items():
        group = samples.loc[indices]
        pooled_rows.append(
            {
                **dict(zip(CELL_KEYS, keys)),
                "slot_id": group["slot_id"].iloc[0],
                "battery_id": group["battery_id"].iloc[0],
                "selected_run_count": group[["experiment_directory", "run_id"]]
                .drop_duplicates()
                .shape[0],
                "pooled_Bideal_forward_rate_pp_per_min": forward_island_fixed_effect_slope(group),
            }
        )

    run_keys = CELL_KEYS + ["experiment_directory", "run_id"]
    run_rows: list[dict] = []
    for keys, indices in samples.groupby(run_keys, sort=True).groups.items():
        group = samples.loc[indices]
        run_rows.append(
            {
                **dict(zip(run_keys, keys)),
                "slot_id": group["slot_id"].iloc[0],
                "battery_id": group["battery_id"].iloc[0],
                "run_Bideal_forward_rate_pp_per_min": forward_island_fixed_effect_slope(group),
            }
        )

    return pd.DataFrame(pooled_rows), pd.DataFrame(run_rows)


def build_flight_time_summary(timing: pd.DataFrame) -> pd.DataFrame:
    """Summarize the recorded flight window without mixing it into rate fitting."""
    run_keys = [
        "formation",
        "inter_drone_spacing_cm",
        "wind_direction",
        "wind_level",
        "experiment_directory",
        "run_id",
    ]
    # Five telemetry streams do not have perfectly identical endpoints. The
    # within-run median is a robust estimate of the common recorded flight window.
    by_run = (
        timing.groupby(run_keys, as_index=False)["total_selected_window_sec"]
        .median()
        .rename(columns={"total_selected_window_sec": "run_flight_time_sec"})
    )
    cell_keys = [
        "formation",
        "inter_drone_spacing_cm",
        "wind_direction",
        "wind_level",
    ]
    return (
        by_run.groupby(cell_keys, as_index=False)["run_flight_time_sec"]
        .agg(flight_time_min_sec="min", flight_time_max_sec="max")
    )


def draw_condition(
    pooled: pd.DataFrame,
    run_rates: pd.DataFrame,
    flight_times: pd.DataFrame,
    wind: str,
    level: int,
    y_max: float,
) -> Path:
    pooled_condition = pooled[
        pooled["wind_direction"].eq(wind) & pooled["wind_level"].eq(level)
    ]
    runs_condition = run_rates[
        run_rates["wind_direction"].eq(wind) & run_rates["wind_level"].eq(level)
    ]
    time_condition = flight_times[
        flight_times["wind_direction"].eq(wind) & flight_times["wind_level"].eq(level)
    ]
    fig, axes = plt.subplots(2, 5, figsize=(17.2, 8.2), sharex=True, sharey=True)

    for row, spacing in enumerate(SPACINGS):
        for column, formation in enumerate(FORMATIONS):
            ax = axes[row, column]
            pooled_cell = pooled_condition[
                pooled_condition["formation"].eq(formation)
                & pooled_condition["inter_drone_spacing_cm"].eq(spacing)
            ]
            run_cell = runs_condition[
                runs_condition["formation"].eq(formation)
                & runs_condition["inter_drone_spacing_cm"].eq(spacing)
            ]
            time_cell = time_condition[
                time_condition["formation"].eq(formation)
                & time_condition["inter_drone_spacing_cm"].eq(spacing)
            ]

            ax.set_title(f"{formation.title()} · {spacing} cm", fontsize=11.5, pad=8)
            ax.set_ylim(0, y_max)
            ax.set_xlim(-0.65, 4.65)
            ax.grid(axis="y", color="#D7DDE3", linewidth=0.8, alpha=0.7)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks(range(5), DRONE_LABELS, rotation=35, ha="right", fontsize=8.5)

            if pooled_cell.empty:
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
                continue

            rates = []
            for drone in DRONES:
                row_value = pooled_cell[pooled_cell["drone_name"].eq(drone)]
                rates.append(
                    float(row_value["pooled_Bideal_forward_rate_pp_per_min"].iloc[0])
                    if not row_value.empty
                    else math.nan
                )

            x = np.arange(5)
            ax.bar(
                x,
                rates,
                width=0.68,
                color=COLORS,
                edgecolor="#263238",
                linewidth=0.55,
                alpha=0.88,
                zorder=2,
            )

            for drone_index, drone in enumerate(DRONES):
                values = run_cell.loc[
                    run_cell["drone_name"].eq(drone),
                    "run_Bideal_forward_rate_pp_per_min",
                ].dropna().to_numpy(float)
                offsets = (
                    np.linspace(-0.13, 0.13, len(values))
                    if len(values) > 1
                    else np.array([0.0])
                )
                ax.scatter(
                    drone_index + offsets,
                    values,
                    s=25,
                    facecolor="white",
                    edgecolor="#111827",
                    linewidth=0.8,
                    zorder=3,
                )

            n_runs = int(pooled_cell["selected_run_count"].max())
            time_min = float(time_cell["flight_time_min_sec"].iloc[0])
            time_max = float(time_cell["flight_time_max_sec"].iloc[0])
            if np.isclose(time_min, time_max, atol=0.05):
                flight_label = f"Flight time: {time_min:.1f} s"
            else:
                flight_label = f"Flight time: {time_min:.1f}–{time_max:.1f} s"
            ax.text(
                0.97,
                0.94,
                f"n={n_runs} run{'s' if n_runs != 1 else ''}\n{flight_label}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#4B5563",
                linespacing=1.35,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            )

    for row in range(2):
        axes[row, 0].set_ylabel(
            "Bideal-normalized forward discharge rate\n(percentage points/min)",
            fontsize=10.5,
        )

    fig.suptitle(f"{wind.title()} wind · Level {level}", fontsize=17, fontweight="semibold", y=0.985)
    fig.text(
        0.5,
        0.945,
        "Bars: pooled slope of repeated forward-only curves; dots: separate run-level slopes. Hover time is excluded.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#374151",
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.895,
        wspace=0.16,
        hspace=0.34,
    )

    path = OUTPUT / f"{wind}_lv{level}_configuration_drone_Bideal_forward_rate.png"
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    return path


def make_contact_sheet(paths: list[Path]) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(16, 11.5))
    for ax, path in zip(axes.flat, paths):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
        wspace=0.015,
        hspace=0.04,
    )
    path = OUTPUT / "all_six_conditions_Bideal_forward_rate_preview.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(INPUT)
    timing = pd.read_csv(TIMING_INPUT)
    samples["wind_direction"] = samples["wind_direction"].str.lower()
    samples["formation"] = samples["formation"].str.lower()
    samples["wind_level"] = pd.to_numeric(samples["wind_level"], errors="raise").astype(int)
    samples["inter_drone_spacing_cm"] = pd.to_numeric(
        samples["inter_drone_spacing_cm"], errors="raise"
    ).astype(int)
    timing["wind_direction"] = timing["wind_direction"].str.lower()
    timing["formation"] = timing["formation"].str.lower()
    timing["wind_level"] = pd.to_numeric(timing["wind_level"], errors="raise").astype(int)
    timing["inter_drone_spacing_cm"] = pd.to_numeric(
        timing["inter_drone_spacing_cm"], errors="raise"
    ).astype(int)

    pooled, run_rates = build_rate_tables(samples)
    flight_times = build_flight_time_summary(timing)
    pooled.to_csv(OUTPUT / "pooled_configuration_drone_Bideal_forward_rates.csv", index=False)
    run_rates.to_csv(OUTPUT / "run_level_configuration_drone_Bideal_forward_rates.csv", index=False)
    flight_times.to_csv(OUTPUT / "configuration_flight_time_ranges.csv", index=False)

    maximum = max(
        float(pooled["pooled_Bideal_forward_rate_pp_per_min"].max()),
        float(run_rates["run_Bideal_forward_rate_pp_per_min"].max()),
    )
    y_max = max(30.0, math.ceil((maximum + 1.0) / 5.0) * 5.0)
    paths = [
        draw_condition(pooled, run_rates, flight_times, wind, level, y_max)
        for wind, level in CONDITIONS
    ]
    make_contact_sheet(paths)

    notes = (
        "Metric: pooled Bideal-normalized forward discharge rate (percentage points/min).\n"
        "Only samples classified as forward movement are fitted; hover time does not enter the forward clock.\n"
        "Each run and each forward movement island has its own intercept.\n"
        "All selected runs contribute equal total weight to the pooled slope.\n"
        "Bars are pooled curve slopes, not means or medians of cumulative battery drops.\n"
        "White dots are slopes fitted separately to each selected run.\n"
        "Flight time is the range across selected runs after taking the median recorded-window duration across the five drone streams within each run.\n"
        f"All six figures share the same 0–{y_max:.0f} pp/min y-axis.\n"
    )
    (OUTPUT / "rate_chart_method.txt").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
