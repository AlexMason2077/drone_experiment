"""Plot forward-only repeated curves with the pooled within-island rate.

This panel uses Front 75 cm, side wind, level 2, nominal slot/drone 5. Only
forward samples enter the plot and fit. The three selected runs occupy distinct
reported-SOC bands and are aligned on a common forward-movement time axis.
Every continuous movement island retains its own intercept while all islands
share the pooled discharge rate used by the algorithm-facing rate table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "output_py"))

from plot_condition_configuration_forward_rates import (  # noqa: E402
    forward_island_fixed_effect_slope,
)


INPUT = ROOT / "analysis_outputs" / "position_energy_method_study" / "method_study_samples.csv.gz"
OUT_DIR = ROOT / "analysis_outputs" / "methodology_figures"

FORMATION = "front"
SPACING_CM = 75
WIND_DIRECTION = "side"
WIND_LEVEL = 2
DRONE_NAME = "drone_5"

RUN_ORDER = ["20260610_212729", "20260610_213126", "20260610_213311"]
RUN_LABELS = {
    "20260610_212729": "Run 1 (start 74%)",
    "20260610_213126": "Run 2 (start 58%)",
    "20260610_213311": "Run 3 (start 45%)",
}
RUN_COLORS = {
    "20260610_212729": "#246B9A",
    "20260610_213126": "#26836A",
    "20260610_213311": "#D68A1C",
}

INK = "#202428"
GRID = "#D9DEE3"


def main() -> None:
    samples = pd.read_csv(INPUT, dtype={"run_id": "string"})
    subset = samples[
        samples["formation"].astype(str).str.lower().eq(FORMATION)
        & pd.to_numeric(samples["inter_drone_spacing_cm"], errors="coerce").eq(SPACING_CM)
        & samples["wind_direction"].astype(str).str.lower().eq(WIND_DIRECTION)
        & pd.to_numeric(samples["wind_level"], errors="coerce").eq(WIND_LEVEL)
        & samples["drone_name"].eq(DRONE_NAME)
        & samples["run_id"].astype(str).isin(RUN_ORDER)
    ].copy()
    moving = subset[
        subset["moving_forward"].astype(bool)
        & pd.to_numeric(subset["movement_island_id"], errors="coerce").ge(0)
    ].copy()
    if moving.empty:
        raise RuntimeError("No forward-only samples found for the requested example")

    pooled_slope = forward_island_fixed_effect_slope(subset)
    if not np.isfinite(pooled_slope):
        raise RuntimeError("The pooled within-island slope could not be estimated")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.15, 3.05))
    island_keys = ["experiment_directory", "run_id", "movement_island_id"]
    displayed_levels: list[float] = []
    for run_id in RUN_ORDER:
        run = moving[moving["run_id"].astype(str).eq(run_id)].copy()
        run_all = subset[subset["run_id"].astype(str).eq(run_id)].sort_values("wall_time_sec")
        color = RUN_COLORS[run_id]
        run_start_clock = float(run["forward_clock_sec"].min())
        reported_start = float(run_all["reported_battery_pct"].iloc[0])
        scale = float(run_all["physical_to_Bideal_scale"].iloc[0])
        # Place each independent run at the equivalent time at which the final
        # pooled line would reach its recorded starting SOC.
        time_offset = max(0.0, (75.0 - reported_start) * 60.0 / pooled_slope)

        island_groups = list(run.groupby(island_keys, sort=False))
        island_groups.sort(key=lambda item: float(item[1]["forward_clock_sec"].min()))
        accumulated_forward_drop = 0.0
        for _, island in island_groups:
            island = island.sort_values("forward_clock_sec")
            x_sec = (
                island["forward_clock_sec"].to_numpy(float)
                - run_start_clock
                + time_offset
            )
            raw_drop = island["standardized_drop_from_window_start_pp"].to_numpy(float)
            if len(x_sec) < 2:
                continue

            # Preserve the run's recorded starting SOC, but accumulate only the
            # changes observed inside forward islands. Any level shift that occurred
            # during an excluded hover interval is removed before joining islands.
            local_forward_drop = raw_drop - raw_drop[0]
            display_level = reported_start - accumulated_forward_drop - local_forward_drop
            ax.step(
                x_sec,
                display_level,
                where="post",
                color=color,
                lw=1.65,
                alpha=0.95,
                zorder=2,
            )
            displayed_levels.extend(display_level.tolist())
            accumulated_forward_drop += max(0.0, float(raw_drop[-1] - raw_drop[0]))

    minimum_display_level = float(min(displayed_levels))
    lower_limit = minimum_display_level - 2.0
    x_fit_max = (75.0 - lower_limit) * 60.0 / pooled_slope
    x_fit = np.linspace(0.0, x_fit_max, 400)
    y_fit = 75.0 - pooled_slope * x_fit / 60.0
    ax.plot(
        x_fit,
        y_fit,
        color="#7C858D",
        lw=1.4,
        ls=(0, (6, 3.5)),
        alpha=0.65,
        zorder=3,
    )

    ax.set_xlim(-2.0, x_fit_max + 2.0)
    ax.set_ylim(lower_limit, 77.0)
    ax.set_xlabel("Aligned forward movement time (s)")
    ax.set_ylabel("Bideal-normalized battery level (%)")
    ax.grid(color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555B61")
    ax.spines["bottom"].set_color("#555B61")

    ax.text(
        0.985,
        0.955,
        f"Forward discharge rate = {pooled_slope:.2f} percentage points/min",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.6},
    )
    legend_handles = [
        Line2D([0], [0], color=RUN_COLORS[run_id], lw=1.6, label=RUN_LABELS[run_id])
        for run_id in RUN_ORDER
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#7C858D",
            lw=1.2,
            ls=(0, (5, 3)),
            alpha=0.75,
            label="Fitted discharge rate",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.992),
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=2.6,
        columnspacing=1.25,
    )
    ax.text(
        0.012,
        0.055,
        "Gaps denote unobserved SOC ranges between independent runs",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#4D555C",
        fontsize=7.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.2},
    )
    fig.text(0.014, 0.985, "(b)", ha="left", va="top", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.82, bottom=0.20)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "fig_forward_only_slope_panel_b.png"
    pdf = OUT_DIR / "fig_forward_only_slope_panel_b.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(png)
    print(pdf)
    print(f"pooled_slope_pp_per_min={pooled_slope:.12f}")


if __name__ == "__main__":
    main()
