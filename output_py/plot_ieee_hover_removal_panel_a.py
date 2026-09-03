"""Create the IEEE methodology panel illustrating trajectory-based hover removal.

The figure uses the same trajectory reconstruction, calibration, and phase-aware
forward-motion mask as the implemented cleaning pipeline.  It is intentionally a
representative processing example, not a configuration-performance comparison.
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

from build_forward_motion_segments import build_forward_mask  # noqa: E402
from build_trajectory_cleaning_segments import (  # noqa: E402
    centered_rolling_median,
    prepare_run_groups,
)


EXPERIMENT_DIRECTORY = "echalon_50_side_lv1_new_001"
RUN_ID = "20260623_185117"
TARGET_DRONE = "drone_4"
OUT_DIR = ROOT / "analysis_outputs" / "methodology_figures"
COORDINATION = (
    ROOT
    / "db_copy_for_cleaning"
    / EXPERIMENT_DIRECTORY
    / f"{EXPERIMENT_DIRECTORY}_{RUN_ID}_all_coordination.csv"
)
TRAJECTORY_SEGMENTS = (
    ROOT
    / "db_copy_for_cleaning"
    / "_cleaning_admin"
    / "trajectory_qc"
    / "trajectory_drone_segments.csv"
)
FORWARD_SEGMENTS = (
    ROOT
    / "db_copy_for_cleaning"
    / "_cleaning_admin"
    / "trajectory_qc"
    / "forward_motion_drone_segments.csv"
)

BLUE = "#246B9A"
BLUE_DARK = "#164A70"
GREY = "#8B949E"
GREY_FILL = "#E8ECEF"
GRID = "#D9DEE3"
INK = "#202428"


def contiguous_groups(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    return list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1))


def main() -> None:
    coordination = pd.read_csv(COORDINATION, low_memory=False)
    prepared = prepare_run_groups(coordination)

    trajectory = pd.read_csv(TRAJECTORY_SEGMENTS, dtype={"run_id": "string"})
    trajectory = trajectory[
        trajectory["experiment_directory"].eq(EXPERIMENT_DIRECTORY)
        & trajectory["run_id"].astype(str).eq(RUN_ID)
    ].set_index("drone_name")

    forward_summary = pd.read_csv(FORWARD_SEGMENTS, dtype={"run_id": "string"})
    forward_summary = forward_summary[
        forward_summary["experiment_directory"].eq(EXPERIMENT_DIRECTORY)
        & forward_summary["run_id"].astype(str).eq(RUN_ID)
    ].set_index("drone_name")

    item = prepared[TARGET_DRONE]
    group = item["group"]
    times = item["times"]
    raw_progress = item["relative"] @ item["run_direction"]
    smooth_progress = centered_rolling_median(raw_progress, 11)

    segment = trajectory.loc[TARGET_DRONE]
    calibration = float(segment["trajectory_distance_calibration_factor"])
    progress = smooth_progress * calibration
    onset = float(segment["motion_onset_sec"])
    finish = float(segment["selected_250cm_end_sec"])
    phases = group["phase"].fillna("").astype(str).to_numpy()
    moving, inside, _ = build_forward_mask(
        times,
        progress,
        phases,
        onset,
        finish,
        threshold_cm_s=2.0,
    )
    row = {
        "times": times,
        "progress": progress,
        "moving": moving,
        "inside": inside,
        "onset": onset,
        "finish": finish,
        "forward_sec": float(forward_summary.loc[TARGET_DRONE, "forward_movement_sec"]),
        "nonforward_sec": float(forward_summary.loc[TARGET_DRONE, "in_flight_nonforward_sec"]),
    }

    x_min = onset - 0.5
    x_max = finish + 0.5

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

    fig, ax = plt.subplots(figsize=(7.15, 2.75))
    times = row["times"]
    progress = row["progress"]
    inside = row["inside"] & np.isfinite(progress)
    moving = row["moving"] & inside
    nonforward = inside & ~moving

    for group in contiguous_groups(nonforward):
        ax.axvspan(
            times[group[0]],
            times[group[-1]],
            facecolor=GREY_FILL,
            edgecolor="none",
            alpha=0.92,
            zorder=0,
        )

    # The full reconstructed path remains visible in grey; forward samples are
    # overlaid in blue without joining across excluded intervals.
    ax.plot(times[inside], progress[inside], color=GREY, lw=1.15, zorder=1)
    for group in contiguous_groups(moving):
        ax.plot(
            times[group],
            progress[group],
            color=BLUE,
            lw=2.15,
            solid_capstyle="round",
            zorder=2,
        )

    ax.axhline(250, color="#5C636A", lw=0.75, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-8, 263)
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    ax.grid(axis="y", color=GRID, lw=0.55, alpha=0.8)
    ax.text(
        0.988,
        0.82,
        f"forward {row['forward_sec']:.1f} s | removed {row['nonforward_sec']:.1f} s",
        transform=ax.transAxes,
        color=INK,
        fontsize=8,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.4},
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555B61")
    ax.spines["bottom"].set_color("#555B61")
    ax.set_xlabel("Wall-clock time from trial recording start (s)")
    ax.set_ylabel("Calibrated along-track progress (cm)")
    fig.text(0.014, 0.985, "(a)", ha="left", va="top", fontsize=10, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=2.0, label="Forward motion (>=2 cm/s)"),
        Line2D([0], [0], color=GREY, lw=1.2, label="Excluded non-forward interval"),
        Line2D(
            [0],
            [0],
            color="#5C636A",
            lw=0.8,
            ls=(0, (4, 3)),
            label="Target distance (250 cm)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.992),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=2.8,
        columnspacing=1.4,
    )

    fig.subplots_adjust(left=0.115, right=0.99, top=0.79, bottom=0.20)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "fig_hover_removal_panel_a.png"
    pdf = OUT_DIR / "fig_hover_removal_panel_a.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
