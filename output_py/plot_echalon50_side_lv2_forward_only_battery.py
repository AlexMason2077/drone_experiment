"""Reference-style battery charts for Echelon 50 cm, side wind, Level 2."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db_copy_for_cleaning"
ADMIN = DB / "_cleaning_admin"
ANALYSIS = ROOT / "analysis_outputs" / "configuration_energy_analysis"
OUT = ANALYSIS / "echalon50_forward_only_examples"

sys.path.insert(0, str(ROOT / "output_py"))
from analyze_configuration_energy import hover_calibration  # noqa: E402
from build_forward_motion_segments import build_forward_mask  # noqa: E402
from build_trajectory_cleaning_segments import (  # noqa: E402
    centered_rolling_median,
    find_coordination_file,
    prepare_run_groups,
)


DRONES = [f"drone_{index}" for index in range(1, 6)]
TRIAL_COLORS = {"002": "#2A6F97", "003": "#D17A22", "004": "#4C956C"}


def trial_number(experiment_directory: str) -> str:
    match = re.search(r"_(\d{3})$", experiment_directory)
    return match.group(1) if match else experiment_directory


def cumulative_moving_time(times: np.ndarray, moving: np.ndarray, inside: np.ndarray) -> np.ndarray:
    dt = np.diff(times, prepend=times[0])
    interval_moving = moving & inside & np.isfinite(dt) & (dt >= 0)
    return np.cumsum(np.where(interval_moving, dt, 0.0))


def cumulative_nonforward_time(times: np.ndarray, moving: np.ndarray, inside: np.ndarray) -> np.ndarray:
    dt = np.diff(times, prepend=times[0])
    interval_nonforward = (~moving) & inside & np.isfinite(dt) & (dt >= 0)
    return np.cumsum(np.where(interval_nonforward, dt, 0.0))


def load_trial_curves() -> tuple[dict[str, dict[str, dict]], pd.DataFrame]:
    trajectory = pd.read_csv(
        ADMIN / "trajectory_qc" / "trajectory_drone_segments.csv",
        dtype={"run_id": "string"},
        low_memory=False,
    )
    forward = pd.read_csv(
        ADMIN / "trajectory_qc" / "forward_motion_drone_segments.csv",
        dtype={"run_id": "string"},
        low_memory=False,
    )
    selected = trajectory[
        trajectory["formation"].eq("echalon")
        & trajectory["inter_drone_spacing_cm"].eq(50)
        & trajectory["wind_direction"].eq("side")
        & trajectory["wind_level"].eq(2)
        & trajectory["trajectory_status"].eq("complete_segmented")
    ].merge(
        forward[
            [
                "experiment_directory",
                "run_id",
                "drone_name",
                "primary_analysis_status",
                "forward_movement_sec",
                "in_flight_nonforward_sec",
            ]
        ],
        on=["experiment_directory", "run_id", "drone_name"],
        validate="one_to_one",
    )
    rates = hover_calibration().set_index("battery_id")["hover_discharge_rate_pp_per_min"]
    selected["hover_rate_pp_per_min"] = selected["battery_id"].map(rates)
    selected["forward_only_drop_pp"] = (
        selected["reported_battery_drop_pct_points"]
        - selected["hover_rate_pp_per_min"] / 60.0 * selected["in_flight_nonforward_sec"]
    ).clip(lower=0.0)
    curves: dict[str, dict[str, dict]] = {drone: {} for drone in DRONES}

    for (directory, run_id), run_rows in selected.groupby(["experiment_directory", "run_id"]):
        coordination_file = find_coordination_file(DB / directory, str(run_id))
        if coordination_file is None:
            continue
        coordination = pd.read_csv(coordination_file, low_memory=False)
        prepared = prepare_run_groups(coordination)
        trial = trial_number(str(directory))
        for _, source in run_rows.iterrows():
            drone_name = str(source["drone_name"])
            item = prepared[drone_name]
            group = item["group"]
            times = item["times"]
            factor = float(source["trajectory_distance_calibration_factor"])
            progress = centered_rolling_median(item["relative"] @ item["run_direction"], 11) * factor
            phases = group["phase"].fillna("").astype(str).to_numpy()
            moving, inside, _ = build_forward_mask(
                times,
                progress,
                phases,
                float(source["motion_onset_sec"]),
                float(source["selected_250cm_end_sec"]),
                2.0,
            )
            battery = pd.to_numeric(group["battery"], errors="coerce").to_numpy(float)
            valid = inside & moving & np.isfinite(battery)
            indices = np.flatnonzero(valid)
            if not len(indices):
                continue
            moving_time = cumulative_moving_time(times, moving, inside)
            nonforward_time = cumulative_nonforward_time(times, moving, inside)
            battery_start = float(source["battery_at_motion_start_pct"])
            raw_cumulative_drop = np.clip(battery_start - battery, 0.0, None)
            restored_hover_drop = (
                float(source["hover_rate_pp_per_min"]) / 60.0 * nonforward_time
            )
            estimated_cumulative_forward_drop = np.clip(
                raw_cumulative_drop - restored_hover_drop,
                0.0,
                float(source["forward_only_drop_pp"]),
            )
            retained_drop = estimated_cumulative_forward_drop[indices]
            retained_drop[-1] = float(source["forward_only_drop_pp"])
            retained_drop = np.maximum.accumulate(retained_drop)
            corrected_battery_level = battery_start - retained_drop
            curves[drone_name][trial] = {
                "moving_time": moving_time[indices],
                "battery_level": corrected_battery_level,
                "primary": source["primary_analysis_status"] == "eligible_primary_75_to_40",
                "run_id": str(run_id),
            }
    return curves, selected


def plot_lines(curves: dict[str, dict[str, dict]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14.2, 10.8), dpi=180, sharey=True)
    for axis_index, (axis, drone_name) in enumerate(zip(axes.flat, DRONES)):
        for trial in sorted(curves[drone_name]):
            item = curves[drone_name][trial]
            axis.step(
                item["moving_time"],
                item["battery_level"],
                where="post",
                color=TRIAL_COLORS.get(trial, "#777777"),
                lw=1.6,
                ls="-" if item["primary"] else "--",
                alpha=0.95 if item["primary"] else 0.75,
                label=(
                    f"trial {trial}"
                    if item["primary"]
                    else f"trial {trial} (outside 75–40%)"
                ),
            )
        axis.set_title(f"{drone_name}: estimated battery level", loc="left", fontsize=10)
        axis.set_xlabel("Cumulative forward-moving time (s)")
        if axis_index % 2 == 0:
            axis.set_ylabel("Estimated battery level after hover removal (%)")
        axis.grid(color="#E1E6EA", lw=0.7, ls="--")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7.5, loc="best")
    axes.flat[-1].axis("off")
    fig.suptitle(
        "Echelon · 50 cm · Side wind · Level 2: forward-only battery lines",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.968,
        "Hover/non-forward time is removed using each battery's 75%–40% hover rate. Trial 004 is shown only as an outside-range sensitivity trace.",
        ha="center",
        fontsize=9,
        color="#52606A",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(
        OUT / "echalon50_side_lv2_forward_only_battery_lines.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_mean_drop(selected: pd.DataFrame) -> None:
    primary = selected[
        selected["primary_analysis_status"].eq("eligible_primary_75_to_40")
    ]
    summary = (
        primary.groupby("drone_name")["forward_only_drop_pp"]
        .agg(["mean", "std", "count"])
        .reindex(DRONES)
        .reset_index()
    )
    summary.to_csv(OUT / "echalon50_side_lv2_forward_only_mean_drop_by_drone.csv", index=False)
    means = summary["mean"].to_numpy(float)
    std = summary["std"].fillna(0.0).to_numpy(float)
    lower = np.minimum(std, means)
    fig, axis = plt.subplots(figsize=(7.4, 5.2), dpi=180)
    bars = axis.bar(
        summary["drone_name"],
        means,
        color="#247567",
        edgecolor="#31574F",
        lw=0.6,
        yerr=np.vstack([lower, std]),
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#30363B"},
    )
    axis.set_title("Mean forward-only battery drop by drone", fontsize=13)
    axis.set_ylabel("Estimated battery drop after hover removal (percentage points)")
    axis.set_xlabel("")
    axis.grid(axis="y", color="#E1E6EA", lw=0.7, ls="--")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.text(
        0.01,
        0.98,
        "Echelon · 50 cm · Side wind · Level 2 · primary trials 002–003; error bars = SD",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#52606A",
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "echalon50_side_lv2_forward_only_mean_drop_by_drone.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves, selected = load_trial_curves()
    plot_lines(curves)
    plot_mean_drop(selected)
    print(
        selected[
            selected["primary_analysis_status"].eq("eligible_primary_75_to_40")
        ][
            [
                "experiment_directory",
                "run_id",
                "drone_name",
                "forward_movement_sec",
                "in_flight_nonforward_sec",
                "reported_battery_drop_pct_points",
                "forward_only_drop_pp",
            ]
        ].round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
