"""Diagnose the Echelon 50 cm side-wind Level 1/2 moving-time difference."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db_copy_for_cleaning"
ANALYSIS = ROOT / "analysis_outputs" / "configuration_energy_analysis"
OUT = ANALYSIS / "echalon50_forward_only_examples"

sys.path.insert(0, str(ROOT / "output_py"))
from build_forward_motion_segments import build_forward_mask  # noqa: E402
from build_trajectory_cleaning_segments import (  # noqa: E402
    centered_rolling_median,
    find_coordination_file,
    prepare_run_groups,
)


def main() -> None:
    drone = pd.read_csv(
        ANALYSIS / "forward_only_primary_drone_metrics.csv", dtype={"run_id": "string"}
    )
    selected = drone[
        drone["formation"].eq("echalon")
        & drone["inter_drone_spacing_cm"].eq(50)
        & drone["wind_direction"].eq("side")
    ].copy()
    rows = []
    for (directory, run_id), run_rows in selected.groupby(["experiment_directory", "run_id"]):
        coordination_file = find_coordination_file(DB / directory, str(run_id))
        coordination = pd.read_csv(coordination_file, low_memory=False)
        prepared = prepare_run_groups(coordination)
        for source in run_rows.itertuples(index=False):
            item = prepared[source.drone_name]
            times = item["times"]
            factor = float(source.trajectory_distance_calibration_factor)
            direction = item["run_direction"]
            perpendicular = np.array([-direction[1], direction[0]])
            forward = centered_rolling_median(item["relative"] @ direction, 11) * factor
            lateral = centered_rolling_median(item["relative"] @ perpendicular, 11) * factor
            phases = item["group"]["phase"].fillna("").astype(str).to_numpy()
            moving, inside, _ = build_forward_mask(
                times,
                forward,
                phases,
                float(source.motion_onset_sec),
                float(source.selected_250cm_end_sec),
                2.0,
            )
            intervals = moving[:-1] & moving[1:] & inside[:-1] & inside[1:]
            forward_path = float(np.abs(np.diff(forward)[intervals]).sum())
            lateral_path = float(np.abs(np.diff(lateral)[intervals]).sum())
            group = item["group"]
            spacing = pd.to_numeric(group["mean_spacing_error"], errors="coerce").to_numpy(float)
            roll = pd.to_numeric(group["roll"], errors="coerce").to_numpy(float)
            rows.append(
                {
                    "wind_level": int(source.wind_level),
                    "run_id": str(run_id),
                    "experiment_directory": directory,
                    "drone_name": source.drone_name,
                    "forward_movement_sec": float(source.forward_movement_sec),
                    "detected_forward_speed_cm_s": float(source.mean_detected_forward_speed_cm_s),
                    "detected_forward_distance_cm": float(source.detected_forward_distance_cm),
                    "lateral_path_per_forward_path": lateral_path / max(forward_path, 1e-9),
                    "mean_spacing_error_cm": float(np.nanmean(np.abs(spacing[moving]))),
                    "mean_absolute_roll_deg": float(np.nanmean(np.abs(roll[moving]))),
                    "segmentation_flag": str(source.forward_segmentation_issue_codes)
                    if pd.notna(source.forward_segmentation_issue_codes)
                    else "",
                }
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "echalon50_side_movement_time_diagnostic_drone_rows.csv", index=False)
    run_summary = (
        detail.groupby(["wind_level", "run_id", "experiment_directory"], as_index=False)
        .agg(
            mean_forward_movement_sec=("forward_movement_sec", "mean"),
            mean_forward_speed_cm_s=("detected_forward_speed_cm_s", "mean"),
            mean_forward_distance_cm=("detected_forward_distance_cm", "mean"),
            mean_lateral_per_forward=("lateral_path_per_forward_path", "mean"),
            mean_spacing_error_cm=("mean_spacing_error_cm", "mean"),
            mean_absolute_roll_deg=("mean_absolute_roll_deg", "mean"),
            flagged_drone_count=("segmentation_flag", lambda values: int(values.ne("").sum())),
        )
    )
    run_summary["run_label"] = (
        "Lv" + run_summary["wind_level"].astype(str) + " · " + run_summary["run_id"].str[-6:]
    )
    run_summary.to_csv(OUT / "echalon50_side_movement_time_diagnostic_runs.csv", index=False)

    colors = run_summary["wind_level"].map({1: "#8CB6CF", 2: "#2A6F97"})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=180)
    metrics = [
        ("mean_forward_movement_sec", "Mean forward-moving time", "Seconds"),
        ("mean_forward_speed_cm_s", "Mean detected forward speed", "cm/s"),
        ("mean_lateral_per_forward", "Lateral correction relative to forward path", "Ratio"),
    ]
    for axis, (field, title, ylabel) in zip(axes, metrics):
        bars = axis.bar(run_summary["run_label"], run_summary[field], color=colors, edgecolor="#40505C", lw=0.6)
        axis.set_title(title, loc="left", fontsize=10.5, weight="bold")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", color="#E1E6EA", lw=0.7, ls="--")
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, run_summary[field]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle(
        "Echelon · 50 cm · Side wind: why Level 2 has longer detected movement",
        x=0.06,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.06,
        0.91,
        "The distance is comparable across runs; Level 2 trial 222335 progresses more slowly and uses more lateral correction.",
        ha="left",
        fontsize=9,
        color="#52606A",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=2.0)
    fig.savefig(
        OUT / "echalon50_side_movement_time_diagnostic.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(run_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
