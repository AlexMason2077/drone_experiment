"""Plot movement-only trajectories and hover-removed energy for Echelon 50 cm."""

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
PER_RUN = OUT / "per_run"

sys.path.insert(0, str(ROOT / "output_py"))
from build_forward_motion_segments import build_forward_mask  # noqa: E402
from build_trajectory_cleaning_segments import (  # noqa: E402
    centered_rolling_median,
    find_coordination_file,
    prepare_run_groups,
)


CONDITIONS = [
    ("head", 1),
    ("head", 2),
    ("side", 1),
    ("side", 2),
    ("tail", 1),
    ("tail", 2),
]
DRONES = [f"drone_{index}" for index in range(1, 6)]
COLORS = {
    "drone_1": "#2A6F97",
    "drone_2": "#D17A22",
    "drone_3": "#6A8E3A",
    "drone_4": "#A64D79",
    "drone_5": "#6B5CA5",
}


def contiguous_groups(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    return [
        group
        for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        if len(group) >= 2
    ]


def select_representatives(run_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, level in CONDITIONS:
        group = run_metrics[
            run_metrics["wind_direction"].eq(direction)
            & run_metrics["wind_level"].eq(level)
        ].copy()
        median_score = float(group["forward_only_hover_equivalent_sec"].median())
        group["distance_to_condition_median"] = (
            group["forward_only_hover_equivalent_sec"] - median_score
        ).abs()
        representative = group.sort_values(
            ["distance_to_condition_median", "run_id"]
        ).iloc[0]
        rows.append(
            {
                **representative.to_dict(),
                "condition_median_forward_only_sec": median_score,
                "selection_rule": "closest run to condition median forward-only score",
            }
        )
    return pd.DataFrame(rows)


def load_run_plot_data(
    experiment_directory: str,
    run_id: str,
    drone_metrics: pd.DataFrame,
) -> dict[str, dict]:
    coordination_file = find_coordination_file(DB / experiment_directory, run_id)
    if coordination_file is None:
        raise FileNotFoundError(f"No coordination file for {experiment_directory} / {run_id}")
    coordination = pd.read_csv(coordination_file, low_memory=False)
    prepared = prepare_run_groups(coordination)
    rows = drone_metrics[
        drone_metrics["experiment_directory"].eq(experiment_directory)
        & drone_metrics["run_id"].astype(str).eq(str(run_id))
    ].set_index("drone_name")
    output: dict[str, dict] = {}
    for drone_name in DRONES:
        source = rows.loc[drone_name]
        item = prepared[drone_name]
        group = item["group"]
        times = item["times"]
        relative = item["relative"]
        direction = item["run_direction"]
        factor = float(source["trajectory_distance_calibration_factor"])
        raw_xy = group[["X_global", "Y_global"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(raw_xy).all(axis=1) & np.isfinite(relative).all(axis=1)
        first = int(np.flatnonzero(valid)[0])
        continuous_xy = np.full_like(relative, np.nan, dtype=float)
        continuous_xy[valid] = raw_xy[first] + relative[valid] * factor
        perpendicular = np.array([-direction[1], direction[0]])
        forward_coordinate = continuous_xy @ direction
        lateral_coordinate = continuous_xy @ perpendicular
        progress = centered_rolling_median(relative @ direction, 11) * factor
        phases = group["phase"].fillna("").astype(str).to_numpy()
        moving, inside, _ = build_forward_mask(
            times,
            progress,
            phases,
            float(source["motion_onset_sec"]),
            float(source["selected_250cm_end_sec"]),
            2.0,
        )
        output[drone_name] = {
            "times": times,
            "forward_coordinate": forward_coordinate,
            "lateral_coordinate": lateral_coordinate,
            "moving": moving & inside & valid,
            "total_energy": float(source["total_hover_equivalent_sec"]),
            "forward_energy": float(source["forward_only_hover_equivalent_sec"]),
            "moving_sec": float(source["forward_movement_sec"]),
            "removed_sec": float(source["in_flight_nonforward_sec"]),
        }
    common_origin = min(
        float(np.nanmin(values["forward_coordinate"][values["moving"]]))
        for values in output.values()
        if values["moving"].any()
    )
    for values in output.values():
        values["forward_coordinate"] = values["forward_coordinate"] - common_origin
    return output


def plot_trajectory_axis(axis: plt.Axes, data: dict[str, dict], title: str) -> None:
    for drone_name in DRONES:
        values = data[drone_name]
        groups = contiguous_groups(values["moving"])
        for group_index, group in enumerate(groups):
            axis.plot(
                values["lateral_coordinate"][group],
                values["forward_coordinate"][group],
                color=COLORS[drone_name],
                lw=1.8,
                alpha=0.92,
                label=drone_name.replace("_", " ").title() if group_index == 0 else None,
            )
        moving_indices = np.flatnonzero(values["moving"])
        if len(moving_indices):
            axis.scatter(
                values["lateral_coordinate"][moving_indices[0]],
                values["forward_coordinate"][moving_indices[0]],
                color=COLORS[drone_name],
                s=17,
                marker="o",
                zorder=4,
            )
            axis.scatter(
                values["lateral_coordinate"][moving_indices[-1]],
                values["forward_coordinate"][moving_indices[-1]],
                facecolor="white",
                edgecolor=COLORS[drone_name],
                linewidth=1.2,
                s=22,
                marker="o",
                zorder=4,
            )
    axis.set_title(title, loc="left", fontsize=10.5, weight="bold")
    axis.set_xlabel("Lateral coordinate (cm)")
    axis.set_ylabel("Forward coordinate (cm)")
    axis.grid(color="#E1E6EA", lw=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_energy_axis(axis: plt.Axes, data: dict[str, dict], title: str) -> None:
    forward = np.array([data[name]["forward_energy"] for name in DRONES])
    total = np.array([data[name]["total_energy"] for name in DRONES])
    removed = np.maximum(0.0, total - forward)
    positions = np.arange(len(DRONES))
    axis.barh(
        positions,
        forward,
        color="#2A6F97",
        edgecolor="#40505C",
        lw=0.6,
        label="Estimated forward-only energy",
    )
    axis.barh(
        positions,
        removed,
        left=forward,
        color="#D9E0E5",
        edgecolor="#7B8790",
        lw=0.6,
        hatch="///",
        label="Removed non-forward energy",
    )
    axis.set_yticks(positions, [name.replace("_", " ").title() for name in DRONES])
    axis.invert_yaxis()
    axis.set_xlim(left=0)
    axis.set_xlabel("Battery-normalized hover-equivalent seconds")
    axis.set_title(title, loc="left", fontsize=10.5, weight="bold")
    axis.grid(axis="x", color="#E1E6EA", lw=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    for position, drone_name in enumerate(DRONES):
        values = data[drone_name]
        axis.text(
            total[position] + 0.6,
            position,
            f"kept {forward[position]:.1f} s | motion {values['moving_sec']:.1f} s",
            va="center",
            fontsize=7.2,
            color="#26333B",
        )


def save_overviews(
    representatives: pd.DataFrame,
    drone_metrics: pd.DataFrame,
) -> None:
    trajectory_fig, trajectory_axes = plt.subplots(3, 2, figsize=(13.5, 15.5), dpi=180)
    energy_fig, energy_axes = plt.subplots(3, 2, figsize=(13.5, 14.2), dpi=180)
    for trajectory_axis, energy_axis, (_, row) in zip(
        trajectory_axes.flat, energy_axes.flat, representatives.iterrows()
    ):
        data = load_run_plot_data(
            str(row["experiment_directory"]), str(row["run_id"]), drone_metrics
        )
        condition = f"{str(row['wind_direction']).title()} wind · Level {int(row['wind_level'])}"
        subtitle = f"{condition} · run {row['run_id']}"
        plot_trajectory_axis(trajectory_axis, data, subtitle)
        plot_energy_axis(energy_axis, data, subtitle)

    handles = [
        plt.Line2D([0], [0], color=COLORS[name], lw=2, label=name.replace("_", " ").title())
        for name in DRONES
    ]
    trajectory_fig.suptitle(
        "Echelon · 50 cm — forward-movement trajectories",
        x=0.08,
        y=0.993,
        ha="left",
        fontsize=17,
    )
    trajectory_fig.text(
        0.08,
        0.970,
        "One median-representative run per wind condition. Hovering, waiting, and non-forward samples are omitted; filled/open circles mark retained start/end points.",
        ha="left",
        fontsize=9.5,
        color="#52606A",
    )
    trajectory_fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=5, frameon=False)
    trajectory_fig.tight_layout(rect=[0, 0, 1, 0.925], h_pad=2.2)
    trajectory_fig.savefig(OUT / "echalon50_forward_only_trajectories.png", bbox_inches="tight", facecolor="white")
    plt.close(trajectory_fig)

    energy_fig.suptitle(
        "Echelon · 50 cm — forward-only energy by drone",
        x=0.08,
        y=0.993,
        ha="left",
        fontsize=17,
    )
    energy_fig.text(
        0.08,
        0.970,
        "One median-representative run per wind condition. Blue is retained; hatched grey is the modeled energy removed with non-forward flight.",
        ha="left",
        fontsize=9.5,
        color="#52606A",
    )
    energy_handles, energy_labels = energy_axes.flat[0].get_legend_handles_labels()
    energy_fig.legend(
        energy_handles,
        energy_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=2,
        frameon=False,
    )
    energy_fig.tight_layout(rect=[0, 0, 1, 0.925], h_pad=2.2)
    energy_fig.savefig(OUT / "echalon50_forward_only_energy.png", bbox_inches="tight", facecolor="white")
    plt.close(energy_fig)


def save_each_run(run_metrics: pd.DataFrame, drone_metrics: pd.DataFrame) -> None:
    PER_RUN.mkdir(parents=True, exist_ok=True)
    for old in PER_RUN.glob("*.png"):
        old.unlink()
    for row in run_metrics.itertuples(index=False):
        data = load_run_plot_data(str(row.experiment_directory), str(row.run_id), drone_metrics)
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), dpi=170)
        condition = f"{str(row.wind_direction).title()} wind · Level {int(row.wind_level)}"
        plot_trajectory_axis(axes[0], data, "Forward-movement coordinates")
        plot_energy_axis(axes[1], data, "Forward-only energy by drone")
        fig.suptitle(
            f"Echelon · 50 cm · {condition} · run {row.run_id}",
            x=0.06,
            ha="left",
            fontsize=15,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.5)
        fig.savefig(
            PER_RUN / f"{row.experiment_directory}_{row.run_id}.png",
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_metrics = pd.read_csv(
        ANALYSIS / "forward_only_primary_run_metrics.csv", dtype={"run_id": "string"}
    )
    drone_metrics = pd.read_csv(
        ANALYSIS / "forward_only_primary_drone_metrics.csv", dtype={"run_id": "string"}
    )
    echalon_runs = run_metrics[
        run_metrics["formation"].eq("echalon")
        & run_metrics["inter_drone_spacing_cm"].eq(50)
    ].copy()
    representatives = select_representatives(echalon_runs)
    representatives.to_csv(OUT / "representative_runs.csv", index=False)
    save_overviews(representatives, drone_metrics)
    save_each_run(echalon_runs, drone_metrics)
    print(
        representatives[[
            "condition",
            "experiment_directory",
            "run_id",
            "forward_only_hover_equivalent_sec",
            "condition_median_forward_only_sec",
        ]].to_string(index=False)
    )
    print(f"Per-run figures: {len(list(PER_RUN.glob('*.png')))}")


if __name__ == "__main__":
    main()
