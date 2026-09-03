"""Compare 50 cm and 75 cm spacing for vee, head wind, level 1."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "swarm_analysis" / "inter_drone_spacing" / "vee_head_lv1"
CHARTS = OUT / "charts"
COLORS = {50: "#2878B5", 75: "#D9911B"}
LABELS = {50: "50 cm", 75: "75 cm"}
DISTANCES = [50, 75]
POSITIONS = [1, 2, 3, 4, 5]


def selected_rows() -> pd.DataFrame:
    path = ROOT / "swarm_analysis" / "multidimensional" / "swarm_rows_with_baseline_adjustment.csv"
    frame = pd.read_csv(path, low_memory=False)
    selected = frame[
        (frame["formation"] == "vee")
        & (frame["wind_direction_short"] == "head")
        & (frame["wind_level"] == 1)
        & (frame["distance"].isin(DISTANCES))
    ].copy()
    selected["distance"] = selected["distance"].astype(int)
    selected["position"] = selected["position"].astype(int)
    return selected


def quartiles(values):
    values = np.asarray(values, dtype=float)
    return np.percentile(values, [25, 50, 75])


def run_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["distance", "experiment_id", "csv_run_id"], as_index=False)
        .agg(
            drone_count=("csv_drone_name", "nunique"),
            duration_sec=("csv_node_duration_sec", "first"),
            mean_start_soc=("csv_battery_hover_start", "mean"),
            total_reported_drop=("csv_battery_drop", "sum"),
            total_expected_baseline_drop=("baseline_expected_drop", "sum"),
            total_excess_vs_baseline=("excess_vs_baseline", "sum"),
            mean_excess_vs_baseline=("excess_vs_baseline", "mean"),
            mean_drop_rate_pct_per_min=("battery_drop_rate_pct_per_min", "mean"),
        )
        .sort_values(["distance", "csv_run_id"])
    )


def plot_run_level(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=240)
    metrics = [
        ("total_excess_vs_baseline", "Total SOC-adjusted excess use\n(% points across five drones)"),
        ("duration_sec", "Mission duration (s)"),
    ]
    fixed_offsets = {
        4: np.array([-0.09, -0.03, 0.03, 0.09]),
        6: np.array([-0.12, -0.072, -0.024, 0.024, 0.072, 0.12]),
    }
    for ax, (metric, ylabel) in zip(axes, metrics):
        for index, distance in enumerate(DISTANCES):
            values = summary.loc[summary["distance"] == distance, metric].to_numpy(float)
            offsets = fixed_offsets.get(len(values), np.linspace(-0.1, 0.1, len(values)))
            ax.scatter(
                index + offsets, values, s=38, color=COLORS[distance], alpha=0.72,
                edgecolor="white", linewidth=0.7, zorder=3,
            )
            q1, median, q3 = quartiles(values)
            ax.vlines(index, q1, q3, color=COLORS[distance], linewidth=6, alpha=0.25, zorder=2)
            ax.hlines(median, index - 0.18, index + 0.18, color=COLORS[distance], linewidth=3, zorder=4)
        ax.set_xticks([0, 1], ["50 cm\n(n=6 runs)", "75 cm\n(n=4 runs)"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDE2E6", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(0, color="#444444", linewidth=1, linestyle="--")
    fig.tight_layout(w_pad=2.6)
    fig.savefig(CHARTS / "01_run_level_energy_and_duration.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def position_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (distance, position), group in rows.groupby(["distance", "position"]):
        q1, median, q3 = quartiles(group["excess_vs_baseline"])
        records.append(
            {
                "distance_cm": int(distance),
                "position": int(position),
                "n_runs": int(group["csv_run_id"].nunique()),
                "q1_excess": q1,
                "median_excess": median,
                "q3_excess": q3,
                "mean_excess": float(group["excess_vs_baseline"].mean()),
                "median_reported_drop": float(group["csv_battery_drop"].median()),
            }
        )
    return pd.DataFrame(records).sort_values(["distance_cm", "position"])


def plot_positions(rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=240)
    offsets = {50: -0.07, 75: 0.07}
    for distance in DISTANCES:
        subset = summary[summary["distance_cm"] == distance].set_index("position").reindex(POSITIONS)
        x = np.asarray(POSITIONS, dtype=float) + offsets[distance]
        med = subset["median_excess"].to_numpy(float)
        lower = med - subset["q1_excess"].to_numpy(float)
        upper = subset["q3_excess"].to_numpy(float) - med
        ax.errorbar(
            x, med, yerr=np.vstack([lower, upper]), color=COLORS[distance],
            marker="o", markersize=6.5, linewidth=2.2, capsize=4,
            label=f"{LABELS[distance]} (median, IQR)", zorder=4,
        )
        for position in POSITIONS:
            vals = rows.loc[
                (rows["distance"] == distance) & (rows["position"] == position),
                "excess_vs_baseline",
            ].to_numpy(float)
            local = np.linspace(-0.025, 0.025, len(vals))
            ax.scatter(
                position + offsets[distance] + local, vals, color=COLORS[distance],
                s=18, alpha=0.22, linewidth=0, zorder=2,
            )
    ax.axhline(0, color="#444444", linewidth=1.1, linestyle="--")
    ax.set_xticks(POSITIONS, [f"Drone {position}" for position in POSITIONS])
    ax.set_xlabel("Position in vee formation")
    ax.set_ylabel("SOC-adjusted excess battery use (% points)")
    ax.grid(color="#DDE2E6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHARTS / "02_position_specific_spacing_effect.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def find_timeseries(experiment_id: str) -> Path:
    matches = list((ROOT / "database_csv" / experiment_id).glob("*_all_battery_timeseries.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one timeseries for {experiment_id}, found {len(matches)}")
    return matches[0]


def progress_curves(summary: pd.DataFrame) -> pd.DataFrame:
    grid = np.linspace(0, 100, 101)
    records = []
    for run in summary.itertuples(index=False):
        raw = pd.read_csv(find_timeseries(run.experiment_id), low_memory=False)
        raw["node_elapsed_time"] = pd.to_numeric(raw["node_elapsed_time"], errors="coerce")
        raw["battery_drop_from_start"] = pd.to_numeric(raw["battery_drop_from_start"], errors="coerce")
        raw = raw.dropna(subset=["node_elapsed_time", "battery_drop_from_start", "drone_name"])
        duration = float(raw["node_elapsed_time"].max())
        drone_curves = []
        for _, drone in raw.groupby("drone_name"):
            drone = drone.sort_values("node_elapsed_time").drop_duplicates("node_elapsed_time", keep="last")
            progress = 100.0 * drone["node_elapsed_time"].to_numpy(float) / duration
            drop = drone["battery_drop_from_start"].to_numpy(float)
            drone_curves.append(np.interp(grid, progress, drop))
        if len(drone_curves) != 5:
            raise RuntimeError(f"{run.experiment_id} has {len(drone_curves)} drone curves")
        mean_curve = np.mean(np.vstack(drone_curves), axis=0)
        for progress, mean_drop in zip(grid, mean_curve):
            records.append(
                {
                    "distance_cm": int(run.distance),
                    "experiment_id": run.experiment_id,
                    "run_id": run.csv_run_id,
                    "mission_progress_pct": progress,
                    "mean_reported_drop_per_drone": mean_drop,
                }
            )
    return pd.DataFrame(records)


def plot_progress(curves: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=240)
    for distance in DISTANCES:
        subset = curves[curves["distance_cm"] == distance]
        pivot = subset.pivot(index="mission_progress_pct", columns="run_id", values="mean_reported_drop_per_drone")
        x = pivot.index.to_numpy(float)
        values = pivot.to_numpy(float)
        for column in pivot.columns:
            ax.plot(x, pivot[column], color=COLORS[distance], linewidth=0.8, alpha=0.20)
        median = np.median(values, axis=1)
        q1 = np.percentile(values, 25, axis=1)
        q3 = np.percentile(values, 75, axis=1)
        ax.fill_between(x, q1, q3, color=COLORS[distance], alpha=0.14, linewidth=0)
        ax.plot(x, median, color=COLORS[distance], linewidth=2.6, label=LABELS[distance])
    ax.set_xlabel("Mission progress (%)")
    ax.set_ylabel("Mean reported battery drop per drone (% points)")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(color="#DDE2E6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", title="Inter-drone spacing")
    fig.tight_layout()
    fig.savefig(CHARTS / "03_battery_drop_over_mission_progress.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_rate_context(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=240)
    fixed_offsets = {
        4: np.array([-0.09, -0.03, 0.03, 0.09]),
        6: np.array([-0.12, -0.072, -0.024, 0.024, 0.072, 0.12]),
    }
    for index, distance in enumerate(DISTANCES):
        subset = summary[summary["distance"] == distance]
        values = subset["mean_drop_rate_pct_per_min"].to_numpy(float)
        offsets = fixed_offsets.get(len(values), np.linspace(-0.1, 0.1, len(values)))
        axes[0].scatter(
            index + offsets, values, s=42, color=COLORS[distance], alpha=0.72,
            edgecolor="white", linewidth=0.7, zorder=3,
        )
        q1, median, q3 = quartiles(values)
        axes[0].vlines(index, q1, q3, color=COLORS[distance], linewidth=6, alpha=0.25)
        axes[0].hlines(median, index - 0.18, index + 0.18, color=COLORS[distance], linewidth=3)
        axes[1].scatter(
            subset["mean_start_soc"], subset["mean_drop_rate_pct_per_min"],
            s=48, color=COLORS[distance], alpha=0.78, edgecolor="white",
            linewidth=0.7, label=f"{LABELS[distance]} (n={len(subset)})",
        )
    axes[0].set_xticks([0, 1], ["50 cm\n(n=6 runs)", "75 cm\n(n=4 runs)"])
    axes[0].set_ylabel("Mean reported battery-drop rate\n(% points per minute per drone)")
    axes[1].set_xlabel("Mean starting reported battery level (%)")
    axes[1].set_ylabel("Mean reported battery-drop rate\n(% points per minute per drone)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(color="#DDE2E6", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.6)
    fig.savefig(CHARTS / "04_drop_rate_and_start_soc.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    rows = selected_rows()
    runs = run_summary(rows)
    positions = position_summary(rows)
    curves = progress_curves(runs)

    runs.to_csv(OUT / "run_level_spacing_summary.csv", index=False)
    positions.to_csv(OUT / "position_spacing_summary.csv", index=False)
    curves.to_csv(OUT / "mission_progress_curves.csv", index=False)
    plot_run_level(runs)
    plot_positions(rows, positions)
    plot_progress(curves)
    plot_rate_context(runs)
    print(f"rows={len(rows)} runs={len(runs)} charts=4 output={OUT}")


if __name__ == "__main__":
    main()
