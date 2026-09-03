"""Plot cumulative battery cost for one representative echalon side-wind run.

Mission-pad waiting/calibration intervals are removed from the time axis. Their
estimated hover consumption is also removed using the per-drone correction in
the pure-forward analysis table.
"""

from pathlib import Path
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORMATION = "echalon"
DISTANCE = 75
LEVEL = 1
WIND = "side"
OUT = ROOT / "swarm_analysis" / "wind_direction" / "echalon_75_lv1"
OUTPUT_PNG = OUT / "echalon_75cm_lv1_side_wind_active_time_battery_cost.png"
OUTPUT_CSV = OUT / "echalon_75cm_lv1_side_wind_active_time_battery_cost.csv"


def unwrap_coordinate(values: pd.Series, threshold: float = 100.0) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    out = np.full(len(raw), np.nan)
    offset = 0.0
    last_raw = np.nan
    last_corrected = np.nan
    for index, value in enumerate(raw):
        if not np.isfinite(value):
            continue
        if np.isfinite(last_raw) and abs(value - last_raw) > threshold:
            offset += last_corrected - (value + offset)
        out[index] = value + offset
        last_raw = value
        last_corrected = out[index]
    return out


def choose_representative_run(motion: pd.DataFrame) -> str:
    selected = motion[
        (motion["formation"] == FORMATION)
        & (motion["distance"] == DISTANCE)
        & (motion["wind_level"] == LEVEL)
        & (motion["wind_direction"] == WIND)
    ].copy()
    per_run = selected.groupby("experiment_id")["pure_forward_drop_250cm"].mean()
    target = per_run.median()
    return str((per_run - target).abs().idxmin())


def active_intervals(coord: pd.DataFrame, commanded_distance: float = 250.0) -> list[tuple[float, float]]:
    coord = coord.sort_values("node_elapsed_time").dropna(
        subset=["node_elapsed_time", "X_global", "Y_global"]
    )
    time = coord["node_elapsed_time"].to_numpy(float)
    raw_xy = coord[["X_global", "Y_global"]].to_numpy(float)
    unwrapped_xy = np.column_stack(
        [unwrap_coordinate(coord["X_global"]), unwrap_coordinate(coord["Y_global"])]
    )

    def endpoints(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        start = np.nanmedian(xy[time <= min(1.0, time.max())], axis=0)
        end = np.nanmedian(xy[time >= max(time.max() - 1.0, 0.0)], axis=0)
        return start, end, float(np.linalg.norm(end - start))

    raw_start, raw_end, raw_total = endpoints(raw_xy)
    unwrap_start, unwrap_end, unwrap_total = endpoints(unwrapped_xy)
    if abs(unwrap_total - commanded_distance) + 5 < abs(raw_total - commanded_distance):
        xy, start, end, total = unwrapped_xy, unwrap_start, unwrap_end, unwrap_total
    else:
        xy, start, end, total = raw_xy, raw_start, raw_end, raw_total

    unit = (end - start) / total
    projected = (xy - start) @ unit
    smoothed = pd.Series(projected).rolling(11, center=True, min_periods=1).median().to_numpy()
    commanded_progress = smoothed / total * commanded_distance

    intervals = []
    segment_length = commanded_distance / 5.0
    for segment in range(5):
        lower = segment * segment_length + 0.1 * segment_length
        upper = segment * segment_length + 0.9 * segment_length
        lower_hits = np.flatnonzero(commanded_progress >= lower)
        upper_hits = np.flatnonzero(commanded_progress >= upper)
        if len(lower_hits) and len(upper_hits):
            start_time = float(time[lower_hits[0]])
            end_time = float(time[upper_hits[0]])
            if end_time > start_time:
                intervals.append((start_time, end_time))
    if len(intervals) != 5:
        raise RuntimeError("Could not identify all five active movement intervals")
    return intervals


def active_elapsed_at(time: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    active = np.zeros_like(time, dtype=float)
    for start, end in intervals:
        active += np.clip(time - start, 0.0, end - start)
    return active


def build_curves(experiment_id: str, motion: pd.DataFrame) -> pd.DataFrame:
    pattern = str(
        ROOT / "db_copy_for_cleaning" / experiment_id / f"{experiment_id}_*_all_battery_timeseries.csv"
    )
    battery_files = glob.glob(pattern)
    coord_files = glob.glob(pattern.replace("battery_timeseries", "coordination"))
    if len(battery_files) != 1 or len(coord_files) != 1:
        raise RuntimeError(f"Expected one battery and coordination file for {experiment_id}")

    battery = pd.read_csv(battery_files[0], low_memory=False)
    coord = pd.read_csv(coord_files[0], low_memory=False)
    meta = motion[motion["experiment_id"] == experiment_id].set_index("drone_name")
    curves = []

    for drone_name, drone_battery in battery.groupby("drone_name"):
        drone_battery = drone_battery.sort_values("node_elapsed_time").copy()
        intervals = active_intervals(coord[coord["drone_name"] == drone_name])
        time = drone_battery["node_elapsed_time"].to_numpy(float)
        active_time = active_elapsed_at(time, intervals)
        row = meta.loc[drone_name]
        raw_drop = (
            drone_battery["battery_start"].to_numpy(float)
            - drone_battery["battery"].to_numpy(float)
        )
        selected_indices = []
        active_only_cost = []
        cumulative_cost = 0.0
        for start, end in intervals:
            indices = np.flatnonzero((time >= start) & (time <= end))
            if not len(indices):
                continue
            segment_battery = drone_battery["battery"].to_numpy(float)[indices]
            segment_cost = np.maximum.accumulate(segment_battery[0] - segment_battery)
            selected_indices.extend(indices.tolist())
            active_only_cost.extend((cumulative_cost + segment_cost).tolist())
            cumulative_cost += float(segment_cost[-1])
        if not selected_indices:
            raise RuntimeError(f"No active-flight battery samples for {drone_name}")

        selected_indices = np.asarray(selected_indices, dtype=int)
        active_only_cost = np.asarray(active_only_cost, dtype=float)
        corrected_total = max(0.0, float(row["pure_forward_drop_250cm"]))
        if active_only_cost[-1] > 0:
            corrected_drop = active_only_cost * corrected_total / active_only_cost[-1]
        else:
            corrected_drop = corrected_total * active_time[selected_indices] / active_time[selected_indices].max()
        keep = np.r_[
            True,
            (np.diff(active_time[selected_indices]) > 0) | (np.diff(corrected_drop) != 0),
        ]
        selected_indices = selected_indices[keep]
        corrected_drop = corrected_drop[keep]

        curves.append(
            pd.DataFrame(
                {
                    "experiment_id": experiment_id,
                    "drone_name": drone_name,
                    "position": int(row["position"]),
                    "battery_id": str(row["battery_id"]),
                    "active_flight_time_sec": active_time[selected_indices],
                    "corrected_cumulative_battery_consumption_pct_points": corrected_drop,
                    "raw_node_elapsed_time_sec": time[selected_indices],
                    "raw_cumulative_battery_consumption_pct_points": raw_drop[selected_indices],
                }
            )
        )
    return pd.concat(curves, ignore_index=True)


def plot(curves: pd.DataFrame) -> None:
    colors = ["#2878B5", "#D9911B", "#4C9E55", "#8B63A8", "#C85A54"]
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=280)
    for index, (position, group) in enumerate(curves.groupby("position", sort=True)):
        group = group.sort_values("active_flight_time_sec")
        battery_id = group["battery_id"].iloc[0]
        ax.step(
            group["active_flight_time_sec"],
            group["corrected_cumulative_battery_consumption_pct_points"],
            where="post",
            linewidth=2.15,
            color=colors[index],
            label=f"Drone {position} ({battery_id})",
        )

    ax.set_xlabel("Active flight time (s)")
    ax.set_ylabel("Cumulative battery consumption (% points)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(color="#DDE2E6", linewidth=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    motion = pd.read_csv(
        ROOT / "swarm_analysis" / "pure_forward" / "pure_forward_drone_rows.csv",
        low_memory=False,
    )
    experiment_id = choose_representative_run(motion)
    curves = build_curves(experiment_id, motion)
    curves.to_csv(OUTPUT_CSV, index=False)
    plot(curves)
    print("representative run", experiment_id)
    print(
        curves.groupby(["position", "battery_id"])
        .agg(
            active_time_sec=("active_flight_time_sec", "max"),
            corrected_drop=("corrected_cumulative_battery_consumption_pct_points", "max"),
        )
        .round(3)
        .to_string()
    )
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
