"""Build an auditable trajectory-based movement index for formal drone runs.

This script does not overwrite raw experiment files.  It reconstructs a continuous
relative path from mission-pad coordinates, identifies terminal-pad completion,
and records the first trajectory-defined 250 cm together with stationary intervals.
The resulting sidecars live under db_copy_for_cleaning/_cleaning_admin/trajectory_qc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db_copy_for_cleaning"
ADMIN = DB / "_cleaning_admin"
OUT = ADMIN / "trajectory_qc"
PLOTS = OUT / "diagnostic_plots"

ELIGIBLE_INPUT_STATUSES = {
    "candidate_pending_trajectory",
    "recoverable_pending_trajectory",
}
DRONES = [f"drone_{i}" for i in range(1, 6)]


def split_codes(value: str) -> list[str]:
    return [item for item in str(value).split(";") if item]


def reconstruct_relative_path(
    times: np.ndarray, raw_xy: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """Integrate plausible local coordinate changes while suppressing pad-frame jumps.

    Mission-pad IDs repeat along some 250/300 cm paths, so absolute X_global/Y_global
    can jump by roughly one grid cycle.  Integrating only physically plausible
    consecutive displacements preserves motion across those frame changes.  Missing
    samples and rejected jumps remain visible in the returned quality fields.
    """

    relative = np.full_like(raw_xy, np.nan, dtype=float)
    valid = np.isfinite(times) & np.isfinite(raw_xy).all(axis=1)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return relative, 0, 0.0

    first = int(indices[0])
    relative[first] = (0.0, 0.0)
    previous_index = first
    previous_raw = raw_xy[first].copy()
    cumulative = np.zeros(2, dtype=float)
    rejected = 0

    for index in indices[1:]:
        delta_t = max(float(times[index] - times[previous_index]), 0.0)
        delta = raw_xy[index] - previous_raw
        # At the usual 10 Hz logging rate, real motion is about 1 cm/sample.
        # The 35 cm floor tolerates corrections; the time term covers short gaps.
        max_plausible_step = max(35.0, 60.0 * delta_t)
        if np.linalg.norm(delta) <= max_plausible_step:
            cumulative = cumulative + delta
        else:
            rejected += 1
        relative[index] = cumulative
        previous_index = int(index)
        previous_raw = raw_xy[index].copy()

    coverage = float(valid.mean()) if len(valid) else 0.0
    return relative, rejected, coverage


def centered_rolling_median(values: np.ndarray, width: int = 11) -> np.ndarray:
    series = pd.Series(values, dtype=float).interpolate(limit_direction="both")
    return series.rolling(width, center=True, min_periods=1).median().to_numpy()


def bridge_boolean_gaps(mask: np.ndarray, times: np.ndarray, max_gap_sec: float) -> np.ndarray:
    result = mask.astype(bool).copy()
    false_indices = np.flatnonzero(~result)
    if not len(false_indices):
        return result
    groups = np.split(false_indices, np.flatnonzero(np.diff(false_indices) > 1) + 1)
    for group in groups:
        left = int(group[0]) - 1
        right = int(group[-1]) + 1
        if left >= 0 and right < len(result) and result[left] and result[right]:
            if times[right] - times[left] <= max_gap_sec:
                result[group] = True
    return result


def remove_short_true_islands(mask: np.ndarray, times: np.ndarray, min_sec: float) -> np.ndarray:
    result = mask.astype(bool).copy()
    true_indices = np.flatnonzero(result)
    if not len(true_indices):
        return result
    groups = np.split(true_indices, np.flatnonzero(np.diff(true_indices) > 1) + 1)
    for group in groups:
        duration = float(times[group[-1]] - times[group[0]])
        if duration < min_sec:
            result[group] = False
    return result


def isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Return an equal-weight PAVA fit without requiring an external package."""

    clean = pd.Series(values, dtype=float).interpolate(limit_direction="both").to_numpy()
    levels: list[float] = []
    weights: list[int] = []
    for value in clean:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged_weight = weights[-2] + weights[-1]
            merged_level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / merged_weight
            levels[-2:] = [merged_level]
            weights[-2:] = [merged_weight]
    return np.concatenate([np.repeat(level, weight) for level, weight in zip(levels, weights)])


def conservative_wait_mask(times: np.ndarray, progress: np.ndarray) -> np.ndarray:
    """Mark only long, very stable progress plateaus as confirmed waiting.

    Mission-pad coordinates are quantized and can jump when the visible pad changes.
    Inferring every instant of motion from their numerical derivative would therefore
    overstate precision.  We instead identify only plateaus that remain within 3 cm
    across roughly three seconds, then require a 2.5-second continuous core.
    Everything else remains movement-or-unresolved rather than being forced into an
    active/stationary classification.
    """

    smooth = centered_rolling_median(progress, 7)
    median_step = float(np.nanmedian(np.diff(times))) if len(times) > 1 else 0.1
    width = max(5, int(round(3.0 / max(median_step, 0.02))))
    if width % 2 == 0:
        width += 1
    series = pd.Series(smooth)
    local_range = (
        series.rolling(width, center=True, min_periods=width // 2)
        .max()
        .sub(series.rolling(width, center=True, min_periods=width // 2).min())
        .to_numpy()
    )
    stationary = np.isfinite(local_range) & (local_range <= 3.0)
    stationary_indices = np.flatnonzero(stationary)
    if not len(stationary_indices):
        return stationary
    groups = np.split(stationary_indices, np.flatnonzero(np.diff(stationary_indices) > 1) + 1)
    for group in groups:
        duration = float(times[group[-1]] - times[group[0]])
        if duration < 2.5:
            stationary[group] = False
    return stationary


def first_crossing_time(times: np.ndarray, values: np.ndarray, threshold: float) -> float:
    indices = np.flatnonzero(values >= threshold)
    if not len(indices):
        return np.nan
    index = int(indices[0])
    if index == 0 or not np.isfinite(values[index - 1]):
        return float(times[index])
    x0, x1 = float(values[index - 1]), float(values[index])
    t0, t1 = float(times[index - 1]), float(times[index])
    if x1 <= x0:
        return t1
    fraction = float(np.clip((threshold - x0) / (x1 - x0), 0.0, 1.0))
    return t0 + fraction * (t1 - t0)


def value_at_or_before(times: np.ndarray, values: np.ndarray, query_time: float) -> float:
    if not len(times) or not np.isfinite(query_time):
        return np.nan
    index = int(np.searchsorted(times, query_time, side="right") - 1)
    index = min(max(index, 0), len(times) - 1)
    return float(values[index])


def state_intervals(
    times: np.ndarray,
    confirmed_wait: np.ndarray,
    onset: float,
    finish: float,
    metadata: dict,
) -> list[dict]:
    inside = (times >= onset) & (times <= finish)
    indices = np.flatnonzero(inside)
    if len(indices) < 2:
        return []
    labels = np.where(confirmed_wait[indices], "confirmed_stationary_wait", "movement_or_unresolved")
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    groups = np.split(np.arange(len(indices)), breaks)
    rows: list[dict] = []
    for sequence, group in enumerate(groups, start=1):
        sample_indices = indices[group]
        start = float(times[sample_indices[0]])
        end = float(times[sample_indices[-1]])
        rows.append(
            {
                **metadata,
                "interval_sequence": sequence,
                "interval_type": str(labels[group[0]]),
                "start_time_sec": start,
                "end_time_sec": end,
                "duration_sec": max(0.0, end - start),
            }
        )
    return rows


def terminal_evidence(group: pd.DataFrame) -> tuple[bool, float, str, bool, bool]:
    phase = group["phase"].fillna("").astype(str)
    time = group["node_elapsed_time"].to_numpy(float)
    verify = phase.str.contains("verify_target_pad", case=False, regex=False)
    landing = phase.str.contains("landing", case=False, regex=False)
    terminal = verify | landing
    if terminal.any():
        arrival_time = float(time[np.flatnonzero(terminal.to_numpy())[0]])
        terminal_source = "verify_target_pad" if verify.any() else "landing"
    else:
        arrival_time = float(np.nanmax(time))
        terminal_source = "none"

    segment_numbers: list[tuple[int, int]] = []
    for item in phase:
        match = re.search(r"segment_(\d+)_of_(\d+)", item)
        if match:
            segment_numbers.append((int(match.group(1)), int(match.group(2))))
    last_segment_complete = bool(
        segment_numbers and max(number for number, _ in segment_numbers) >= max(total for _, total in segment_numbers)
    )

    target_pad = pd.to_numeric(group.get("target_pad"), errors="coerce")
    mid = pd.to_numeric(group.get("mid"), errors="coerce")
    target_values = target_pad.dropna()
    target_seen = False
    if len(target_values):
        target = float(target_values.mode().iloc[0])
        late = time >= np.nanquantile(time, 0.7)
        target_seen = bool(np.any(late & np.isclose(mid.to_numpy(float), target, equal_nan=False)))

    complete = bool(terminal.any() or (last_segment_complete and target_seen))
    return complete, arrival_time, terminal_source, last_segment_complete, target_seen


def prepare_run_groups(coordination: pd.DataFrame) -> dict[str, dict]:
    prepared: dict[str, dict] = {}
    net_vectors = []
    for drone_name, group in coordination.groupby("drone_name", sort=True):
        group = group.sort_values("node_elapsed_time").drop_duplicates("node_elapsed_time", keep="last")
        times = pd.to_numeric(group["node_elapsed_time"], errors="coerce").to_numpy(float)
        raw_xy = group[["X_global", "Y_global"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        relative, rejected_jumps, coverage = reconstruct_relative_path(times, raw_xy)
        valid = np.isfinite(relative).all(axis=1)
        if valid.any():
            tail_start = np.nanquantile(times[valid], 0.9)
            net = np.nanmedian(relative[valid & (times >= tail_start)], axis=0)
            if np.linalg.norm(net) > 10:
                net_vectors.append(net / np.linalg.norm(net))
        prepared[str(drone_name)] = {
            "group": group,
            "times": times,
            "relative": relative,
            "rejected_jumps": rejected_jumps,
            "coverage": coverage,
        }

    if net_vectors:
        direction = np.nanmedian(np.vstack(net_vectors), axis=0)
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm else np.array([0.0, 1.0])
    else:
        direction = np.array([0.0, 1.0])
    for item in prepared.values():
        item["run_direction"] = direction
    return prepared


def analyze_drone(
    item: dict,
    run_meta: pd.Series,
    manual_calibration_keys: set[tuple[str, str, str]],
) -> tuple[dict, list[dict], dict]:
    group: pd.DataFrame = item["group"]
    times: np.ndarray = item["times"]
    relative: np.ndarray = item["relative"]
    direction: np.ndarray = item["run_direction"]
    drone_name = str(group["drone_name"].iloc[0])
    calibration_key = (str(run_meta["experiment_directory"]), str(run_meta["run_id"]), drone_name)
    issues: list[str] = []

    command_values = pd.to_numeric(group.get("node_forward_distance_cm"), errors="coerce").dropna()
    commanded = float(command_values.median()) if len(command_values) else float(run_meta["commanded_distance_cm"])
    raw_progress = relative @ direction
    smooth_progress = centered_rolling_median(raw_progress, 11)
    monotone_raw_progress = isotonic_non_decreasing(smooth_progress)

    complete, arrival_time, terminal_source, last_segment_complete, target_seen = terminal_evidence(group)
    arrival_window = (times >= max(float(np.nanmin(times)), arrival_time - 1.0)) & (
        times <= min(float(np.nanmax(times)), arrival_time + 1.5)
    )
    endpoint_progress = float(np.nanmedian(monotone_raw_progress[arrival_window])) if arrival_window.any() else np.nan
    if not np.isfinite(endpoint_progress):
        endpoint_progress = float(np.nanmax(monotone_raw_progress))

    # The terminal-pad observation anchors the full commanded distance.  Spatial
    # timing still comes entirely from the measured trajectory, including 250/300.
    calibration_factor = commanded / endpoint_progress if endpoint_progress > 0 else np.nan
    automatic_geometry_plausible = bool(
        np.isfinite(calibration_factor) and 0.70 <= calibration_factor <= 1.35
    )
    manual_calibration_applied = bool(
        complete
        and calibration_key in manual_calibration_keys
        and np.isfinite(calibration_factor)
        and 0.50 <= calibration_factor <= 2.00
    )
    geometry_usable = automatic_geometry_plausible or manual_calibration_applied
    if not complete:
        issues.append("no_terminal_completion_evidence")
    if not automatic_geometry_plausible:
        issues.append("trajectory_endpoint_geometry_implausible")
    elif abs(calibration_factor - 1.0) > 0.10:
        issues.append("trajectory_distance_calibration_over_10pct")
    if manual_calibration_applied:
        issues.append("user_confirmed_coordinate_frame_calibration_applied")
    if item["coverage"] < 0.95:
        issues.append("mission_pad_coordinate_coverage_below_95pct")
    if item["rejected_jumps"]:
        issues.append("mission_pad_coordinate_jumps_reconstructed")

    normalized_progress = smooth_progress * calibration_factor if geometry_usable else np.full_like(smooth_progress, np.nan)
    monotone_progress = monotone_raw_progress * calibration_factor if geometry_usable else np.full_like(smooth_progress, np.nan)
    # At a verified 250 cm terminal pad, 249 cm is used as the numerical crossing
    # tolerance; for 300 cm commands the spatial 250 cm crossing remains exact.
    finish_threshold = 249.0 if commanded <= 250.5 else 250.0
    finish = first_crossing_time(times, monotone_progress, finish_threshold) if geometry_usable else np.nan
    onset = first_crossing_time(times, monotone_progress, 7.5) if geometry_usable else np.nan
    confirmed_wait = conservative_wait_mask(times, normalized_progress) if geometry_usable else np.zeros(len(times), dtype=bool)
    if np.isfinite(onset) and np.isfinite(finish):
        confirmed_wait = confirmed_wait & (times >= onset) & (times <= finish)
    if not np.isfinite(finish):
        issues.append("cannot_locate_trajectory_250cm")
    if not np.isfinite(onset):
        issues.append("cannot_locate_sustained_motion_onset")

    confirmed_wait_duration = 0.0
    selected_duration = np.nan
    if np.isfinite(onset) and np.isfinite(finish) and finish > onset:
        selected_duration = float(finish - onset)
        interval_dt = np.diff(times)
        interval_wait = confirmed_wait[:-1] & (times[:-1] >= onset) & (times[1:] <= finish)
        confirmed_wait_duration = float(np.sum(interval_dt[interval_wait]))
    movement_or_unresolved_duration = (
        max(0.0, float(selected_duration - confirmed_wait_duration)) if np.isfinite(selected_duration) else np.nan
    )

    battery = pd.to_numeric(group.get("battery"), errors="coerce").to_numpy(float)
    battery_start = value_at_or_before(times, battery, onset)
    battery_end = value_at_or_before(times, battery, finish)
    battery_drop = battery_start - battery_end if np.isfinite(battery_start + battery_end) else np.nan
    if np.isfinite(battery_start) and battery_start > 75:
        issues.append("movement_starts_above_75pct")
    if np.isfinite(battery_end) and battery_end < 40:
        issues.append("movement_ends_below_40pct")

    if not complete:
        status = "incomplete_no_terminal_evidence"
    elif not geometry_usable:
        status = "needs_manual_geometry_review"
    elif not np.isfinite(onset) or not np.isfinite(finish) or finish <= onset:
        status = "needs_manual_segmentation_review"
    elif item["coverage"] < 0.80:
        status = "needs_manual_coordinate_review"
    else:
        status = "complete_segmented"

    battery_id_values = group.get("battery_id", pd.Series(dtype=str)).dropna().astype(str)
    battery_id = battery_id_values.mode().iloc[0] if len(battery_id_values) else ""
    metadata = {
        "experiment_directory": str(run_meta["experiment_directory"]),
        "run_id": str(run_meta["run_id"]),
        "drone_name": drone_name,
    }
    row = {
        **metadata,
        "formation": run_meta["formation"],
        "inter_drone_spacing_cm": run_meta["inter_drone_spacing_cm"],
        "wind_direction": run_meta["wind_direction"],
        "wind_level": run_meta["wind_level"],
        "battery_id": battery_id,
        "commanded_distance_cm": commanded,
        "selected_distance_cm": 250.0,
        "trajectory_status": status,
        "completion_evidence": complete,
        "terminal_source": terminal_source,
        "last_segment_complete": last_segment_complete,
        "target_pad_seen_late": target_seen,
        "motion_onset_sec": onset,
        "selected_250cm_end_sec": finish,
        "pre_movement_wait_sec": onset - float(np.nanmin(times)) if np.isfinite(onset) else np.nan,
        "selected_window_sec": selected_duration,
        "confirmed_stationary_wait_sec": confirmed_wait_duration,
        "movement_or_unresolved_sec": movement_or_unresolved_duration,
        "endpoint_observed_progress_cm": endpoint_progress,
        "trajectory_distance_calibration_factor": calibration_factor,
        "automatic_geometry_check_passed": automatic_geometry_plausible,
        "manual_coordinate_frame_calibration_applied": manual_calibration_applied,
        "manual_calibration_source": "manual_trajectory_calibrations.csv" if manual_calibration_applied else "",
        "coordinate_coverage_fraction": item["coverage"],
        "rejected_coordinate_jump_count": item["rejected_jumps"],
        "battery_at_motion_start_pct": battery_start,
        "battery_at_250cm_end_pct": battery_end,
        "reported_battery_drop_pct_points": battery_drop,
        "within_75_to_40_range": bool(
            np.isfinite(battery_start + battery_end) and battery_start <= 75 and battery_end >= 40
        ),
        "issue_codes": ";".join(dict.fromkeys(issues)),
    }
    intervals = (
        state_intervals(times, confirmed_wait, onset, finish, metadata)
        if np.isfinite(onset) and np.isfinite(finish) and finish > onset
        else []
    )
    plot_data = {
        "times": times,
        "progress": normalized_progress if np.isfinite(normalized_progress).any() else smooth_progress,
        "distance_normalized": bool(np.isfinite(normalized_progress).any()),
        "confirmed_wait": confirmed_wait,
        "onset": onset,
        "finish": finish,
        "status": status,
    }
    return row, intervals, plot_data


def find_coordination_file(directory: Path, run_id: str) -> Path | None:
    exact = sorted(directory.glob(f"*{run_id}*_all_coordination.csv"))
    if exact:
        return exact[0]
    candidates = sorted(directory.glob("*_all_coordination.csv"))
    return candidates[0] if candidates else None


def plot_run_diagnostic(run_key: tuple[str, str], plot_rows: dict[str, dict], status: str) -> None:
    directory, run_id = run_key
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=170)
    colors = plt.get_cmap("tab10").colors
    for index, drone_name in enumerate(DRONES):
        item = plot_rows.get(drone_name)
        if not item:
            continue
        times = item["times"]
        progress = item["progress"]
        ax.plot(times, progress, color=colors[index], lw=1.25, alpha=0.65, label=drone_name)
        confirmed_wait = item["confirmed_wait"] & np.isfinite(progress)
        ax.plot(times[confirmed_wait], progress[confirmed_wait], color=colors[index], lw=3.2, alpha=0.95)
        if np.isfinite(item["finish"]):
            ax.scatter([item["finish"]], [250], color=colors[index], s=20, zorder=4)
    ax.axhline(250, color="#30363b", lw=1.2, ls="--")
    ax.set_xlabel("Node elapsed time (s)")
    ax.set_ylabel("Reconstructed forward progress (cm)")
    ax.set_title(f"{directory} · {run_id}\n{status}", loc="left", fontsize=11)
    ax.text(
        0.99,
        0.03,
        "Thick segments: conservatively detected stationary waits",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4f5962",
    )
    ax.grid(color="#dfe4e8", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(PLOTS / f"{directory}_{run_id}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    for old_plot in PLOTS.glob("*.png"):
        old_plot.unlink()
    inventory = pd.read_csv(ADMIN / "run_inventory.csv", dtype={"run_id": "string"}, low_memory=False)
    candidates = inventory[inventory["overall_status"].isin(ELIGIBLE_INPUT_STATUSES)].copy()
    calibration_file = ADMIN / "manual_trajectory_calibrations.csv"
    calibrations = pd.read_csv(calibration_file, dtype={"run_id": "string"})
    accepted_calibrations = calibrations[
        calibrations["action"].eq("accept_coordinate_frame_calibration")
    ]
    manual_calibration_keys = set(
        accepted_calibrations[["experiment_directory", "run_id", "drone_name"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )

    drone_rows: list[dict] = []
    interval_rows: list[dict] = []
    plot_cache: dict[tuple[str, str], dict[str, dict]] = {}
    file_issues: list[dict] = []

    for _, run_meta in candidates.iterrows():
        directory_name = str(run_meta["experiment_directory"])
        run_id = str(run_meta["run_id"])
        coordination_file = find_coordination_file(DB / directory_name, run_id)
        if coordination_file is None:
            file_issues.append(
                {"experiment_directory": directory_name, "run_id": run_id, "issue_code": "coordination_file_missing"}
            )
            continue
        coordination = pd.read_csv(
            coordination_file,
            dtype={"run_id": "string", "drone_name": "string", "battery_id": "string", "phase": "string"},
            low_memory=False,
        )
        coordination = coordination[coordination["run_id"].astype(str) == run_id].copy()
        if coordination.empty:
            file_issues.append(
                {"experiment_directory": directory_name, "run_id": run_id, "issue_code": "run_id_absent_from_coordination"}
            )
            continue
        prepared = prepare_run_groups(coordination)
        run_plot_rows: dict[str, dict] = {}
        for drone_name in DRONES:
            if drone_name not in prepared:
                file_issues.append(
                    {
                        "experiment_directory": directory_name,
                        "run_id": run_id,
                        "issue_code": f"coordination_missing_{drone_name}",
                    }
                )
                continue
            row, intervals, plot_data = analyze_drone(
                prepared[drone_name], run_meta, manual_calibration_keys
            )
            drone_rows.append(row)
            interval_rows.extend(intervals)
            run_plot_rows[drone_name] = plot_data
        plot_cache[(directory_name, run_id)] = run_plot_rows

    drones = pd.DataFrame(drone_rows)
    intervals = pd.DataFrame(interval_rows)
    file_issue_frame = pd.DataFrame(
        file_issues,
        columns=["experiment_directory", "run_id", "issue_code"],
    )

    run_rows: list[dict] = []
    for _, run_meta in candidates.iterrows():
        directory_name = str(run_meta["experiment_directory"])
        run_id = str(run_meta["run_id"])
        group = drones[(drones["experiment_directory"] == directory_name) & (drones["run_id"] == run_id)]
        statuses = group["trajectory_status"].value_counts().to_dict() if len(group) else {}
        issue_codes = []
        for item in group.get("issue_codes", pd.Series(dtype=str)):
            issue_codes.extend(split_codes(item))
        run_file_issues = file_issue_frame[
            (file_issue_frame.get("experiment_directory", pd.Series(dtype=str)) == directory_name)
            & (file_issue_frame.get("run_id", pd.Series(dtype=str)) == run_id)
        ] if len(file_issue_frame) else pd.DataFrame()
        if len(run_file_issues):
            issue_codes.extend(run_file_issues["issue_code"].astype(str).tolist())

        if len(group) < 5:
            run_status = "needs_manual_trajectory_review"
        elif (group["trajectory_status"] == "complete_segmented").all():
            run_status = "eligible_cleaned_250cm"
        elif group["trajectory_status"].str.startswith("incomplete").any():
            run_status = "excluded_incomplete_five_drone_run"
        else:
            run_status = "needs_manual_trajectory_review"
        run_rows.append(
            {
                "experiment_directory": directory_name,
                "run_id": run_id,
                "formation": run_meta["formation"],
                "inter_drone_spacing_cm": run_meta["inter_drone_spacing_cm"],
                "wind_direction": run_meta["wind_direction"],
                "wind_level": run_meta["wind_level"],
                "input_status": run_meta["overall_status"],
                "trajectory_run_status": run_status,
                "drone_rows_found": len(group),
                "complete_segmented_drone_count": int(statuses.get("complete_segmented", 0)),
                "median_pre_movement_wait_sec": group["pre_movement_wait_sec"].median() if len(group) else np.nan,
                "max_pre_movement_wait_sec": group["pre_movement_wait_sec"].max() if len(group) else np.nan,
                "median_confirmed_stationary_wait_sec": group["confirmed_stationary_wait_sec"].median() if len(group) else np.nan,
                "max_confirmed_stationary_wait_sec": group["confirmed_stationary_wait_sec"].max() if len(group) else np.nan,
                "all_five_within_75_to_40_range": bool(len(group) == 5 and group["within_75_to_40_range"].all()),
                "issue_codes": ";".join(dict.fromkeys(issue_codes)),
            }
        )
    runs = pd.DataFrame(run_rows)

    master = inventory.merge(
        runs[
            [
                "experiment_directory",
                "run_id",
                "trajectory_run_status",
                "all_five_within_75_to_40_range",
                "median_confirmed_stationary_wait_sec",
                "max_confirmed_stationary_wait_sec",
            ]
        ],
        on=["experiment_directory", "run_id"],
        how="left",
    )

    def cleaning_decision(row: pd.Series) -> str:
        if row["overall_status"] == "excluded_marked_outlier":
            return "excluded_marked_outlier"
        if str(row["overall_status"]).startswith("retained_no_wind"):
            return "held_no_wind_not_analyzed"
        trajectory_status = row.get("trajectory_run_status")
        if trajectory_status == "eligible_cleaned_250cm":
            return "eligible_after_trajectory_cleaning"
        if trajectory_status == "excluded_incomplete_five_drone_run":
            return "excluded_incomplete_five_drone_run"
        if trajectory_status == "needs_manual_trajectory_review":
            return "held_manual_trajectory_review"
        return "held_unresolved"

    master["cleaning_decision"] = master.apply(cleaning_decision, axis=1)
    master["primary_analysis_status"] = np.where(
        (master["cleaning_decision"] == "eligible_after_trajectory_cleaning")
        & master["all_five_within_75_to_40_range"].fillna(False),
        "eligible_primary_75_to_40",
        np.where(
            master["cleaning_decision"] == "eligible_after_trajectory_cleaning",
            "held_outside_75_to_40_for_sensitivity_only",
            "not_primary_eligible",
        ),
    )

    drones.to_csv(OUT / "trajectory_drone_segments.csv", index=False)
    intervals.to_csv(OUT / "trajectory_movement_intervals.csv", index=False)
    runs.to_csv(OUT / "trajectory_run_status.csv", index=False)
    file_issue_frame.to_csv(OUT / "trajectory_file_issues.csv", index=False)
    master.to_csv(ADMIN / "cleaning_master_run_index.csv", index=False)

    # Plot a small, inspectable audit set: one passing run per formation plus every
    # run rejected as incomplete (capped to keep the folder manageable).
    selected_keys: list[tuple[str, str]] = []
    passing = runs[runs["trajectory_run_status"] == "eligible_cleaned_250cm"]
    for _, row in passing.sort_values(["formation", "experiment_directory", "run_id"]).groupby("formation").head(1).iterrows():
        selected_keys.append((str(row["experiment_directory"]), str(row["run_id"])))
    flagged = runs[runs["trajectory_run_status"] != "eligible_cleaned_250cm"].head(15)
    selected_keys.extend((str(row.experiment_directory), str(row.run_id)) for row in flagged.itertuples())
    calibrated_keys = drones[drones["manual_coordinate_frame_calibration_applied"]][
        ["experiment_directory", "run_id"]
    ].drop_duplicates()
    selected_keys.extend(
        (str(row.experiment_directory), str(row.run_id))
        for row in calibrated_keys.itertuples()
    )
    selected_keys = list(dict.fromkeys(selected_keys))
    for key in selected_keys:
        match = runs[(runs["experiment_directory"] == key[0]) & (runs["run_id"] == key[1])]
        status = str(match["trajectory_run_status"].iloc[0]) if len(match) else "unknown"
        plot_run_diagnostic(key, plot_cache.get(key, {}), status)

    summary = {
        "input_formal_candidate_runs": int(len(candidates)),
        "input_expected_drone_rows": int(len(candidates) * 5),
        "trajectory_drone_rows_created": int(len(drones)),
        "run_status_counts": {str(key): int(value) for key, value in runs["trajectory_run_status"].value_counts().items()},
        "drone_status_counts": {str(key): int(value) for key, value in drones["trajectory_status"].value_counts().items()},
        "trajectory_eligible_runs_all_five_within_75_to_40": int(
            (
                (runs["trajectory_run_status"] == "eligible_cleaned_250cm")
                & runs["all_five_within_75_to_40_range"]
            ).sum()
        ),
        "drone_rows_with_coordinate_jumps": int((drones["rejected_coordinate_jump_count"] > 0).sum()),
        "drone_rows_with_distance_calibration_over_10pct": int(
            drones["issue_codes"].str.contains("trajectory_distance_calibration_over_10pct", na=False).sum()
        ),
        "user_confirmed_coordinate_frame_calibrations": int(
            drones["manual_coordinate_frame_calibration_applied"].sum()
        ),
        "median_pre_movement_wait_sec": float(drones["pre_movement_wait_sec"].median()),
        "median_confirmed_stationary_wait_sec": float(drones["confirmed_stationary_wait_sec"].median()),
        "diagnostic_plot_count": len(selected_keys),
        "notes": [
            "Raw files were not edited or deleted.",
            "B12 and B15 remain distinct battery IDs.",
            "Times are trajectory-derived; 10 cm/s multiplied by 25 s is not used.",
            "Coordinate frame jumps are reconstructed from plausible consecutive motion and separately flagged.",
            "Seven user-confirmed coordinate-frame jumps use explicit row-level calibration overrides.",
            "Waiting labels are conservative: unresolved time is not automatically called forward motion.",
            "No-wind and previously marked outlier runs are not included in this pass.",
        ],
    }
    (OUT / "trajectory_cleaning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    flagged_table = runs[runs["trajectory_run_status"] != "eligible_cleaned_250cm"][
        ["experiment_directory", "run_id", "trajectory_run_status", "complete_segmented_drone_count"]
    ]
    flagged_lines = [
        f"- `{row.experiment_directory}` / `{row.run_id}`: {row.trajectory_run_status} "
        f"({row.complete_segmented_drone_count}/5 drones segmented)"
        for row in flagged_table.itertuples()
    ]
    report = "\n".join(
        [
            "# Trajectory cleaning report",
            "",
            "## Outcome",
            "",
            f"- Formal candidate runs reviewed: {len(candidates)}",
            f"- Five-drone runs with an auditable first 250 cm: {(runs.trajectory_run_status == 'eligible_cleaned_250cm').sum()}",
            f"- Runs held for manual trajectory review: {(runs.trajectory_run_status == 'needs_manual_trajectory_review').sum()}",
            f"- Runs excluded for incomplete five-drone completion: {(runs.trajectory_run_status == 'excluded_incomplete_five_drone_run').sum()}",
            f"- User-confirmed coordinate-frame calibrations applied: {drones.manual_coordinate_frame_calibration_applied.sum()}",
            f"- Trajectory-eligible runs with all five drones inside the 75%-40% battery range: {((runs.trajectory_run_status == 'eligible_cleaned_250cm') & runs.all_five_within_75_to_40_range).sum()}",
            "",
            "## Method",
            "",
            "- Raw files were never edited. Every decision is stored in sidecar CSV files.",
            "- Repeating mission-pad coordinate frames were reconstructed by integrating only physically plausible consecutive coordinate changes; every rejected jump remains flagged.",
            "- Completion requires terminal-phase evidence or a completed final movement segment with the target pad observed late in the run.",
            "- A verified terminal pad anchors the commanded 250/300 cm distance. For 300 cm commands, the first spatial crossing of 250 cm is selected.",
            "- The seven user-confirmed coordinate-frame cases are explicitly listed in manual_trajectory_calibrations.csv; their original reconstructed distance and applied factor remain in the drone-level audit table.",
            "- Motion onset is the first trajectory crossing of 7.5 cm. It is not inferred from 10 cm/s multiplied by 25 seconds.",
            "- Waiting removal is conservative: only stable plateaus lasting at least about 2.5 seconds are labeled confirmed stationary waits. All other selected time is movement-or-unresolved.",
            "- Battery IDs remain literal; B12 is not relabeled as B15.",
            "",
            "## Important limitations",
            "",
            "- Mission-pad telemetry is quantized and sometimes changes coordinate frame. The movement boundary is defensible, but sub-second motor-on/off timing is not observable.",
            "- Confirmed waiting is a lower bound. Later energy analysis should show both uncorrected results and a conservative waiting-corrected sensitivity result.",
            "- A trajectory status of eligible does not automatically make the run primary-analysis eligible; the 75%-40% battery-range rule is tracked separately.",
            "- B12 and B15 remain partly confounded with experiment date/configuration and require the previously documented hover-baseline normalization plus a drone-5-excluded sensitivity analysis.",
            "",
            "## Runs not automatically accepted",
            "",
            *(flagged_lines or ["- None"]),
            "",
        ]
    )
    (OUT / "trajectory_qc_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
