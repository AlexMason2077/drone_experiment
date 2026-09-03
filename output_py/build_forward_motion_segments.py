"""Identify forward-moving time inside each trajectory-selected 250 cm segment.

This is a stricter second-stage segmentation.  It preserves the existing trajectory
cleaning outputs and adds auditable sidecars that separate forward movement from
in-flight non-forward time (hovering, waiting, or lateral-only correction).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db_copy_for_cleaning"
ADMIN = DB / "_cleaning_admin"
TRAJECTORY_DIR = ADMIN / "trajectory_qc"
TRAJECTORY_SEGMENTS = TRAJECTORY_DIR / "trajectory_drone_segments.csv"
OUT_SEGMENTS = TRAJECTORY_DIR / "forward_motion_drone_segments.csv"
OUT_INTERVALS = TRAJECTORY_DIR / "forward_motion_intervals.csv"
OUT_SENSITIVITY = TRAJECTORY_DIR / "forward_motion_threshold_sensitivity.csv"
OUT_REASON_SUMMARY = TRAJECTORY_DIR / "forward_motion_hover_reason_summary.csv"

sys.path.insert(0, str(ROOT / "output_py"))
from build_trajectory_cleaning_segments import (  # noqa: E402
    bridge_boolean_gaps,
    centered_rolling_median,
    find_coordination_file,
    isotonic_non_decreasing,
    prepare_run_groups,
    remove_short_true_islands,
)


PRIMARY_SPEED_THRESHOLD_CM_S = 2.0
BRIDGE_GAP_SEC = 1.0
MIN_MOVEMENT_ISLAND_SEC = 0.5
SENSITIVITY_THRESHOLDS = (1.0, 2.0, 3.0, 4.0)


def collector_motion_mode(formation: str, spacing_cm: float) -> str:
    """Map a run to the movement-control branch used by data_collector.py."""

    normalized = str(formation or "").strip().lower()
    spacing = int(round(float(spacing_cm)))
    if normalized == "front" or normalized in {"echalon", "echelon", "echolon"}:
        return "parallel_segment_commands"
    if normalized == "vee" and spacing == 75:
        return "parallel_segment_commands"
    if normalized == "column":
        return "staggered_release_with_spacing_gate"
    if normalized == "diamond" and spacing == 75:
        return "staggered_shape_layer_release"
    return "staggered_release"


def expected_wait_causes(motion_mode: str) -> str:
    common = "marker_or_position_correction;post_arrival_group_synchronization"
    if motion_mode == "staggered_release_with_spacing_gate":
        return (
            "current_pad_group_lock;programmed_release_delay;column_spacing_safety_gate;"
            + common
        )
    if motion_mode in {"staggered_release", "staggered_shape_layer_release"}:
        return "current_pad_group_lock;programmed_release_delay;" + common
    return common


def node_segment_number(phase: str) -> int | None:
    match = re.search(r"node_segment_(\d+)_of_(\d+)", str(phase or ""))
    return int(match.group(1)) if match else None


def phase_gradient_velocity(
    times: np.ndarray,
    progress: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Estimate along-track speed independently inside each collector phase."""

    velocity = np.full(len(times), np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(times) & np.isfinite(progress))
    if not len(valid_indices):
        return velocity
    phase_values = phases[valid_indices]
    boundaries = np.flatnonzero(
        (np.diff(valid_indices) > 1) | (phase_values[1:] != phase_values[:-1])
    ) + 1
    for block in np.split(valid_indices, boundaries):
        if len(block) < 3:
            continue
        block_times = times[block]
        block_progress = progress[block]
        velocity[block] = np.gradient(block_progress, block_times)
    return velocity


def build_forward_mask(
    times: np.ndarray,
    progress: np.ndarray,
    phases: np.ndarray,
    onset: float,
    finish: float,
    threshold_cm_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a phase-aware forward mask and smoothed forward velocity.

    The collector synchronizes the swarm after every mission-pad segment.  Mask
    cleanup is therefore performed independently inside each logged phase so a
    short programmed hold at a segment boundary cannot be bridged into motion.
    """

    inside = (
        np.isfinite(times)
        & np.isfinite(progress)
        & (times >= onset)
        & (times <= finish)
    )
    velocity = phase_gradient_velocity(times, progress, phases)
    candidate = inside & np.isfinite(velocity) & (velocity >= threshold_cm_s)
    moving = np.zeros(len(times), dtype=bool)
    inside_indices = np.flatnonzero(inside)
    if len(inside_indices):
        phase_values = phases[inside_indices]
        boundaries = np.flatnonzero(
            (np.diff(inside_indices) > 1) | (phase_values[1:] != phase_values[:-1])
        ) + 1
        for block in np.split(inside_indices, boundaries):
            local = candidate[block]
            local = bridge_boolean_gaps(local, times[block], BRIDGE_GAP_SEC)
            local = remove_short_true_islands(local, times[block], MIN_MOVEMENT_ISLAND_SEC)
            moving[block] = local
    return moving, inside, velocity


def classify_samples(
    phases: np.ndarray,
    moving: np.ndarray,
    inside: np.ndarray,
    motion_mode: str,
) -> np.ndarray:
    """Reason-code non-forward samples using the collector's segment lifecycle."""

    labels = np.full(len(phases), "outside_selected_window", dtype=object)
    labels[inside & moving] = "forward_movement"
    inside_indices = np.flatnonzero(inside)
    if not len(inside_indices):
        return labels

    phase_values = phases[inside_indices]
    boundaries = np.flatnonzero(
        (np.diff(inside_indices) > 1) | (phase_values[1:] != phase_values[:-1])
    ) + 1
    for block in np.split(inside_indices, boundaries):
        nonforward = block[~moving[block]]
        if not len(nonforward):
            continue
        if node_segment_number(phases[block[0]]) is None:
            labels[nonforward] = "nonforward_outside_node_segment_phase"
            continue
        forward = block[moving[block]]
        if not len(forward):
            labels[nonforward] = "node_segment_without_detected_forward_motion"
            continue
        first_forward = int(forward[0])
        last_forward = int(forward[-1])
        before = nonforward[nonforward < first_forward]
        between = nonforward[(nonforward > first_forward) & (nonforward < last_forward)]
        after = nonforward[nonforward > last_forward]
        if motion_mode.startswith("staggered"):
            labels[before] = "pre_release_or_group_pad_lock_hover"
        else:
            labels[before] = "pre_command_or_marker_wait"
        labels[between] = "mid_segment_nonforward_or_correction_wait"
        labels[after] = "post_arrival_group_sync_hover"
    return labels


def durations_and_distance(
    times: np.ndarray,
    monotone_progress: np.ndarray,
    moving: np.ndarray,
    inside: np.ndarray,
) -> tuple[float, float, float]:
    dt = np.diff(times)
    valid_intervals = inside[:-1] & inside[1:] & np.isfinite(dt) & (dt >= 0)
    moving_intervals = valid_intervals & moving[:-1]
    moving_sec = float(dt[moving_intervals].sum())
    total_sec = float(dt[valid_intervals].sum())
    nonforward_sec = max(0.0, total_sec - moving_sec)
    progress_delta = np.clip(np.diff(monotone_progress), 0.0, None)
    moving_distance = float(progress_delta[moving_intervals & np.isfinite(progress_delta)].sum())
    return moving_sec, nonforward_sec, moving_distance


def battery_drop_by_state(
    times: np.ndarray,
    battery: np.ndarray,
    moving: np.ndarray,
    inside: np.ndarray,
) -> tuple[float, float, int]:
    """Assign observed integer SOC decreases to the state at their timestamp.

    This is retained only as a sensitivity field because Tello SOC updates are
    quantized and may lag the physical energy use that caused the decrement.
    """

    dt = np.diff(times)
    valid = (
        inside[:-1]
        & inside[1:]
        & np.isfinite(dt)
        & (dt >= 0)
        & np.isfinite(battery[:-1])
        & np.isfinite(battery[1:])
    )
    decreases = np.clip(battery[:-1] - battery[1:], 0.0, None)
    moving_drop = float(decreases[valid & moving[:-1]].sum())
    nonforward_drop = float(decreases[valid & ~moving[:-1]].sum())
    upward_jump_count = int(np.sum(valid & ((battery[1:] - battery[:-1]) > 0)))
    return moving_drop, nonforward_drop, upward_jump_count


def state_intervals(
    times: np.ndarray,
    phases: np.ndarray,
    labels: np.ndarray,
    inside: np.ndarray,
    metadata: dict,
) -> list[dict]:
    indices = np.flatnonzero(inside)
    if len(indices) < 2:
        return []
    selected_labels = labels[indices]
    selected_phases = phases[indices]
    breaks = np.flatnonzero(
        (selected_labels[1:] != selected_labels[:-1])
        | (selected_phases[1:] != selected_phases[:-1])
    ) + 1
    groups = np.split(np.arange(len(indices)), breaks)
    rows = []
    for sequence, group in enumerate(groups, start=1):
        sample_indices = indices[group]
        start = float(times[sample_indices[0]])
        end = float(times[sample_indices[-1]])
        rows.append(
            {
                **metadata,
                "interval_sequence": sequence,
                "collector_phase": str(selected_phases[group[0]]),
                "node_segment_number": node_segment_number(selected_phases[group[0]]),
                "interval_type": str(selected_labels[group[0]]),
                "start_time_sec": start,
                "end_time_sec": end,
                "duration_sec": max(0.0, end - start),
            }
        )
    return rows


def label_durations(
    times: np.ndarray,
    labels: np.ndarray,
    inside: np.ndarray,
) -> dict[str, float]:
    dt = np.diff(times)
    valid = inside[:-1] & inside[1:] & np.isfinite(dt) & (dt >= 0)
    durations: dict[str, float] = {}
    for label in np.unique(labels[:-1][valid]):
        durations[str(label)] = float(dt[valid & (labels[:-1] == label)].sum())
    return durations


def main() -> None:
    trajectory = pd.read_csv(TRAJECTORY_SEGMENTS, dtype={"run_id": "string"}, low_memory=False)
    eligible = trajectory[trajectory["trajectory_status"].eq("complete_segmented")].copy()
    master = pd.read_csv(
        ADMIN / "cleaning_master_run_index.csv", dtype={"run_id": "string"}, low_memory=False
    )
    eligibility = master[
        ["experiment_directory", "run_id", "primary_analysis_status", "cleaning_decision"]
    ]
    eligible = eligible.merge(
        eligibility,
        on=["experiment_directory", "run_id"],
        how="left",
        validate="many_to_one",
    )

    output_rows: list[dict] = []
    interval_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    missing_files: list[dict] = []

    for (directory, run_id), run_rows in eligible.groupby(
        ["experiment_directory", "run_id"], sort=True
    ):
        coordination_file = find_coordination_file(DB / str(directory), str(run_id))
        if coordination_file is None:
            missing_files.append({"experiment_directory": directory, "run_id": run_id})
            continue
        coordination = pd.read_csv(coordination_file, low_memory=False)
        prepared = prepare_run_groups(coordination)

        for _, source in run_rows.iterrows():
            drone_name = str(source["drone_name"])
            item = prepared.get(drone_name)
            if item is None:
                missing_files.append(
                    {"experiment_directory": directory, "run_id": run_id, "drone_name": drone_name}
                )
                continue

            group = item["group"]
            times = item["times"]
            phases = group.get("phase", pd.Series("", index=group.index)).fillna("").astype(str).to_numpy()
            raw_progress = item["relative"] @ item["run_direction"]
            calibration_factor = float(source["trajectory_distance_calibration_factor"])
            progress = centered_rolling_median(raw_progress, 11) * calibration_factor
            monotone_progress = isotonic_non_decreasing(progress)
            onset = float(source["motion_onset_sec"])
            finish = float(source["selected_250cm_end_sec"])
            motion_mode = collector_motion_mode(
                str(source["formation"]), float(source["inter_drone_spacing_cm"])
            )

            moving, inside, velocity = build_forward_mask(
                times,
                progress,
                phases,
                onset,
                finish,
                PRIMARY_SPEED_THRESHOLD_CM_S,
            )
            labels = classify_samples(phases, moving, inside, motion_mode)
            moving_sec, nonforward_sec, moving_distance = durations_and_distance(
                times, monotone_progress, moving, inside
            )
            reason_durations = label_durations(times, labels, inside)
            battery = pd.to_numeric(group.get("battery"), errors="coerce").to_numpy(float)
            movement_drop, nonforward_drop, upward_jumps = battery_drop_by_state(
                times, battery, moving, inside
            )
            total_sec = moving_sec + nonforward_sec
            mean_forward_speed = moving_distance / moving_sec if moving_sec > 0 else np.nan
            issues = []
            if moving_sec < 18.0 or moving_sec > 36.0:
                issues.append("forward_movement_duration_outside_18_to_36_sec")
            if moving_distance < 205.0:
                issues.append("detected_forward_distance_below_205cm")
            if upward_jumps:
                issues.append("reported_battery_upward_jump_inside_selected_window")
            segment_phase = np.array([node_segment_number(value) is not None for value in phases])
            selected_interval = inside[:-1] & inside[1:]
            interval_dt = np.diff(times)
            valid_selected_interval = selected_interval & np.isfinite(interval_dt) & (interval_dt >= 0)
            segment_phase_sec = float(
                interval_dt[valid_selected_interval & segment_phase[:-1]].sum()
            )
            segment_phase_coverage = segment_phase_sec / total_sec if total_sec > 0 else np.nan
            if np.isfinite(segment_phase_coverage) and segment_phase_coverage < 0.95:
                issues.append("node_segment_phase_coverage_below_95pct")

            metadata = {
                "experiment_directory": str(directory),
                "run_id": str(run_id),
                "drone_name": drone_name,
            }
            output_rows.append(
                {
                    **metadata,
                    "formation": source["formation"],
                    "inter_drone_spacing_cm": source["inter_drone_spacing_cm"],
                    "wind_direction": source["wind_direction"],
                    "wind_level": source["wind_level"],
                    "battery_id": source["battery_id"],
                    "primary_analysis_status": source["primary_analysis_status"],
                    "collector_motion_mode": motion_mode,
                    "collector_expected_wait_causes": expected_wait_causes(motion_mode),
                    "forward_speed_threshold_cm_s": PRIMARY_SPEED_THRESHOLD_CM_S,
                    "bridge_gap_sec": BRIDGE_GAP_SEC,
                    "minimum_movement_island_sec": MIN_MOVEMENT_ISLAND_SEC,
                    "node_segment_phase_coverage_fraction": segment_phase_coverage,
                    "node_segments_observed": len(
                        {number for number in map(node_segment_number, phases[inside]) if number is not None}
                    ),
                    "selected_window_sec_recomputed": total_sec,
                    "forward_movement_sec": moving_sec,
                    "in_flight_nonforward_sec": nonforward_sec,
                    "pre_release_or_group_pad_lock_hover_sec": reason_durations.get(
                        "pre_release_or_group_pad_lock_hover", 0.0
                    ),
                    "pre_command_or_marker_wait_sec": reason_durations.get(
                        "pre_command_or_marker_wait", 0.0
                    ),
                    "mid_segment_nonforward_or_correction_wait_sec": reason_durations.get(
                        "mid_segment_nonforward_or_correction_wait", 0.0
                    ),
                    "post_arrival_group_sync_hover_sec": reason_durations.get(
                        "post_arrival_group_sync_hover", 0.0
                    ),
                    "other_nonforward_sec": (
                        reason_durations.get("nonforward_outside_node_segment_phase", 0.0)
                        + reason_durations.get("node_segment_without_detected_forward_motion", 0.0)
                    ),
                    "forward_movement_fraction": moving_sec / total_sec if total_sec > 0 else np.nan,
                    "detected_forward_distance_cm": moving_distance,
                    "mean_detected_forward_speed_cm_s": mean_forward_speed,
                    "reported_drop_during_forward_events_pp": movement_drop,
                    "reported_drop_during_nonforward_events_pp": nonforward_drop,
                    "battery_upward_jump_count": upward_jumps,
                    "forward_segmentation_issue_codes": ";".join(issues),
                }
            )
            interval_rows.extend(state_intervals(times, phases, labels, inside, metadata))

            for threshold in SENSITIVITY_THRESHOLDS:
                threshold_mask, threshold_inside, _ = build_forward_mask(
                    times, progress, phases, onset, finish, threshold
                )
                threshold_moving, threshold_nonforward, threshold_distance = durations_and_distance(
                    times, monotone_progress, threshold_mask, threshold_inside
                )
                sensitivity_rows.append(
                    {
                        **metadata,
                        "threshold_cm_s": threshold,
                        "forward_movement_sec": threshold_moving,
                        "in_flight_nonforward_sec": threshold_nonforward,
                        "detected_forward_distance_cm": threshold_distance,
                    }
                )

    output = pd.DataFrame(output_rows)
    intervals = pd.DataFrame(interval_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    output.to_csv(OUT_SEGMENTS, index=False)
    intervals.to_csv(OUT_INTERVALS, index=False)
    sensitivity.to_csv(OUT_SENSITIVITY, index=False)

    primary = output[output["primary_analysis_status"].eq("eligible_primary_75_to_40")]
    reason_summary = (
        primary.groupby(
            ["formation", "inter_drone_spacing_cm", "collector_motion_mode"],
            as_index=False,
        )
        .agg(
            drone_rows=("drone_name", "size"),
            median_forward_movement_sec=("forward_movement_sec", "median"),
            median_total_nonforward_sec=("in_flight_nonforward_sec", "median"),
            median_pre_release_or_group_pad_lock_hover_sec=(
                "pre_release_or_group_pad_lock_hover_sec",
                "median",
            ),
            median_pre_command_or_marker_wait_sec=(
                "pre_command_or_marker_wait_sec",
                "median",
            ),
            median_mid_segment_nonforward_or_correction_wait_sec=(
                "mid_segment_nonforward_or_correction_wait_sec",
                "median",
            ),
            median_post_arrival_group_sync_hover_sec=(
                "post_arrival_group_sync_hover_sec",
                "median",
            ),
        )
        .round(3)
    )
    reason_summary.to_csv(OUT_REASON_SUMMARY, index=False)
    threshold_summary = (
        sensitivity.merge(
            eligibility[["experiment_directory", "run_id", "primary_analysis_status"]],
            on=["experiment_directory", "run_id"],
            how="left",
            validate="many_to_one",
        )
        .query("primary_analysis_status == 'eligible_primary_75_to_40'")
        .groupby("threshold_cm_s")["forward_movement_sec"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    summary = {
        "segmentation_definition": (
            "Within the trajectory-selected first 250 cm, forward movement is a smoothed "
            "forward-progress velocity of at least 2 cm/s. Classification is performed "
            "independently inside each data_collector node-segment phase, so release waits "
            "and end-of-segment synchronization holds cannot be bridged across phase boundaries. "
            "Gaps up to 1.0 s are bridged only within the same phase to handle mission-pad "
            "coordinate quantization, and movement islands shorter than 0.5 s are removed."
        ),
        "collector_logic_source": "data_collector.py",
        "collector_wait_mechanisms_used_for_reason_coding": [
            "group current-pad lock before a staggered segment",
            "formation-specific programmed release delays",
            "column spacing safety gate",
            "marker/position correction waits",
            "post-arrival waiting until every drone completes the segment",
        ],
        "eligible_drone_rows_processed": int(len(output)),
        "primary_drone_rows": int(len(primary)),
        "primary_runs": int(primary[["experiment_directory", "run_id"]].drop_duplicates().shape[0]),
        "primary_forward_movement_sec_median": float(primary["forward_movement_sec"].median()),
        "primary_in_flight_nonforward_sec_median": float(primary["in_flight_nonforward_sec"].median()),
        "primary_detected_forward_distance_cm_median": float(
            primary["detected_forward_distance_cm"].median()
        ),
        "primary_pre_release_or_group_pad_lock_hover_sec_median": float(
            primary["pre_release_or_group_pad_lock_hover_sec"].median()
        ),
        "primary_pre_command_or_marker_wait_sec_median": float(
            primary["pre_command_or_marker_wait_sec"].median()
        ),
        "primary_mid_segment_nonforward_or_correction_wait_sec_median": float(
            primary["mid_segment_nonforward_or_correction_wait_sec"].median()
        ),
        "primary_post_arrival_group_sync_hover_sec_median": float(
            primary["post_arrival_group_sync_hover_sec"].median()
        ),
        "primary_node_segment_phase_coverage_median": float(
            primary["node_segment_phase_coverage_fraction"].median()
        ),
        "primary_rows_with_segmentation_flags": int(
            primary["forward_segmentation_issue_codes"].fillna("").ne("").sum()
        ),
        "missing_coordination_items": missing_files,
        "threshold_sensitivity": threshold_summary.round(3).to_dict(orient="records"),
        "important_limitations": [
            "Mission-pad positions are integer-quantized, so sub-second motion state is not observable.",
            "Reason codes use collector phases plus observed movement timing; they distinguish where a wait occurs but do not prove the exact low-level cause of every pause.",
            "Sub-second non-forward gaps inside one node-segment phase cannot be separated reliably from coordinate quantization; the 1.0 s bridge is never allowed to cross a collector phase boundary.",
            "Non-forward time includes hovering, waiting, and lateral-only correction; all are excluded by design.",
            "Reported SOC is integer-valued, so battery-decrement timing is retained only as a sensitivity field.",
            "The primary energy estimate subtracts battery-specific hover-baseline energy during non-forward time rather than attributing individual SOC steps by timestamp.",
        ],
    }
    (TRAJECTORY_DIR / "forward_motion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = f"""# Forward-motion segmentation report

## Definition

{summary['segmentation_definition']}

## Primary analysis coverage

- Primary five-drone runs: {summary['primary_runs']}
- Primary drone rows: {summary['primary_drone_rows']}
- Median forward-moving time: {summary['primary_forward_movement_sec_median']:.2f} s
- Median excluded in-flight non-forward time: {summary['primary_in_flight_nonforward_sec_median']:.2f} s
- Median trajectory advancement assigned to forward movement: {summary['primary_detected_forward_distance_cm_median']:.2f} cm
- Median node-segment phase coverage: {100 * summary['primary_node_segment_phase_coverage_median']:.2f}%
- Rows carrying a segmentation review flag: {summary['primary_rows_with_segmentation_flags']}

## Collector-informed hover causes

- Staggered formations can hover while the group acquires its current pads, during their programmed release delay, and after arrival while waiting for the remaining drones.
- Column formations can additionally hover behind the spacing safety gate until the preceding drone has cleared the required distance.
- Parallel Front, Echelon, and Vee-75 segments do not use staggered release, but an early-arriving drone still waits for the slowest drone before the next segment begins.
- Marker loss, target-pad verification, and position-correction loops can add further non-forward intervals in either control branch.

## Interpretation

The commanded 250 cm at 10 cm/s implies about 25 s of ideal forward movement. Time not classified as forward progression is excluded from the primary energy score, including formation-induced waiting after the common timer starts. The revised cleanup is deliberately stricter at the collector's segment boundaries and preserves auditable reason-coded intervals.

## Limitations

- Mission-pad positions are integer-quantized, so sub-second state changes are not directly observable.
- The 2 cm/s threshold is an operational definition, not a physical motor-state sensor.
- The interval reason code identifies the collector stage in which non-forward time occurred; it should not be read as proof that a specific command alone caused the entire interval.
- Battery percentage updates are quantized and may lag the underlying energy use; individual percentage-point drops are therefore not assigned as the primary energy measure.
"""
    (TRAJECTORY_DIR / "forward_motion_qc_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
