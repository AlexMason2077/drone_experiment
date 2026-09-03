"""Compare alternative estimators for configuration-slot battery consumption.

The study is diagnostic: it reads the cleaned 250 cm trajectories and the current
run-selection table, but does not change either.  Outputs are written under
analysis_outputs/position_energy_method_study.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import lsq_linear
except ImportError:  # pragma: no cover
    lsq_linear = None


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db_copy_for_cleaning"
ADMIN = DB / "_cleaning_admin"
TRAJECTORY_DIR = ADMIN / "trajectory_qc"
MODEL_DIR = ROOT / "analysis_outputs" / "forward_discharge_rate_modeling"
OUT = ROOT / "analysis_outputs" / "position_energy_method_study"

sys.path.insert(0, str(ROOT / "output_py"))
from build_forward_motion_segments import (  # noqa: E402
    PRIMARY_SPEED_THRESHOLD_CM_S,
    build_forward_mask,
    durations_and_distance,
)
from build_trajectory_cleaning_segments import (  # noqa: E402
    centered_rolling_median,
    find_coordination_file,
    isotonic_non_decreasing,
    prepare_run_groups,
)


RUN_KEYS = ["experiment_directory", "run_id"]
CELL_KEYS = ["formation", "inter_drone_spacing_cm", "wind_direction", "wind_level"]
SLOT_ORDER = [1, 2, 3, 4, 5]


def slot_id(formation: str, drone_name: str) -> str:
    return f"{str(formation).lower()}_{str(drone_name).split('_')[-1]}"


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    intervals = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1e-9:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def intersect_all(interval_sets: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    if not interval_sets:
        return []
    result = merge_intervals(interval_sets[0])
    for other in interval_sets[1:]:
        other = merge_intervals(other)
        intersection: list[tuple[float, float]] = []
        left = right = 0
        while left < len(result) and right < len(other):
            start = max(result[left][0], other[right][0])
            end = min(result[left][1], other[right][1])
            if end > start:
                intersection.append((start, end))
            if result[left][1] <= other[right][1]:
                left += 1
            else:
                right += 1
        result = intersection
        if not result:
            break
    return result


def interval_overlap(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    return float(sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals))


def through_origin_slope(time_sec: np.ndarray, drop_pp: np.ndarray) -> float:
    x = np.asarray(time_sec, float) / 60.0
    y = np.asarray(drop_pp, float)
    valid = np.isfinite(x) & np.isfinite(y)
    denominator = float(np.sum(x[valid] ** 2))
    if denominator <= 0:
        return math.nan
    return max(0.0, float(np.sum(x[valid] * y[valid]) / denominator))


def build_curves() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_table = pd.read_csv(
        MODEL_DIR / "selected_runs_by_database_cell.csv",
        dtype={"run_id": "string"},
        low_memory=False,
    )
    selected = selected_table[selected_table["selection_status"].eq("selected")][
        RUN_KEYS + CELL_KEYS + ["soc_selection_stratum"]
    ].copy()
    trajectory = pd.read_csv(
        TRAJECTORY_DIR / "trajectory_drone_segments.csv",
        dtype={"run_id": "string"},
        low_memory=False,
    )
    trajectory = trajectory.merge(
        selected[RUN_KEYS + ["soc_selection_stratum"]],
        on=RUN_KEYS,
        how="inner",
        validate="many_to_one",
    )
    scales = (
        pd.read_csv(MODEL_DIR / "battery_ideal_normalization.csv")
        .set_index("battery_id")["scale_physical_drop_to_Bideal"]
        .to_dict()
    )

    run_drone_rows: list[dict] = []
    sample_rows: list[dict] = []
    run_rows: list[dict] = []

    for (directory, run_id), run_source in trajectory.groupby(RUN_KEYS, sort=True):
        path = find_coordination_file(DB / str(directory), str(run_id))
        if path is None:
            raise RuntimeError(f"Missing coordination file: {directory}/{run_id}")
        coordination = pd.read_csv(path, low_memory=False)
        prepared = prepare_run_groups(coordination)
        drone_data: dict[str, dict] = {}

        for _, source in run_source.iterrows():
            drone_name = str(source["drone_name"])
            item = prepared[drone_name]
            group = item["group"]
            times = item["times"]
            phases = group.get("phase", pd.Series("", index=group.index)).fillna("").astype(str).to_numpy()
            raw_progress = item["relative"] @ item["run_direction"]
            progress = centered_rolling_median(raw_progress, 11) * float(
                source["trajectory_distance_calibration_factor"]
            )
            monotone = isotonic_non_decreasing(progress)
            moving, inside, _ = build_forward_mask(
                times,
                progress,
                phases,
                float(source["motion_onset_sec"]),
                float(source["selected_250cm_end_sec"]),
                PRIMARY_SPEED_THRESHOLD_CM_S,
            )
            battery = pd.to_numeric(group.get("battery"), errors="coerce").to_numpy(float)
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
            moving_intervals = valid & moving[:-1]
            first_drop_indices = np.flatnonzero(moving_intervals & (decreases > 0))
            first_drop_wall = (
                float(times[first_drop_indices[0] + 1]) if len(first_drop_indices) else math.nan
            )
            movement_intervals = merge_intervals(
                [(times[index], times[index + 1]) for index in np.flatnonzero(moving_intervals)]
            )
            moving_sec, nonforward_sec, moving_distance = durations_and_distance(
                times, monotone, moving, inside
            )
            selected_battery = battery[inside & np.isfinite(battery)]
            start_battery = float(selected_battery[0]) if len(selected_battery) else math.nan
            end_battery = float(selected_battery[-1]) if len(selected_battery) else math.nan
            scale = float(scales[str(source["battery_id"])])
            forward_drop = float(decreases[moving_intervals].sum())
            total_drop = max(0.0, start_battery - end_battery)
            total_sec = moving_sec + nonforward_sec

            forward_clock = np.zeros(len(times), dtype=float)
            nonforward_clock = np.zeros(len(times), dtype=float)
            for index in range(len(dt)):
                forward_clock[index + 1] = forward_clock[index]
                nonforward_clock[index + 1] = nonforward_clock[index]
                if valid[index]:
                    if moving[index]:
                        forward_clock[index + 1] += float(dt[index])
                    else:
                        nonforward_clock[index + 1] += float(dt[index])

            island_ids = np.full(len(times), -1, dtype=int)
            current_island = -1
            previous = -2
            previous_phase = ""
            for index in np.flatnonzero(inside & moving):
                phase = str(phases[index])
                if index != previous + 1 or phase != previous_phase:
                    current_island += 1
                island_ids[index] = current_island
                previous = int(index)
                previous_phase = phase

            metadata = {
                "experiment_directory": str(directory),
                "run_id": str(run_id),
                "formation": str(source["formation"]),
                "inter_drone_spacing_cm": float(source["inter_drone_spacing_cm"]),
                "wind_direction": str(source["wind_direction"]),
                "wind_level": source["wind_level"],
                "drone_name": drone_name,
                "slot_id": slot_id(source["formation"], drone_name),
                "battery_id": str(source["battery_id"]),
                "physical_to_Bideal_scale": scale,
                "soc_selection_stratum": source["soc_selection_stratum"],
            }

            initial_battery = start_battery
            for index in np.flatnonzero(inside & np.isfinite(battery)):
                sample_rows.append(
                    {
                        **metadata,
                        "wall_time_sec": float(times[index]),
                        "forward_clock_sec": float(forward_clock[index]),
                        "nonforward_clock_sec": float(nonforward_clock[index]),
                        "reported_battery_pct": float(battery[index]),
                        "standardized_drop_from_window_start_pp": max(
                            0.0, (initial_battery - float(battery[index])) * scale
                        ),
                        "moving_forward": bool(moving[index]),
                        "movement_island_id": int(island_ids[index]),
                        "collector_phase": str(phases[index]),
                        "normalized_progress_cm": float(monotone[index]),
                    }
                )

            drone_data[drone_name] = {
                "metadata": metadata,
                "times": times,
                "battery": battery,
                "valid": valid,
                "moving": moving,
                "inside": inside,
                "decreases": decreases,
                "movement_intervals": movement_intervals,
                "first_drop_wall": first_drop_wall,
                "forward_clock": forward_clock,
                "scale": scale,
                "moving_sec": moving_sec,
                "nonforward_sec": nonforward_sec,
                "moving_distance": moving_distance,
                "forward_drop": forward_drop,
                "total_drop": total_drop,
                "total_sec": total_sec,
                "start_battery": start_battery,
                "end_battery": end_battery,
            }

        first_drop_values = [item["first_drop_wall"] for item in drone_data.values()]
        common_observed_onset = (
            max(first_drop_values) if all(np.isfinite(value) for value in first_drop_values) else math.nan
        )
        common_intervals = intersect_all(
            [item["movement_intervals"] for item in drone_data.values()]
        )
        common_forward_sec = float(sum(end - start for start, end in common_intervals))
        run_rows.append(
            {
                "experiment_directory": str(directory),
                "run_id": str(run_id),
                **{column: run_source.iloc[0][column] for column in CELL_KEYS},
                "all_five_have_forward_drop": bool(np.isfinite(common_observed_onset)),
                "common_observed_drop_onset_wall_sec": common_observed_onset,
                "strict_all_five_forward_overlap_sec": common_forward_sec,
                "drone_count_without_forward_drop": int(
                    sum(not np.isfinite(value) for value in first_drop_values)
                ),
            }
        )

        for drone_name, item in drone_data.items():
            times = item["times"]
            battery = item["battery"]
            valid = item["valid"]
            moving = item["moving"]
            decreases = item["decreases"]
            scale = item["scale"]
            metadata = item["metadata"]

            post_onset_sec = post_onset_drop = 0.0
            if np.isfinite(common_observed_onset):
                for index in np.flatnonzero(valid & moving[:-1]):
                    start, end = float(times[index]), float(times[index + 1])
                    overlap = max(0.0, end - max(start, common_observed_onset))
                    if overlap > 0:
                        post_onset_sec += overlap
                        if end > common_observed_onset:
                            post_onset_drop += float(decreases[index])

            simultaneous_sec = simultaneous_drop = 0.0
            for index in np.flatnonzero(valid):
                start, end = float(times[index]), float(times[index + 1])
                overlap = interval_overlap(start, end, common_intervals)
                if overlap > 0:
                    simultaneous_sec += overlap
                    simultaneous_drop += float(decreases[index])

            run_drone_rows.append(
                {
                    **metadata,
                    "start_reported_soc_pct": item["start_battery"],
                    "end_reported_soc_pct": item["end_battery"],
                    "first_forward_drop_wall_sec": item["first_drop_wall"],
                    "common_observed_drop_onset_wall_sec": common_observed_onset,
                    "total_selected_window_sec": item["total_sec"],
                    "forward_movement_sec": item["moving_sec"],
                    "in_flight_nonforward_sec": item["nonforward_sec"],
                    "detected_forward_distance_cm": item["moving_distance"],
                    "total_window_drop_Bideal_pp": item["total_drop"] * scale,
                    "forward_event_drop_Bideal_pp": item["forward_drop"] * scale,
                    "post_common_observed_onset_forward_sec": post_onset_sec,
                    "post_common_observed_onset_drop_Bideal_pp": post_onset_drop * scale,
                    "strict_all_five_forward_sec": simultaneous_sec,
                    "strict_all_five_forward_drop_Bideal_pp": simultaneous_drop * scale,
                    "M0_total_window_rate_pp_per_min": (
                        item["total_drop"] * scale / (item["total_sec"] / 60.0)
                        if item["total_sec"] > 0
                        else math.nan
                    ),
                    "M1_forward_endpoint_rate_pp_per_min": (
                        item["forward_drop"] * scale / (item["moving_sec"] / 60.0)
                        if item["moving_sec"] > 0
                        else math.nan
                    ),
                    "M2_post_common_observed_onset_rate_pp_per_min": (
                        post_onset_drop * scale / (post_onset_sec / 60.0)
                        if post_onset_sec > 0
                        else math.nan
                    ),
                    "M3_strict_all_five_forward_rate_pp_per_min": (
                        simultaneous_drop * scale / (simultaneous_sec / 60.0)
                        if simultaneous_sec > 0
                        else math.nan
                    ),
                    "M4_forward_distance_drop_pp_per_m": (
                        item["forward_drop"] * scale / (item["moving_distance"] / 100.0)
                        if item["moving_distance"] > 0
                        else math.nan
                    ),
                }
            )

    return pd.DataFrame(run_drone_rows), pd.DataFrame(sample_rows), pd.DataFrame(run_rows)


def fixed_effect_slope(group: pd.DataFrame) -> float:
    """Estimate forward slope using a separate intercept per movement island."""
    group = group[group["moving_forward"] & group["movement_island_id"].ge(0)].copy()
    numerator = denominator = 0.0
    for _, island in group.groupby(["experiment_directory", "run_id", "drone_name", "movement_island_id"]):
        x = island["forward_clock_sec"].to_numpy(float) / 60.0
        y = island["standardized_drop_from_window_start_pp"].to_numpy(float)
        if len(x) < 3 or np.ptp(x) <= 0:
            continue
        x = x - np.mean(x)
        y = y - np.mean(y)
        weight = 1.0 / max(len(group[group["run_id"].eq(island["run_id"].iloc[0])]), 1)
        numerator += weight * float(np.sum(x * y))
        denominator += weight * float(np.sum(x**2))
    if denominator <= 0:
        return math.nan
    return max(0.0, numerator / denominator)


def add_island_slopes(run_drone: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    slopes = []
    for keys, group in samples.groupby(RUN_KEYS + ["drone_name"], sort=True):
        slopes.append({**dict(zip(RUN_KEYS + ["drone_name"], keys)), "M5_forward_island_FE_rate_pp_per_min": fixed_effect_slope(group)})
    return run_drone.merge(
        pd.DataFrame(slopes), on=RUN_KEYS + ["drone_name"], how="left", validate="one_to_one"
    )


def fit_two_clock_model(group: pd.DataFrame) -> tuple[dict[str, float], float, float]:
    """Fit slot-specific forward rates plus one nuisance non-forward rate.

    A fixed intercept is removed separately for each run x drone curve.  Samples
    are weighted so each run contributes equal total weight.
    """
    slots = sorted(group["slot_id"].unique())
    columns = [f"forward::{slot}" for slot in slots] + ["nonforward"]
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    for (directory, run_id, drone), curve in group.groupby(RUN_KEYS + ["drone_name"]):
        n = len(curve)
        if n < 3:
            continue
        x = np.zeros((n, len(columns)), dtype=float)
        slot = str(curve["slot_id"].iloc[0])
        x[:, slots.index(slot)] = curve["forward_clock_sec"].to_numpy(float) / 60.0
        x[:, -1] = curve["nonforward_clock_sec"].to_numpy(float) / 60.0
        y = curve["standardized_drop_from_window_start_pp"].to_numpy(float)
        x = x - np.mean(x, axis=0, keepdims=True)
        y = y - np.mean(y)
        run_sample_count = max(len(group[group["run_id"].eq(run_id)]), 1)
        w = np.full(n, 1.0 / run_sample_count, dtype=float)
        x_parts.append(x)
        y_parts.append(y)
        w_parts.append(w)
    if not x_parts:
        return {slot: math.nan for slot in slots}, math.nan, math.nan
    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    weights = np.concatenate(w_parts)
    xw = x * np.sqrt(weights[:, None])
    yw = y * np.sqrt(weights)
    if lsq_linear is not None:
        solution = lsq_linear(xw, yw, bounds=(0.0, np.inf), method="trf").x
    else:
        solution = np.maximum(0.0, np.linalg.lstsq(xw, yw, rcond=None)[0])
    fitted = x @ solution
    residual = y - fitted
    rmse = float(np.sqrt(np.average(residual**2, weights=weights)))
    condition_number = float(np.linalg.cond(xw))
    return dict(zip(slots, solution[:-1])), float(solution[-1]), condition_number


def build_two_clock_estimates(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_level = []
    for keys, group in samples.groupby(CELL_KEYS + ["run_id"], dropna=False, sort=True):
        rates, hover_rate, condition_number = fit_two_clock_model(group)
        for slot, rate in rates.items():
            run_level.append(
                {
                    **dict(zip(CELL_KEYS + ["run_id"], keys)),
                    "slot_id": slot,
                    "M6_two_clock_FE_rate_pp_per_min": rate,
                    "M6_nuisance_nonforward_rate_pp_per_min": hover_rate,
                    "M6_design_condition_number": condition_number,
                }
            )
    pooled = []
    for keys, group in samples.groupby(CELL_KEYS, dropna=False, sort=True):
        rates, hover_rate, condition_number = fit_two_clock_model(group)
        for slot, rate in rates.items():
            pooled.append(
                {
                    **dict(zip(CELL_KEYS, keys)),
                    "slot_id": slot,
                    "M6_two_clock_FE_rate_pp_per_min": rate,
                    "M6_nuisance_nonforward_rate_pp_per_min": hover_rate,
                    "M6_design_condition_number": condition_number,
                    "selected_run_count": group[RUN_KEYS].drop_duplicates().shape[0],
                }
            )
    return pd.DataFrame(run_level), pd.DataFrame(pooled)


def rank_stability(run_rates: pd.DataFrame, rate_column: str) -> tuple[float, int]:
    correlations = []
    comparable_cells = 0
    for _, cell in run_rates.dropna(subset=[rate_column]).groupby(CELL_KEYS, dropna=False):
        pivot = cell.pivot_table(index="slot_id", columns="run_id", values=rate_column, aggfunc="first")
        if pivot.shape[0] < 3 or pivot.shape[1] < 2:
            continue
        comparable_cells += 1
        correlation = pivot.corr(method="spearman").to_numpy()
        values = correlation[np.triu_indices_from(correlation, k=1)]
        correlations.extend(values[np.isfinite(values)].tolist())
    return (float(np.median(correlations)) if correlations else math.nan, comparable_cells)


def method_summary(run_drone: pd.DataFrame, two_clock_run: pd.DataFrame) -> pd.DataFrame:
    base = run_drone.copy()
    two = two_clock_run.merge(
        base[RUN_KEYS + CELL_KEYS + ["slot_id", "drone_name"]],
        on=CELL_KEYS + ["run_id", "slot_id"],
        how="left",
        validate="one_to_one",
    )
    definitions = [
        ("M0 total selected-window rate", base, "M0_total_window_rate_pp_per_min", "total_selected_window_sec"),
        ("M1 forward-event endpoint rate", base, "M1_forward_endpoint_rate_pp_per_min", "forward_movement_sec"),
        ("M2 after all-five observed-drop onset", base, "M2_post_common_observed_onset_rate_pp_per_min", "post_common_observed_onset_forward_sec"),
        ("M3 strict all-five simultaneous-forward rate", base, "M3_strict_all_five_forward_rate_pp_per_min", "strict_all_five_forward_sec"),
        ("M5 within-forward-island fixed-effects rate", base, "M5_forward_island_FE_rate_pp_per_min", "forward_movement_sec"),
        ("M6 two-clock fixed-effects rate", two, "M6_two_clock_FE_rate_pp_per_min", None),
    ]
    rows = []
    denominator = len(base)
    for name, frame, rate, duration in definitions:
        valid = frame[np.isfinite(pd.to_numeric(frame[rate], errors="coerce"))].copy()
        stability, comparable_cells = rank_stability(frame, rate)
        rows.append(
            {
                "method": name,
                "available_run_drone_rows": len(valid),
                "coverage_fraction_of_780": len(valid) / denominator,
                "zero_rate_fraction_among_available": float(np.isclose(valid[rate], 0.0).mean()) if len(valid) else math.nan,
                "median_estimated_rate_pp_per_min": valid[rate].median() if len(valid) else math.nan,
                "median_effective_duration_sec": valid[duration].median() if duration and len(valid) else math.nan,
                "median_pairwise_spearman_slot_rank_across_runs": stability,
                "cells_with_replicate_rank_comparison": comparable_cells,
            }
        )
    return pd.DataFrame(rows)


def simulate_quantization(
    seed: int = 20260815,
    repetitions: int = 3000,
    runs_per_condition: int = 3,
) -> pd.DataFrame:
    """Compare pooled repeated-run estimators when only integer SOC is observed."""
    rng = np.random.default_rng(seed)
    true_rates = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    times = np.arange(0.0, 25.0001, 0.1)
    records = defaultdict(list)
    for _ in range(repetitions):
        run_curves: list[list[np.ndarray]] = []
        common_onsets: list[float] = []
        for _run in range(runs_per_condition):
            curves = []
            firsts = []
            for rate in true_rates:
                fractional_start = 70.0 + rng.uniform(0.0, 1.0)
                latent = fractional_start - rate * times / 60.0
                reported = np.floor(latent)
                drop = reported[0] - reported
                curves.append(drop)
                positive = np.flatnonzero(drop > 0)
                firsts.append(times[positive[0]] if len(positive) else math.nan)
            run_curves.append(curves)
            common_onsets.append(max(firsts) if all(np.isfinite(firsts)) else math.nan)

        for slot, rate in enumerate(true_rates, start=1):
            slot_curves = [curves[slot - 1] for curves in run_curves]
            total_minutes = runs_per_condition * times[-1] / 60.0
            endpoint = float(sum(drop[-1] for drop in slot_curves) / total_minutes)
            origin_numerator = origin_denominator = 0.0
            intercept_numerator = intercept_denominator = 0.0
            crop_drop_total = crop_time_total = 0.0
            for drop, common_onset in zip(slot_curves, common_onsets):
                x = times / 60.0
                origin_numerator += float(np.sum(x * drop))
                origin_denominator += float(np.sum(x**2))
                x_center = x - np.mean(x)
                y_center = drop - np.mean(drop)
                intercept_numerator += float(np.sum(x_center * y_center))
                intercept_denominator += float(np.sum(x_center**2))
                if np.isfinite(common_onset) and common_onset < times[-1]:
                    mask = times >= common_onset
                    crop_t = times[mask] - common_onset
                    crop_drop = drop[mask] - drop[mask][0]
                    crop_drop_total += float(crop_drop[-1])
                    crop_time_total += float(crop_t[-1] / 60.0)
            origin = origin_numerator / origin_denominator
            intercept_slope = intercept_numerator / intercept_denominator
            crop = crop_drop_total / crop_time_total if crop_time_total > 0 else math.nan
            for method, estimate in [
                ("endpoint", endpoint),
                ("through_origin_curve", origin),
                ("free_intercept_curve", intercept_slope),
                ("post_all_five_first_drop", crop),
            ]:
                records[(method, slot, rate)].append(estimate)
    rows = []
    for (method, slot, true_rate), estimates in records.items():
        values = np.asarray(estimates, float)
        valid = np.isfinite(values)
        rows.append(
            {
                "method": method,
                "slot": slot,
                "true_rate_pp_per_min": true_rate,
                "simulation_repetitions": repetitions,
                "runs_pooled_per_condition": runs_per_condition,
                "coverage_fraction": float(valid.mean()),
                "mean_estimate_pp_per_min": float(np.mean(values[valid])),
                "bias_pp_per_min": float(np.mean(values[valid] - true_rate)),
                "rmse_pp_per_min": float(np.sqrt(np.mean((values[valid] - true_rate) ** 2))),
            }
        )
    return pd.DataFrame(rows)


def overlay_curves_for_example(run_drone: pd.DataFrame, samples: pd.DataFrame) -> Path:
    mask = (
        samples["formation"].eq("diamond")
        & samples["inter_drone_spacing_cm"].eq(75.0)
        & samples["wind_direction"].eq("tail")
        & pd.to_numeric(samples["wind_level"], errors="coerce").eq(1)
    )
    data = samples[mask].copy()
    detail = run_drone[
        run_drone["formation"].eq("diamond")
        & run_drone["inter_drone_spacing_cm"].eq(75.0)
        & run_drone["wind_direction"].eq("tail")
        & pd.to_numeric(run_drone["wind_level"], errors="coerce").eq(1)
    ].copy()
    colors = ["#0072B2", "#E69F00", "#009E73"]
    run_ids = sorted(data["run_id"].unique())
    color_map = dict(zip(run_ids, colors))
    fig, axes = plt.subplots(5, 2, figsize=(12.8, 14.0), dpi=220, sharex="col", sharey=True)

    for row_index, slot_number in enumerate(SLOT_ORDER):
        slot = f"diamond_{slot_number}"
        slot_data = data[data["slot_id"].eq(slot)]
        left, right = axes[row_index]
        for run_id, curve in slot_data.groupby("run_id", sort=True):
            moving = curve[curve["moving_forward"]].copy()
            left.step(
                moving["forward_clock_sec"],
                moving["standardized_drop_from_window_start_pp"],
                where="post",
                color=color_map[run_id],
                linewidth=1.35,
                alpha=0.85,
                label=f"run {run_id}",
            )
            row = detail[(detail["run_id"].eq(run_id)) & detail["slot_id"].eq(slot)]
            onset = float(row["common_observed_drop_onset_wall_sec"].iloc[0])
            if np.isfinite(onset):
                post = moving[moving["wall_time_sec"].ge(onset)].copy()
                if not post.empty:
                    post["post_forward_clock_sec"] = (
                        post["forward_clock_sec"] - post["forward_clock_sec"].iloc[0]
                    )
                    post["post_drop_pp"] = (
                        post["standardized_drop_from_window_start_pp"]
                        - post["standardized_drop_from_window_start_pp"].iloc[0]
                    )
                    right.step(
                        post["post_forward_clock_sec"],
                        post["post_drop_pp"],
                        where="post",
                        color=color_map[run_id],
                        linewidth=1.35,
                        alpha=0.85,
                        label=f"run {run_id}",
                    )
        left.set_ylabel(f"{slot}\nBideal drop (pp)")
        for axis in (left, right):
            axis.grid(True, color="#D9DEE3", linewidth=0.55, alpha=0.8)
            axis.spines[["top", "right"]].set_visible(False)
        if row_index == 0:
            left.set_title("A. From detected forward-motion start")
            right.set_title("B. After all five have shown ≥1 reported SOC drop")
            left.legend(frameon=False, fontsize=8, loc="upper left")
            right.legend(frameon=False, fontsize=8, loc="upper left")
    axes[-1, 0].set_xlabel("Per-drone forward-motion clock (s)")
    axes[-1, 1].set_xlabel("Forward-motion clock after common observed-drop onset (s)")
    fig.suptitle(
        "Repeated-run discharge curves: diamond 75 cm, tail wind, level 1\n"
        "Panel B omits any run in which at least one drone never reports a forward SOC drop",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.965])
    path = OUT / "diamond_75_tail_lv1_repeated_curve_alignment_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_method_coverage(summary: pd.DataFrame) -> Path:
    labels = [
        "Total window",
        "Forward endpoint",
        "After all-five\nfirst drop",
        "Strict all-five\nforward overlap",
        "Forward-island FE",
        "Two-clock FE",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=220)
    x = np.arange(len(summary))
    axes[0].bar(x, summary["coverage_fraction_of_780"] * 100, color="#26766A")
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].set_ylabel("Available run × drone rows (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Usable-data coverage")
    axes[1].bar(x, summary["zero_rate_fraction_among_available"] * 100, color="#C46A3A")
    axes[1].set_xticks(x, labels, rotation=22, ha="right")
    axes[1].set_ylabel("Zero estimated-rate rows (%)")
    axes[1].set_ylim(0, max(25, summary["zero_rate_fraction_among_available"].max() * 115))
    axes[1].set_title("Sensitivity to integer-SOC flat curves")
    for axis in axes:
        axis.grid(True, axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = OUT / "method_coverage_and_zero_rate_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_drone, samples, run_summary = build_curves()
    run_drone = add_island_slopes(run_drone, samples)
    two_clock_run, two_clock_pooled = build_two_clock_estimates(samples)
    summary = method_summary(run_drone, two_clock_run)
    simulation = simulate_quantization()

    run_drone.to_csv(OUT / "method_rates_run_drone.csv", index=False)
    run_summary.to_csv(OUT / "method_run_coverage.csv", index=False)
    two_clock_run.to_csv(OUT / "two_clock_run_slot_rates.csv", index=False)
    two_clock_pooled.to_csv(OUT / "two_clock_condition_slot_rates.csv", index=False)
    summary.to_csv(OUT / "method_comparison_summary.csv", index=False)
    simulation.to_csv(OUT / "integer_soc_quantization_simulation.csv", index=False)
    samples.to_csv(
        OUT / "method_study_samples.csv.gz",
        index=False,
        compression="gzip",
    )
    overlay_curves_for_example(run_drone, samples)
    plot_method_coverage(summary)

    print(summary.to_string(index=False))
    print("\nRUN COVERAGE")
    print(run_summary[["all_five_have_forward_drop", "strict_all_five_forward_overlap_sec"]].describe(include="all").to_string())
    print("\nSIMULATION")
    print(simulation.groupby("method").agg(bias=("bias_pp_per_min", "mean"), rmse=("rmse_pp_per_min", "mean"), coverage=("coverage_fraction", "mean")).to_string())
    print(OUT)


if __name__ == "__main__":
    main()
