"""Build Bideal-normalized forward-only discharge curves and modeling tables.

The analysis grain is one formal run x one drone/formation slot.  Hovering and
other non-forward intervals are removed by reusing the phase-aware trajectory
segmentation in ``build_forward_motion_segments.py``.  Reported SOC decreases
are accumulated only while the drone is classified as moving forward.

This script never edits raw or cleaned experiment files.  It writes auditable
analysis products under ``analysis_outputs/forward_discharge_rate_modeling``.
"""

from __future__ import annotations

import json
import math
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
TRAJECTORY_DIR = ADMIN / "trajectory_qc"
OUT = ROOT / "analysis_outputs" / "forward_discharge_rate_modeling"

sys.path.insert(0, str(ROOT / "output_py"))
from build_forward_motion_segments import (  # noqa: E402
    PRIMARY_SPEED_THRESHOLD_CM_S,
    battery_drop_by_state,
    build_forward_mask,
    durations_and_distance,
)
from build_trajectory_cleaning_segments import (  # noqa: E402
    centered_rolling_median,
    find_coordination_file,
    isotonic_non_decreasing,
    prepare_run_groups,
)
from generate_hover_battery_charts import (  # noqa: E402
    SELECTED_MEAN_BATTERIES,
    cleaning_reason,
    find_hover_timeseries,
    load_hover_timeseries,
    mean_trace_for_battery,
)
from plot_hover_baseline_linear_range import clipped_segment  # noqa: E402


BATTERIES = tuple(SELECTED_MEAN_BATTERIES)
UPPER_SOC = 75.0
MIN_LOWER_SOC = 30.0
MAX_LOWER_SOC = 40.0
MIN_BASELINE_R2 = 0.97
CELL_COLUMNS = [
    "formation",
    "inter_drone_spacing_cm",
    "wind_direction",
    "wind_level",
]
RUN_COLUMNS = ["experiment_directory", "run_id"]


def fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    coeff = np.polyfit(x, y, 1)
    fitted = np.polyval(coeff, x)
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": float(coeff[0]),
        "intercept": float(coeff[1]),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def fit_battery_normalization() -> tuple[pd.DataFrame, dict[str, float], dict[str, dict]]:
    """Fit each physical battery and map its rate to the virtual Bideal rate."""

    traces = []
    for path in find_hover_timeseries(DB / "baselines"):
        trace, _ = load_hover_timeseries(path, max_points=10000)
        if trace is not None and not cleaning_reason(trace):
            traces.append(trace)

    rows: list[dict] = []
    curve_details: dict[str, dict] = {}
    for battery_id in BATTERIES:
        group = [trace for trace in traces if trace["batteryId"] == battery_id]
        mean_trace = mean_trace_for_battery(battery_id, group, max_points=10000)
        if mean_trace is None:
            raise RuntimeError(f"No usable complete hover baseline for {battery_id}")

        candidates: list[dict] = []
        for lower in np.arange(MIN_LOWER_SOC, MAX_LOWER_SOC + 0.1, 1.0):
            segment = clipped_segment(mean_trace["points"], UPPER_SOC, float(lower))
            if segment is None:
                continue
            x, y = segment
            linear = fit_line(x, y)
            quadratic = np.polyfit(x, y, 2)
            quadratic_fitted = np.polyval(quadratic, x)
            quadratic_rmse = float(np.sqrt(np.mean((y - quadratic_fitted) ** 2)))
            improvement = (
                (linear["rmse"] - quadratic_rmse) / linear["rmse"]
                if linear["rmse"] > 0
                else 0.0
            )
            candidates.append(
                {
                    "lower_soc_pct": float(lower),
                    **linear,
                    "quadratic_rmse": quadratic_rmse,
                    "quadratic_rmse_improvement_fraction": improvement,
                    "quadratic_coefficient": float(quadratic[0]),
                    "point_count": len(x),
                    "x": x,
                    "y": y,
                }
            )

        # R² is the pre-agreed acceptance gate.  Quadratic improvement remains an
        # audit field rather than a second, newly invented exclusion rule.
        accepted = [item for item in candidates if item["r_squared"] >= MIN_BASELINE_R2]
        if not accepted:
            diagnostics = [
                (item["lower_soc_pct"], item["r_squared"], item["quadratic_rmse_improvement_fraction"])
                for item in candidates
            ]
            raise RuntimeError(
                f"{battery_id} has no accepted 75%-to-30/40% linear baseline: {diagnostics}"
            )
        chosen = min(accepted, key=lambda item: item["lower_soc_pct"])
        curve_details[battery_id] = chosen
        rows.append(
            {
                "battery_id": battery_id,
                "clean_complete_hover_trace_count": len(group),
                "upper_soc_pct": UPPER_SOC,
                "chosen_lower_soc_pct": chosen["lower_soc_pct"],
                "physical_battery_discharge_rate_pp_per_min": abs(chosen["slope"]),
                "linear_fit_intercept_pct": chosen["intercept"],
                "linear_fit_r_squared": chosen["r_squared"],
                "linear_fit_rmse_pp": chosen["rmse"],
                "quadratic_fit_rmse_pp": chosen["quadratic_rmse"],
                "quadratic_rmse_improvement_fraction": chosen[
                    "quadratic_rmse_improvement_fraction"
                ],
                "quadratic_coefficient": chosen["quadratic_coefficient"],
                "fit_point_count": chosen["point_count"],
                "baseline_acceptance_rule": (
                    f"R2>={MIN_BASELINE_R2:.2f}; lowest accepted SOC >=30%; "
                    "quadratic improvement retained as diagnostic"
                ),
            }
        )

    normalization = pd.DataFrame(rows)
    ideal_rate = float(
        normalization["physical_battery_discharge_rate_pp_per_min"].median()
    )
    normalization["Bideal_discharge_rate_pp_per_min"] = ideal_rate
    normalization["scale_physical_drop_to_Bideal"] = (
        ideal_rate / normalization["physical_battery_discharge_rate_pp_per_min"]
    )
    scales = dict(
        zip(normalization["battery_id"], normalization["scale_physical_drop_to_Bideal"])
    )
    return normalization, scales, curve_details


def build_candidate_runs(
    normalization: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select valid formal runs and choose at most three SOC-representative runs/cell."""

    master = pd.read_csv(
        ADMIN / "cleaning_master_run_index.csv", dtype={"run_id": "string"}, low_memory=False
    )
    trajectory = pd.read_csv(
        TRAJECTORY_DIR / "trajectory_drone_segments.csv",
        dtype={"run_id": "string"},
        low_memory=False,
    )
    bounds = normalization.set_index("battery_id")["chosen_lower_soc_pct"].to_dict()
    trajectory["battery_specific_lower_soc_pct"] = trajectory["battery_id"].map(bounds)
    trajectory["drone_soc_range_eligible"] = (
        trajectory["trajectory_status"].eq("complete_segmented")
        & trajectory["battery_at_motion_start_pct"].le(UPPER_SOC)
        & trajectory["battery_at_250cm_end_pct"].ge(
            trajectory["battery_specific_lower_soc_pct"]
        )
    )

    run_trajectory = (
        trajectory.groupby(RUN_COLUMNS, as_index=False)
        .agg(
            trajectory_drone_count=("drone_name", "nunique"),
            all_five_drone_soc_range_eligible=("drone_soc_range_eligible", "all"),
            run_start_soc_min_pct=("battery_at_motion_start_pct", "min"),
            run_start_soc_median_pct=("battery_at_motion_start_pct", "median"),
            run_start_soc_mean_pct=("battery_at_motion_start_pct", "mean"),
            run_start_soc_max_pct=("battery_at_motion_start_pct", "max"),
            run_end_soc_min_pct=("battery_at_250cm_end_pct", "min"),
            flat_full_window_drone_count=(
                "reported_battery_drop_pct_points",
                lambda values: int(np.sum(np.isclose(pd.to_numeric(values), 0.0))),
            ),
        )
    )
    run_trajectory["all_five_complete"] = run_trajectory["trajectory_drone_count"].eq(5)

    run_table = master.merge(run_trajectory, on=RUN_COLUMNS, how="left", validate="one_to_one")
    run_table["formal_clean_trajectory_candidate"] = (
        run_table["scope_status"].eq("formal_analysis_candidate")
        & run_table["cleaning_decision"].eq("eligible_after_trajectory_cleaning")
        & run_table["all_five_complete"].fillna(False)
        & run_table["all_five_drone_soc_range_eligible"].fillna(False)
        & ~run_table["wind_direction"].astype(str).str.contains("no[_ ]?wind", case=False, regex=True)
    )
    run_table["selection_status"] = "not_eligible"
    run_table["selection_reason"] = "outside_formal_complete_battery_range_rules"
    run_table["soc_selection_stratum"] = ""

    eligible = run_table[run_table["formal_clean_trajectory_candidate"]].copy()
    selected_keys: set[tuple[str, str]] = set()
    for _, cell in eligible.groupby(CELL_COLUMNS, dropna=False, sort=True):
        cell = cell.sort_values(
            ["run_start_soc_median_pct", "experiment_directory", "run_id"]
        ).copy()
        candidate_count = len(cell)
        keys = list(zip(cell["experiment_directory"].astype(str), cell["run_id"].astype(str)))
        run_table.loc[
            run_table.set_index(RUN_COLUMNS).index.isin(keys), "cell_eligible_run_count"
        ] = candidate_count

        if candidate_count <= 3:
            chosen = cell
            strata = ["all_available"] * candidate_count
        else:
            # A frozen integer-SOC trace is retained when the cell has <=3 runs.  In
            # replicated cells it is avoided only when at least three alternatives exist.
            nonfrozen = cell[cell["flat_full_window_drone_count"].eq(0)]
            pool = nonfrozen if len(nonfrozen) >= 3 else cell
            pool = pool.sort_values(
                ["run_start_soc_median_pct", "experiment_directory", "run_id"]
            )
            target_quantiles = [0.0, 0.5, 1.0]
            available = list(pool.index)
            chosen_indices: list[int] = []
            strata = []
            for label, quantile in zip(["low", "middle", "high"], target_quantiles):
                target = float(pool["run_start_soc_median_pct"].quantile(quantile))
                possible = pool.loc[[idx for idx in available if idx not in chosen_indices]].copy()
                possible["distance_to_target"] = (
                    possible["run_start_soc_median_pct"] - target
                ).abs()
                choice = int(
                    possible.sort_values(
                        ["distance_to_target", "run_start_soc_median_pct", "run_id"]
                    ).index[0]
                )
                chosen_indices.append(choice)
                strata.append(label)
            chosen = pool.loc[chosen_indices]

        for (_, row), stratum in zip(chosen.iterrows(), strata):
            key = (str(row["experiment_directory"]), str(row["run_id"]))
            selected_keys.add(key)
            mask = (
                run_table["experiment_directory"].astype(str).eq(key[0])
                & run_table["run_id"].astype(str).eq(key[1])
            )
            run_table.loc[mask, "selection_status"] = "selected"
            run_table.loc[mask, "selection_reason"] = (
                "all_eligible_runs_retained" if candidate_count <= 3 else "SOC_representative_run"
            )
            run_table.loc[mask, "soc_selection_stratum"] = stratum

        unselected = cell[
            ~cell.apply(
                lambda row: (str(row["experiment_directory"]), str(row["run_id"]))
                in selected_keys,
                axis=1,
            )
        ]
        for _, row in unselected.iterrows():
            key = (str(row["experiment_directory"]), str(row["run_id"]))
            mask = (
                run_table["experiment_directory"].astype(str).eq(key[0])
                & run_table["run_id"].astype(str).eq(key[1])
            )
            reason = (
                "flat_reported_SOC_with_alternatives"
                if candidate_count > 3
                and row["flat_full_window_drone_count"] > 0
                and len(cell[cell["flat_full_window_drone_count"].eq(0)]) >= 3
                else "not_one_of_three_SOC_representative_runs"
            )
            run_table.loc[mask, "selection_status"] = "eligible_not_selected"
            run_table.loc[mask, "selection_reason"] = reason

    selected = run_table[run_table["selection_status"].eq("selected")].copy()
    return run_table, selected


def through_origin_fit(time_sec: np.ndarray, drop_pp: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(time_sec) & np.isfinite(drop_pp)
    x = time_sec[valid] / 60.0
    y = drop_pp[valid]
    denominator = float(np.sum(x**2))
    if denominator <= 0:
        return math.nan, math.nan
    slope = float(np.sum(x * y) / denominator)
    fitted = slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return max(0.0, slope), r_squared


def curve_for_drone(
    source: pd.Series,
    item: dict,
    scale: float,
) -> tuple[dict, list[dict]]:
    group: pd.DataFrame = item["group"]
    times = item["times"]
    phases = group.get("phase", pd.Series("", index=group.index)).fillna("").astype(str).to_numpy()
    raw_progress = item["relative"] @ item["run_direction"]
    factor = float(source["trajectory_distance_calibration_factor"])
    progress = centered_rolling_median(raw_progress, 11) * factor
    monotone_progress = isotonic_non_decreasing(progress)
    onset = float(source["motion_onset_sec"])
    finish = float(source["selected_250cm_end_sec"])
    moving, inside, _ = build_forward_mask(
        times, progress, phases, onset, finish, PRIMARY_SPEED_THRESHOLD_CM_S
    )
    moving_sec, nonforward_sec, moving_distance = durations_and_distance(
        times, monotone_progress, moving, inside
    )
    battery = pd.to_numeric(group.get("battery"), errors="coerce").to_numpy(float)
    forward_event_drop, nonforward_event_drop, upward_jumps = battery_drop_by_state(
        times, battery, moving, inside
    )

    dt = np.diff(times)
    decreases = np.clip(battery[:-1] - battery[1:], 0.0, None)
    valid = (
        inside[:-1]
        & inside[1:]
        & np.isfinite(dt)
        & (dt >= 0)
        & np.isfinite(battery[:-1])
        & np.isfinite(battery[1:])
    )
    moving_intervals = valid & moving[:-1]
    forward_clock = 0.0
    cumulative_raw = 0.0
    curve = [
        {
            "curve_sample_sequence": 0,
            "forward_clock_sec": 0.0,
            "cumulative_forward_drop_raw_pp": 0.0,
            "cumulative_forward_drop_Bideal_pp": 0.0,
        }
    ]
    sequence = 0
    for index in np.flatnonzero(moving_intervals):
        forward_clock += float(dt[index])
        cumulative_raw += float(decreases[index])
        sequence += 1
        curve.append(
            {
                "curve_sample_sequence": sequence,
                "forward_clock_sec": forward_clock,
                "cumulative_forward_drop_raw_pp": cumulative_raw,
                "cumulative_forward_drop_Bideal_pp": cumulative_raw * scale,
            }
        )

    curve_frame = pd.DataFrame(curve)
    slope, r_squared = through_origin_fit(
        curve_frame["forward_clock_sec"].to_numpy(float),
        curve_frame["cumulative_forward_drop_Bideal_pp"].to_numpy(float),
    )
    endpoint_rate = (
        cumulative_raw * scale / (forward_clock / 60.0) if forward_clock > 0 else math.nan
    )
    inside_battery = battery[inside & np.isfinite(battery)]
    start_soc = float(inside_battery[0]) if len(inside_battery) else math.nan
    end_soc = float(inside_battery[-1]) if len(inside_battery) else math.nan
    metadata = {
        "experiment_directory": str(source["experiment_directory"]),
        "run_id": str(source["run_id"]),
        "formation": str(source["formation"]),
        "inter_drone_spacing_cm": float(source["inter_drone_spacing_cm"]),
        "wind_direction": str(source["wind_direction"]),
        "wind_level": source["wind_level"],
        "drone_name": str(source["drone_name"]),
        "slot_id": f"{str(source['formation']).lower()}_{str(source['drone_name']).split('_')[-1]}",
        "battery_id": str(source["battery_id"]),
    }
    flags = []
    if np.isclose(cumulative_raw, 0.0):
        flags.append("flat_forward_SOC_curve")
    if upward_jumps:
        flags.append("reported_battery_upward_jump")
    if moving_sec < 18.0 or moving_sec > 36.0:
        flags.append("forward_duration_outside_18_to_36_sec")
    if moving_distance < 205.0:
        flags.append("detected_forward_distance_below_205cm")

    summary = {
        **metadata,
        "start_reported_soc_pct": start_soc,
        "end_reported_soc_pct": end_soc,
        "forward_movement_sec": moving_sec,
        "in_flight_nonforward_removed_sec": nonforward_sec,
        "detected_forward_distance_cm": moving_distance,
        "physical_to_Bideal_scale": scale,
        "reported_drop_during_forward_events_raw_pp": forward_event_drop,
        "reported_drop_during_nonforward_events_raw_pp": nonforward_event_drop,
        "cumulative_forward_drop_Bideal_pp": cumulative_raw * scale,
        "endpoint_forward_discharge_rate_Bideal_pp_per_min": max(0.0, endpoint_rate),
        "curve_slope_Bideal_pp_per_min": slope,
        "curve_through_origin_r_squared": r_squared,
        "curve_sample_count": len(curve_frame),
        "reported_battery_upward_jump_count": upward_jumps,
        "flat_forward_SOC_curve": bool(np.isclose(cumulative_raw, 0.0)),
        "curve_qc_flags": ";".join(flags),
    }
    curve_rows = [{**metadata, **row} for row in curve]
    return summary, curve_rows


def build_curves(
    selected_runs: pd.DataFrame,
    trajectory: pd.DataFrame,
    scales: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict] = []
    curves: list[dict] = []
    selected_columns = RUN_COLUMNS + ["soc_selection_stratum", "selection_reason"]
    source = trajectory.merge(
        selected_runs[selected_columns], on=RUN_COLUMNS, how="inner", validate="many_to_one"
    )
    for (directory, run_id), run_rows in source.groupby(RUN_COLUMNS, sort=True):
        path = find_coordination_file(DB / str(directory), str(run_id))
        if path is None:
            raise RuntimeError(f"Missing coordination file for {directory}/{run_id}")
        coordination = pd.read_csv(path, low_memory=False)
        prepared = prepare_run_groups(coordination)
        for _, row in run_rows.iterrows():
            drone_name = str(row["drone_name"])
            if drone_name not in prepared:
                raise RuntimeError(f"Missing {drone_name} in coordination file {path}")
            battery_id = str(row["battery_id"])
            if battery_id not in scales:
                raise RuntimeError(f"No Bideal normalization for {battery_id}")
            summary, curve_rows = curve_for_drone(row, prepared[drone_name], scales[battery_id])
            summary["soc_selection_stratum"] = row["soc_selection_stratum"]
            summary["run_selection_reason"] = row["selection_reason"]
            for curve_row in curve_rows:
                curve_row["soc_selection_stratum"] = row["soc_selection_stratum"]
            summaries.append(summary)
            curves.extend(curve_rows)
    return pd.DataFrame(summaries), pd.DataFrame(curves)


def pooled_position_estimates(curves: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    """Fit one equal-run-weight curve slope for each condition x slot."""

    rows = []
    grouping = CELL_COLUMNS + ["slot_id"]
    for keys, group in curves.groupby(grouping, dropna=False, sort=True):
        weighted_x_y = 0.0
        weighted_x2 = 0.0
        for _, run_curve in group.groupby(RUN_COLUMNS, sort=False):
            x = run_curve["forward_clock_sec"].to_numpy(float) / 60.0
            y = run_curve["cumulative_forward_drop_Bideal_pp"].to_numpy(float)
            n = max(len(run_curve), 1)
            weighted_x_y += float(np.sum(x * y) / n)
            weighted_x2 += float(np.sum(x**2) / n)
        slope = weighted_x_y / weighted_x2 if weighted_x2 > 0 else math.nan
        selector = np.ones(len(summaries), dtype=bool)
        for column, value in zip(grouping, keys):
            selector &= summaries[column].eq(value).to_numpy()
        slot_runs = summaries[selector]
        rows.append(
            {
                **dict(zip(grouping, keys)),
                "selected_run_count": slot_runs[RUN_COLUMNS].drop_duplicates().shape[0],
                "pooled_equal_run_weight_curve_slope_Bideal_pp_per_min": max(0.0, slope),
                "individual_run_curve_slope_min_pp_per_min": slot_runs[
                    "curve_slope_Bideal_pp_per_min"
                ].min(),
                "individual_run_curve_slope_median_pp_per_min": slot_runs[
                    "curve_slope_Bideal_pp_per_min"
                ].median(),
                "individual_run_curve_slope_max_pp_per_min": slot_runs[
                    "curve_slope_Bideal_pp_per_min"
                ].max(),
                "flat_forward_curve_run_count": int(slot_runs["flat_forward_SOC_curve"].sum()),
                "starting_soc_min_pct": slot_runs["start_reported_soc_pct"].min(),
                "starting_soc_max_pct": slot_runs["start_reported_soc_pct"].max(),
            }
        )
    return pd.DataFrame(rows)


def plot_example_cell(
    curves: pd.DataFrame,
    estimates: pd.DataFrame,
    formation: str,
    spacing: float,
    wind: str,
    level: int,
    filename: str,
) -> Path | None:
    mask = (
        curves["formation"].str.lower().eq(formation.lower())
        & curves["inter_drone_spacing_cm"].eq(float(spacing))
        & curves["wind_direction"].str.lower().eq(wind.lower())
        & pd.to_numeric(curves["wind_level"], errors="coerce").eq(level)
    )
    data = curves[mask].copy()
    if data.empty:
        return None
    slots = sorted(data["slot_id"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), dpi=220, sharex=True, sharey=True)
    axes = axes.ravel()
    colors = {"low": "#0072B2", "middle": "#E69F00", "high": "#009E73"}
    run_palette = ["#0072B2", "#E69F00", "#009E73"]
    for axis, slot in zip(axes, slots):
        slot_data = data[data["slot_id"].eq(slot)]
        for run_index, ((directory, run_id), run_curve) in enumerate(
            slot_data.groupby(RUN_COLUMNS, sort=True)
        ):
            label = str(run_curve["soc_selection_stratum"].iloc[0])
            color = colors.get(label, run_palette[run_index % len(run_palette)])
            legend_label = (
                f"{label}: {run_id}" if label != "all_available" else f"run {run_id}"
            )
            axis.step(
                run_curve["forward_clock_sec"],
                run_curve["cumulative_forward_drop_Bideal_pp"],
                where="post",
                color=color,
                alpha=0.72,
                linewidth=1.25,
                label=legend_label,
            )
        est_mask = (
            estimates["formation"].str.lower().eq(formation.lower())
            & estimates["inter_drone_spacing_cm"].eq(float(spacing))
            & estimates["wind_direction"].str.lower().eq(wind.lower())
            & pd.to_numeric(estimates["wind_level"], errors="coerce").eq(level)
            & estimates["slot_id"].eq(slot)
        )
        if est_mask.any():
            slope = float(
                estimates.loc[
                    est_mask, "pooled_equal_run_weight_curve_slope_Bideal_pp_per_min"
                ].iloc[0]
            )
            xmax = max(float(slot_data["forward_clock_sec"].max()), 1.0)
            axis.plot(
                [0, xmax], [0, slope * xmax / 60.0], color="#222222", linewidth=2.1,
                linestyle="--", label=f"pooled slope: {slope:.2f} pp/min"
            )
        axis.set_title(slot.replace("_", " "))
        axis.grid(True, color="#D9DEE3", linewidth=0.55, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7.0, loc="upper left")
    for axis in axes[len(slots):]:
        axis.axis("off")
    fig.supxlabel("Forward-motion clock (s)")
    fig.supylabel("Cumulative Bideal-normalized battery drop (percentage points)")
    fig.suptitle(
        f"{formation.title()} formation · {spacing:g} cm · {wind} · level {level}\n"
        "Hover and other non-forward intervals removed",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    path = OUT / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_data_dictionary(normalization: pd.DataFrame, run_count: int, drone_count: int) -> None:
    ideal_rate = float(normalization["Bideal_discharge_rate_pp_per_min"].iloc[0])
    text = f"""# Forward-only discharge-rate modeling table

## Purpose

This dataset compares formation slots using the shape and slope of the reported
battery-discharge curve, rather than treating a simple mean battery drop as the
main result. All drops are transformed to the virtual standard battery **Bideal**.

## Bideal normalization

- Bideal hover-discharge rate: **{ideal_rate:.6f} percentage points/minute**.
- Bideal is the median accepted hover-baseline rate of B10--B15; it is not a real battery.
- For physical battery `b`, `standardized drop = observed drop × (Bideal rate / battery-b rate)`.
- Each battery has an independent scale factor, including B12 and B15.
- Baseline fitting starts with 75%--30% and raises the lower bound only if needed.
- Acceptance requires R² >= {MIN_BASELINE_R2:.2f}. Quadratic-fit RMSE improvement is
  retained as a curvature diagnostic rather than used as a new exclusion rule.

## Forward-only clock and curve

- The first trajectory-defined 250 cm is selected from reconstructed mission-pad coordinates.
- Forward motion is detected independently inside each logged node-segment phase at >=
  {PRIMARY_SPEED_THRESHOLD_CM_S:g} cm/s.
- Gaps <=1.0 s are bridged only within the same phase; movement islands <0.5 s are removed.
- The forward-motion clock advances only in forward intervals.
- Positive SOC decreases are accumulated only during those forward intervals.
- Waiting, pad-lock hover, correction/lateral-only intervals, and post-arrival synchronization
  do not advance the clock and their observed SOC decreases are not included.
- Positive battery jumps are ignored in cumulative drop and retained as QC flags.

## Primary rate

`curve_slope_Bideal_pp_per_min` is a through-origin fit to each cumulative forward-only
curve. `pooled_equal_run_weight_curve_slope_Bideal_pp_per_min` fits the selected curves
for one condition × slot while giving each run equal total weight. Individual run slopes
and their range remain in the outputs; the pooled slope is not a simple mean of drops.

## Run selection

- Same database condition/cell = formation + spacing + wind direction + wind level.
- Cells with at most three eligible successful runs keep all runs, including a flat
  integer-SOC trace.
- Cells with more than three eligible runs select three representative starting-SOC
  runs (low, middle, high). A flat trace is avoided only if at least three non-flat
  alternatives exist. The same selected run IDs are used for all five slots.

## Output grain and audit fields

- Selected runs: {run_count}
- Run × drone rows: {drone_count}
- `forward_curve_samples.csv`: every cumulative curve point.
- `forward_discharge_rate_run_drone.csv`: one row per selected run × drone.
- `position_discharge_rate_estimates.csv`: one row per condition × slot.
- `selected_runs_by_database_cell.csv`: all run-selection decisions and reasons.
- `battery_ideal_normalization.csv`: physical-battery fits and scale factors.

## Important limitation

DJI Tello reports integer battery percentages. An SOC decrement can be delayed relative
to the physical consumption that caused it and can land just after a motion/hover boundary.
Therefore flat curves and `reported_drop_during_nonforward_events_raw_pp` must remain visible
as sensitivity/QC evidence. Starting SOC is also retained because discharge behavior may
vary across the battery range.
"""
    (OUT / "data_dictionary.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    normalization, scales, _ = fit_battery_normalization()
    run_table, selected_runs = build_candidate_runs(normalization)
    trajectory = pd.read_csv(
        TRAJECTORY_DIR / "trajectory_drone_segments.csv",
        dtype={"run_id": "string"},
        low_memory=False,
    )
    selected_trajectory = trajectory[
        trajectory["trajectory_status"].eq("complete_segmented")
    ].copy()
    summaries, curves = build_curves(selected_runs, selected_trajectory, scales)
    estimates = pooled_position_estimates(curves, summaries)

    normalization.to_csv(OUT / "battery_ideal_normalization.csv", index=False)
    run_table.to_csv(OUT / "selected_runs_by_database_cell.csv", index=False)
    summaries.to_csv(OUT / "forward_discharge_rate_run_drone.csv", index=False)
    curves.to_csv(OUT / "forward_curve_samples.csv", index=False)
    estimates.to_csv(OUT / "position_discharge_rate_estimates.csv", index=False)
    write_data_dictionary(
        normalization,
        selected_runs[RUN_COLUMNS].drop_duplicates().shape[0],
        len(summaries),
    )

    plot_example_cell(
        curves,
        estimates,
        "diamond",
        75,
        "tail",
        1,
        "example_diamond_75_tail_lv1_forward_discharge_curves.png",
    )
    plot_example_cell(
        curves,
        estimates,
        "echalon",
        50,
        "side",
        1,
        "example_echalon_50_side_lv1_forward_discharge_curves.png",
    )

    print(f"Bideal rate: {normalization['Bideal_discharge_rate_pp_per_min'].iloc[0]:.6f} pp/min")
    print(f"Selected runs: {selected_runs[RUN_COLUMNS].drop_duplicates().shape[0]}")
    print(f"Run x drone rows: {len(summaries)}")
    print(f"Curve samples: {len(curves)}")
    print(f"Position estimates: {len(estimates)}")
    print(OUT)


if __name__ == "__main__":
    main()
