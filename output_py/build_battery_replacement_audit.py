from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db_copy_for_cleaning"
ADMIN = DB / "_cleaning_admin"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_hover_battery_charts import (  # noqa: E402
    cleaning_reason,
    find_hover_timeseries,
    load_hover_timeseries,
    mean_trace_for_battery,
)


def clipped_segment(points: list[dict], upper: float = 75.0, lower: float = 40.0):
    xs = np.asarray([float(point["t"]) for point in points], dtype=float)
    ys = np.asarray([float(point["battery"]) for point in points], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    def crossing(level: float):
        candidates = np.where((ys[:-1] >= level) & (ys[1:] <= level))[0]
        if not len(candidates):
            return None
        index = int(candidates[0])
        if ys[index + 1] == ys[index]:
            return xs[index + 1]
        return xs[index] + (level - ys[index]) * (xs[index + 1] - xs[index]) / (
            ys[index + 1] - ys[index]
        )

    upper_time = crossing(upper)
    lower_time = crossing(lower)
    if upper_time is None or lower_time is None or lower_time <= upper_time:
        return None
    mask = (xs > upper_time) & (xs < lower_time)
    x = np.concatenate(([upper_time], xs[mask], [lower_time])) - upper_time
    y = np.concatenate(([upper], ys[mask], [lower]))
    return x / 60.0, y


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def drone5_start_end(run: dict[str, str]):
    directory = DB / run["experiment_directory"]
    for path in directory.glob("drones/*/*_battery.csv"):
        for row in read_csv(path):
            if row.get("run_id") == run["run_id"] and row.get("drone_name") == "drone_5":
                return (
                    row.get("battery_id", ""),
                    float(row["battery_hover_start"]),
                    float(row["battery_hover_end"]),
                )
    for path in directory.glob("drones/*/*_coordination.csv"):
        rows = [
            row
            for row in read_csv(path)
            if row.get("run_id") == run["run_id"]
            and row.get("drone_name") == "drone_5"
            and row.get("battery", "")
        ]
        if rows:
            return rows[0].get("battery_id", ""), float(rows[0]["battery"]), float(rows[-1]["battery"])
    return None


def main() -> None:
    run_inventory = read_csv(ADMIN / "run_inventory.csv")
    eligible_statuses = {"candidate_pending_trajectory", "recoverable_pending_trajectory"}
    eligible = [
        row
        for row in run_inventory
        if row["scope_status"] == "formal_analysis_candidate"
        and row["overall_status"] in eligible_statuses
    ]

    battery_runs: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    cell_battery_counts: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    for run in eligible:
        result = drone5_start_end(run)
        if result is None:
            continue
        battery_id, start, end = result
        battery_runs[battery_id].append((start, end))
        cell = (
            run["formation"],
            run["inter_drone_spacing_cm"],
            run["wind_direction"],
            run["wind_level"],
        )
        cell_battery_counts[cell][battery_id] += 1

    traces = []
    for path in find_hover_timeseries(DB / "baselines"):
        trace, _ = load_hover_timeseries(path, max_points=2000)
        if trace is not None and not cleaning_reason(trace):
            traces.append(trace)

    metrics = []
    for battery_id in ["B12", "B15"]:
        group = [trace for trace in traces if trace["batteryId"] == battery_id]
        mean_trace = mean_trace_for_battery(battery_id, group, max_points=2000)
        if mean_trace is None:
            continue
        segment = clipped_segment(mean_trace["points"])
        if segment is None:
            continue
        x, y = segment
        coefficient = np.polyfit(x, y, 1)
        fitted = np.polyval(coefficient, x)
        residuals = y - fitted
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
        starts_ends = battery_runs.get(battery_id, [])
        metrics.append(
            {
                "battery_id": battery_id,
                "clean_hover_trace_count": len(group),
                "duration_75_to_40_min": float(x[-1]),
                "linear_discharge_rate_pp_per_min": abs(float(coefficient[0])),
                "linear_fit_r_squared": r_squared,
                "linear_fit_rmse_pp": float(np.sqrt(np.mean(residuals**2))),
                "eligible_formal_run_count": len(starts_ends),
                "runs_fully_within_75_to_40": sum(
                    40 <= start <= 75 and 40 <= end <= 75 for start, end in starts_ends
                ),
                "runs_ending_below_40": sum(end < 40 for _, end in starts_ends),
                "start_soc_min": min((start for start, _ in starts_ends), default=""),
                "start_soc_max": max((start for start, _ in starts_ends), default=""),
                "end_soc_min": min((end for _, end in starts_ends), default=""),
                "end_soc_max": max((end for _, end in starts_ends), default=""),
            }
        )

    with (ADMIN / "battery_replacement_hover_calibration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = list(metrics[0]) if metrics else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)

    both = {
        cell: counts
        for cell, counts in cell_battery_counts.items()
        if counts.get("B12", 0) and counts.get("B15", 0)
    }
    b12_only = sum(
        bool(counts.get("B12")) and not counts.get("B15")
        for counts in cell_battery_counts.values()
    )
    b15_only = sum(
        bool(counts.get("B15")) and not counts.get("B12")
        for counts in cell_battery_counts.values()
    )
    metric_by_battery = {row["battery_id"]: row for row in metrics}
    rate_ratio = (
        metric_by_battery["B12"]["linear_discharge_rate_pp_per_min"]
        / metric_by_battery["B15"]["linear_discharge_rate_pp_per_min"]
    )
    summary = {
        "confirmed_history": "Drone 5 used B15 until it was damaged, then changed to B12.",
        "last_b15_formal_run_id": max(
            row["run_id"]
            for row in eligible
            if "drone_5:B15" in row["drone_battery_map"]
        ),
        "first_b12_formal_run_id": min(
            row["run_id"]
            for row in eligible
            if "drone_5:B12" in row["drone_battery_map"]
        ),
        "b12_vs_b15_hover_rate_ratio": rate_ratio,
        "b12_hover_rate_higher_pct": 100.0 * (rate_ratio - 1.0),
        "eligible_configuration_condition_cell_count": len(cell_battery_counts),
        "cells_with_both_batteries": len(both),
        "b12_only_cells": b12_only,
        "b15_only_cells": b15_only,
        "matched_cells": [
            {
                "formation": cell[0],
                "spacing_cm": cell[1],
                "wind_direction": cell[2],
                "wind_level": cell[3],
                "run_counts": dict(counts),
            }
            for cell, counts in sorted(both.items())
        ],
        "cleaning_policy": {
            "map_b12_to_b15": False,
            "keep_actual_battery_id": True,
            "add_battery_era": True,
            "primary_normalization": "Convert each battery drop with that battery's own hover discharge curve to battery-specific equivalent hover seconds.",
            "primary_soc_window": "Use movement segments within 75%-40%; flag segments that cross below 40%.",
            "sensitivity_checks": [
                "Repeat the analysis without drone 5's contribution.",
                "Report raw percentage-point drop alongside battery-normalized energy.",
                "Do not claim the replacement effect is fully removed because battery and date/configuration are nearly confounded.",
            ],
        },
    }
    (ADMIN / "battery_replacement_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    b12 = metric_by_battery["B12"]
    b15 = metric_by_battery["B15"]
    report = f"""# Drone 5 battery replacement audit

## Confirmed history

Drone 5 used B15 until the battery was damaged, then used B12. The observed formal-run transition is from `{summary['last_b15_formal_run_id']}` to `{summary['first_b12_formal_run_id']}`.

## Baseline evidence

- B12: {b12['clean_hover_trace_count']} clean hover traces; 75%-40% rate {b12['linear_discharge_rate_pp_per_min']:.3f} percentage points/min; R²={b12['linear_fit_r_squared']:.4f}.
- B15: {b15['clean_hover_trace_count']} clean hover traces; 75%-40% rate {b15['linear_discharge_rate_pp_per_min']:.3f} percentage points/min; R²={b15['linear_fit_r_squared']:.4f}.
- B12's fitted hover discharge rate is {summary['b12_hover_rate_higher_pct']:.1f}% higher than B15's, so raw percentage-point drops are not directly comparable.

## Identifiability limitation

Among {summary['eligible_configuration_condition_cell_count']} eligible formation × spacing × wind × level cells, only {summary['cells_with_both_batteries']} cell contains runs from both battery eras. Battery identity is therefore almost completely confounded with date and configuration coverage. Battery-specific calibration can reduce the battery-capacity bias, but it cannot prove that all era-related differences have been removed.

## Cleaning policy

1. Keep B12 and B15 as separate physical batteries; never map B12 to B15.
2. Add `battery_era` and retain the actual battery ID on every drone/run row.
3. Convert SOC drop using each battery's own hover curve to equivalent hover seconds.
4. Use the 75%-40% movement window as the primary analysis window; flag or exclude movement segments crossing below 40%.
5. Run a sensitivity analysis that removes drone 5's contribution. This is not the primary result, but it shows whether formation rankings depend on the replacement battery.
6. Do not estimate a standalone causal B12-versus-B15 effect from the swarm data because there is almost no matched configuration coverage.
"""
    (ADMIN / "battery_replacement_policy.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
