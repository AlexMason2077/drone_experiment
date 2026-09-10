"""Recompute the approved forward-flight cohort using each battery's own middle.

Read-only inputs. Writes a separate candidate table; never changes frozen data,
telemetry, flight control, training, or online consumers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "output_py"))
from battery_normalization import BatteryNormalizer
from build_forward_discharge_rate_modeling_table import (
    DB, PRIMARY_SPEED_THRESHOLD_CM_S, build_forward_mask, centered_rolling_median,
    curve_for_drone, find_coordination_file, prepare_run_groups, through_origin_fit,
)

FROZEN = ROOT / "frozen_data/discharge_rates/medium_bideal_v1_20260908"
MODEL = ROOT / "analysis_results/battery_normalization_extended_v3_20260909/model.json"
TRAJECTORY = DB / "_cleaning_admin/trajectory_qc/trajectory_drone_segments.csv"
DEFAULT_OUT = ROOT / "analysis_results/medium_forward_bideal_v3_20260909"
KEY = ["experiment_directory", "run_id", "drone_name"]
CELL = ["wind_direction", "wind_level", "formation", "inter_drone_spacing_cm"]
RATE = "discharge_rate_Bideal_pp_per_min"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def medium_intervals(times, battery, moving, inside, upper, lower):
    """Keep whole observed forward intervals in (lower, upper].

    A decrease ending exactly at lower is included; a plateau starting at lower
    belongs to the next SOC band. Straddling intervals are not interpolated.
    Upward SOC steps retain zero discharge and time as in the previous pipeline,
    with an explicit QC count. No missing-time or battery points are fabricated.
    """
    dt = np.diff(times)
    start, end = battery[:-1], battery[1:]
    valid = (inside[:-1] & inside[1:] & moving[:-1]
             & np.isfinite(dt) & (dt >= 0)
             & np.isfinite(start) & np.isfinite(end))
    selected = (valid & (start <= upper) & (start > lower)
                & (end <= upper) & (end >= lower))
    return dt, selected


def summarize(source, item, previous, normalizer):
    meta = {c: previous[c] for c in KEY + CELL + ["battery_id"]}
    meta["position"] = int(str(previous.drone_name).split("_")[-1])
    meta["old_run_rate_Bideal_pp_per_min"] = float(previous.curve_slope_Bideal_pp_per_min)
    meta["old_forward_duration_s"] = float(previous.forward_movement_sec)
    meta["old_qc_flags"] = str(previous.curve_qc_flags) if pd.notna(previous.curve_qc_flags) else ""
    meta["old_upward_jump_count"] = int(previous.reported_battery_upward_jump_count)

    # This independent replay must match the historical forward segmentation.
    replay, _ = curve_for_drone(source, item, float(previous.physical_to_Bideal_scale))
    for field in ("curve_slope_Bideal_pp_per_min", "forward_movement_sec"):
        if not math.isclose(float(replay[field]), float(previous[field]), rel_tol=1e-8, abs_tol=1e-8):
            raise AssertionError(f"Historical replay mismatch: {meta} {field}: {replay[field]} vs {previous[field]}")
    meta["historical_replay_verified"] = True
    if previous.battery_id not in normalizer.curves:
        return dict(meta, status="missing_battery_calibration", qc_flags=""), []
    curve = normalizer.curve_for(previous.battery_id, previous.drone_name)
    upper, lower = curve.boundaries[1:3]
    scale = normalizer.reference.rates_pp_min[1] / curve.rates_pp_min[1]
    meta.update(medium_upper_soc=upper, medium_lower_soc=lower,
                battery_medium_baseline_pp_min=curve.rates_pp_min[1],
                Bideal_medium_baseline_pp_min=normalizer.reference.rates_pp_min[1],
                physical_to_Bideal_medium_scale=scale)

    group, times = item["group"], item["times"]
    if "battery_id" in group:
        ids = set(group.battery_id.dropna().astype(str))
        if ids and ids != {str(previous.battery_id)}:
            raise AssertionError(f"Source battery identity mismatch: {meta} vs {ids}")
    phases = group.get("phase", pd.Series("", index=group.index)).fillna("").astype(str).to_numpy()
    progress = centered_rolling_median(item["relative"] @ item["run_direction"], 11)
    progress = progress * float(source.trajectory_distance_calibration_factor)
    moving, inside, _ = build_forward_mask(times, progress, phases,
        float(source.motion_onset_sec), float(source.selected_250cm_end_sec), PRIMARY_SPEED_THRESHOLD_CM_S)
    battery = pd.to_numeric(group.battery, errors="coerce").to_numpy(float)
    dt, selected = medium_intervals(times, battery, moving, inside, upper, lower)
    ids = np.flatnonzero(selected)
    duration = float(dt[selected].sum())
    meta.update(medium_forward_duration_s=duration, selected_interval_count=len(ids),
                retained_forward_fraction=duration/meta["old_forward_duration_s"])
    if duration <= 0:
        return dict(meta, status="no_forward_time_in_own_medium", qc_flags=""), []

    raw = np.maximum(0.0, battery[ids]-battery[ids+1])
    clock = np.r_[0.0, np.cumsum(dt[ids])]
    cumulative = np.r_[0.0, np.cumsum(raw)]
    raw_slope, _ = through_origin_fit(clock, cumulative)
    rate, r2 = through_origin_fit(clock, cumulative*scale)
    assert math.isclose(rate, raw_slope*scale, rel_tol=1e-12, abs_tol=1e-12)
    # Integral normalization agrees with a constant scale for own-middle samples.
    equiv = sum(curve.equivalent_seconds(float(battery[i]), float(min(battery[i], battery[i+1]))) for i in ids)
    assert math.isclose(equiv*normalizer.reference.rates_pp_min[1]/60, cumulative[-1]*scale,
                        rel_tol=1e-10, abs_tol=1e-10)
    upward = int((battery[ids+1] > battery[ids]).sum())
    flags = []
    if duration < 18:
        flags.append("medium_forward_duration_below_18s")
    if cumulative[-1] == 0:
        flags.append("flat_medium_SOC_trace")
    if upward:
        flags.append("upward_SOC_steps_zero_drop_as_legacy")
    if duration + 1e-6 < meta["old_forward_duration_s"]:
        flags.append("partial_forward_segment_in_own_medium")
    meta.update(status="included", medium_observed_start_soc=float(battery[ids[0]]),
        medium_observed_end_soc=float(battery[ids[-1]+1]),
        raw_medium_discharge_rate_pp_min=raw_slope,
        raw_medium_drop_pp=float(cumulative[-1]), Bideal_medium_drop_pp=float(cumulative[-1]*scale),
        **{RATE: rate}, curve_through_origin_r_squared=r2, medium_upward_step_count=upward,
        qc_flags=";".join(flags))
    evidence = [dict(**{c: meta[c] for c in KEY}, interval_sequence=j+1,
        source_elapsed_start_s=float(times[i]), source_elapsed_end_s=float(times[i+1]),
        observed_soc_start=float(battery[i]), observed_soc_end=float(battery[i+1]),
        forward_clock_s=float(clock[j+1]), cumulative_raw_drop_pp=float(cumulative[j+1]))
        for j, i in enumerate(ids)]
    return meta, evidence


def aggregate(details, cells):
    usable = details[details.status.eq("included")]
    stats = usable.groupby(CELL+["position"])[RATE].agg(["mean", "count", "min", "max", "std"])
    rates, counts, coverage = [], [], []
    for _, cell in cells.sort_values(CELL).iterrows():
        base = {c: cell[c] for c in CELL}
        r, n = dict(base), dict(base)
        for position in range(1,6):
            key = tuple(base[c] for c in CELL)+(position,)
            subset = details[(details[CELL] == pd.Series(base)).all(axis=1) & details.position.eq(position)]
            s = stats.loc[key] if key in stats.index else None
            r[f"position_{position}_{RATE}"] = float(s["mean"]) if s is not None else None
            n[f"position_{position}_run_count"] = int(s["count"]) if s is not None else 0
            good = subset[subset.status.eq("included")]
            coverage.append(dict(**base, position=position,
                candidate_run_count=len(subset), included_run_count=len(good),
                missing_calibration_count=int(subset.status.eq("missing_battery_calibration").sum()),
                no_medium_count=int(subset.status.eq("no_forward_time_in_own_medium").sum()),
                flat_trace_count=int(good.qc_flags.str.contains("flat_medium_SOC_trace").sum()),
                short_trace_count=int(good.qc_flags.str.contains("below_18s").sum()),
                rate_mean=float(s["mean"]) if s is not None else None,
                rate_min=float(s["min"]) if s is not None else None,
                rate_max=float(s["max"]) if s is not None else None,
                rate_std=float(s["std"]) if s is not None else None))
        rates.append(r); counts.append(n)
    return pd.DataFrame(rates), pd.DataFrame(counts), pd.DataFrame(coverage)


def calculate(output, model_path=MODEL):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    model_path = Path(model_path).resolve()
    normalizer = BatteryNormalizer.load(model_path)
    fixed_files = [model_path, TRAJECTORY, FROZEN/"source_run_drone_rates.csv",
                   FROZEN/"source_runs.csv", FROZEN/"manifest.json", FROZEN/"views/discharge_rates_wide.csv"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in fixed_files}
    previous = pd.read_csv(FROZEN/"source_run_drone_rates.csv")
    trajectory = pd.read_csv(TRAJECTORY)
    assert not previous.duplicated(KEY).any()
    assert not trajectory.duplicated(KEY).any()
    lookup = trajectory.set_index(KEY, drop=False)
    records, evidence = [], []
    groups = previous.groupby(KEY[:2], sort=True)
    for counter, ((directory, run_id), run) in enumerate(groups,1):
        path = find_coordination_file(DB/str(directory), str(run_id))
        if path is None:
            raise FileNotFoundError(f"Missing coordination: {directory} / {run_id}")
        hashes[str(path.relative_to(ROOT))] = digest(path)
        prepared = prepare_run_groups(pd.read_csv(path,low_memory=False))
        for _, old in run.iterrows():
            key = tuple(old[c] for c in KEY)
            source = lookup.loc[key]
            assert source.trajectory_status == "complete_segmented"
            for col in CELL+["battery_id"]:
                assert str(source[col]) == str(old[col]), (key,col,source[col],old[col])
            result, points = summarize(source,prepared[old.drone_name],old,normalizer)
            records.append(result);evidence.extend(points)
        if counter % 25 == 0:
            print(f"Recomputed {counter}/{len(groups)} source runs",flush=True)
    details = pd.DataFrame(records)
    cells = previous[CELL].drop_duplicates()
    rates, counts, coverage = aggregate(details,cells)
    # Explicit independent equal-run mean and count checks for every output cell.
    for _, row in rates.iterrows():
        matching = details[(details[CELL] == row[CELL]).all(axis=1) & details.status.eq("included")]
        for p in range(1,6):
            vals = matching[matching.position.eq(p)][RATE].tolist()
            value = row[f"position_{p}_{RATE}"]
            assert (pd.isna(value) and not vals) or math.isclose(float(value),math.fsum(vals)/len(vals), rel_tol=1e-12,abs_tol=1e-12)
    for path, expected in hashes.items():
        assert digest(ROOT/path) == expected, f"Source changed during analysis: {path}"
    used = details[details.status.eq("included")]
    ratecols = [c for c in rates if c.startswith("position_")]
    summary = dict(original_runs=len(groups), original_drone_records=len(details),
        included_drone_records=len(used), included_distinct_runs=len(used[KEY[:2]].drop_duplicates()),
        status_counts=details.status.value_counts().to_dict(), condition_rows=len(rates),
        complete_five_position_rows=int(rates[ratecols].notna().all(axis=1).sum()),
        available_position_rates=int(rates[ratecols].notna().sum().sum()),
        missing_position_rates=int(rates[ratecols].isna().sum().sum()),
        short_medium_records=int(used.qc_flags.str.contains("below_18s").sum()),
        flat_medium_records=int(used.qc_flags.str.contains("flat_medium_SOC_trace").sum()),
        upward_step_records=int(used.medium_upward_step_count.gt(0).sum()),
        partial_medium_records=int(used.qc_flags.str.contains("partial_forward").sum()),
        median_medium_forward_duration_s=float(used.medium_forward_duration_s.median()))
    manifest = dict(version=output.name, summary=summary,
        normalization_model_version=normalizer.version, normalization_model_sha256=normalizer.sha256,
        scope="Exactly the previous frozen 134-run forward-flight cohort, not wind tunnel",
        source_segment_cm=250, decision_interval_s=25, time_rescaled=False,
        medium_selection="Own battery middle: interval start in (lower,upper], end in [lower,upper]. No boundary interpolation.",
        Bideal_medium_soc_range=[normalizer.reference.boundaries[2],normalizer.reference.boundaries[1]],
        rate_definition="Origin-constrained OLS of cumulative forward-only SOC drops against cumulative forward time, then multiplied by Bideal middle baseline rate / own middle baseline rate.",
        aggregation="Arithmetic mean of the included run-specific slopes per condition and position, equal run weight as in frozen table.",
        units="Bideal SOC percentage points per minute",
        warnings="Legacy QC warnings retained. SOC increases contribute zero decrease, as before. Flat traces are recorded zero, not proof of zero energy use. Short/partial traces have higher uncertainty.",
        unsupported_batteries=sorted(set(details[details.status.eq("missing_battery_calibration")].battery_id)),
        consumer_paths_changed=False, input_files_sha256=hashes,
        processing_code_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            [Path(__file__), ROOT/"battery_normalization.py", ROOT/"output_py/build_forward_discharge_rate_modeling_table.py",
             ROOT/"output_py/build_forward_motion_segments.py",ROOT/"output_py/build_trajectory_cleaning_segments.py"]})
    calibration=[]
    for b, curve in normalizer.curves.items():
        calibration.append(dict(battery_id=b,drone_id=normalizer.pairs[b],
            medium_upper_soc=curve.boundaries[1],medium_lower_soc=curve.boundaries[2],
            baseline_medium_pp_min=curve.rates_pp_min[1],
            physical_to_Bideal_medium_scale=normalizer.reference.rates_pp_min[1]/curve.rates_pp_min[1]))
    output.mkdir(parents=True)
    for filename,frame in [("discharge_rates_wide.csv",rates),("sample_counts_wide.csv",counts),
                           ("run_drone_rates.csv",details),("coverage.csv",coverage),
                           ("selected_forward_intervals.csv",pd.DataFrame(evidence)),
                           ("battery_calibration.csv",pd.DataFrame(calibration))]:
        frame.to_csv(output/filename,index=False,float_format="%.12g")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    bundle=dict(summary=summary,model_version=normalizer.version,model_sha256=normalizer.sha256,
        rates=json.loads(rates.to_json(orient="records")),counts=json.loads(counts.to_json(orient="records")),
        coverage=json.loads(coverage.to_json(orient="records")),calibration=calibration,
        reference=dict(medium_upper_soc=normalizer.reference.boundaries[1],
            medium_lower_soc=normalizer.reference.boundaries[2],rate=normalizer.reference.rates_pp_min[1]))
    (output/"workbook_data.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(rates.head(8).round(4).to_string(index=False))
    return output


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--model",type=Path,default=MODEL)
    args=parser.parse_args()
    calculate(args.output, args.model)
