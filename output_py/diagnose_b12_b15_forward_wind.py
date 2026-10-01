"""Offline diagnosis only: preserve calibration, flight code and training inputs.

Compare baseline definitions, a matched B12 SOC band across two dates, and
existing forward/wind measurements. No curve averaging or calibration activation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/wind_tunnel_three_stage_mpl")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "output_py"))
from fit_wind_tunnel_three_stage_rate_first_trial import fit_raw_stage
from generate_hover_battery_charts import (
    cleaning_reason, find_hover_timeseries, load_hover_timeseries,
    mean_trace_for_battery,
)
from plot_hover_baseline_linear_range import clipped_segment

OUT = ROOT / "analysis_results/b12_b15_forward_wind_diagnosis_20260929"
MODEL = ROOT / "analysis_results/battery_normalization_v3_with_b15_20260909/model.json"
OLD = ROOT / "frozen_data/discharge_rates/medium_bideal_v1_20260908"
FORWARD = ROOT / "analysis_results/medium_forward_bideal_v3_with_b15_20260909"
STAGING = ROOT / "analysis_results/50cm_stage_curves_20260929/measured_run_stage_rates.csv"
WIND = ROOT / "database/wind_tunnel_diamond_50_head_lv2_002/wind_tunnel_diamond_50_head_lv2_002_20260928_232343_all_coordination.csv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    if OUT.exists():
        raise FileExistsError("Preserve earlier diagnostics; use a new output directory.")
    model = json.loads(MODEL.read_text())
    forward_manifest = json.loads((FORWARD / "manifest.json").read_text())
    model_hash = sha(MODEL)
    assert model_hash == forward_manifest["normalization_model_sha256"]
    old_cal = pd.read_csv(OLD / "battery_calibration.csv").set_index("battery_id")
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [
        MODEL, OLD / "battery_calibration.csv", FORWARD / "manifest.json",
        FORWARD / "run_drone_rates.csv", FORWARD / "selected_forward_intervals.csv",
        STAGING, WIND, ROOT / "TRAINING_EXCLUSIONS.md",
    ]}

    # Replay the old baseline source selection and its pooled 75..30% fit.
    traces = {"B12": [], "B15": []}
    memberships = []
    for path in find_hover_timeseries(ROOT / "db_copy_for_cleaning/baselines"):
        if not any(f"_{b}/" in str(path) for b in traces):
            continue
        trace, _ = load_hover_timeseries(path, max_points=10000)
        if trace is None or cleaning_reason(trace):
            continue
        traces[trace["batteryId"]].append(trace)
        memberships.append({"battery_id": trace["batteryId"],
            "drone_name": trace["droneName"], "run_id": trace["runId"],
            "source_file": str(path.relative_to(ROOT))})
        sources[str(path.relative_to(ROOT))] = sha(path)
    baseline_versions = []
    for battery, group in traces.items():
        old = old_cal.loc[battery]
        pooled = mean_trace_for_battery(battery, group, max_points=10000)
        x, y = clipped_segment(pooled["points"], old.upper_soc_pct, old.chosen_lower_soc_pct)
        rate = abs(float(np.polyfit(x, y, 1)[0]))
        assert math.isclose(rate, old.physical_battery_discharge_rate_pp_per_min,
                            rel_tol=1e-9, abs_tol=1e-9)
        own = model["batteries"][battery]
        scale = model["reference"]["rates_pp_min"][1] / own["rates_pp_min"][1]
        baseline_versions.append({"battery_id": battery,
            "old_trace_count": len(group), "old_upper_soc": old.upper_soc_pct,
            "old_lower_soc": old.chosen_lower_soc_pct,
            "old_baseline_rate_pp_min": rate,
            "old_reference_rate_pp_min": old.Bideal_discharge_rate_pp_per_min,
            "old_scale": old.scale_physical_drop_to_Bideal,
            "v3_drone_name": own["drone_id"],
            "v3_medium_upper_soc": own["boundaries_soc"][1],
            "v3_medium_lower_soc": own["boundaries_soc"][2],
            "v3_medium_baseline_rate_pp_min": own["rates_pp_min"][1],
            "v3_reference_medium_rate_pp_min": model["reference"]["rates_pp_min"][1],
            "v3_medium_scale": scale,
            "scale_change_percent_if_raw_window_unchanged": 100 * (scale / old.scale_physical_drop_to_Bideal - 1)})

    # Identical band and estimator across dates. This is a diagnostic band,
    # not a newly optimized or activated historical three-stage calibration.
    bands = []
    for battery, path, ranges, role in [
        ("B12", ROOT / "database/baselines/drone_5_B12/drone_5_B12_hover_lv1_20260609_174758_20260609_174758_all_coordination.csv", [(79,48),(75,55)], "historical_diagnostic"),
        ("B12", ROOT / model["batteries"]["B12"]["source"], [(79,48),(75,55)], "v3_calibration_source"),
        ("B15", ROOT / model["batteries"]["B15"]["source"], [(80,54),(75,55)], "v3_calibration_source"),
    ]:
        sources[str(path.relative_to(ROOT))] = sha(path)
        raw = pd.read_csv(path)
        tcol = "node_elapsed_time" if "node_elapsed_time" in raw else "elapsed_time"
        g = raw[raw.phase.eq("hover_to_10_percent")].sort_values(tcol).drop_duplicates(tcol)
        for upper, lower in ranges:
            samples = pd.DataFrame({"raw_soc": g.battery.to_numpy(float),
                                    "fit_time_s": g[tcol].to_numpy(float)})
            fitted, part = fit_raw_stage(samples, upper, lower)
            t_upper = float(g.loc[g.battery.le(upper), tcol].iloc[0])
            t_lower = float(g.loc[g.battery.le(lower), tcol].iloc[0])
            bands.append({"battery_id": battery, "drone_name": "drone_5",
                "run_id": str(g.run_id.iloc[0]), "source_file": str(path.relative_to(ROOT)),
                "role": role, "upper_soc": upper, "lower_soc": lower,
                "crossing_duration_s": t_lower-t_upper,
                "crossing_rate_pp_min": (upper-lower)*60/(t_lower-t_upper),
                "time_weighted_OLS_rate_pp_min": fitted["raw_rate_pp_min"],
                "OLS_r_squared": fitted["r_squared"],
                "median_logged_h_cm": float(g.loc[g.battery.between(lower,upper), "h"].median()),
                "telemetry_endpoint": str(g.drone_ip.iloc[0])})

    rates = pd.read_csv(STAGING)
    exclusions = set(__import__("re").findall(
        r"`((?:column|diamond)_50_(?:head|side|tail)_lv2)`",
        (ROOT / "TRAINING_EXCLUSIONS.md").read_text()))
    assert len(exclusions) == 5
    rates = rates[~rates.condition_id.isin(exclusions)]
    middle = rates[rates.stage.eq("Medium") & rates.position.eq(5)]
    same_b12 = middle[middle.battery_id.eq("B12")]
    grouped = same_b12.groupby(["condition_id", "protocol"]).agg(
        runs=("run_id","nunique"), raw_rate_pp_min=("raw_rate_pp_min","mean"),
        normalized_rate_pp_min=("normalized_rate_pp_min","mean"),
        median_observed_duration_s=("duration_s","median"))
    matched = []
    for condition in grouped.index.get_level_values(0).unique():
        if not all((condition,p) in grouped.index for p in ["forward_250cm","wind_tunnel"]):
            continue
        f, w = grouped.loc[(condition,"forward_250cm")], grouped.loc[(condition,"wind_tunnel")]
        matched.append({"condition_id": condition, "battery_id": "B12", "position": 5,
            "forward_runs": int(f.runs), "wind_runs": int(w.runs),
            "forward_normalized_pp_min": f.normalized_rate_pp_min,
            "wind_normalized_pp_min": w.normalized_rate_pp_min,
            "wind_over_forward_ratio": w.normalized_rate_pp_min/f.normalized_rate_pp_min,
            "forward_median_duration_s": f.median_observed_duration_s,
            "wind_median_duration_s": w.median_observed_duration_s})
    matched = pd.DataFrame(matched)

    # Inspect only the one pilot condition; replay source hashes for both old runs.
    forward = pd.read_csv(FORWARD / "run_drone_rates.csv")
    q = forward[(forward.formation.eq("diamond")) & forward.inter_drone_spacing_cm.eq(50)
        & forward.wind_direction.eq("head") & forward.wind_level.eq(2) & forward.position.eq(5)]
    assert len(q) == 2 and set(q.battery_id) == {"B15"}
    intervals = pd.read_csv(FORWARD / "selected_forward_intervals.csv")
    pilot = []
    for row in q.itertuples():
        raw_path = ROOT / f"db_copy_for_cleaning/{row.experiment_directory}/{row.experiment_directory}_{row.run_id}_all_coordination.csv"
        expected = forward_manifest["input_files_sha256"][str(raw_path.relative_to(ROOT))]
        assert sha(raw_path) == expected
        sources[str(raw_path.relative_to(ROOT))] = expected
        original = pd.read_csv(raw_path, low_memory=False)
        observed = original[original.drone_name.eq("drone_5")]
        i = intervals[(intervals.run_id.eq(row.run_id)) & intervals.drone_name.eq("drone_5")]
        first_drop = float(i.loc[i.cumulative_raw_drop_pp.gt(0),"forward_clock_s"].iloc[0])
        # The raw cumulative trace reproduces the existing origin-constrained slope.
        clock = np.r_[0,i.forward_clock_s.to_numpy(float)]
        drop = np.r_[0,i.cumulative_raw_drop_pp.to_numpy(float)]
        replay = 60 * np.dot(clock,drop)/np.dot(clock,clock)
        assert math.isclose(replay,row.raw_medium_discharge_rate_pp_min,rel_tol=1e-10)
        pilot.append({"protocol": "forward_250cm", "run_id": row.run_id,
            "battery_id": row.battery_id, "source_file": str(raw_path.relative_to(ROOT)),
            "raw_rate_pp_min": replay,
            "normalized_rate_pp_min": row.discharge_rate_Bideal_pp_per_min,
            "observed_duration_s": row.medium_forward_duration_s,
            "selected_SOC_drop_pp": row.raw_medium_drop_pp,
            "first_reported_forward_drop_at_s": first_drop,
            "origin_fit_r_squared": row.curve_through_origin_r_squared,
            "observed_start_SOC": row.medium_observed_start_soc,
            "observed_end_SOC": row.medium_observed_end_soc,
            "telemetry_endpoint": str(observed.drone_ip.iloc[0])})
    w = middle[middle.condition_id.eq("diamond_50_head_lv2") & middle.protocol.eq("wind_tunnel")]
    assert len(w) == 1
    w = w.iloc[0]
    wr = pd.read_csv(WIND,low_memory=False)
    pilot.append({"protocol":"wind_tunnel","run_id":w.run_id,"battery_id":w.battery_id,
        "source_file":str(WIND.relative_to(ROOT)),"raw_rate_pp_min":w.raw_rate_pp_min,
        "normalized_rate_pp_min":w.normalized_rate_pp_min,
        "observed_duration_s":w.duration_s,
        "telemetry_endpoint":str(wr.loc[wr.drone_name.eq("drone_5"),"drone_ip"].iloc[0])})

    b = pd.DataFrame(bands)
    old_b12 = b[(b.battery_id.eq("B12")) & b.upper_soc.eq(79) & b.role.eq("historical_diagnostic")].iloc[0]
    new_b12 = b[(b.battery_id.eq("B12")) & b.upper_soc.eq(79) & b.role.eq("v3_calibration_source")].iloc[0]
    result = {
        "scope":"Read-only diagnosis; no forward/wind averaging, baseline activation, training or hardware actions",
        "model_hash_shared_by_forward_v3_and_wind_pilot":model_hash,
        "b12_matched_79_to_48_crossing_rate_increase_percent":100*(new_b12.crossing_rate_pp_min/old_b12.crossing_rate_pp_min-1),
        "b12_matched_79_to_48_time_weighted_OLS_increase_percent":100*(new_b12.time_weighted_OLS_rate_pp_min/old_b12.time_weighted_OLS_rate_pp_min-1),
        "same_B12_matched_conditions":len(matched),
        "same_B12_wind_higher_conditions":int(matched.wind_over_forward_ratio.gt(1).sum()),
        "same_B12_wind_lower_conditions":int(matched.wind_over_forward_ratio.lt(1).sum()),
        "same_B12_ratio_min":float(matched.wind_over_forward_ratio.min()),
        "same_B12_ratio_max":float(matched.wind_over_forward_ratio.max()),
        "pilot_old_forward_battery":"B15", "pilot_new_wind_battery":"B12",
        "pilot_forward_mean_normalized_pp_min":float(q.discharge_rate_Bideal_pp_per_min.mean()),
        "pilot_wind_normalized_pp_min":float(w.normalized_rate_pp_min),
        "pilot_normalized_wind_over_forward_increase_percent":100*(w.normalized_rate_pp_min/q.discharge_rate_Bideal_pp_per_min.mean()-1),
        "proposed_mapping":"raw configuration stage rate / contemporaneous same-aircraft same-battery stage baseline rate * fixed ideal reference stage rate",
        "mapping_status":"Method recommendation only. No retrospective epoch baseline or protocol transfer has been validated or activated.",
        "limits":[
            "One complete D5/B12 baseline per date; age, hardware, environmental effects and noise are not separately identifiable.",
            "79..48 is a matched diagnostic band, not a newly optimized historical three-stage segmentation.",
            "Telemetry endpoint changes do not prove a physical aircraft change or establish a serial identity.",
            "Forward fits keep reported drops only during detected motion; some drops happen outside motion.",
            "Initial no-drop periods, short windows and integer SOC quantization affect forward slope estimates.",
            "The available B12 matched conditions have only one wind repeat and one to three old forward repeats.",
            "The original 134-run forward table contains a now-excluded safety condition; this diagnostic applies all five current exclusions."
        ]
    }
    OUT.mkdir(parents=True)
    for name,data in [
        ("old_baseline_membership.csv",pd.DataFrame(memberships)),
        ("baseline_definitions.csv",pd.DataFrame(baseline_versions)),
        ("matched_SOC_baseline_records.csv",b),
        ("same_B12_matched_conditions.csv",matched),
        ("diamond_head_lv2_position5_records.csv",pd.DataFrame(pilot)),
    ]:
        data.to_csv(OUT/name,index=False)
    (OUT/"findings.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    (OUT/"manifest.json").write_text(json.dumps({"status":"diagnostic_only_no_dataset_changes",
        "inputs_sha256":sources,"processing_script_sha256":sha(__file__),
        "current_exclusions":sorted(exclusions)},indent=2)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    run()
