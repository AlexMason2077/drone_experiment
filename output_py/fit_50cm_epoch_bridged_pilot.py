"""One-condition offline pilot: bridge historical baselines to fixed v3.

Keep existing forward raw Medium slopes; use a contemporaneous same-drone
calibration over the identical own-battery SOC band. Map each old relative
drain factor to the new position's battery, then to fixed Bideal. Average the
two protocol means only where both have observations. Never modify inputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from process_50cm_stage_curves import (
    ROOT, MODEL, FORWARD, NAMES, BatteryNormalizer, plt, sha, write_json,
    plot_curve, exclusion_ids, observed_hover_frames, fit_blocks,
)

CID = "diamond_50_head_lv2"
WIND_PILOT = ROOT / "analysis_results/50cm_wind_only_pilot_20260929" / CID
OUT = ROOT / "analysis_results/50cm_epoch_bridged_pilot_20260929" / CID
HISTORICAL_RUNS = {
    "B11":"20260513_143201", "B10":"20260513_180556",
    "B13":"20260513_163826", "B14":"20260513_144257",
    "B15":"20260513_150444",
}


def main():
    if OUT.exists():
        raise FileExistsError("Preserve previous output; choose a new directory.")
    assert CID not in exclusion_ids()
    normalizer = BatteryNormalizer.load(MODEL)
    curve_path = WIND_PILOT / "curve_stage_rates.csv"
    observed_path = WIND_PILOT / "observed_stage_rates.csv"
    wind_manifest = json.loads((WIND_PILOT / "manifest.json").read_text())
    forward_manifest = json.loads((FORWARD / "manifest.json").read_text())
    assert normalizer.sha256 == forward_manifest["normalization_model_sha256"]
    assert wind_manifest["normalization_model_version"] == normalizer.version
    paths = [MODEL, curve_path, observed_path, WIND_PILOT / "manifest.json",
             FORWARD / "run_drone_rates.csv", FORWARD / "manifest.json",
             FORWARD / "selected_forward_intervals.csv", ROOT / "TRAINING_EXCLUSIONS.md"]
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    wind = pd.read_csv(observed_path)
    wind_curve = pd.read_csv(curve_path)
    f = pd.read_csv(FORWARD / "run_drone_rates.csv")
    f = f[f.formation.eq("diamond") & f.wind_direction.eq("head")
        & f.wind_level.eq(2) & f.inter_drone_spacing_cm.eq(50)].copy()
    assert len(f) == 10 and f.run_id.nunique() == 2
    wind_lookup = wind[wind.stage.eq("Medium")].set_index("position")
    assert set(wind_lookup.index) == set(range(1,6))

    # Verify wind rates directly from the real record, independently of the CSV.
    wind_source = ROOT / "database/wind_tunnel_diamond_50_head_lv2_002/wind_tunnel_diamond_50_head_lv2_002_20260928_232343_all_coordination.csv"
    hashes[str(wind_source.relative_to(ROOT))] = sha(wind_source)
    raw_wind = pd.read_csv(wind_source,low_memory=False)
    wind_replay_error = 0.0
    for drone,g in raw_wind.groupby("drone_name"):
        position = int(drone.split("_")[-1])
        row = wind_lookup.loc[position]
        own = normalizer.curve_for(row.battery_id, drone)
        frame,_ = observed_hover_frames(g)
        replay = fit_blocks(frame[frame.raw_soc.between(own.boundaries[2],own.boundaries[1])])
        wind_replay_error = max(wind_replay_error,abs(replay["raw_rate_pp_min"]-row.raw_rate_pp_min))
    assert wind_replay_error < 1e-9

    bridges, records = [], []
    for position,group in f[f.status.eq("included")].groupby("position"):
        battery = str(group.battery_id.iloc[0])
        drone = str(group.drone_name.iloc[0])
        assert group.battery_id.nunique() == 1 and group.drone_name.nunique() == 1
        old_own = normalizer.curve_for(battery,drone)
        upper,lower = old_own.boundaries[1:3]
        run_id = HISTORICAL_RUNS[battery]
        path = ROOT / f"database/baselines/{drone}_{battery}/{drone}_{battery}_hover_{run_id}_timeseries.csv"
        hashes[str(path.relative_to(ROOT))] = sha(path)
        raw = pd.read_csv(path)
        assert set(raw.battery_id) == {battery} and set(raw.drone_name) == {drone}
        g = raw[raw.phase.eq("hover_to_10_percent")].sort_values("elapsed_time").drop_duplicates("elapsed_time")
        t_upper = float(g.loc[g.battery.le(upper),"elapsed_time"].iloc[0])
        t_lower = float(g.loc[g.battery.le(lower),"elapsed_time"].iloc[0])
        old_baseline_rate = 60*(upper-lower)/(t_lower-t_upper)
        # Endpoint/crossing-time definition matches the existing v3 baseline
        # segment parameter. Reuse stage boundaries; do not optimize new knots.
        wr = wind_lookup.loc[position]
        new_own = normalizer.curve_for(str(wr.battery_id),drone)
        new_baseline_rate = float(new_own.rates_pp_min[1])
        ref_rate = float(normalizer.reference.rates_pp_min[1])
        scale = new_baseline_rate / old_baseline_rate
        bridges.append(dict(position=int(position),drone_name=drone,stage="Medium",
            old_battery_id=battery,new_battery_id=wr.battery_id,
            old_baseline_run_id=run_id,old_baseline_source=str(path.relative_to(ROOT)),
            old_medium_upper_soc=upper,old_medium_lower_soc=lower,
            old_baseline_duration_s=t_lower-t_upper,old_baseline_rate_pp_min=old_baseline_rate,
            new_baseline_rate_pp_min=new_baseline_rate,bridge_scale=scale,
            bridged_old_baseline_rate_pp_min=old_baseline_rate*scale,
            new_reference_medium_rate_pp_min=ref_rate,
            baseline_definition="first crossing times at fixed own-Medium boundaries",
            target_definition="existing fixed v3 position battery baseline"))
        assert math.isclose(old_baseline_rate*scale,new_baseline_rate,rel_tol=1e-12)
        for row in group.itertuples():
            source = ROOT / f"db_copy_for_cleaning/{row.experiment_directory}/{row.experiment_directory}_{row.run_id}_all_coordination.csv"
            actual = sha(source)
            assert actual == forward_manifest["input_files_sha256"][str(source.relative_to(ROOT))]
            hashes[str(source.relative_to(ROOT))] = actual
            rate = float(row.raw_medium_discharge_rate_pp_min)
            relative = rate / old_baseline_rate
            bridged_raw = rate*scale
            normalized = bridged_raw/new_baseline_rate*ref_rate
            assert math.isclose(normalized,relative*ref_rate,rel_tol=1e-12)
            # Verify existing v3 coefficients, but do not scale them a second time.
            previous = rate/old_own.rates_pp_min[1]*ref_rate
            assert math.isclose(previous,row.discharge_rate_Bideal_pp_per_min,rel_tol=1e-9)
            records.append(dict(position=int(position),drone_name=drone,stage="Medium",
                experiment_id=row.experiment_directory,run_id=row.run_id,
                old_battery_id=battery,new_battery_id=wr.battery_id,
                measured_raw_rate_pp_min=rate,old_baseline_rate_pp_min=old_baseline_rate,
                old_relative_drain_factor=relative,bridge_scale=scale,
                adjusted_raw_rate_on_new_battery_pp_min=bridged_raw,
                bridged_normalized_rate_pp_min=normalized,
                previous_v3_normalized_rate_pp_min=row.discharge_rate_Bideal_pp_per_min,
                observed_forward_duration_s=row.medium_forward_duration_s,
                observed_forward_SOC_drop_pp=row.raw_medium_drop_pp,
                original_forward_quality_flags=row.qc_flags,
                rate_origin="existing_raw_medium_slope_transferred_by_historical_baseline_ratio"))
    records = pd.DataFrame(records)
    bridges = pd.DataFrame(bridges)
    assert len(bridges) == 5 and len(records) == 8
    forward_mean = records.groupby("position").agg(
        forward_normalized_pp_min=("bridged_normalized_rate_pp_min","mean"),
        forward_run_count=("run_id","nunique"),
        previous_v3_mean_pp_min=("previous_v3_normalized_rate_pp_min","mean"))
    combined = forward_mean.join(wind_lookup[["normalized_rate_pp_min"]].rename(
        columns={"normalized_rate_pp_min":"wind_normalized_pp_min"}))
    combined["combined_normalized_pp_min"] = (
        combined.forward_normalized_pp_min+combined.wind_normalized_pp_min)/2
    combined["wind_run_count"] = 1
    combined["forward_weight"] = combined["wind_weight"] = .5
    combined["reference_medium_duration_s"] = 60*30/combined.combined_normalized_pp_min
    result_curve = wind_curve.copy()
    result_curve["source_policy"] = "existing_wind_only_stage"
    result_curve["forward_run_count"] = 0
    result_curve["is_modeled"] = result_curve.rate_origin.str.contains("modeled")
    for idx,row in result_curve.iterrows():
        if row.stage == "Medium":
            result_curve.loc[idx,"normalized_rate_pp_min"] = combined.loc[row.position,"combined_normalized_pp_min"]
            result_curve.loc[idx,"rate_origin"] = "equal_protocol_mean_after_historical_baseline_bridge"
            result_curve.loc[idx,"source_policy"] = "50_percent_bridged_forward_mean_plus_50_percent_wind_mean"
            result_curve.loc[idx,"forward_run_count"] = combined.loc[row.position,"forward_run_count"]
    result_curve["stage_duration_s"] = 60*(result_curve.reference_upper_soc-result_curve.reference_lower_soc)/result_curve.normalized_rate_pp_min
    knots = []
    for pos,g in result_curve.groupby("position"):
        by = g.set_index("stage")
        time = np.r_[0,np.cumsum([by.loc[s,"stage_duration_s"] for s in NAMES])]
        assert np.all(np.diff(time)>0)
        for index,(t,soc) in enumerate(zip(time,normalizer.reference.boundaries)):
            knots.append(dict(position=int(pos),knot=index,time_s=float(t),normalized_soc=float(soc)))
    high_low = result_curve.stage.ne("Medium")
    assert np.array_equal(result_curve.loc[high_low,"normalized_rate_pp_min"],wind_curve.loc[high_low,"normalized_rate_pp_min"])
    summary = result_curve.pivot(index="position",columns="stage",values="normalized_rate_pp_min").reindex(columns=NAMES)
    summary["reference_100_to_20_s"] = result_curve.groupby("position").stage_duration_s.sum()
    OUT.mkdir(parents=True)
    for filename,data,index in [
        ("baseline_bridge.csv",bridges,False), ("bridged_forward_rates.csv",records,False),
        ("forward_source_audit.csv",f,False), ("medium_protocol_means.csv",combined,True),
        ("curve_stage_rates.csv",result_curve,False),("curve_knots.csv",pd.DataFrame(knots),False),
        ("five_position_summary.csv",summary,True),
    ]:
        data.to_csv(OUT/filename,index=index)

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,ax = plt.subplots(figsize=(11,6.8))
    plot_curve(ax,result_curve,normalizer)
    for name,u,l in zip(NAMES,normalizer.reference.boundaries,normalizer.reference.boundaries[1:]):
        ax.text(.985,(u+l)/2,name,transform=ax.get_yaxis_transform(),ha="right",va="center",
            fontsize=11,color="#64757F",bbox=dict(facecolor="white",edgecolor="none",alpha=.9,pad=2))
    fig.suptitle("Diamond · 50 cm · Headwind · Level 2",x=.11,y=.966,ha="left",fontsize=17,fontweight="semibold")
    fig.text(.11,.912,"Fixed reference battery · Historical baseline transfer before averaging Medium rates",fontsize=10.5,color="#52636C")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.16),ncol=3,frameon=False,fontsize=11)
    fig.text(.11,.026,"High/Low retain wind-source rates; P3/P4 High retain the existing model completion.",fontsize=9,color="#52636C")
    fig.subplots_adjust(left=.11,right=.96,bottom=.23,top=.855)
    for ext in ["png","pdf","svg"]:
        fig.savefig(OUT/f"bridged_averaged_three_stage.{ext}",dpi=300,facecolor="white")
    plt.close(fig)
    validation = dict(single_condition_only=True,
        wind_medium_raw_replay_max_error=wind_replay_error,
        bridge_maps_historical_baseline_to_fixed_target=True,
        old_relative_factor_preserved=True,
        medium_average_max_error=float(np.max(np.abs(combined.combined_normalized_pp_min-(combined.forward_normalized_pp_min+combined.wind_normalized_pp_min)/2))),
        existing_forward_raw_slopes_reused_without_refitting=True,
        previous_v3_normalized_slopes_not_scaled_twice=True,
        high_low_wind_coefficients_unchanged=True,
        reference_and_all_inputs_unchanged=all(sha(ROOT/p)==h for p,h in hashes.items()),
        modeled_high_rows=int(result_curve.is_modeled.sum()),
        training_dataset_modified=False,raw_diagnostic_figure_added_to_dataset=False)
    assert validation["reference_and_all_inputs_unchanged"] and validation["medium_average_max_error"]<1e-12
    write_json(OUT/"validation.json",validation)
    write_json(OUT/"manifest.json",dict(status="single_condition_baseline_bridge_pilot_awaiting_review",
        condition_id=CID,reference_model_version=normalizer.version,reference_model_sha256=normalizer.sha256,
        source_hashes=hashes,historical_baseline_runs=HISTORICAL_RUNS,
        bridge_formula="r_old_adjusted = r_old_observed * r_new_baseline / r_old_baseline",
        normalization_formula="r_old_Bideal = r_old_adjusted / r_new_baseline * r_reference = r_old_observed / r_old_baseline * r_reference",
        averaging_formula="0.5 * mean(old_transferred_rates) + 0.5 * mean(wind_rates), within the same position and stage",
        supported_average_stages=["Medium"],old_raw_forward_fit_changed=False,
        high_low_policy="Retain existing wind-only rates, including two previously modeled High completions; no invented old High/Low data",
        assumptions=["The same position's relative configuration load can transfer between calibration dates and, for P5, between B15 and B12.",
            "Historical same-drone May 13 hover records are suitable contemporary calibration references for the May 20 forward example.",
            "Corresponding own-battery Medium bands represent comparable normalized stages across dates."],
        limitations=["Cross-protocol relative-load transfer is assumed, not independently demonstrated by this average.",
            "There is only one selected historical full baseline per battery and one wind repeat in this pilot.",
            "Short or few-drop forward records retain their existing quality flags.",
            "The current reference model is unchanged and remains a calibration candidate without independent validation."],
        safety_conditions_excluded=sorted(exclusion_ids()),bulk_processing_performed=False,
        active_dataset_updated=False,trained=False,flight_control_changed=False))
    print(combined.round(6).to_string())
    print(summary.round(6).to_string())
    print(json.dumps(validation,indent=2))
    print(OUT/"bridged_averaged_three_stage.png")


if __name__ == "__main__":
    main()
