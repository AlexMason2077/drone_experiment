"""Correct a whole-run exclusion error without changing experimental records.

Reads the already reviewed separate 50-cm package and actual missed runs.
Collection coverage and fitting eligibility are different fields. No flight,
controller, active training dataset, baseline mutation or protocol averaging.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from process_50cm_stage_curves import (
    ROOT, MODEL, CELL, GRAIN, NAMES, FORMS, FORM_NAMES, WIND_NAMES,
    BatteryNormalizer, plt, sha, write_json, cell_id, condition_from_raw,
    exclusion_ids, fit_blocks, calibrated_rate, plot_curve,
)
from process_separate_forward_wind_latest_baseline import complete_wind_curves, wind_figure

PRIOR = ROOT / "analysis_results/50cm_processed_fixed_baseline_20260929"
DEFAULT_OUT = ROOT / "analysis_results/50cm_selection_corrected_fixed_baseline_20260929"
FIXED_HASH = "5ea37d52e43b4c283f3ce6f956b2b669c1878030e32afc92fc9749fe66af0fe5"
RECOVERED = (
    "database/wind_tunnel_front_50_head_lv2_002/wind_tunnel_front_50_head_lv2_002_20260829_211154_merged_002_005_all_coordination.csv",
    "database/wind_tunnel_vee_50_head_lv1_002/wind_tunnel_vee_50_head_lv1_002_20260909_164806_all_coordination.csv",
    "database/wind_tunnel_vee_50_tail_lv2_001/wind_tunnel_vee_50_tail_lv2_001_20260925_201109_all_coordination.csv",
    "database/wind_tunnel_echalon_50_head_lv1_001/wind_tunnel_echalon_50_head_lv1_001_20260908_205751_all_coordination.csv",
    "database/wind_tunnel_echalon_50_head_lv1_002/wind_tunnel_echalon_50_head_lv1_002_20260908_205928_all_coordination.csv",
    "database/wind_tunnel_echalon_50_head_lv1_003/wind_tunnel_echalon_50_head_lv1_003_20260908_210026_all_coordination.csv",
    "database/wind_tunnel_echalon_50_head_lv1_004/wind_tunnel_echalon_50_head_lv1_004_20260908_210136_all_coordination.csv",
)
LABEL_CORRECTION = dict(experiment_id="wind_tunnel_vee_50_side_lv3_003",
    run_id="20260927_223215", original_wind_level=3, actual_wind_level=2,
    authorization="User explicitly confirmed actual Level2 in this session on 2026-09-29",
    original_condition_id="vee_50_side_lv3", corrected_condition_id="vee_50_side_lv2",
    original_records_unchanged=True)


def select_real_windows(group: pd.DataFrame, merged: bool = False):
    """Hover windows plus telemetry-confirmed holding under a latched alarm.

    Fault-phase samples are supplemental, quality-flagged observed discharge.
    They require a recognized pad and both pad-Z and ToF at 60--100 cm.
    A different pad or a latched fault remains an explicit quality flag.
    These offline screening limits do not prove stable
    formation or safe flight. Missing/rejected rows break the fitted clock block.
    """
    if merged:
        group = group[group.record_origin.eq("observed")].copy()
        components = group.groupby(["source_experiment_id", "source_run_id"], sort=False)
        clock = "source_elapsed_time"
    else:
        components = [("original", group)]
        clock = "elapsed_time"
    frames, audits = [], []
    for ci, (component, g) in enumerate(components):
        g = g.sort_values(clock).reset_index(drop=True)
        # No post-landing samples from that component enter the discharge fit.
        grounded = g.phase.isin(["wind_tunnel_uncommanded_landed", "wind_tunnel_landed",
                                "wind_tunnel_landing_20_percent"])
        # The first landing_20 sample may follow an actual hover sample at 20%.
        first_ground = grounded[grounded].index.min() if grounded.any() else len(g)
        before_ground = g.index < first_ground
        nominal = g.phase.eq("wind_tunnel_hover")
        fault = g.phase.eq("wind_tunnel_control_fault")
        valid_fault = fault & g.mid.isin(range(1, 9)) & g.z.between(60, 100) & g.tof.between(60, 100)
        eligible = (nominal | valid_fault) & before_ground
        h = g[eligible].copy()
        if h.empty:
            audits.append(dict(component=str(component), status="no_selected_observed_window"))
            continue
        h["clock"] = pd.to_numeric(h[clock], errors="raise")
        h["raw_soc"] = pd.to_numeric(h.battery, errors="raise")
        if h.clock.duplicated().any() or h.raw_soc.diff().gt(0).any():
            raise ValueError("Duplicate clock or SOC rebound in selected real window")
        drops = h.index[h.raw_soc.diff().lt(0)]
        if not len(drops):
            audits.append(dict(component=str(component), status="no_observed_SOC_decrease"))
            continue
        start = int(drops[0])
        floor = h.index[h.raw_soc.le(20)]
        stop = int(floor[0]) if len(floor) else int(h.index[-1])
        kept = h.loc[(h.index >= start) & (h.index <= stop) & h.raw_soc.between(20, 100)].copy()
        # Marker changes also break blocks: a frame jump is not a discharge-time bridge.
        episode = (~eligible).cumsum() + g.mid.ne(g.mid.shift()).cumsum()
        kept["block_id"] = [f"{ci}_{int(episode.loc[i])}" for i in kept.index]
        kept["fit_time_s"] = kept.clock - float(h.loc[start, "clock"])
        kept["sample_phase"] = kept.phase
        kept["source_component"] = str(component)
        kept["source_clock_s"] = kept.clock
        kept["telemetry_verified_fault_sample"] = kept.phase.eq("wind_tunnel_control_fault")
        kept["different_assigned_pad_sample"] = kept.mid.ne(kept.target_pad)
        frames.append(kept[["fit_time_s", "raw_soc", "block_id", "sample_phase", "source_component",
                            "source_clock_s", "telemetry_verified_fault_sample", "different_assigned_pad_sample"]])
        audits.append(dict(component=str(component), status="retained_real_windows",
            eligible_samples_before_plateau_removal=len(h), retained_samples=len(kept),
            initial_plateau_removed_s=float(h.loc[start, "clock"] - h.clock.iloc[0]),
            first_selected_soc=float(kept.raw_soc.iloc[0]), last_selected_soc=float(kept.raw_soc.iloc[-1]),
            own_20_percent_observed=bool(len(floor)),
            verified_fault_samples=int(kept.telemetry_verified_fault_sample.sum()),
            different_assigned_pad_samples=int(kept.different_assigned_pad_sample.sum()),
            excluded_fault_samples=int((fault & ~valid_fault).sum()),
            own_selected_window_duration_s=float(kept.fit_time_s.iloc[-1])))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), audits


def replay_wls(frame):
    """Independently replay slope with weighted least squares on centered blocks."""
    xx, yy = [], []
    for _, g in frame.groupby("block_id", sort=False):
        g = g.sort_values("fit_time_s")
        if len(g) < 3:
            continue
        t, y = g.fit_time_s.to_numpy(float), g.raw_soc.to_numpy(float)
        dt = np.diff(t)
        w = np.r_[dt[0], dt[:-1] + dt[1:], dt[-1]] / 2
        tc = t - np.dot(w, t) / w.sum()
        yc = y - np.dot(w, y) / w.sum()
        xx.extend((tc * np.sqrt(w)).tolist())
        yy.extend((yc * np.sqrt(w)).tolist())
    coef = np.linalg.lstsq(np.asarray(xx).reshape(-1, 1), np.asarray(yy), rcond=None)[0][0]
    return float(-60 * coef)


def inventory(registry, hashes, excluded):
    rows = []
    for path in sorted((ROOT / "database").glob("wind_tunnel_*_50_*/*_all_coordination.csv")):
        rel = str(path.relative_to(ROOT)); hashes[rel] = sha(path)
        raw = pd.read_csv(path, low_memory=False)
        exp = path.parent.name
        row = dict(experiment_id=exp, source_file=rel, raw_rows=len(raw),
                   numbered_record=bool(exp.rsplit("_", 1)[-1].isdigit()))
        if raw.empty:
            row.update(status="empty_candidate")
        else:
            try:
                meta = condition_from_raw(raw)
            except ValueError as error:
                row.update(status="outside_requested_condition_grid", detail=str(error))
                rows.append(row)
                continue
            row.update(meta, condition_id=cell_id(meta),
                safety_excluded=cell_id(meta) in excluded,
                first_soc=float(raw.battery.max()), last_soc=float(raw.battery.min()),
                duration_s=float(raw.elapsed_time.max()),
                phases=json.dumps(raw.phase.value_counts().to_dict()),
                registry_outlier=bool(registry.get(exp, {}).get("is_outlier", False)),
                status="nonempty_collected_record")
            if exp == LABEL_CORRECTION["experiment_id"] and meta["run_id"] == LABEL_CORRECTION["run_id"]:
                row.update(original_wind_level=3, original_condition_id=cell_id(meta),
                    wind_level=2, condition_id="vee_50_side_lv2",
                    wind_level_metadata_correction="user_confirmed_actual_Level2")
        rows.append(row)
    return pd.DataFrame(rows)


def main(out: Path):
    if out.exists():
        raise FileExistsError("Preserve earlier exports: choose a new output directory")
    normalizer = BatteryNormalizer.load(MODEL)
    assert normalizer.sha256 == FIXED_HASH
    excluded = exclusion_ids()
    regpath = ROOT / "database/experiment_registry.json"
    regdata = json.loads(regpath.read_text())["experiments"]
    registry = {r["experiment_id"]: r for r in (regdata.values() if isinstance(regdata, dict) else regdata)}
    prior_manifest = json.loads((PRIOR / "manifest.json").read_text())
    for p, digest in prior_manifest["source_hashes"].items():
        assert sha(ROOT / p) == digest, f"Reviewed input changed: {p}"
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [MODEL, regpath,
        ROOT / "TRAINING_EXCLUSIONS.md", Path(__file__),
        ROOT / "output_py/process_50cm_stage_curves.py",
        ROOT / "output_py/process_separate_forward_wind_latest_baseline.py"]}
    for p in PRIOR.rglob("*"):
        if p.is_file():
            hashes[str(p.relative_to(ROOT))] = sha(p)
    collected = inventory(registry, hashes, excluded)
    prior_rates = pd.read_csv(PRIOR / "wind_tunnel/observed_run_stage_rates.csv")
    prior_rates["original_wind_level"] = prior_rates.wind_level
    prior_rates["original_condition_id"] = prior_rates.condition_id
    corrected = prior_rates.experiment_id.eq(LABEL_CORRECTION["experiment_id"]) & prior_rates.run_id.eq(LABEL_CORRECTION["run_id"])
    assert corrected.any() and prior_rates.loc[corrected, "wind_level"].eq(3).all()
    prior_rates.loc[corrected, "wind_level"] = 2
    prior_rates.loc[corrected, "condition_id"] = "vee_50_side_lv2"
    prior_rates.loc[corrected, "wind_level_metadata_correction"] = "user_confirmed_actual_Level2"
    prior_rates["selection_method"] = "previously_reviewed_original_hover_blocks"
    prior_rates["telemetry_verified_fault_sample_count"] = 0
    prior_rates["fault_phase_quality_flag"] = False
    new_rows, new_audits, support, selected_points, replay_errors = [], [], [], [], []
    for rel in RECOVERED:
        path = ROOT / rel
        raw = pd.read_csv(path, low_memory=False)
        meta = condition_from_raw(raw)
        cid, exp = cell_id(meta), path.parent.name
        assert cid not in excluded and meta["spacing_cm"] == 50
        assert not prior_rates.condition_id.eq(cid).any()
        merged = "_merged_" in path.name
        if merged:
            assert registry[exp]["merge_metadata"]["status"] == "canonical_merged_run"
            manifest = path.with_name(path.name.replace("_all_coordination.csv", "_merge_manifest.json"))
            hashes[str(manifest.relative_to(ROOT))] = sha(manifest)
            for source in json.loads(manifest.read_text())["source_files"]:
                assert sha(ROOT / source["path"]) == source["sha256"]
        for drone, g in raw.groupby("drone_name", sort=True):
            ident = g.battery_id.dropna().unique()
            assert len(ident) == 1
            battery = str(ident[0]); position = int(drone.split("_")[-1])
            own = normalizer.curve_for(battery, drone)
            frame, windows = select_real_windows(g, merged)
            dmeta = dict(**meta, protocol="wind_tunnel", experiment_id=exp,
                source_file=rel, condition_id=cid, drone_name=drone, position=position,
                battery_id=battery, original_battery_id=battery, battery_identity_correction="")
            new_audits.append(dict(**dmeta, status="recovered_actual_discharge_windows" if not frame.empty else "no_usable_discharge",
                window_details=json.dumps(windows), retained_samples=len(frame)))
            if frame.empty:
                continue
            points = frame.copy()
            for k in ["source_file", "condition_id", "experiment_id", "run_id", "drone_name", "position", "battery_id"]:
                points[k] = dmeta[k]
            selected_points.append(points)
            for i, (stage, up, lo) in enumerate(zip(NAMES, own.boundaries, own.boundaries[1:])):
                part = frame[frame.raw_soc.between(lo, up)]
                fit = fit_blocks(part)
                if fit is None:
                    continue
                error = abs(replay_wls(part) - fit["raw_rate_pp_min"])
                replay_errors.append(error)
                fault_count = int(part.telemetry_verified_fault_sample.sum())
                pad_change_count = int(part.different_assigned_pad_sample.sum())
                new_rows.append(dict(**dmeta, stage=stage,
                    selection_method="real_hover_and_telemetry_verified_fault_windows",
                    original_wind_level=meta["wind_level"], original_condition_id=cid,
                    rate_origin="independent_time_weighted_OLS_of_selected_real_discharge_blocks",
                    telemetry_verified_fault_sample_count=fault_count,
                    different_assigned_pad_sample_count=pad_change_count,
                    fault_phase_quality_flag=bool(fault_count),
                    training_eligibility="requires_position_control_quality_review" if fault_count or pad_change_count else "candidate_not_activated",
                    **calibrated_rate(fit, battery, drone, i, normalizer)))
                support.append(dict(condition_id=cid, experiment_id=exp, run_id=meta["run_id"],
                    drone_name=drone, position=position, battery_id=battery, stage=stage,
                    own_stage_upper_soc=up, own_stage_lower_soc=lo,
                    observed_upper_soc=float(part.raw_soc.max()), observed_lower_soc=float(part.raw_soc.min()),
                    missing_upper_soc_pp=max(0., up-float(part.raw_soc.max())),
                    missing_lower_soc_pp=max(0., float(part.raw_soc.min())-lo),
                    stage_span_fully_observed=bool(part.raw_soc.max() >= up and part.raw_soc.min() <= lo),
                    own_20_endpoint_observed=any(w.get("own_20_percent_observed", False) for w in windows),
                    initial_unchanged_SOC_seconds_removed=sum(w.get("initial_plateau_removed_s", 0.) for w in windows),
                    fitted_rate_from_real_observations=True, full_reference_stage_duration_is_calculated=True,
                    telemetry_verified_fault_sample_count=fault_count,
                    different_assigned_pad_sample_count=pad_change_count,
                    formation_stability_unverified=bool(fault_count or pad_change_count)))
    added = pd.DataFrame(new_rows)
    rates = pd.concat([prior_rates, added], ignore_index=True)
    assert not rates.duplicated(["experiment_id", "run_id", "drone_name", "stage"]).any()
    means = rates.groupby(GRAIN, as_index=False).agg(
        normalized_rate_pp_min=("normalized_rate_pp_min", "mean"),
        observed_run_count=("run_id", "nunique"), weak_run_count=("weak_support", "sum"),
        min_run_rate_pp_min=("normalized_rate_pp_min", "min"), max_run_rate_pp_min=("normalized_rate_pp_min", "max"),
        run_rate_std_pp_min=("normalized_rate_pp_min", "std"),
        fault_flagged_run_count=("fault_phase_quality_flag", "sum"))
    means["condition_id"] = means.apply(cell_id, axis=1)
    curves, knots = complete_wind_curves(means, normalizer)
    coverage = []
    for formation in FORMS:
        for wind in ["head", "tail", "side"]:
            for level in [1, 2]:
                meta = dict(formation=formation, wind_direction=wind, wind_level=level, spacing_cm=50)
                cid = cell_id(meta)
                c = collected[collected.condition_id.eq(cid) & collected.numbered_record]
                g = curves[curves.condition_id.eq(cid)]
                status = ("excluded_safety_condition" if cid in excluded else
                    "collected_has_stage_curves" if len(g) else
                    "collected_no_fitted_discharge_stage")
                coverage.append(dict(**meta, condition_id=cid, collection_record_found=bool(len(c)),
                    numbered_nonempty_files=len(c), analysis_status=status,
                    positions_with_curves=int(g.position.nunique()), observed_stage_count=int((~g.is_modeled).sum()),
                    modeled_stage_count=int(g.is_modeled.sum()), fault_flagged_stage_count=int(g.fault_flagged_run_count.sum())))
    coverage = pd.DataFrame(coverage)
    assert coverage[~coverage.condition_id.isin(excluded)].collection_record_found.all()
    assert not set(curves.condition_id) & excluded
    assert curves.groupby(["condition_id", "position"]).size().eq(3).all()
    assert curves.groupby("condition_id").position.nunique().eq(5).all()
    assert np.isfinite(curves.normalized_rate_pp_min).all() and curves.normalized_rate_pp_min.gt(0).all()
    for (cid, pos), g in knots.groupby(["condition_id", "position"]):
        g = g.sort_values("knot")
        target = curves[curves.condition_id.eq(cid) & curves.position.eq(pos)].set_index("stage").loc[list(NAMES)]
        assert np.allclose(-60 * np.diff(g.normalized_soc) / np.diff(g.time_s), target.normalized_rate_pp_min)
    oldcurves = pd.read_csv(PRIOR / "wind_tunnel/curve_stage_rates.csv")
    oldcurves.loc[oldcurves.condition_id.eq("vee_50_side_lv3"), "wind_level"] = 2
    oldcurves.loc[oldcurves.condition_id.eq("vee_50_side_lv3"), "condition_id"] = "vee_50_side_lv2"
    comparison = oldcurves.merge(curves, on=GRAIN, suffixes=("_old", "_new"), validate="one_to_one")
    old_error = float(abs(comparison.normalized_rate_pp_min_old - comparison.normalized_rate_pp_min_new).max())
    assert len(comparison) == len(oldcurves) and old_error < 1e-10
    assert max(replay_errors) < 1e-8
    out.mkdir(parents=True)
    wind = out / "wind_tunnel"; wind.mkdir(); (wind / "figures").mkdir()
    shutil.copytree(PRIOR / "forward_250cm", out / "forward_250cm")
    shutil.copyfile(MODEL, out / "fixed_september_baseline.json")
    all_support = pd.concat([pd.read_csv(PRIOR / "wind_tunnel/stage_observation_support.csv"), pd.DataFrame(support)], ignore_index=True)
    old_windows = pd.read_csv(PRIOR / "wind_tunnel/wind_drone_window_audit.csv")
    audit = pd.read_csv(PRIOR / "wind_tunnel/wind_run_audit.csv")
    for rel in RECOVERED:
        mask = audit.source_file.eq(rel)
        audit.loc[mask, "previous_exclusion_reason"] = audit.loc[mask, "status"]
        included_count = int(added.source_file.eq(rel).sum())
        audit.loc[mask, "status"] = "recovered_per_drone_real_windows" if included_count else "collected_without_estimable_retained_discharge_slope"
        audit.loc[mask, "included_stage_count"] = included_count
    labelmask = audit.experiment_id.eq(LABEL_CORRECTION["experiment_id"]) & audit.run_id.eq(LABEL_CORRECTION["run_id"])
    audit.loc[labelmask, "original_wind_level"] = 3
    audit.loc[labelmask, "wind_level"] = 2
    audit.loc[labelmask, "original_condition_id"] = "vee_50_side_lv3"
    audit.loc[labelmask, "condition_id"] = "vee_50_side_lv2"
    audit.loc[labelmask, "wind_level_metadata_correction"] = "user_confirmed_actual_Level2"
    old_windows.loc[old_windows.condition_id.eq("vee_50_side_lv3"), "original_condition_id"] = "vee_50_side_lv3"
    old_windows.loc[old_windows.condition_id.eq("vee_50_side_lv3"), "wind_level"] = 2
    old_windows.loc[old_windows.condition_id.eq("vee_50_side_lv3"), "condition_id"] = "vee_50_side_lv2"
    all_support.loc[all_support.condition_id.eq("vee_50_side_lv3"), "original_condition_id"] = "vee_50_side_lv3"
    all_support.loc[all_support.condition_id.eq("vee_50_side_lv3"), "condition_id"] = "vee_50_side_lv2"
    for filename, table in [
        ("observed_run_stage_rates.csv", rates), ("recovered_observed_stage_rates.csv", added),
        ("condition_position_stage_means.csv", means), ("curve_stage_rates.csv", curves),
        ("three_stage_curve_knots.csv", knots), ("condition_coverage.csv", coverage),
        ("collection_inventory.csv", collected), ("wind_run_audit.csv", audit),
        ("wind_drone_window_audit.csv", pd.concat([old_windows, pd.DataFrame(new_audits)], ignore_index=True)),
        ("stage_observation_support.csv", all_support),
        ("recovered_selected_real_samples.csv", pd.concat(selected_points, ignore_index=True)),
    ]:
        table.to_csv(wind / filename, index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42, "svg.fonttype": "none"})
    with PdfPages(wind / "all_wind_three_stage_curves.pdf") as pdf:
        for cid, g in curves.groupby("condition_id", sort=True):
            fig = wind_figure(g, normalizer)
            pdf.savefig(fig)
            for ext in ["png", "pdf", "svg"]:
                fig.savefig(wind / "figures" / f"{cid}_wind_three_stage.{ext}", dpi=300, facecolor="white")
            plt.close(fig)
    fig, axes = plt.subplots(5, 6, figsize=(22.5, 17.5))
    for i, form in enumerate(FORMS):
        for j, (direction, level) in enumerate([(w, l) for w in ["head", "tail", "side"] for l in [1, 2]]):
            ax = axes[i, j]; cid = f"{form}_50_{direction}_lv{level}"
            ax.set_title(f"{FORM_NAMES[form]} · {WIND_NAMES[direction]} · L{level}", fontsize=10, loc="left")
            g = curves[curves.condition_id.eq(cid)]
            if len(g):
                plot_curve(ax, g, normalizer, small=True)
            else:
                label = ("Safety-excluded" if cid in excluded else "Record found\nConfirm full-run identity" if cid.startswith("echalon")
                         else "Record found\nConfirm L2 / recorded L3")
                ax.set_facecolor("#F0F1F2" if cid in excluded else "#FAF7F0")
                ax.text(.5, .5, label, transform=ax.transAxes, ha="center", va="center", color="#64757F", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
    handles, labels = next(ax for ax in axes.flat if ax.lines).get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(.5, .015))
    fig.suptitle("50 cm · Wind-tunnel three-stage discharge curves", x=.055, y=.984, ha="left", fontsize=22, fontweight="semibold")
    fig.text(.055, .956, "All safe configurations have records · Fixed September baseline · Fit stage rates before reference time", fontsize=12, color="#52636C")
    fig.subplots_adjust(left=.055, right=.985, top=.923, bottom=.071, hspace=.6, wspace=.43)
    for ext in ["png", "pdf"]:
        fig.savefig(wind / f"50cm_wind_overview.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    for p in (PRIOR / "forward_250cm").rglob("*"):
        if p.is_file():
            assert sha(p) == sha(out / "forward_250cm" / p.relative_to(PRIOR / "forward_250cm"))
    assert sha(MODEL) == sha(out / "fixed_september_baseline.json") == FIXED_HASH
    for p, digest in hashes.items():
        assert sha(ROOT / p) == digest, f"Input changed during recovery: {p}"
    summary = dict(collected_safe_level1_2_conditions=int(coverage[~coverage.condition_id.isin(excluded)].collection_record_found.sum()),
        safe_level1_2_conditions_with_stage_curves=int(curves[curves.wind_level.isin([1, 2])].condition_id.nunique()),
        actual_additional_Level3_conditions=int(curves[curves.wind_level.eq(3)].condition_id.nunique()),
        wind_level_metadata_corrections=[LABEL_CORRECTION],
        recovered_conditions=sorted(added.condition_id.unique().tolist()), recovered_observed_stage_rates=len(added),
        observed_stage_rates=len(rates), curve_stage_coefficients=len(curves),
        missing_whole_stage_model_completions=int(curves.is_modeled.sum()),
        collected_conditions_without_estimated_discharge=coverage[coverage.positions_with_curves.eq(0) & ~coverage.condition_id.isin(excluded)].condition_id.tolist(),
        raw_input_files_audited=len(collected), safety_excluded_conditions=sorted(excluded),
        forward_adjustment_unchanged=True, averaging_applied=False)
    validation = dict(status="passed_offline_checks_for_recovered_selection", sources_unchanged=True,
        fixed_september_baseline_unchanged=True, all_safe_conditions_have_collected_records=True,
        prior_stage_rates_unchanged=True, prior_curve_rate_max_error_pp_min=old_error,
        recovered_raw_WLS_replay_max_error_pp_min=max(replay_errors), every_plotted_condition_has_five_positions=True,
        all_curves_have_three_stages=True, stage_time_slope_identity=True,
        excluded_conditions_absent=True, simulated_gap_rows_used=0,
        user_confirmed_Level3_metadata_corrected_to_actual_Level2=True, actual_missing_sources_not_invented=True,
        original_raw_records_or_controller_changed=False, active_training_dataset_changed=False)
    write_json(out / "summary.json", summary)
    write_json(out / "validation.json", validation)
    manifest = dict(status="corrected_all_50cm_separate_candidate_analysis_not_activated",
        normalization_model_sha256=FIXED_HASH, source_hashes=hashes, summary=summary,
        source_previous_package=str(PRIOR.relative_to(ROOT)),
        corrected_error="A fault or landing label on one drone no longer excludes an entire collected configuration.",
        initial_plateau_policy="Remove through first selected observed SOC decrease per component; retain later integer-SOC plateaus.",
        selection="Previously reviewed results plus recovered per-drone long and short trials; observed hover blocks, with qualified fault-latched windows separately flagged.",
        verified_fault_gate="Recognized Mission Pad and z/ToF both 60..100 cm. Assigned-pad mismatches and fault states remain quality flags, not independent validation of a stable formation or exact spacing.",
        wind_level_metadata_corrections=[LABEL_CORRECTION],
        short_record_user_authorization="User approved fitting stages from separate short observations; no need for each run to reach 20%.",
        partial_stage_policy="Actual slopes fitted only inside own battery SOC stage; missing stage completed only in curve model using same-condition/position relative factor.",
        time_definition="60 * reference stage SOC drop / normalized stage rate; fit rates before calculating time.",
        uncertainty="Fault-latched holding segments, partial SOC spans and changing membership after a peer lands need review before training; measured rates retain source quality flags.",
        cross_protocol_averaging_applied=False, original_raw_records_changed=False,
        baseline_changed=False, active_training_dataset_changed=False)
    write_json(out / "manifest.json", manifest)
    (out / "README.md").write_text(f"""# 50厘米筛选修正结果

**安全范围的25个Level1/2配置全部有非空编号记录。之前把筛选未纳入写成缺少采集，是表述与筛选错误。**

本包保留旧250厘米换算结果，固定九月份baseline字节未变。两种协议未平均，训练集、控制器及原始实验未改。

## 这次修正

恢复Front head Level2、Vee head Level1、Vee tail Level2三个长记录配置，并将Echelon head Level1现有短记录逐机、逐阶段检查。原来一架出现异常标签就删除整个实验，现在逐无人机提取实际窗口；合并Front只用observed来源块，3095个模拟间隙行完全不拟合。

Vee head Level1和Echelon head Level1部分control_fault为持续状态标签，之后仍有实际空中耗电观测。仅纳入识别到已知Mission Pad且pad相对高度与ToF同时为60–100厘米的该类记录，保留fault_phase_quality_flag及different_assigned_pad_sample_count。不同Pad切换分开拟合；发生故障或离开分配Pad的实测率需定位控制质量复核，不自动认定为稳定编队训练标签。这个离线筛选不等于证明编队稳定、间距精确或实机安全。没有把异常落地、起飞、移动或定位修正段当作hover。

每个真实块单独截距；不把恢复或模拟间隔计入拟合时间。开头SOC不变的段去掉，后面整数SOC平台保留；按各自电池阶段先拟合率，再标准化，最后算参考100/82/52/20时间。缺整段只补在曲线模型系数表，实测阶段表不添观测。

本包有{summary['safe_level1_2_conditions_with_stage_curves']}个Level1/2五机三阶段结果。新增{len(added)}条真实阶段率，原来293条及21个条件的数值逐项保持。

## 用户确认与短记录处理

- Echelon 50 head Level1：按用户确认允许使用各阶段的短记录，不要求同次飞行耗尽。实测拟合来自有实际SOC下降的可用窗口，缺整段只在模型系数中按同位置相对负载完成。004在本地Git首次提交也是约6秒的起飞记录，未找到完整历史副本；其中D3有47→44%下降，但阶段为takeoff且切到别机Pad，未冒充80厘米定位悬停Low观测。
- Vee 50 side：用户已确认Level3 003的实际风速为Level2，派生数据修正为Level2。CSV原文件未改，原Level3、实验ID、run_id及用户确认保留在wind_level_metadata_corrections.json。

## 文件

- wind_tunnel/all_wind_three_stage_curves.pdf、figures/：更新的每条件三段折线；PNG 3300×2040以及PDF/SVG。
- wind_tunnel/collection_inventory.csv、condition_coverage.csv：采集存在与分析归属分开列。
- wind_tunnel/observed_run_stage_rates.csv：真实阶段拟合。recovered_observed_stage_rates.csv只列新增记录。
- wind_tunnel/recovered_selected_real_samples.csv：新增拟合所用的每个真实样本与实际源时钟。
- wind_tunnel/curve_stage_rates.csv：曲线系数，缺整段来源在is_modeled和completion_anchor_stage。
- wind_tunnel/stage_observation_support.csv：实测SOC覆盖及初始平台删除秒数。
- forward_250cm/：上次已审核的281条50cm前进Medium率与24条件，逐文件复制未变。
- summary.json、validation.json、manifest.json：范围、复算、只读输入哈希及用户明确确认的标签修正。

旧包保留便于追溯；本包替代它的采集覆盖和筛选结论。25条件都有分析曲线，不代表每个阶段都有完整SOC实测，也未写入训练标签。特别是Echelon head Level1的缺阶段完成不应混入纯实测训练表。
""", encoding="utf-8")
    write_json(out / "wind_level_metadata_corrections.json", [LABEL_CORRECTION])
    write_json(out / "file_checksums.json", {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    archive = out.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError("Archive already exists")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, str(Path(out.name) / p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(str(archive))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    main(parser.parse_args().output.resolve())
