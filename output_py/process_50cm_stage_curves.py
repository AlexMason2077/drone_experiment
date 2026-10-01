"""Offline 50-cm stage-rate extraction and requested mixed-protocol averages.

Read-only raw inputs, existing trajectory QC and battery calibration. No aircraft,
application, network, training or dataset-activation imports. Measured rate tables
and completed three-stage curve coefficients are separate products.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/wind_tunnel_three_stage_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battery_normalization import BatteryNormalizer

MODEL = ROOT / "analysis_results/battery_normalization_v3_with_b15_20260909/model.json"
ADMIN = ROOT / "db_copy_for_cleaning/_cleaning_admin"
FORWARD = ROOT / "analysis_results/medium_forward_bideal_v3_with_b15_20260909"
DEFAULT_OUT = ROOT / "analysis_results/50cm_stage_curves_20260929"
NAMES = ("High", "Medium", "Low")
CELL = ["formation", "wind_direction", "wind_level", "spacing_cm"]
GRAIN = CELL + ["position", "stage"]
COLORS = ("#0072B2", "#D55E00", "#009E73", "#7A55A3", "#CC79A7")
FORMS = ("front", "column", "vee", "echalon", "diamond")
WIND_NAMES = {"head": "Headwind", "tail": "Tailwind", "side": "Sidewind"}
FORM_NAMES = {"front": "Front", "column": "Column", "vee": "Vee", "echalon": "Echelon", "diamond": "Diamond"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def jsonable(value):
    return json.loads(pd.Series(value, dtype=object).to_json()) if isinstance(value, dict) else value


def cell_id(meta):
    return f"{meta['formation']}_{int(meta['spacing_cm'])}_{meta['wind_direction']}_lv{int(meta['wind_level'])}"


def condition_from_raw(raw):
    vals = {}
    for key in ("formation", "wind_direction", "wind_speed", "inter_drone_distance_cm", "run_id"):
        unique = raw[key].dropna().astype(str).unique()
        if len(unique) != 1:
            raise ValueError(f"Inconsistent metadata {key}: {unique}")
        vals[key] = unique[0]
    level = re.fullmatch(r"Level\s*(\d+)", vals["wind_speed"], re.I)
    wind = vals["wind_direction"].strip().lower().removesuffix(" wind")
    formation = vals["formation"].lower().replace("echelon", "echalon")
    if not level or wind not in WIND_NAMES or formation not in FORMS:
        raise ValueError("Unsupported condition metadata")
    return dict(formation=formation, wind_direction=wind, wind_level=int(level[1]),
                spacing_cm=float(vals["inter_drone_distance_cm"]), run_id=vals["run_id"])


def exclusion_ids():
    text = (ROOT / "TRAINING_EXCLUSIONS.md").read_text()
    ids = set(re.findall(r"^- `([^`]+)`", text, flags=re.M))
    assert "column_50_tail_lv2" in ids and len(ids) == 5
    return ids


def fit_blocks(frame):
    """Independent time-weighted OLS with a separate intercept per real block.

    A restart, recovery or omitted non-hover phase is never a fitted time bridge.
    For one block this exactly matches the approved pilot's independent fit.
    """
    numerator = denominator = 0.0
    blocks = []
    for block_id, g in frame.groupby("block_id", sort=False):
        g = g.sort_values("fit_time_s")
        if len(g) < 3:
            continue
        t = g.fit_time_s.to_numpy(float)
        y = g.raw_soc.to_numpy(float)
        dt = np.diff(t)
        if not np.isfinite(np.r_[t, y]).all() or np.any(dt <= 0):
            raise ValueError("Nonfinite or nonincreasing fitted block")
        weights = np.r_[dt[0]/2, (t[2:]-t[:-2])/2, dt[-1]/2]
        mt, my = np.average(t, weights=weights), np.average(y, weights=weights)
        numerator += float(np.sum(weights * (t-mt) * (y-my)))
        denominator += float(np.sum(weights * (t-mt)**2))
        blocks.append((block_id, g, weights, mt, my))
    if not blocks or denominator <= 0 or frame.raw_soc.nunique() < 2:
        return None
    slope = numerator / denominator
    rate = -60 * slope
    if not np.isfinite(rate) or rate <= 0:
        return None
    residual = variance = total_weight = duration = 0.0
    block_fits = []
    for bid, g, w, mt, my in blocks:
        t, y = g.fit_time_s.to_numpy(float), g.raw_soc.to_numpy(float)
        predicted = my + slope * (t-mt)
        residual += float(np.sum(w*(y-predicted)**2))
        variance += float(np.sum(w*(y-my)**2))
        total_weight += float(w.sum())
        duration += float(t[-1]-t[0])
        block_fits.append(dict(block_id=str(bid), origin_s=float(t[0]), end_s=float(t[-1]),
                              intercept_at_origin=float(my+slope*(t[0]-mt)), samples=len(g)))
    levels = int(frame.raw_soc.nunique())
    return dict(raw_rate_pp_min=float(rate), duration_s=duration, sample_count=sum(len(b[1]) for b in blocks),
                distinct_soc_levels=levels, rmse_pp=float(np.sqrt(residual/total_weight)),
                r_squared=float(1-residual/variance) if variance else np.nan,
                weak_support=bool(duration < 20 or levels < 4), block_count=len(blocks),
                fitted_blocks=json.dumps(block_fits))


def calibrated_rate(fit, battery, drone, stage_i, normalizer):
    own = normalizer.curve_for(battery, drone)
    factor = fit["raw_rate_pp_min"] / own.rates_pp_min[stage_i]
    return dict(**fit, own_upper_soc=own.boundaries[stage_i], own_lower_soc=own.boundaries[stage_i+1],
                own_baseline_rate_pp_min=own.rates_pp_min[stage_i],
                reference_baseline_rate_pp_min=normalizer.reference.rates_pp_min[stage_i],
                relative_drain_factor=factor,
                normalized_rate_pp_min=factor * normalizer.reference.rates_pp_min[stage_i])


def observed_hover_frames(group, merged=False):
    """Select real hover phases; discard only the INITIAL constant-SOC period.

    Each canonical merge component has its own clock and initial plateau.
    Later integer-SOC plateaus remain in the fit. Each drone ends at its own 20%.
    """
    if merged:
        group = group[group.record_origin.eq("observed")].copy()
        components = group.groupby(["source_experiment_id", "source_run_id"], sort=False)
        clock_col = "source_elapsed_time"
    else:
        components = [("original", group)]
        clock_col = "elapsed_time"
    frames, audits = [], []
    for comp_i, (component, g) in enumerate(components):
        g = g.sort_values(clock_col).reset_index(drop=True)
        hover = g.phase.eq("wind_tunnel_hover")
        h = g[hover].copy()
        if h.empty:
            continue
        h["clock"] = pd.to_numeric(h[clock_col], errors="raise")
        if h.clock.duplicated().any() or h.battery.diff().gt(0).any():
            raise ValueError("Duplicate hover timestamp or SOC rebound")
        drops = h.index[h.battery.diff().lt(0)]
        if not len(drops):
            audits.append(dict(component=str(component), status="no_observed_SOC_decrease", hover_rows=len(h)))
            continue
        first = int(drops[0])
        floor = h.index[h.battery.le(20)]
        end = int(floor[0]) if len(floor) else int(h.index[-1])
        retained = h.loc[(h.index >= first) & (h.index <= end)].copy()
        # Different hover episodes retain individual intercepts; excluded recovery
        # and centering time cannot become hidden hovering discharge time.
        episode = (~hover).cumsum()
        retained["block_id"] = [f"{comp_i}_{int(episode.loc[i])}" for i in retained.index]
        retained["fit_time_s"] = retained.clock-float(h.loc[first, "clock"])
        retained["raw_soc"] = pd.to_numeric(retained.battery, errors="raise")
        retained = retained[retained.raw_soc.between(20, 100)]
        frames.append(retained[["fit_time_s", "raw_soc", "block_id"]])
        audits.append(dict(component=str(component), status="retained", original_hover_rows=len(h),
                           initial_plateau_removed_s=float(h.loc[first,"clock"]-h.clock.iloc[0]),
                           retained_first_soc=float(retained.raw_soc.iloc[0]), retained_last_soc=float(retained.raw_soc.iloc[-1]),
                           own_20_percent_observed=bool(len(floor)), retained_rows=len(retained),
                           own_hover_duration_s=float(retained.fit_time_s.iloc[-1])))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["fit_time_s","raw_soc","block_id"])), audits


def extract_wind(normalizer, registry, excluded, hashes):
    rows, run_audit, drone_audit = [], [], []
    files = sorted((ROOT / "database").glob("wind_tunnel_*_50_*/*_all_coordination.csv"))
    for path in files:
        rel = str(path.relative_to(ROOT)); hashes[rel] = sha(path)
        exp = path.parent.name
        audit = dict(protocol="wind_tunnel", experiment_id=exp, source_file=rel, file_bytes=path.stat().st_size)
        try:
            raw = pd.read_csv(path, low_memory=False)
            audit["raw_rows"] = len(raw)
            if raw.empty:
                audit["status"] = "excluded_empty_file"; run_audit.append(audit); continue
            meta = condition_from_raw(raw); audit.update(meta, condition_id=cell_id(meta))
            reason = None
            reg = registry.get(exp, {})
            merged = "_merged_" in path.name
            if meta["spacing_cm"] != 50:
                reason = "excluded_not_50cm_metadata"
            elif cell_id(meta) in excluded:
                reason = "excluded_safety_condition"
            elif not re.search(r"_\d{3}$", exp):
                reason = "excluded_prepare_or_non_numbered_attempt"
            elif merged and reg.get("merge_metadata", {}).get("status") != "canonical_merged_run":
                reason = "excluded_unverified_merged_file"
            elif reg.get("merge_metadata", {}).get("status") == "canonical_merged_run" and not merged:
                reason = "excluded_component_already_represented_by_canonical_merge"
            elif reg.get("is_outlier", False):
                reason = "excluded_registry_outlier_or_merged_component"
            elif raw.phase.str.contains("control_fault|uncommanded_landed", na=False).any():
                reason = "excluded_control_fault_or_uncommanded_landing"
            elif not ((raw.phase.eq("wind_tunnel_hover")) & raw.battery.le(20)).any():
                reason = "excluded_interrupted_attempt_no_hover_reserve_endpoint"
            if reason:
                audit["status"] = reason; run_audit.append(audit); continue
            if merged:
                if "record_origin" not in raw or "source_elapsed_time" not in raw:
                    raise ValueError("Merged provenance absent")
                audit["simulated_rows_excluded"] = int((~raw.record_origin.eq("observed")).sum())
                manifest_path = path.with_name(path.name.replace("_all_coordination.csv", "_merge_manifest.json"))
                hashes[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
                manifest = json.loads(manifest_path.read_text())
                for source in manifest["source_files"]:
                    if sha(ROOT/source["path"]) != source["sha256"]:
                        raise ValueError("Canonical merge source hash changed")
            included = 0
            for drone, g in raw.groupby("drone_name", sort=True):
                ident = g.battery_id.dropna().astype(str).unique()
                if len(ident) != 1:
                    raise ValueError("Multiple batteries for one drone")
                original_battery = ident[0]; battery = original_battery
                correction = ""
                if (drone == "drone_5" and battery == "B06" and
                    "Drone 5 physically used B12" in reg.get("notes", "") and
                    "historical metadata entered in error" in reg.get("notes", "")):
                    battery = "B12"; correction = "registry_explicit_physical_B12_metadata_correction"
                dmeta = dict(**meta, protocol="wind_tunnel", experiment_id=exp, source_file=rel,
                             condition_id=cell_id(meta), drone_name=drone, position=int(drone.split("_")[-1]),
                             battery_id=battery, original_battery_id=original_battery, battery_identity_correction=correction)
                try:
                    own = normalizer.curve_for(battery, drone)
                    frame, windows = observed_hover_frames(g, merged)
                    da = dict(**dmeta, window_details=json.dumps(windows), retained_samples=len(frame))
                    if frame.empty:
                        da["status"]="no_usable_hover_SOC_change"; drone_audit.append(da); continue
                    da["status"]="retained_independent_hover_windows"
                    drone_audit.append(da)
                    for i, (stage, up, lo) in enumerate(zip(NAMES, own.boundaries, own.boundaries[1:])):
                        part = frame[frame.raw_soc.between(lo,up)]
                        fit = fit_blocks(part)
                        if fit is not None:
                            rows.append(dict(**dmeta, stage=stage, rate_origin="measured_hover_blocks_independent_time_weighted_OLS",
                                             **calibrated_rate(fit,battery,drone,i,normalizer)))
                            included += 1
                except (ValueError, KeyError) as exc:
                    drone_audit.append(dict(**dmeta,status="excluded_drone_invalid_window_or_calibration", detail=str(exc)))
            audit.update(status="included_measured_hover_stages" if included else "excluded_no_usable_stage", included_stage_count=included)
            run_audit.append(audit)
        except (ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            audit.update(status="excluded_empty_or_invalid_schema", detail=str(exc)); run_audit.append(audit)
    return rows, run_audit, drone_audit


def extract_forward(normalizer, registry, excluded, hashes):
    """Reuse already-computed 250-cm normalized rates, without refitting them."""
    paths = [FORWARD/name for name in ["run_drone_rates.csv", "discharge_rates_wide.csv", "manifest.json", "selected_forward_intervals.csv"]]
    for path in paths: hashes[str(path.relative_to(ROOT))] = sha(path)
    previous_manifest=json.loads((FORWARD/"manifest.json").read_text())
    if previous_manifest["normalization_model_sha256"] != normalizer.sha256:
        raise ValueError("Previously computed forward rates use a different battery model")
    raw=pd.read_csv(FORWARD/"run_drone_rates.csv")
    raw=raw[raw.inter_drone_spacing_cm.eq(50)].copy()
    rows,audits=[],[]
    for _,previous in raw.iterrows():
        meta=dict(formation=str(previous.formation),wind_direction=str(previous.wind_direction),
                  wind_level=int(previous.wind_level),spacing_cm=50,protocol="forward_250cm",
                  experiment_id=str(previous.experiment_directory),run_id=str(previous.run_id),
                  drone_name=str(previous.drone_name),position=int(previous.position),
                  battery_id=str(previous.battery_id),original_battery_id=str(previous.battery_id),battery_identity_correction="",
                  source_file=str((FORWARD/"run_drone_rates.csv").relative_to(ROOT)))
        meta["condition_id"]=cell_id(meta)
        if meta["condition_id"] in excluded:
            audits.append(dict(**meta,status="excluded_safety_condition"));continue
        if previous.status != "included":
            audits.append(dict(**meta,status="excluded_previous_"+str(previous.status)));continue
        own=normalizer.curve_for(meta["battery_id"],meta["drone_name"])
        value=float(previous.discharge_rate_Bideal_pp_per_min)
        if not np.isfinite(value) or value<0:
            raise ValueError("Previously computed forward coefficient invalid")
        expected=float(previous.raw_medium_discharge_rate_pp_min)/own.rates_pp_min[1]*normalizer.reference.rates_pp_min[1]
        if not np.isclose(value,expected,rtol=1e-9,atol=1e-9):
            raise ValueError("Previously computed forward normalization does not match current reference")
        rows.append(dict(**meta,stage="Medium",raw_rate_pp_min=float(previous.raw_medium_discharge_rate_pp_min),
                    normalized_rate_pp_min=value,own_upper_soc=own.boundaries[1],own_lower_soc=own.boundaries[2],
                    own_baseline_rate_pp_min=own.rates_pp_min[1],reference_baseline_rate_pp_min=normalizer.reference.rates_pp_min[1],
                    relative_drain_factor=float(previous.raw_medium_discharge_rate_pp_min)/own.rates_pp_min[1],
                    duration_s=float(previous.medium_forward_duration_s),sample_count=int(previous.selected_interval_count)+1,
                    distinct_soc_levels=np.nan,rmse_pp=np.nan,r_squared=float(previous.curve_through_origin_r_squared),
                    weak_support=bool(previous.medium_forward_duration_s<20 or previous.raw_medium_drop_pp<3),
                    block_count=np.nan,fitted_blocks="",rate_origin="reused_existing_normalized_250cm_medium_coefficient_unchanged",
                    legacy_forward_qc_flags=previous.qc_flags,measured_forward_drop_pp=float(previous.raw_medium_drop_pp)))
        audits.append(dict(**meta,status="reused_existing_forward_coefficient_unchanged",raw_medium_drop_pp=float(previous.raw_medium_drop_pp),
                    previous_status=previous.status,legacy_qc_flags=previous.qc_flags))
    result=pd.DataFrame(rows)
    actual=result.groupby(CELL+["position"]).normalized_rate_pp_min.mean()
    wide=pd.read_csv(FORWARD/"discharge_rates_wide.csv")
    wide=wide[wide.inter_drone_spacing_cm.eq(50)]
    errors=[]
    for _,cell in wide.iterrows():
        key=(str(cell.formation),str(cell.wind_direction),int(cell.wind_level),50)
        for position in range(1,6):
            fullkey=key+(position,)
            if fullkey in actual.index:
                errors.append(abs(actual.loc[fullkey]-float(cell[f"position_{position}_discharge_rate_Bideal_pp_per_min"])))
    if not errors or max(errors)>1e-8:
        raise ValueError("Reused forward protocol means differ from the existing aggregate table")
    intervals=pd.read_csv(FORWARD/"selected_forward_intervals.csv")
    keys=result[["experiment_id","run_id","drone_name"]].rename(columns={"experiment_id":"experiment_directory"})
    intervals=intervals.merge(keys,on=["experiment_directory","run_id","drone_name"],how="inner",validate="many_to_one")
    return rows,audits,intervals.to_dict("records")

def extract(out):
    if out.exists():raise FileExistsError("Use a new output folder to preserve prior results")
    out.mkdir(parents=True)
    regdata=json.loads((ROOT/"database/experiment_registry.json").read_text())["experiments"]
    registry={r["experiment_id"]:r for r in (regdata.values() if isinstance(regdata,dict) else regdata)}
    excluded=exclusion_ids();normalizer=BatteryNormalizer.load(MODEL)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [MODEL,ROOT/"battery_normalization.py",ROOT/"TRAINING_EXCLUSIONS.md",ROOT/"database/experiment_registry.json",Path(__file__),
             ROOT/"output_py/recompute_medium_rates_v3.py"]}
    w,wa,wd=extract_wind(normalizer,registry,excluded,hashes)
    f,fa,fs=extract_forward(normalizer,registry,excluded,hashes)
    rates=pd.DataFrame(w+f)
    assert not set(rates.condition_id)&excluded
    assert rates.spacing_cm.eq(50).all()
    assert not rates.duplicated(["protocol","experiment_id","run_id","drone_name","stage"]).any()
    assert np.isfinite(rates.normalized_rate_pp_min).all() and rates.normalized_rate_pp_min.ge(0).all()
    rates.to_csv(out/"measured_run_stage_rates.csv",index=False)
    pd.DataFrame(wa).to_csv(out/"wind_run_audit.csv",index=False)
    pd.DataFrame(wd).to_csv(out/"wind_drone_window_audit.csv",index=False)
    pd.DataFrame(fa).to_csv(out/"forward_run_drone_audit.csv",index=False)
    pd.DataFrame(fs).to_csv(out/"forward_selected_intervals.csv",index=False)
    means=rates.groupby(["protocol"]+GRAIN,dropna=False).agg(normalized_rate_pp_min=("normalized_rate_pp_min","mean"),
              observed_run_count=("normalized_rate_pp_min","size"),rate_sd_pp_min=("normalized_rate_pp_min","std"),
              weak_support_run_count=("weak_support","sum")).reset_index()
    means["condition_id"]=means.apply(cell_id,axis=1)
    means.to_csv(out/"protocol_stage_means.csv",index=False)
    assert all(sha(ROOT/p)==digest for p,digest in hashes.items())
    write_json(out/"extraction_manifest.json",dict(status="extracted_pending_average_policy",model_version=normalizer.version,
        reference=normalizer.reference.to_dict(),training_exclusions=sorted(excluded),source_hashes=hashes,
        measured_stage_rows=len(rates),wind_input_files=len(wa),method="independent stages first, per-battery stage normalization, then averages and stage times",
        initial_plateau_policy="wind: remove through each drone's first observed SOC decrease; later plateaus retained. Forward: existing processed coefficients reused unchanged as explicitly requested",
        forward_method="reuse prior v3 normalized Medium coefficients unchanged, with existing forward-only trajectory segmentation and origin-constrained OLS; repeated-condition means verified against prior wide table",
        measured_vs_modeled="measured_run_stage_rates contains fitted real observations only, never missing-stage completions or simulated merge bridges",
        source_protocols="static wind-tunnel hovering and previously trajectory-filtered 250cm forward flight; any mix is a descriptive user-requested average, not direct forward-flight validation",
        diagnostic_figure_in_dataset=False,diamond_head_lv2_002_included=True,active_dataset_updated=False,trained=False))
    print(json.dumps(dict(status="extracted",out=str(out),rate_rows=len(rates),protocol_counts=rates.groupby("protocol").size().to_dict(),
                           wind_file_status=pd.DataFrame(wa).status.value_counts().to_dict(),forward_drone_status=pd.DataFrame(fa).status.value_counts().to_dict()),indent=2))


def combined_means(rates, policy):
    combined=[]
    for key,g in rates.groupby(GRAIN,sort=True):
        meta=dict(zip(GRAIN,key));means=g.groupby("protocol").normalized_rate_pp_min.mean()
        rate=float(means.mean()) if policy=="equal_protocol" else float(g.normalized_rate_pp_min.mean())
        counts=g.protocol.value_counts().to_dict()
        combined.append(dict(**meta,condition_id=cell_id(meta),normalized_rate_pp_min=rate,
                             wind_run_count=int(counts.get("wind_tunnel",0)),forward_run_count=int(counts.get("forward_250cm",0)),
                             wind_mean_pp_min=float(means.get("wind_tunnel",np.nan)),forward_mean_pp_min=float(means.get("forward_250cm",np.nan)),
                             source_protocol_count=len(means),rate_origin="average_observed_normalized_stage_rates",completion_anchor_stage="",average_policy=policy))
    return pd.DataFrame(combined)


def complete_curves(observed, normalizer):
    completed=[]
    for key,g in observed.groupby(CELL+["position"],sort=True):
        by={r.stage:r.to_dict() for _,r in g.iterrows()}
        for i,stage in enumerate(NAMES):
            if stage in by:row=by[stage].copy()
            else:
                anchor="Medium" if "Medium" in by else min(by,key=lambda s:abs(NAMES.index(s)-i))
                row=by[anchor].copy();j=NAMES.index(anchor)
                row.update(stage=stage,normalized_rate_pp_min=by[anchor]["normalized_rate_pp_min"]/normalizer.reference.rates_pp_min[j]*normalizer.reference.rates_pp_min[i],
                           wind_run_count=0,forward_run_count=0,wind_mean_pp_min=np.nan,forward_mean_pp_min=np.nan,source_protocol_count=0,
                           rate_origin="modeled_missing_stage_same_position_anchor_relative_factor",completion_anchor_stage=anchor)
            row.update(reference_upper_soc=normalizer.reference.boundaries[i],reference_lower_soc=normalizer.reference.boundaries[i+1],
                       stage_duration_s=60*(normalizer.reference.boundaries[i]-normalizer.reference.boundaries[i+1])/row["normalized_rate_pp_min"])
            completed.append(row)
    return pd.DataFrame(completed)


def plot_curve(ax,group,normalizer,small=False,baseline=True):
    for position,g in group.groupby("position",sort=True):
        by=g.set_index("stage");rates=np.array([by.loc[s,"normalized_rate_pp_min"] for s in NAMES])
        durations=60*np.diff(-np.asarray(normalizer.reference.boundaries))/rates
        t=np.r_[0.,np.cumsum(durations)]
        ax.plot(t,normalizer.reference.boundaries,lw=1.5 if small else 2.4,color=COLORS[int(position)-1],
                marker="o",markersize=2.2 if small else 4,label=f"P{int(position)} / D{int(position)}")
    if baseline:
        t=np.r_[0.,np.cumsum(60*np.diff(-np.asarray(normalizer.reference.boundaries))/normalizer.reference.rates_pp_min)]
        ax.plot(t,normalizer.reference.boundaries,color="#48555D",lw=1.2 if small else 1.6,ls=(0,(5,3)),label="Reference battery")
    for bound in normalizer.reference.boundaries[1:-1]:ax.axhline(bound,color="#9DADB5",lw=.7,ls=(0,(4,4)),zorder=0)
    ax.set_ylim(16,104);ax.set_xlim(left=0)
    ax.set_yticks([20,52,82,100]);ax.set_xlabel("Time (s)",fontsize=9 if small else 12)
    ax.set_ylabel("Normalized SOC (%)",fontsize=9 if small else 12)
    ax.tick_params(labelsize=8 if small else 10);ax.grid(color="#DCE3E7",lw=.5,alpha=.65)
    ax.spines[["top","right"]].set_visible(False);ax.set_axisbelow(True)


def finish(out,policy):
    manifest=json.loads((out/"extraction_manifest.json").read_text())
    if not all(sha(ROOT/p)==digest for p,digest in manifest["source_hashes"].items()):
        raise ValueError("Source or method changed after extraction; re-extract into a new folder")
    if (out/"manifest.json").exists():raise FileExistsError("A finished result already exists")
    normalizer=BatteryNormalizer.load(MODEL);rates=pd.read_csv(out/"measured_run_stage_rates.csv")
    observed=combined_means(rates,policy);completed=complete_curves(observed,normalizer)
    observed.to_csv(out/"combined_observed_stage_rates.csv",index=False)
    completed.to_csv(out/"completed_curve_stage_rates.csv",index=False)
    curves=[];coverage=[]
    for condition,group in completed.groupby("condition_id",sort=True):
        first=group.iloc[0]
        for pos,g in group.groupby("position",sort=True):
            by=g.set_index("stage");t=np.r_[0.,np.cumsum([by.loc[s,"stage_duration_s"] for s in NAMES])]
            for j,(ts,soc) in enumerate(zip(t,normalizer.reference.boundaries)):
                curves.append(dict(condition_id=condition,position=int(pos),knot=j,time_s=float(ts),normalized_soc=float(soc)))
        coverage.append(dict(**{c:first[c] for c in CELL},condition_id=condition,position_count=group.position.nunique(),
                             observed_stage_count=int(group.rate_origin.eq("average_observed_normalized_stage_rates").sum()),
                             modeled_stage_count=int(group.rate_origin.ne("average_observed_normalized_stage_rates").sum()),
                             mixed_protocol_stage_count=int(group.source_protocol_count.eq(2).sum())))
    pd.DataFrame(curves).to_csv(out/"three_stage_curve_knots.csv",index=False)
    pd.DataFrame(coverage).to_csv(out/"condition_coverage.csv",index=False)
    figdir=out/"figures";figdir.mkdir()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"pdf.fonttype":42,"svg.fonttype":"none"})
    with PdfPages(out/"all_condition_curves.pdf") as pdf:
        for condition,g in completed.groupby("condition_id",sort=True):
            first=g.iloc[0];title=f"{FORM_NAMES[first.formation]} · 50 cm · {WIND_NAMES[first.wind_direction]} · Level {int(first.wind_level)}"
            fig,ax=plt.subplots(figsize=(11,6.8));plot_curve(ax,g,normalizer)
            fig.suptitle(title,x=.11,y=.966,ha="left",fontsize=17,fontweight="semibold")
            fig.text(.11,.912,"Three battery stages · Rates averaged on the reference-battery scale",fontsize=10.5,color="#52636C")
            ax.legend(loc="upper center",bbox_to_anchor=(.5,-.16),ncol=3,frameon=False)
            fig.subplots_adjust(left=.11,right=.96,bottom=.23,top=.855)
            for ext in ("png","pdf","svg"):fig.savefig(figdir/f"{condition}.{ext}",dpi=300,facecolor="white")
            pdf.savefig(fig);plt.close(fig)
    excluded=set(manifest["training_exclusions"])
    fig,axes=plt.subplots(5,6,figsize=(22.5,17.5))
    for i,formation in enumerate(FORMS):
        for j,(wind,level) in enumerate([(w,l) for w in ("head","tail","side") for l in (1,2)]):
            ax=axes[i,j];cid=f"{formation}_50_{wind}_lv{level}";g=completed[completed.condition_id.eq(cid)]
            ax.set_title(f"{FORM_NAMES[formation]} · {WIND_NAMES[wind]} · L{level}",fontsize=10,loc="left")
            if cid in excluded:
                ax.set_facecolor("#F0F1F2");ax.text(.5,.5,"Excluded condition",transform=ax.transAxes,ha="center",color="#7F858A",fontsize=10)
                ax.set_xticks([]);ax.set_yticks([])
            elif g.empty:
                ax.text(.5,.5,"No supported stage rates",transform=ax.transAxes,ha="center",fontsize=10)
                ax.set_xticks([]);ax.set_yticks([])
            else:plot_curve(ax,g,normalizer,small=True)
    handles,labels=next(ax for ax in axes.flat if ax.lines).get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=6,frameon=False,bbox_to_anchor=(.5,.015))
    fig.suptitle("50 cm · Normalized three-stage discharge curves",x=.055,y=.984,ha="left",fontsize=22,fontweight="semibold")
    fig.text(.055,.956,"Matching configuration, wind, position and battery stage; average rates before calculating stage duration",fontsize=12,color="#52636C")
    fig.subplots_adjust(left=.055,right=.985,top=.923,bottom=.071,hspace=.60,wspace=.43)
    for ext in ("png","pdf"):fig.savefig(out/f"50cm_overview.{ext}",dpi=300,facecolor="white")
    plt.close(fig)
    validation=dict(status="passed_offline_analysis_checks",source_hashes_unchanged=True,only_50cm=True,
                    excluded_conditions_absent_from_rate_tables=not bool(set(completed.condition_id)&excluded),
                    measured_rate_grain_unique=not rates.duplicated(["protocol","experiment_id","run_id","drone_name","stage"]).any(),
                    all_curve_rates_positive=bool(completed.normalized_rate_pp_min.gt(0).all()),
                    every_curve_has_three_stages=bool(completed.groupby(["condition_id","position"]).size().eq(3).all()),
                    normalized_rate_identity_max_error=float(np.max(np.abs(rates.normalized_rate_pp_min-rates.raw_rate_pp_min/rates.own_baseline_rate_pp_min*rates.reference_baseline_rate_pp_min))),
                    duration_identity_max_error_s=float(np.max(np.abs(completed.stage_duration_s-60*(completed.reference_upper_soc-completed.reference_lower_soc)/completed.normalized_rate_pp_min))),
                    simulated_bridge_rows_used=0,diagnostic_figure_in_dataset=False,active_training_or_dataset_modified=False)
    assert all(validation[k] for k in ["excluded_conditions_absent_from_rate_tables","measured_rate_grain_unique","all_curve_rates_positive","every_curve_has_three_stages"])
    write_json(out/"validation.json",validation)
    manifest.update(status="candidate_analysis_completed_not_activated",average_policy=policy,
        safe_L1_L2_conditions=int(sum(c["wind_level"] in (1,2) for c in coverage)),additional_recorded_L3_conditions=int(sum(c["wind_level"]==3 for c in coverage)),
        plotted_conditions=len(coverage),completed_curve_stage_rows=len(completed),modeled_stage_rows=int(completed.source_protocol_count.eq(0).sum()),
        both_protocols_observed_stage_rows=int(completed.source_protocol_count.eq(2).sum()),
        missing_stage_completion="Same position/configuration relative drain factor from Medium if available, otherwise nearest observed stage; applies only after measured rates have been averaged. It is not an observation or independent validation.",
        forward_comparability="Forward source mostly supports Medium SOC. Stages with only one measured protocol retain that protocol, without pretending a two-protocol mean exists.")
    write_json(out/"manifest.json",manifest)
    (out/"README.md").write_text(f"""# 50 cm 三阶段耗电曲线\n\n此目录为独立派生分析，没有更新训练数据或在线模型。实测SOC对照图没有加入；Diamond顶风Level2 `_002` 实测记录保留。\n\n## 方法\n\n1. 原始CSV只读，250cm直接复用已经计算好的标准化系数和均值，不重拟合。风洞只用hover，剔除起飞、定位、故障、异常降落、prepare、中断试飞以及当前五项安全排除条件。每架飞机保留自己的记录；不会按最早降落的飞机截断全部记录。\n2. 风洞每架飞机从首次实测SOC下降之后开始；开头不变电量的时间去除，后续整数SOC平台仍保留。风洞恢复/重启分成独立真实时间块，各块独立截距，不把未观测间隔拟合为飞行时间。合并Front顶风记录只用observed行，模拟补点全部剔除。\n3. 各物理电池自己的High/Medium/Low区间分别拟合耗电率。风洞使用独立时间加权OLS；前进直接保留已有Medium过原点拟合和阶段标准化系数。前进原有时间窗口与拟合方法没有更改；与原有各条件位置均值逐项核对一致。\n4. 标准化阶段耗电率 = 实测阶段耗电率 / 该物理电池基线阶段耗电率 × 已有参考电池阶段耗电率。参考边界100/82/52/20，单位pp/min。\n5. 当前平均方式 `{policy}`：{'先在同条件/位置/阶段内分别平均两类的重复实验，再对两类均值各取50%。只有一类有实测时使用那一类，表中保留来源数量。' if policy=='equal_protocol' else '同条件/位置/阶段的每个有效实验阶段拟合等权平均。表中保留两类数量与各自均值。'}\n6. 先得到平均后的耗电率，再计算阶段时间=60×阶段SOC下降/阶段耗电率。不平均原始时间轴、不平均起始SOC、不平均耗尽时间。\n7. 未观察到的阶段仅在曲线系数表中按同配置同位置的Medium相对耗电因子（无Medium时取最近可用阶段）完成。实测阶段表不填假观测。图不显示拓展标签，系数表及manifest完整保留来源。\n\n## 文件\n\n- `50cm_overview.png` / `.pdf`：Level1/2总览；灰色单元为排除条件。\n- `all_condition_curves.pdf`：逐条件五位置曲线；`figures/`另有300dpi PNG/PDF/SVG。Level3原始记录单独按Level3绘制，未并入Level2。\n- `measured_run_stage_rates.csv`：真实观测拟合的每实验/位置/阶段系数；弱支撑短阶段有标记。\n- `protocol_stage_means.csv`：风洞、前进250cm各自重复实验均值。\n- `combined_observed_stage_rates.csv`：所选平均策略得到的可观测阶段均值。\n- `completed_curve_stage_rates.csv`：最终曲线系数，包含缺失阶段模型补全来源。\n- `three_stage_curve_knots.csv`：100、82、52、20四个折线节点。\n- `condition_coverage.csv`：每条件实测、模型补全、双来源平均的覆盖情况。\n- 三个audit文件记录全部候选、排除原因及独立耗电窗口；`forward_selected_intervals.csv`为已有250cm选中区间的只读派生副本，保留此前时间和耗电累计依据。\n- `manifest.json`、`validation.json`：只读输入校验及离线计算检查。\n\n## 解释限制\n\n这次平均合并了风洞悬停与实际前进两种实验协议，是用户要求的描述性组合系数；它不等同于已经通过实机验证的前进耗电模型。历史250cm记录主要在中电量区间，不能声称全部高/低电量阶段也有两类实测平均。缺失阶段的补全假设、短阶段支撑和不同飞机落地后的编队成员变化都应在训练或论文结论使用前考虑。已有参考电池模型保留其candidate状态及95→100基线延伸来源，不声称本次对其重新独立验证。\n""")
    notebook={"cells":[{"cell_type":"markdown","metadata":{},"source":["# 50 cm 三阶段分析复核\n", "只读取派生表；实测拟合与模型补全分开。原始数据及训练模型未修改。"]},
        {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["from pathlib import Path\nimport pandas as pd\nimport numpy as np\n",f"folder = Path({str(out)!r})\n", "rates = pd.read_csv(folder/'measured_run_stage_rates.csv')\ncurves = pd.read_csv(folder/'completed_curve_stage_rates.csv')\n", "assert rates.spacing_cm.eq(50).all()\nassert curves.normalized_rate_pp_min.gt(0).all()\n", "assert np.allclose(rates.normalized_rate_pp_min, rates.raw_rate_pp_min/rates.own_baseline_rate_pp_min*rates.reference_baseline_rate_pp_min)\n", "display(pd.read_csv(folder/'condition_coverage.csv'))\n"]},
        {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["from IPython.display import Image, display\ndisplay(Image(filename=str(folder/'50cm_overview.png')))\n"]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
    write_json(out/"review_50cm_stage_curves.ipynb",notebook)
    print(json.dumps({k:manifest[k] for k in ["status","average_policy","plotted_conditions","safe_L1_L2_conditions","additional_recorded_L3_conditions","completed_curve_stage_rows","modeled_stage_rows","both_protocols_observed_stage_rows"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--extract",action="store_true");parser.add_argument("--finish",choices=["equal_protocol","equal_run"])
    args=parser.parse_args()
    if args.extract:extract(args.out.resolve())
    if args.finish:finish(args.out.resolve(),args.finish)
    if not args.extract and not args.finish:parser.error("Choose --extract and/or --finish")
