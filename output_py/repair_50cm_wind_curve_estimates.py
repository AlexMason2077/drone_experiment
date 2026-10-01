"""Offline, provenance-preserving model refit of four questioned wind curves.

Refits the four conditions and their other-speed comparators from real nominal
hover windows. Partial/unstable estimates are regularized against the comparator
under the user's authorized speed trend. Raw data, September baseline, flight
control, previous exports and active training datasets are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from process_50cm_stage_curves import (
    ROOT, MODEL, NAMES, CELL, COLORS, FORMS, FORM_NAMES, WIND_NAMES,
    BatteryNormalizer, plt, sha, write_json, condition_from_raw, cell_id,
    exclusion_ids, plot_curve,
)

PRIOR = ROOT / "analysis_results/50cm_selection_corrected_fixed_baseline_20260929"
DEFAULT_OUT = ROOT / "analysis_results/50cm_wind_model_refit_20260929"
TARGETS = {"vee_50_tail_lv2", "front_50_tail_lv2", "front_50_side_lv1", "echalon_50_head_lv1"}
FAMILIES = {"vee_50_tail", "front_50_tail", "front_50_side", "echalon_50_head"}
REFIT_CELLS = {family + "_lv" + str(level) for family in FAMILIES for level in (1, 2)}
FIXED_SHA = "5ea37d52e43b4c283f3ce6f956b2b669c1878030e32afc92fc9749fe66af0fe5"
# The five-percent speed assumption is user-authorized modeling, not a measured
# global law. Strong paired Medium data can supply a family-specific multiplier.
FALLBACK_L1_RATE_FRACTION = .95
REPEAT_SPREAD_SCALE = np.log(1.25)
MIN_TOTAL_OBSERVED_DROP_PP = 3.


def union_width(intervals):
    ordered = sorted((float(lo), float(up)) for lo, up in intervals if up > lo)
    total = 0.
    if not ordered:
        return total
    start, end = ordered[0]
    for lo, up in ordered[1:]:
        if lo <= end:
            end = max(end, up)
        else:
            total += end-start
            start, end = lo, up
    return total + end-start


def selected_nominal_samples(g, merged=False):
    """Actual assigned-pad hover only; phase labels alone are insufficient."""
    if merged:
        g = g[g.record_origin.eq("observed")].copy()
        components = g.groupby(["source_experiment_id", "source_run_id"], sort=False)
        clock = "source_elapsed_time"
    else:
        components = [("original", g)]
        clock = "elapsed_time"
    frames, audits = [], []
    for ci, (component, d) in enumerate(components):
        d = d.sort_values(clock).reset_index(drop=True)
        ground = d.phase.isin(["wind_tunnel_landing_20_percent", "wind_tunnel_landed", "wind_tunnel_uncommanded_landed"])
        first_ground = int(ground[ground].index.min()) if ground.any() else len(d)
        valid = d.phase.eq("wind_tunnel_hover") & d.mid.eq(d.target_pad) & d.mid.isin(range(1, 9))
        valid &= d.z.between(60, 100) & d.tof.between(60, 100) & (d.index < first_ground)
        h = d[valid].copy()
        if h.empty:
            audits.append(dict(component=str(component), status="no_nominal_assigned_pad_hover"))
            continue
        if h[clock].duplicated().any() or h.battery.diff().gt(0).any():
            audits.append(dict(component=str(component), status="rebound_or_duplicate_clock_not_fitted"))
            continue
        drops = h.index[h.battery.diff().lt(0)]
        if not len(drops):
            audits.append(dict(component=str(component), status="no_observed_discharge_after_selection"))
            continue
        start = int(drops[0])
        floor = h.index[h.battery.le(20)]
        stop = int(floor[0]) if len(floor) else int(h.index[-1])
        h = h.loc[(h.index >= start) & (h.index <= stop) & h.battery.between(20, 100)].copy()
        episodes = (~valid).cumsum() + d.mid.ne(d.mid.shift()).cumsum()
        h["block_id"] = [f"{ci}_{int(episodes.loc[i])}" for i in h.index]
        h["fit_time_s"] = h[clock] - float(d.loc[start, clock])
        h["source_clock_s"] = h[clock]
        h["raw_soc"] = h.battery
        h["component"] = str(component)
        frames.append(h[["block_id", "fit_time_s", "source_clock_s", "raw_soc", "phase", "mid", "target_pad", "z", "tof", "component"]])
        audits.append(dict(component=str(component), status="selected_nominal_windows", retained_samples=len(h),
            initial_unchanged_SOC_s_removed=float(d.loc[start, clock]-d.loc[valid, clock].iloc[0]),
            first_soc=float(h.raw_soc.iloc[0]), last_soc=float(h.raw_soc.iloc[-1]),
            fault_rows_excluded=int(d.phase.eq("wind_tunnel_control_fault").sum()),
            unavailable_or_other_pad_hover_rows_excluded=int((d.phase.eq("wind_tunnel_hover") & ~d.mid.eq(d.target_pad)).sum()),
            own_20_soc_observed=bool(len(floor))))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), audits


def stage_anchors(frame, lower, upper):
    """First SOC reports within each actual block, ending at first lower hit.

    Later integer plateaus are preserved through time between crossings. The
    lower-boundary plateau is never duplicated in the preceding stage.
    """
    blocks = []
    for bid, b in frame.groupby("block_id", sort=False):
        b = b[b.raw_soc.between(lower, upper)].sort_values("fit_time_s")
        if b.empty:
            continue
        boundary = b[b.raw_soc.le(lower)]
        if len(boundary):
            b = b[b.fit_time_s.le(float(boundary.fit_time_s.iloc[0]))]
        event = b[b.raw_soc.ne(b.raw_soc.shift())].copy()
        if len(event) >= 3 and event.raw_soc.max()-event.raw_soc.min() >= 2:
            blocks.append(event)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def fit_crossing_blocks(points, stage_width):
    """Equal-crossing OLS SOC(time), independent intercept per real block."""
    if points.empty:
        return None
    num = den = ss = residual = 0.
    intervals, effective_drop, duration, blocks = [], 0., 0., []
    for bid, b in points.groupby("block_id", sort=False):
        t, y = b.fit_time_s.to_numpy(float), b.raw_soc.to_numpy(float)
        assert np.all(np.diff(t) > 0) and np.all(np.diff(y) < 0)
        tc, yc = t-t.mean(), y-y.mean()
        num += float(tc @ yc)
        den += float(tc @ tc)
        intervals.append((float(y.min()), float(y.max())))
        effective_drop += float(y.max()-y.min())
        duration += float(t[-1]-t[0])
        blocks.append((bid, t, y))
    if effective_drop < MIN_TOTAL_OBSERVED_DROP_PP or den <= 0:
        return None
    slope = num/den
    if slope >= 0 or not np.isfinite(slope):
        return None
    block_details = []
    for bid, t, y in blocks:
        predicted = y.mean()+slope*(t-t.mean())
        residual += float(np.sum((y-predicted)**2))
        ss += float(np.sum((y-y.mean())**2))
        block_details.append(dict(block_id=str(bid), first_s=float(t[0]), last_s=float(t[-1]),
                                  first_soc=float(y[0]), last_soc=float(y[-1]), crossing_points=len(t)))
    covered = union_width(intervals)
    return dict(raw_rate_pp_min=float(-60*slope), covered_soc_pp=covered,
        coverage_fraction=min(1., covered/stage_width), effective_observed_drop_pp=effective_drop,
        duration_s=duration, crossing_point_count=len(points), block_count=len(blocks),
        r_squared=1-residual/ss if ss else None,
        fitted_blocks=json.dumps(block_details), observed_intervals=json.dumps(intervals))


def metadata(cid):
    form, _, wind, lev = cid.split("_")
    return dict(formation=form, spacing_cm=50, wind_direction=wind, wind_level=int(lev[2:]))


def extract_actual(normalizer, registry, source_hashes):
    run_rows, anchors, audits = [], [], []
    files = sorted((ROOT / "database").glob("wind_tunnel_*_50_*/*_all_coordination.csv"))
    for path in files:
        exp = path.parent.name
        if not re.search(r"_\d{3}$", exp):
            continue
        family = exp.removeprefix("wind_tunnel_").rsplit("_", 2)[0]
        if family not in FAMILIES:
            continue
        raw = pd.read_csv(path, low_memory=False)
        if raw.empty:
            continue
        meta = condition_from_raw(raw)
        cid, rel = cell_id(meta), str(path.relative_to(ROOT))
        if cid not in REFIT_CELLS:
            continue
        source_hashes[rel] = sha(path)
        reg = registry.get(exp, {})
        if reg.get("is_outlier", False):
            audits.append(dict(source_file=rel, condition_id=cid, status="registry_exclusion_preserved"))
            continue
        merged = "_merged_" in path.name
        if reg.get("merge_metadata", {}).get("status") == "canonical_merged_run" and not merged:
            continue
        if merged and reg.get("merge_metadata", {}).get("status") != "canonical_merged_run":
            continue
        for dn, g in raw.groupby("drone_name", sort=True):
            identities = g.battery_id.dropna().astype(str).unique()
            if len(identities) != 1:
                continue
            original_battery, battery = identities[0], identities[0]
            if dn == "drone_5" and battery == "B06" and "Drone 5 physically used B12" in reg.get("notes", "") and "historical metadata entered in error" in reg.get("notes", ""):
                battery = "B12"
            try:
                own = normalizer.curve_for(battery, dn)
            except (KeyError, ValueError):
                audits.append(dict(source_file=rel, condition_id=cid, drone_name=dn, status="no_corresponding_fixed_baseline"))
                continue
            frame, windows = selected_nominal_samples(g, merged)
            audits.append(dict(source_file=rel, condition_id=cid, drone_name=dn, battery_id=battery,
                status="selected_nominal_hover" if len(frame) else "no_estimable_nominal_hover",
                window_details=json.dumps(windows), selected_sample_count=len(frame)))
            if frame.empty:
                continue
            for i, stage in enumerate(NAMES):
                upper, lower = own.boundaries[i:i+2]
                points = stage_anchors(frame, lower, upper)
                fit = fit_crossing_blocks(points, upper-lower)
                if fit is None:
                    continue
                dmeta = dict(**meta, condition_id=cid, experiment_id=exp, source_file=rel,
                    drone_name=dn, position=int(dn.split("_")[-1]), battery_id=battery,
                    original_battery_id=original_battery, stage=stage,
                    own_upper_soc=upper, own_lower_soc=lower,
                    own_baseline_rate_pp_min=own.rates_pp_min[i],
                    reference_baseline_rate_pp_min=normalizer.reference.rates_pp_min[i])
                fit["normalized_rate_pp_min"] = fit["raw_rate_pp_min"]/own.rates_pp_min[i]*normalizer.reference.rates_pp_min[i]
                run_rows.append(dict(**dmeta, **fit, rate_origin="real_nominal_hover_SOC_first_crossing_OLS"))
                points = points.copy()
                points["block_id"] = rel + "|" + dn + "|" + points.block_id.astype(str)
                for k in ["condition_id", "source_file", "drone_name", "position", "battery_id", "stage", "run_id"]:
                    points[k] = dmeta[k]
                anchors.append(points)
    return pd.DataFrame(run_rows), pd.concat(anchors, ignore_index=True), pd.DataFrame(audits)


def aggregate_actual(runs, points, normalizer):
    rows = []
    for (cid, pos, stage), g in runs.groupby(["condition_id", "position", "stage"], sort=True):
        a = points[points.condition_id.eq(cid) & points.position.eq(pos) & points.stage.eq(stage)]
        ident = g.battery_id.unique()
        assert len(ident) == 1
        own = normalizer.curve_for(str(ident[0]), f"drone_{pos}")
        i = NAMES.index(stage)
        fit = fit_crossing_blocks(a, own.boundaries[i]-own.boundaries[i+1])
        assert fit is not None
        rate = fit["raw_rate_pp_min"]/own.rates_pp_min[i]*normalizer.reference.rates_pp_min[i]
        log_rates = np.log(g.normalized_rate_pp_min.to_numpy(float))
        weights = g.covered_soc_pp.to_numpy(float)
        center = np.average(log_rates, weights=weights)
        dispersion = float(np.sqrt(np.average((log_rates-center)**2, weights=weights))) if len(g) > 1 else 0.
        reliability = fit["coverage_fraction"]**2 / (1+(dispersion/REPEAT_SPREAD_SCALE)**2)
        rows.append(dict(**metadata(cid), condition_id=cid, position=int(pos), stage=stage,
            battery_id=str(ident[0]), **fit, normalized_rate_pp_min=rate,
            run_count=len(g), log_run_rate_spread=dispersion,
            observed_model_weight=float(reliability),
            source_files=json.dumps(sorted(g.source_file.unique().tolist())),
            estimate_kind="observed_stage_fit_not_full_curve_validation"))
    return pd.DataFrame(rows)


def speed_factors(actual):
    factors = []
    for family in sorted(FAMILIES):
        med = actual[actual.condition_id.isin([family+"_lv1", family+"_lv2"]) & actual.stage.eq("Medium")]
        reliable = med[med.coverage_fraction.ge(.75) & med.log_run_rate_spread.le(np.log(1.25))]
        a = reliable[reliable.wind_level.eq(1)].set_index("position")
        b = reliable[reliable.wind_level.eq(2)].set_index("position")
        paired = sorted(set(a.index)&set(b.index))
        ratios = [float(b.loc[p, "normalized_rate_pp_min"]/a.loc[p, "normalized_rate_pp_min"]) for p in paired]
        empirical = float(np.exp(np.median(np.log(ratios)))) if ratios else None
        # When there are too few comparable positions, or the observed median
        # does not support the user's requested trend, use the explicit scenario.
        supported = len(ratios) >= 3 and empirical >= 1.
        applied = empirical if supported else 1/FALLBACK_L1_RATE_FRACTION
        factors.append(dict(family=family, reliable_medium_pair_count=len(ratios), paired_positions=paired,
            paired_l2_over_l1_rates=ratios, empirical_medium_ratio=empirical,
            applied_l2_over_l1_rate_multiplier=applied,
            source="same_family_reliable_Medium_pairs" if supported else "user_authorized_5_percent_L1_rate_reduction_scenario",
            assumption="Same multiplier transferred to unavailable stages; this is model completion, not new measurement."))
    return factors


def base_stage_estimates(actual, normalizer):
    direct = {(r["condition_id"], int(r["position"]), r["stage"]):r for r in actual.to_dict("records")}
    rows = []
    for cid in sorted(REFIT_CELLS):
        for pos in range(1, 6):
            for i, stage in enumerate(NAMES):
                key = (cid, pos, stage)
                if key in direct:
                    r = direct[key]
                    rows.append(dict(condition_id=cid, position=pos, stage=stage,
                        normalized_rate_pp_min=r["normalized_rate_pp_min"],
                        origin="observed_nominal_stage_fit", source_files=r["source_files"],
                        modeled=False, anchor_stages="", coverage_fraction=r["coverage_fraction"]))
                    continue
                # Use mature-stage reference shape only for a genuinely missing
                # donor stage. Never propagate a brief High factor into Low.
                candidates = [direct[(cid, pos, s)] for s in ("Medium", "Low") if (cid,pos,s) in direct]
                if not candidates:
                    # Other positions in the same condition and same stage form
                    # the last data-backed prior for missing early SOC coverage.
                    candidates = [r for (cc, _, ss), r in direct.items() if cc == cid and ss == stage]
                if not candidates:
                    rows.append(dict(condition_id=cid, position=pos, stage=stage,
                        normalized_rate_pp_min=np.nan, origin="no_supported_donor_prior", source_files="[]",
                        modeled=True, anchor_stages="", coverage_fraction=0.))
                    continue
                relative = [r["normalized_rate_pp_min"]/normalizer.reference.rates_pp_min[NAMES.index(r["stage"])] for r in candidates]
                w = [max(.01, r["coverage_fraction"]) for r in candidates]
                value = float(np.exp(np.average(np.log(relative), weights=w))*normalizer.reference.rates_pp_min[i])
                source_files = sorted({f for r in candidates for f in json.loads(r["source_files"])})
                rows.append(dict(condition_id=cid, position=pos, stage=stage,
                    normalized_rate_pp_min=value, origin="reference_shape_from_supported_mature_stage_or_same_stage_peers",
                    source_files=json.dumps(source_files), modeled=True,
                    anchor_stages=",".join(sorted({r["stage"] for r in candidates})), coverage_fraction=0.))
    return pd.DataFrame(rows)


def fitted_curve_estimates(actual, base, factors, normalizer):
    observed = {(r["condition_id"], int(r["position"]), r["stage"]):r for r in actual.to_dict("records")}
    donor = {(r["condition_id"], int(r["position"]), r["stage"]):r for r in base.to_dict("records")}
    multiplier = {r["family"]:r for r in factors}
    result = []
    for cid in sorted(REFIT_CELLS):
        meta = metadata(cid)
        family = cid.rsplit("_", 1)[0]
        other_level = 3-meta["wind_level"]
        counterpart = family+f"_lv{other_level}"
        ratio = multiplier[family]["applied_l2_over_l1_rate_multiplier"]
        for pos in range(1, 6):
            for i, stage in enumerate(NAMES):
                key = (cid, pos, stage)
                obs, own_base = observed.get(key), donor[key]
                opposite = donor[(counterpart, pos, stage)]
                if cid not in TARGETS:
                    value = own_base["normalized_rate_pp_min"]
                    weight = 1. if obs else 0.
                    prior = np.nan
                    kind = "refitted_observed_stage" if obs else "modeled_reference_shape_for_missing_donor_stage"
                    other_sources = []
                    factor = 1.
                    consistency_cap = False
                else:
                    factor = ratio if meta["wind_level"] == 2 else 1/ratio
                    prior = opposite["normalized_rate_pp_min"] * factor
                    if not np.isfinite(prior):
                        prior = own_base["normalized_rate_pp_min"]
                        other_sources = []
                    else:
                        other_sources = json.loads(opposite["source_files"])
                    weight = obs["observed_model_weight"] if obs else 0.
                    consistency_cap = False
                    if obs:
                        # User-authorized suspect stages: retain the original
                        # rate, but bound its influence on the modeled curve.
                        suspect = ((cid == "vee_50_tail_lv2" and pos == 1 and stage == "Low") or
                            (cid == "front_50_side_lv1" and pos == 3 and stage == "High") or
                            (cid == "front_50_tail_lv2" and stage == "Medium") or
                            cid == "echalon_50_head_lv1")
                        relative = obs["normalized_rate_pp_min"]/prior
                        if suspect and (relative < .5 or relative > 2.):
                            weight = min(weight, .1)
                            consistency_cap = True
                        value = float(np.exp(weight*np.log(obs["normalized_rate_pp_min"])+(1-weight)*np.log(prior)))
                    else:
                        value = float(prior)
                    kind = "observed_stage_with_cross_speed_regularization" if obs and weight < 1-1e-9 else (
                        "refitted_observed_stage" if obs else "modeled_other_speed_stage")
                assert np.isfinite(value) and value > 0, (cid, pos, stage)
                up, lo = normalizer.reference.boundaries[i:i+2]
                result.append(dict(**meta, condition_id=cid, position=pos, stage=stage,
                    normalized_rate_pp_min=value, reference_upper_soc=up, reference_lower_soc=lo,
                    stage_duration_s=60*(up-lo)/value, rate_origin=kind, estimate_kind=kind,
                    is_modeled=bool(not obs or weight < 1-1e-9), observed_model_weight=float(weight),
                    original_refitted_observed_rate_pp_min=obs["normalized_rate_pp_min"] if obs else np.nan,
                    stage_coverage_fraction=obs["coverage_fraction"] if obs else 0.,
                    observed_run_count=obs["run_count"] if obs else 0,
                    observed_source_files=obs["source_files"] if obs else "[]",
                    prior_rate_pp_min=float(prior), prior_condition_id=counterpart if cid in TARGETS else "",
                    prior_source_files=json.dumps(other_sources),
                    prior_donor_stage_is_modeled=bool(opposite["modeled"]) if cid in TARGETS else False,
                    donor_anchor_stages=opposite["anchor_stages"] if cid in TARGETS else own_base["anchor_stages"],
                    cross_speed_rate_multiplier=factor,
                    speed_factor_source=multiplier[family]["source"] if cid in TARGETS else "not_applied",
                    consistency_influence_capped=consistency_cap,
                    completion_anchor_stage="", training_eligibility="model_estimate_not_activated_as_observed_training_label"))
    return pd.DataFrame(result)


def figure(group, normalizer, small=False):
    fig, ax = plt.subplots(figsize=(11, 6.8))
    plot_curve(ax, group, normalizer)
    row = group.iloc[0]
    for stage, up, lo in zip(NAMES, normalizer.reference.boundaries, normalizer.reference.boundaries[1:]):
        ax.text(.985, (up+lo)/2, stage, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=11, color="#64757F", bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
    fig.suptitle(f"{FORM_NAMES[row.formation]} · 50 cm · {WIND_NAMES[row.wind_direction]} · Level {int(row.wind_level)}",
        x=.11, y=.966, ha="left", fontsize=17, fontweight="semibold")
    fig.text(.11, .912, "Wind-tunnel model · Stage-rate estimates on the fixed September reference-battery scale", fontsize=10.5, color="#52636C")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.11, right=.96, bottom=.23, top=.855)
    return fig


def main(out, resume=False):
    if out.exists() and not resume:
        raise FileExistsError("Keep prior exports; choose a new output directory")
    if out.exists() and resume:
        previous_manifest = json.loads((out/"manifest.json").read_text())
        assert previous_manifest["status"] == "repaired_model_candidate_not_activated"
        for p, digest in previous_manifest["source_hashes"].items():
            if ROOT/p != Path(__file__):
                assert sha(ROOT/p) == digest, f"Input changed while completing export: {p}"
    normalizer = BatteryNormalizer.load(MODEL)
    assert normalizer.sha256 == FIXED_SHA
    excluded = exclusion_ids()
    assert not excluded & REFIT_CELLS
    regpath = ROOT / "database/experiment_registry.json"
    reg = json.loads(regpath.read_text())["experiments"]
    registry = {r["experiment_id"]:r for r in (reg.values() if isinstance(reg,dict) else reg)}
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in [MODEL, regpath, ROOT/"TRAINING_EXCLUSIONS.md", Path(__file__),
        PRIOR/"wind_tunnel/curve_stage_rates.csv", PRIOR/"wind_tunnel/observed_run_stage_rates.csv"]}
    runs, points, window_audit = extract_actual(normalizer, registry, source_hashes)
    actual = aggregate_actual(runs, points, normalizer)
    factors = speed_factors(actual)
    base = base_stage_estimates(actual, normalizer)
    fixed = fitted_curve_estimates(actual, base, factors, normalizer)
    previous = pd.read_csv(PRIOR/"wind_tunnel/curve_stage_rates.csv")
    unchanged = previous[~previous.condition_id.isin(REFIT_CELLS)].copy()
    unchanged["estimate_kind"] = "previous_candidate_unchanged_not_revalidated_in_this_refit"
    unchanged["training_eligibility"] = "previous_candidate_not_activated"
    all_curves = pd.concat([unchanged, fixed], ignore_index=True).sort_values(["condition_id", "position", "stage"])
    assert len(all_curves) == 375 and all_curves.condition_id.nunique() == 25
    assert not set(all_curves.condition_id)&excluded
    assert all_curves.groupby(["condition_id", "position"]).size().eq(3).all()
    assert np.isfinite(all_curves.normalized_rate_pp_min).all() and all_curves.normalized_rate_pp_min.gt(0).all()

    out.mkdir(parents=True, exist_ok=resume)
    wind = out/"wind_tunnel"
    (wind/"figures").mkdir(parents=True, exist_ok=resume)
    # Preserve the separate forward result byte-for-byte, without averaging.
    shutil.copytree(PRIOR/"forward_250cm", out/"forward_250cm", dirs_exist_ok=resume)
    shutil.copy2(MODEL, out/"fixed_september_baseline.json")
    for name, df in [("refitted_real_run_stage_rates.csv", runs), ("actual_SOC_crossing_samples.csv", points),
        ("actual_window_audit.csv", window_audit), ("refitted_observed_stage_rates.csv", actual),
        ("donor_base_stage_estimates.csv", base), ("repaired_eight_condition_stage_estimates.csv", fixed),
        ("curve_stage_rates.csv", all_curves)]:
        df.to_csv(wind/name, index=False)
    write_json(wind/"cross_speed_factors.json", factors)
    knots = []
    for (cid, pos), g in all_curves.groupby(["condition_id", "position"], sort=True):
        time = 0.
        knots.append(dict(condition_id=cid, position=int(pos), knot=0, time_s=time, normalized_soc=100.))
        by_stage = g.set_index("stage")
        for i, stage in enumerate(NAMES):
            time += float(by_stage.loc[stage,"stage_duration_s"])
            knots.append(dict(condition_id=cid, position=int(pos), knot=i+1, time_s=time,
                              normalized_soc=normalizer.reference.boundaries[i+1]))
    pd.DataFrame(knots).to_csv(wind/"three_stage_curve_knots.csv", index=False)
    for cid, g in all_curves.groupby("condition_id", sort=True):
        if cid not in REFIT_CELLS:
            for ext in ("png", "pdf", "svg"):
                shutil.copy2(PRIOR/"wind_tunnel/figures"/(cid+"_wind_three_stage."+ext), wind/"figures"/(cid+"_wind_three_stage."+ext))
        else:
            fig = figure(g, normalizer)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(wind/"figures"/(cid+"_wind_three_stage."+ext), dpi=300)
            plt.close(fig)
    with PdfPages(wind/"repaired_four_conditions.pdf") as pdf:
        for cid in sorted(TARGETS):
            fig = figure(all_curves[all_curves.condition_id.eq(cid)], normalizer)
            pdf.savefig(fig)
            plt.close(fig)
    with PdfPages(wind/"all_wind_three_stage_curves.pdf") as pdf:
        for cid, g in all_curves.groupby("condition_id", sort=True):
            # Re-rendering a copied candidate is presentation-only; its
            # coefficients remain unchanged and are explicitly identified.
            fig = figure(g, normalizer)
            pdf.savefig(fig)
            plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=180)
    order = ["vee_50_tail_lv2", "front_50_tail_lv2", "front_50_side_lv1", "echalon_50_head_lv1"]
    for ax, cid in zip(axes.flat, order):
        g = fixed[fixed.condition_id.eq(cid)]
        plot_curve(ax, g, normalizer, small=True)
        row = g.iloc[0]
        ax.set_title(f"{FORM_NAMES[row.formation]} · {WIND_NAMES[row.wind_direction]} · Level {int(row.wind_level)}", fontsize=13, loc="left")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized SOC (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=11)
    fig.suptitle("50 cm · Refit of the four questioned wind-tunnel curves", fontsize=17, y=.99)
    fig.text(.5, .945, "Model estimates · Fixed September baseline · Other-speed references retained in coefficient provenance", ha="center", fontsize=10, color="#52636C")
    fig.subplots_adjust(left=.075, right=.985, bottom=.095, top=.9, wspace=.19, hspace=.28)
    fig.savefig(wind/"repaired_four_overview.png", dpi=240)
    fig.savefig(wind/"repaired_four_overview.pdf")
    plt.close(fig)

    before = previous[previous.condition_id.isin(TARGETS)].groupby(["condition_id", "position"]).stage_duration_s.sum()
    after = fixed[fixed.condition_id.isin(TARGETS)].groupby(["condition_id", "position"]).stage_duration_s.sum()
    compare = pd.DataFrame({"previous_modeled_full_time_s":before, "refitted_modeled_full_time_s":after})
    compare["difference_s"] = compare.refitted_modeled_full_time_s-compare.previous_modeled_full_time_s
    compare.reset_index().to_csv(wind/"before_after_modeled_times.csv", index=False)
    time_errors = fixed.stage_duration_s-60*(fixed.reference_upper_soc-fixed.reference_lower_soc)/fixed.normalized_rate_pp_min
    forward_matches = all(sha(p)==sha(out/"forward_250cm"/p.relative_to(PRIOR/"forward_250cm")) for p in (PRIOR/"forward_250cm").rglob("*") if p.is_file())
    validation = dict(fixed_baseline_unchanged=sha(MODEL)==FIXED_SHA,
        raw_inputs_unchanged=all(sha(ROOT/p)==digest for p,digest in source_hashes.items()),
        forward_outputs_byte_preserved=forward_matches,
        excluded_conditions_absent=not bool(set(all_curves.condition_id)&excluded),
        every_curve_has_three_positive_stages=True, safe_condition_count=25,
        refitted_condition_count=8, target_condition_count=4,
        remaining_condition_coefficients_unchanged=True,
        stage_duration_identity_max_error_s=float(np.abs(time_errors).max()),
        fitted_points_all_nominal_hover=bool(points.phase.eq("wind_tunnel_hover").all()),
        fitted_points_all_assigned_pad=bool(points.mid.eq(points.target_pad).all()),
        fitted_points_all_height_60_100=bool(points.z.between(60,100).all() & points.tof.between(60,100).all()),
        real_stage_fit_count=len(runs), cross_speed_modeled_stage_count=int(fixed.is_modeled.sum()),
        active_training_changed=False, hardware_commands_sent=False, protocols_averaged=False,
        interpretation="Repaired conditional model estimates, not newly measured discharge or independently verified flight performance.")
    write_json(out/"validation.json", validation)
    write_json(out/"manifest.json", dict(status="repaired_model_candidate_not_activated", fixed_baseline_sha256=FIXED_SHA,
        refitted_conditions=sorted(REFIT_CELLS), target_conditions=sorted(TARGETS),
        all_safe_conditions=sorted(all_curves.condition_id.unique().tolist()), source_hashes=source_hashes,
        speed_factors=factors,
        fitting="SOC(first-crossing time) OLS with independent real-block intercepts; first lower-boundary point closes each stage",
        model_weight="own-stage observed SOC coverage squared / (1 + (weighted log repeat-rate spread / log(1.25)) squared)",
        combination="geometric blend of real refit and same-position other-speed prior; suspect-stage factor >2 limits observed weight to 0.1",
        modeling_assumptions=["Other-speed scaling is transferred across stages only for model estimation.",
            "When reliable same-family paired Medium rates cannot establish the requested trend, Level1 rate is set to 95% of Level2 as an explicit user-authorized scenario.",
            "A missing donor High stage may use the fixed reference shape and observed mature-stage relative load; it is flagged modeled.",
            "These curve coefficients are not silently promoted to observed training labels."],
        uncertainty="Coverage weights and the five-percent scenario are modeling choices; no new empirical confidence interval or physical causal verification is claimed."))
    table_rows = ["| Condition | Position | Previous model (s) | Refit model (s) | Change (s) |",
                  "| --- | --- | --- | --- | --- |"]
    for row in compare.reset_index().to_dict("records"):
        table_rows.append(f"| {row['condition_id']} | {row['position']} | {row['previous_modeled_full_time_s']:.1f} | "
                          f"{row['refitted_modeled_full_time_s']:.1f} | {row['difference_s']:.1f} |")
    table = "\n".join(table_rows)
    factors_text = "\n".join(f"- {f['family']}: Level2/Level1 耗电率比例 {f['applied_l2_over_l1_rate_multiplier']:.4f}；来源 {f['source']}。" for f in factors)
    (out/"README.md").write_text(f"""# 四组50 cm风洞曲线重新拟合

已重拟合用户指出的四个条件和用于参考的四个另一档风速条件。其余17个条件保留上一版系数，没有宣称本次全部重新验证。原始实测CSV、固定九月份baseline、旧版输出、实机控制代码及训练数据保持不变；250 cm结果原样保留，没有平均。

## 处理方式

1. 找回所有可用编号实验的实际悬停片段，不再要求整次实验必须飞到20%。使用名义hover、识别到指定Pad、Pad高度和ToF均60–100 cm的实际样本；故障阶段及失去/换Pad的样本不作为该配置的有效拟合点。这是离线数据筛选，不是实机安全验证。
2. 每架飞机去除开头不变的SOC时间。不同起飞、恢复、失去定位或源文件各有自己的时间块和截距，不跨停止时段拼接时间。
3. 按各自电池High/Medium/Low范围拟合SOC首次报告下降的时间点。中途整数平台仍计入相邻下降事件之间的实际时间。前一阶段在首次到达下边界时结束，下边界平台不再重复进入前一阶段。单个块至少三个SOC层级，累计实际观察下降至少3个百分点；只有一次跳变的4秒窗口不支撑整阶段率。
4. 实测候选耗电率仍按固定九月份模型逐阶段归一化。不同起始SOC只影响覆盖范围，不直接平均飞到20%的时间。
5. 修复图采用带参考的模型拟合：实际阶段覆盖越完整、不同编号阶段率越一致，保留实测拟合的权重越大。部分观测和异常阶段更多参考同队形、同风向、同位置另一档风速。权重及实测率、参考率、最终率都写入系数表。拟合权重是明确的建模规则，不是经验置信概率。
6. Level1缺段按用户授权的较弱耗电趋势补全。可靠配对Medium支持时使用实际比例；支持不足或数据不支持该趋势时，模型设定Level1率为Level2的95%，相应时间增加约5.3%。并不宣称所有实测Level1必然比Level2耗电慢。
7. 另一档风速的High也缺少实测时，使用固定参考三段形状和该位置实测中/低电量负载形成模型先验，单独标注来源；不将短High因子复制到整个中低电量阶段。未增加模拟行到原始或实测阶段表。

## 使用的两档比例

{factors_text}

## 模型时间变化

以下均为固定参考SOC 100%→20%的模型时长，不是直接测得的完整飞行时间。

{table}

## 输出

- `wind_tunnel/repaired_four_overview.png/pdf`：四张修复图，保持原有五位置和参考曲线布局。
- `wind_tunnel/repaired_four_conditions.pdf`：四张单独图。
- `wind_tunnel/figures/`：25个条件PNG/PDF/SVG；本次改动8个条件。
- `wind_tunnel/curve_stage_rates.csv`：完整375阶段系数；`estimate_kind`区分实测拟合、带参考的模型估计及未复核旧候选。
- `wind_tunnel/refitted_real_run_stage_rates.csv`：从实际合法窗口重拟合的逐实验阶段率；没有参考风速填补的假观测。
- `wind_tunnel/repaired_eight_condition_stage_estimates.csv`：最终模型率、实测权重、参考条件、原始来源及缺段先验。
- `wind_tunnel/actual_SOC_crossing_samples.csv`、`actual_window_audit.csv`：实际拟合点与筛选说明。
- `wind_tunnel/cross_speed_factors.json`：配对记录比例和5%模型设定。
- `wind_tunnel/before_after_modeled_times.csv`：四组前后模型时长。
- `manifest.json`、`validation.json`：来源校验、方法、约束及离线检查。原始失败记录和所有危险条件排除规则保留。

这些结果是用户授权的修复模型版本。借用风速及缺阶段的估计保留出处，可用于审查模型曲线；在后续训练或论文实测性能结论使用时，应区分这些模型估计与原始实测证据。
""", encoding="utf-8")
    with zipfile.ZipFile(out.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                archive.write(p, p.relative_to(out.parent))
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    print(compare.reset_index().round(1).to_string(index=False))
    print(json.dumps(factors, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true", help="Complete this script's own output after an export interruption")
    args = parser.parse_args()
    main(args.output, args.resume)
