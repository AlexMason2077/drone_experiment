"""Two independent offline products on the fixed September battery scale.

Step 1 transfers existing historical 250-cm Medium coefficients and curves.
Step 2 fits real 50-cm wind-tunnel hover stages. Neither step blends protocols.
Raw records, the calibration model, controllers and active datasets are read-only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from process_50cm_stage_curves import (
    ROOT, MODEL, FORWARD, NAMES, CELL, GRAIN, COLORS, FORMS, FORM_NAMES,
    WIND_NAMES, BatteryNormalizer, plt, sha, write_json, cell_id,
    exclusion_ids, extract_wind, plot_curve,
)

OUT_ROOT = ROOT / "analysis_results/separate_baseline_processing_20260929"
FIXED_MODEL_HASH = "5ea37d52e43b4c283f3ce6f956b2b669c1878030e32afc92fc9749fe66af0fe5"
HISTORICAL = {
    "B11": "database/baselines/drone_1_B11/drone_1_B11_hover_20260513_143201_timeseries.csv",
    "B10": "database/baselines/drone_2_B10/drone_2_B10_hover_20260513_180556_timeseries.csv",
    "B13": "database/baselines/drone_3_B13/drone_3_B13_hover_20260513_163826_timeseries.csv",
    "B14": "database/baselines/drone_4_B14/drone_4_B14_hover_20260513_144257_timeseries.csv",
    "B15": "database/baselines/drone_5_B15/drone_5_B15_hover_20260513_150444_timeseries.csv",
    "B12": "database/baselines/drone_5_B12/drone_5_B12_hover_lv1_20260609_174758_20260609_174758_all_coordination.csv",
}
TARGET_BATTERY = {1: "B11", 2: "B10", 3: "B13", 4: "B14", 5: "B12"}
KEY = ["experiment_directory", "run_id", "drone_name"]


def inputs():
    normalizer = BatteryNormalizer.load(MODEL)
    assert normalizer.sha256 == FIXED_MODEL_HASH
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (
        MODEL, ROOT / "TRAINING_EXCLUSIONS.md", ROOT / "battery_normalization.py",
        Path(__file__), ROOT / "output_py/process_50cm_stage_curves.py",
    )}
    return normalizer, exclusion_ids(), hashes


def common_manifest(normalizer, hashes, protocol):
    return dict(status="completed_separate_candidate_not_activated", included_protocols=[protocol],
        normalization_model_version=normalizer.version,
        normalization_model_sha256=normalizer.sha256,
        reference_boundaries_soc=list(normalizer.reference.boundaries),
        reference_rates_pp_min=list(normalizer.reference.rates_pp_min),
        baseline_modified=False, cross_protocol_averaging_applied=False,
        active_dataset_modified=False, trained=False, flight_control_changed=False,
        source_hashes=hashes, safety_conditions_excluded=sorted(exclusion_ids()))


def finish_input_check(hashes):
    assert all(sha(ROOT / p) == h for p, h in hashes.items())


def prepare_output(path):
    if path.exists():
        raise FileExistsError(f"Preserve existing results: {path}")
    path.mkdir(parents=True)
    (path / "figures").mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "pdf.fonttype": 42, "svg.fonttype": "none"})


def historical_bridges(normalizer, hashes):
    rows = []
    for battery, relative_path in HISTORICAL.items():
        drone = normalizer.pairs[battery]
        position = int(drone.split("_")[-1])
        own = normalizer.curve_for(battery, drone)
        target_id = TARGET_BATTERY[position]
        target = normalizer.curve_for(target_id, drone)
        source = ROOT / relative_path
        hashes[relative_path] = sha(source)
        raw = pd.read_csv(source, low_memory=False)
        assert set(raw.battery_id.dropna()) == {battery}
        assert set(raw.drone_name.dropna()) == {drone}
        hover = raw[raw.phase.eq("hover_to_10_percent")].sort_values("elapsed_time")
        assert not hover.elapsed_time.duplicated().any()
        assert not hover.battery.diff().gt(0).any()
        for i, stage in enumerate(NAMES):
            # The existing model's High slope is calibrated over 95% downwards;
            # its 100% endpoint is an extension, not a 5-pp observed discharge.
            upper = min(95.0, own.boundaries[i])
            lower = own.boundaries[i + 1]
            t_upper = float(hover.loc[hover.battery.le(upper), "elapsed_time"].iloc[0])
            t_lower = float(hover.loc[hover.battery.le(lower), "elapsed_time"].iloc[0])
            duration = t_lower - t_upper
            assert duration > 0
            old_rate = 60 * (upper - lower) / duration
            same_battery_new = float(own.rates_pp_min[i])
            target_new = float(target.rates_pp_min[i])
            scale = target_new / old_rate
            rows.append(dict(position=position, drone_name=drone, old_battery_id=battery,
                new_position_battery_id=target_id, stage=stage,
                old_baseline_source=relative_path, old_observed_upper_soc=upper,
                old_observed_lower_soc=lower, old_baseline_duration_s=duration,
                old_baseline_rate_pp_min=old_rate,
                same_battery_current_baseline_rate_pp_min=same_battery_new,
                same_battery_epoch_scale=same_battery_new / old_rate,
                new_position_baseline_rate_pp_min=target_new,
                old_to_new_position_scale=scale,
                adjusted_old_baseline_rate_pp_min=old_rate * scale,
                reference_rate_pp_min=float(normalizer.reference.rates_pp_min[i]),
                available_forward_stage=(stage == "Medium"),
                baseline_fit_definition="SOC drop / first-crossing elapsed time at fixed own-stage boundaries",
                identity_transfer="same logical position; B15 to current B12 at P5" if battery == "B15" else "same drone and battery ID across dates"))
    result = pd.DataFrame(rows)
    assert np.allclose(result.adjusted_old_baseline_rate_pp_min,
                       result.new_position_baseline_rate_pp_min, rtol=1e-12)
    return result


def forward_figure(group, normalizer):
    fig, ax = plt.subplots(figsize=(11, 6.8))
    upper, lower = normalizer.reference.boundaries[1:3]
    for row in group.sort_values("position").itertuples():
        if row.normalized_rate_pp_min <= 0:
            continue
        duration = 60 * (upper - lower) / row.normalized_rate_pp_min
        ax.plot([0, duration], [upper, lower], color=COLORS[row.position-1], lw=2.4,
                marker="o", markersize=4, label=f"P{row.position} / D{row.position}")
    duration = 60 * (upper - lower) / normalizer.reference.rates_pp_min[1]
    ax.plot([0, duration], [upper, lower], color="#48555D", lw=1.6,
            ls=(0, (5, 3)), label="Reference battery")
    ax.set(xlabel="Time (s)", ylabel="Normalized SOC (%)", ylim=(48, 86))
    ax.set_xlim(left=0)
    ax.set_yticks([52, 60, 70, 82])
    ax.grid(color="#E3E8EB", lw=.7)
    ax.spines[["top", "right"]].set_visible(False)
    row = group.iloc[0]
    fig.suptitle(f"{FORM_NAMES[row.formation]} · {int(row.spacing_cm)} cm · {WIND_NAMES[row.wind_direction]} · Level {int(row.wind_level)}",
        x=.11, y=.966, ha="left", fontsize=17, fontweight="semibold")
    fig.text(.11, .912, "250 cm forward data · Baseline transfer · Medium stage only", fontsize=11, color="#52636C")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=3, frameon=False, fontsize=11)
    fig.text(.11, .026, "Stage duration is calculated from the transferred rate; original forward time is unchanged.", fontsize=9, color="#52636C")
    fig.subplots_adjust(left=.11, right=.96, bottom=.23, top=.855)
    return fig


def process_forward(out):
    normalizer, excluded, hashes = inputs()
    source_paths = [FORWARD / n for n in ("run_drone_rates.csv", "selected_forward_intervals.csv", "manifest.json")]
    for p in source_paths:
        hashes[str(p.relative_to(ROOT))] = sha(p)
    previous_manifest = json.loads((FORWARD / "manifest.json").read_text())
    assert previous_manifest["normalization_model_sha256"] == normalizer.sha256
    original = pd.read_csv(FORWARD / "run_drone_rates.csv")
    frame = original.rename(columns={"inter_drone_spacing_cm": "spacing_cm"}).copy()
    frame["condition_id"] = frame.apply(cell_id, axis=1)
    frame["source_status"] = frame.status
    frame.loc[frame.condition_id.isin(excluded), "status"] = "excluded_safety_condition"
    safe = frame[~frame.condition_id.isin(excluded)]
    source_pairs = safe[["experiment_directory", "run_id"]].drop_duplicates()
    for row in source_pairs.itertuples():
        p = ROOT / f"db_copy_for_cleaning/{row.experiment_directory}/{row.experiment_directory}_{row.run_id}_all_coordination.csv"
        relative_path = str(p.relative_to(ROOT))
        digest = sha(p)
        assert digest == previous_manifest["input_files_sha256"][relative_path]
        hashes[relative_path] = digest
    bridges = historical_bridges(normalizer, hashes)
    medium = bridges[bridges.stage.eq("Medium")].set_index(["drone_name", "old_battery_id"])
    rows = []
    for previous in frame[frame.status.eq("included")].itertuples():
        bridge = medium.loc[(previous.drone_name, previous.battery_id)]
        raw = float(previous.raw_medium_discharge_rate_pp_min)
        adjusted = raw * bridge.old_to_new_position_scale
        normalized = adjusted / bridge.new_position_baseline_rate_pp_min * normalizer.reference.rates_pp_min[1]
        current_own = normalizer.curve_for(previous.battery_id, previous.drone_name)
        assert math.isclose(previous.discharge_rate_Bideal_pp_per_min,
            raw / current_own.rates_pp_min[1] * normalizer.reference.rates_pp_min[1], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(normalized, raw / bridge.old_baseline_rate_pp_min * normalizer.reference.rates_pp_min[1], rel_tol=1e-12, abs_tol=1e-12)
        rows.append(dict(condition_id=previous.condition_id, formation=previous.formation,
            spacing_cm=int(previous.spacing_cm), wind_direction=previous.wind_direction,
            wind_level=int(previous.wind_level), experiment_directory=previous.experiment_directory,
            run_id=previous.run_id, drone_name=previous.drone_name, position=int(previous.position),
            old_battery_id=previous.battery_id, new_position_battery_id=bridge.new_position_battery_id,
            protocol="forward_250cm", stage="Medium", observed_raw_rate_pp_min=raw,
            old_baseline_rate_pp_min=float(bridge.old_baseline_rate_pp_min),
            new_position_baseline_rate_pp_min=float(bridge.new_position_baseline_rate_pp_min),
            old_to_new_position_scale=float(bridge.old_to_new_position_scale),
            old_relative_drain_factor=raw / bridge.old_baseline_rate_pp_min,
            adjusted_raw_rate_on_new_battery_pp_min=adjusted, normalized_rate_pp_min=normalized,
            previous_v3_normalized_rate_pp_min=previous.discharge_rate_Bideal_pp_per_min,
            original_forward_duration_s=previous.medium_forward_duration_s,
            original_raw_drop_pp=previous.raw_medium_drop_pp,
            adjusted_raw_drop_on_new_battery_pp=previous.raw_medium_drop_pp * bridge.old_to_new_position_scale,
            normalized_drop_pp=previous.raw_medium_drop_pp * normalizer.reference.rates_pp_min[1] / bridge.old_baseline_rate_pp_min,
            original_curve_r_squared=previous.curve_through_origin_r_squared,
            original_quality_flags=previous.qc_flags if pd.notna(previous.qc_flags) else "",
            weak_support=bool(previous.medium_forward_duration_s < 18 or previous.raw_medium_drop_pp < 3),
            rate_origin="existing_raw_Medium_fit_transferred_by_stage_baseline_ratio"))
    rates = pd.DataFrame(rows)
    points = pd.read_csv(FORWARD / "selected_forward_intervals.csv").merge(
        rates[KEY + ["condition_id", "position", "old_battery_id", "new_position_battery_id",
            "old_to_new_position_scale", "old_baseline_rate_pp_min", "new_position_baseline_rate_pp_min"]],
        on=KEY, how="inner", validate="many_to_one")
    points["adjusted_cumulative_raw_drop_pp"] = points.cumulative_raw_drop_pp * points.old_to_new_position_scale
    points["normalized_cumulative_drop_pp"] = points.adjusted_cumulative_raw_drop_pp / points.new_position_baseline_rate_pp_min * normalizer.reference.rates_pp_min[1]
    replay_errors, time_errors = [], []
    lookup = rates.set_index(KEY)
    for key, g in points.groupby(KEY, sort=False):
        t, d = g.forward_clock_s.to_numpy(), g.cumulative_raw_drop_pp.to_numpy()
        replay = 60 * float(np.dot(t, d)) / float(np.dot(t, t))
        replay_errors.append(abs(replay - lookup.loc[key, "observed_raw_rate_pp_min"]))
        time_errors.append(abs(float(t[-1]) - lookup.loc[key, "original_forward_duration_s"]))
    assert len(replay_errors) == len(rates) and max(replay_errors) < 1e-8 and max(time_errors) < 1e-8
    means = rates.groupby(GRAIN, as_index=False).agg(
        normalized_rate_pp_min=("normalized_rate_pp_min", "mean"),
        observed_run_count=("run_id", "nunique"),
        min_run_rate_pp_min=("normalized_rate_pp_min", "min"),
        max_run_rate_pp_min=("normalized_rate_pp_min", "max"),
        run_rate_std_pp_min=("normalized_rate_pp_min", "std"),
        weak_run_count=("weak_support", "sum"),
        zero_observed_rate_count=("observed_raw_rate_pp_min", lambda x: int(x.eq(0).sum())))
    means["condition_id"] = means.apply(cell_id, axis=1)
    means["reference_upper_soc"], means["reference_lower_soc"] = normalizer.reference.boundaries[1:3]
    means["calculated_medium_duration_s"] = np.where(means.normalized_rate_pp_min.gt(0),
        60 * (means.reference_upper_soc - means.reference_lower_soc) / means.normalized_rate_pp_min, np.nan)
    prepare_output(out)
    for filename, data in [
        ("baseline_stage_transfer_factors.csv", bridges), ("adjusted_run_drone_rates.csv", rates),
        ("adjusted_forward_curve_points.csv", points), ("forward_record_audit.csv", frame),
        ("condition_position_medium_means.csv", means),
    ]:
        data.to_csv(out / filename, index=False)
    means.pivot(index=CELL + ["condition_id"], columns="position", values="normalized_rate_pp_min").rename(
        columns=lambda pos: f"position_{pos}_normalized_pp_min").to_csv(out / "discharge_rates_wide.csv")
    with PdfPages(out / "all_forward_medium_curves.pdf") as pdf:
        for cid, group in means.groupby("condition_id", sort=True):
            fig = forward_figure(group, normalizer)
            pdf.savefig(fig)
            if cid == "diamond_50_head_lv2":
                for ext in ("png", "pdf", "svg"):
                    fig.savefig(out / "figures" / f"{cid}_forward_medium.{ext}", dpi=300, facecolor="white")
            plt.close(fig)
    finish_input_check(hashes)
    validation = dict(status="passed_offline_checks", input_hashes_unchanged=True,
        baseline_model_unchanged=True, no_wind_experiment_inputs_read=True,
        cross_protocol_average_applied=False, all_existing_raw_slopes_replayed=True,
        raw_slope_replay_max_error_pp_min=max(replay_errors),
        actual_forward_time_replay_max_error_s=max(time_errors),
        baseline_bridge_max_error_pp_min=float(np.max(np.abs(bridges.adjusted_old_baseline_rate_pp_min - bridges.new_position_baseline_rate_pp_min))),
        relative_factor_preserved=True, normalized_twice=False,
        excluded_conditions_absent=not bool(set(rates.condition_id) & excluded),
        raw_zero_slopes_preserved=int(rates.observed_raw_rate_pp_min.eq(0).sum()),
        unsupported_High_Low_not_filled=True, active_training_unchanged=True)
    manifest = common_manifest(normalizer, hashes, "forward_250cm")
    manifest.update(original_candidate_runs=int(original.run_id.nunique()),
        safe_runs=int(safe.run_id.nunique()), included_drone_rates=len(rates),
        safe_conditions=int(means.condition_id.nunique()),
        spacing_cm_values=sorted(int(x) for x in rates.spacing_cm.unique()),
        supported_observed_stages=["Medium"],
        safe_records_without_Medium=int(safe.status.ne("included").sum()),
        safety_excluded_records=int(frame.status.eq("excluded_safety_condition").sum()),
        transfer_formula="r_adjusted = r_old_raw * b_new_position_stage / b_historical_same_drone_stage",
        normalization_formula="r_normalized = r_adjusted / b_new_position_stage * b_fixed_reference_stage",
        historical_baseline_policy="Same logical drone+battery, May 13 for B10/B11/B13/B14/B15, June 09 for old D5+B12; rates measured over current own-stage SOC bands",
        B15_transfer="Keep old B15 identity; use current P5+B12 as the explicit target battery. Relative load transfer across batteries is an assumption.",
        aggregation="Equal run weights within identical condition, position and Medium stage; no wind inputs",
        initial_plateau_policy="Retain approved forward segmentation and slopes; this step only changes calibration. Wind initial-plateau trimming is performed independently in step 2.",
        uncertainties=["Historical source is one baseline run per battery, not a separately validated estimate of ageing.",
            "Date-to-date and B15-to-B12 relative-load transfer are assumptions.",
            "Existing short, partial and integer-SOC flat-trace flags are preserved; zero measured slopes are not evidence of zero energy use.",
            "Reference-stage time is computed from rates; it is not the observed 250-cm segment duration."])
    write_json(out / "validation.json", validation)
    write_json(out / "manifest.json", manifest)
    (out / "README.md").write_text(f"""# 第一步：历史250厘米数据换算到固定九月份baseline

仅处理已有250厘米前进结果。{manifest['safe_runs']}组安全范围内实验、{len(rates)}条有效Medium耗电率、{manifest['safe_conditions']}个条件；50和75厘米都保留。另{manifest['safe_records_without_Medium']}条本机没有Medium前进时间，留在audit，不填假观测。当前五项危险条件全部排除，原始记录保留。

## 换算

逐位置、逐电池、逐阶段：比例 = 九月份目标基线阶段率 / 同无人机旧电池基线阶段率。旧前进原始率乘这个比例，得到新目标电池尺度的率；再除新基线率、乘固定参考电池阶段率，得到统一参考尺度。直接使用已有raw率，未对已经标准化的率再乘比例。原始前进时间、SOC下降、分段和质量标记均保留。

旧基线采用B10/B11/B13/B14/B15的5月13日本机完整悬停记录、D5+B12的6月9日记录；用与当前模型相同的本电池SOC阶段边界重新计算旧基线的率。避开旧版把不同无人机的B12混池、以及把低电量部分并入中电量的比例。没有修改任何九月份model参数。P5旧B15明确映射到当前P5+B12；这是假定相同位置的相对负载可跨电池转移，不能视为已经证实的物理因果。

## 文件

- baseline_stage_transfer_factors.csv：18条分阶段旧→新比例；只有Medium用于这批已有前进系数。
- adjusted_run_drone_rates.csv：逐实验逐位置的原始率、调整后率、参考尺度率和质量标记。
- adjusted_forward_curve_points.csv：{len(points)}个已有前进观测区间的累计下降换算，实际时间不变。调整/标准化下降为计算值，不是新增SOC实测。
- condition_position_medium_means.csv / discharge_rates_wide.csv：同条件同位置的前进实验均值；为以后第三步准备，当前未与风洞平均。
- all_forward_medium_curves.pdf：每个条件的Medium参考曲线，82→52的时间由率计算；不是把各实验时间拉到相同长度。
- figures/diamond_50_head_lv2_forward_medium.*：示例PNG 3300×2040及PDF/SVG。
- forward_record_audit.csv / manifest.json / validation.json：来源、排除原因、只读输入哈希和离线计算校验。

这批已有系数只有Medium观测，未补造High/Low。17条原始零斜率保留，整数SOC没有跳档不表示真实能耗为零；短/部分阶段同样保留质量标记。输出为独立候选分析，没有更新训练数据、模型、控制或数据库。
""")
    print(json.dumps({k: manifest[k] for k in ("safe_runs", "included_drone_rates", "safe_conditions", "safe_records_without_Medium", "safety_excluded_records")}, indent=2))


def complete_wind_curves(observed_means, normalizer):
    rows, knots = [], []
    for key, group in observed_means.groupby(CELL + ["position"], sort=True):
        by_stage = {r["stage"]: r for r in group.to_dict("records")}
        elapsed = 0.0
        cid, position = cell_id(dict(zip(CELL + ["position"], key))), int(key[-1])
        knots.append(dict(condition_id=cid, position=position, knot=0, time_s=0., normalized_soc=100.))
        for i, stage in enumerate(NAMES):
            if stage in by_stage:
                row = by_stage[stage].copy()
                row.update(is_modeled=False, completion_anchor_stage="", rate_origin="mean_of_observed_wind_stage_fits")
            else:
                anchor = "Medium" if "Medium" in by_stage else min(by_stage, key=lambda s: abs(NAMES.index(s) - i))
                row = by_stage[anchor].copy()
                relative = row["normalized_rate_pp_min"] / normalizer.reference.rates_pp_min[NAMES.index(anchor)]
                row.update(stage=stage, normalized_rate_pp_min=relative * normalizer.reference.rates_pp_min[i],
                    observed_run_count=0, weak_run_count=0, min_run_rate_pp_min=np.nan,
                    max_run_rate_pp_min=np.nan, run_rate_std_pp_min=np.nan,
                    is_modeled=True, completion_anchor_stage=anchor,
                    rate_origin="modeled_missing_stage_from_same_position_wind_relative_factor")
            upper, lower = normalizer.reference.boundaries[i:i+2]
            duration = 60 * (upper - lower) / row["normalized_rate_pp_min"]
            row.update(reference_upper_soc=upper, reference_lower_soc=lower, stage_duration_s=duration)
            rows.append(row)
            elapsed += duration
            knots.append(dict(condition_id=cid, position=position, knot=i+1, time_s=elapsed, normalized_soc=lower))
    return pd.DataFrame(rows), pd.DataFrame(knots)


def wind_figure(group, normalizer):
    fig, ax = plt.subplots(figsize=(11, 6.8))
    plot_curve(ax, group, normalizer)
    for stage, upper, lower in zip(NAMES, normalizer.reference.boundaries, normalizer.reference.boundaries[1:]):
        ax.text(.985, (upper+lower)/2, stage, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=11, color="#64757F", bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
    row = group.iloc[0]
    fig.suptitle(f"{FORM_NAMES[row.formation]} · 50 cm · {WIND_NAMES[row.wind_direction]} · Level {int(row.wind_level)}",
        x=.11, y=.966, ha="left", fontsize=17, fontweight="semibold")
    fig.text(.11, .912, "Wind-tunnel data · Independent stage rates on the fixed reference-battery scale", fontsize=10.5, color="#52636C")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.11, right=.96, bottom=.23, top=.855)
    return fig


def process_wind(out):
    normalizer, excluded, hashes = inputs()
    registry_path = ROOT / "database/experiment_registry.json"
    hashes[str(registry_path.relative_to(ROOT))] = sha(registry_path)
    data = json.loads(registry_path.read_text())["experiments"]
    registry = {r["experiment_id"]: r for r in (data.values() if isinstance(data, dict) else data)}
    rows, run_audit, drone_audit = extract_wind(normalizer, registry, excluded, hashes)
    rates = pd.DataFrame(rows)
    assert rates.spacing_cm.eq(50).all() and set(rates.protocol) == {"wind_tunnel"}
    assert not rates.duplicated(["experiment_id", "run_id", "drone_name", "stage"]).any()
    means = rates.groupby(GRAIN, as_index=False).agg(
        normalized_rate_pp_min=("normalized_rate_pp_min", "mean"),
        observed_run_count=("run_id", "nunique"), weak_run_count=("weak_support", "sum"),
        min_run_rate_pp_min=("normalized_rate_pp_min", "min"),
        max_run_rate_pp_min=("normalized_rate_pp_min", "max"),
        run_rate_std_pp_min=("normalized_rate_pp_min", "std"))
    means["condition_id"] = means.apply(cell_id, axis=1)
    curves, knots = complete_wind_curves(means, normalizer)
    coverage = []
    for formation in FORMS:
        for wind in ("head", "tail", "side"):
            for level in (1, 2):
                meta = dict(formation=formation, wind_direction=wind, wind_level=level, spacing_cm=50)
                cid = cell_id(meta); g = curves[curves.condition_id.eq(cid)]
                coverage.append(dict(**meta, condition_id=cid,
                    status="excluded_safety_condition" if cid in excluded else ("has_wind_stage_curves" if not g.empty else "no_supported_complete_wind_record"),
                    positions_with_curves=int(g.position.nunique()), observed_stage_count=int((~g.is_modeled).sum()),
                    modeled_stage_count=int(g.is_modeled.sum()), weak_observed_stage_count=int(g.weak_run_count.sum())))
    for cid, g in curves[~curves.wind_level.isin([1, 2])].groupby("condition_id"):
        meta = {k: g[k].iloc[0] for k in CELL}
        coverage.append(dict(**meta, condition_id=cid, status="additional_actual_level_kept_separate",
            positions_with_curves=int(g.position.nunique()), observed_stage_count=int((~g.is_modeled).sum()),
            modeled_stage_count=int(g.is_modeled.sum()), weak_observed_stage_count=int(g.weak_run_count.sum())))
    coverage = pd.DataFrame(coverage)
    for key, g in knots.groupby(["condition_id", "position"]):
        g = g.sort_values("knot")
        target = curves[(curves.condition_id.eq(key[0])) & curves.position.eq(key[1])].set_index("stage").loc[list(NAMES)]
        assert np.diff(g.time_s).min() > 0 and np.diff(g.normalized_soc).max() < 0
        assert np.allclose(-60*np.diff(g.normalized_soc)/np.diff(g.time_s), target.normalized_rate_pp_min)
    prepare_output(out)
    for filename, table in [("observed_run_stage_rates.csv", rates), ("condition_position_stage_means.csv", means),
        ("curve_stage_rates.csv", curves), ("three_stage_curve_knots.csv", knots),
        ("condition_coverage.csv", coverage), ("wind_run_audit.csv", pd.DataFrame(run_audit)),
        ("wind_drone_window_audit.csv", pd.DataFrame(drone_audit))]:
        table.to_csv(out / filename, index=False)
    with PdfPages(out / "all_wind_three_stage_curves.pdf") as pdf:
        for cid, group in curves.groupby("condition_id", sort=True):
            fig = wind_figure(group, normalizer)
            pdf.savefig(fig)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(out / "figures" / f"{cid}_wind_three_stage.{ext}", dpi=300, facecolor="white")
            plt.close(fig)
    fig, axes = plt.subplots(5, 6, figsize=(22.5, 17.5))
    for i, formation in enumerate(FORMS):
        for j, (wind, level) in enumerate([(w, l) for w in ("head", "tail", "side") for l in (1, 2)]):
            ax = axes[i, j]; cid = f"{formation}_50_{wind}_lv{level}"
            g = curves[curves.condition_id.eq(cid)]
            ax.set_title(f"{FORM_NAMES[formation]} · {WIND_NAMES[wind]} · L{level}", fontsize=10, loc="left")
            if cid in excluded or g.empty:
                ax.set_facecolor("#F0F1F2" if cid in excluded else "white")
                label = "Excluded condition" if cid in excluded else "No supported complete record"
                ax.text(.5, .5, label, transform=ax.transAxes, ha="center", fontsize=9, color="#64757F")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                plot_curve(ax, g, normalizer, small=True)
    handles, labels = next(ax for ax in axes.flat if ax.lines).get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(.5, .015))
    fig.suptitle("50 cm · Wind-tunnel three-stage discharge curves", x=.055, y=.984, ha="left", fontsize=22, fontweight="semibold")
    fig.text(.055, .956, "Wind data only · Fixed September baseline · Stage rates fitted before reference time is calculated", fontsize=12, color="#52636C")
    fig.subplots_adjust(left=.055, right=.985, top=.923, bottom=.071, hspace=.60, wspace=.43)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"50cm_wind_overview.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    pilot = ROOT / "analysis_results/50cm_wind_only_pilot_20260929/diamond_50_head_lv2/curve_stage_rates.csv"
    hashes[str(pilot.relative_to(ROOT))] = sha(pilot)
    check = curves[curves.condition_id.eq("diamond_50_head_lv2")].merge(
        pd.read_csv(pilot)[["position", "stage", "normalized_rate_pp_min"]], on=["position", "stage"], suffixes=("_new", "_pilot"), validate="one_to_one")
    pilot_error = float(np.max(np.abs(check.normalized_rate_pp_min_new - check.normalized_rate_pp_min_pilot)))
    assert len(check) == 15 and pilot_error < 1e-8
    finish_input_check(hashes)
    validation = dict(status="passed_offline_checks", input_hashes_unchanged=True,
        baseline_model_unchanged=True, no_250cm_experiment_inputs_read=True,
        cross_protocol_average_applied=False, all_curves_have_three_stages=True,
        all_curves_monotone=True, knot_slope_rate_identity_passed=True,
        pilot_replay_max_rate_error_pp_min=pilot_error,
        normalization_identity_max_error_pp_min=float(np.max(np.abs(rates.normalized_rate_pp_min - rates.raw_rate_pp_min / rates.own_baseline_rate_pp_min * rates.reference_baseline_rate_pp_min))),
        excluded_conditions_absent=not bool(set(curves.condition_id) & excluded),
        simulated_observations_used=0, modeled_rates_separate_from_observed=True,
        original_diagnostic_figure_added_to_dataset=False, active_training_unchanged=True)
    manifest = common_manifest(normalizer, hashes, "wind_tunnel")
    manifest.update(scope="All local 50cm wind-tunnel CSV candidates independently audited",
        candidate_files=len(run_audit), included_runs=int(rates.run_id.nunique()),
        observed_stage_rate_rows=len(rates), plotted_conditions=int(curves.condition_id.nunique()),
        level1_2_conditions=int(curves[curves.wind_level.isin([1, 2])].condition_id.nunique()),
        additional_actual_level_conditions=int(curves[~curves.wind_level.isin([1, 2])].condition_id.nunique()),
        curve_stage_rows=len(curves), modeled_stage_rows=int(curves.is_modeled.sum()),
        missing_safe_level1_2_conditions=coverage[coverage.status.eq("no_supported_complete_wind_record")].condition_id.tolist(),
        run_status_counts=pd.DataFrame(run_audit).status.value_counts().to_dict(),
        first_plateau_policy="Remove initial unchanged hovering SOC period through first observed decrease; retain subsequent integer-SOC plateaus",
        fit_method="Independent time-weighted stage OLS with separate intercepts for real hovering blocks",
        time_definition="Only after stage rates are fitted and normalized, time = 60 * reference stage SOC drop / normalized stage rate",
        missing_stage_policy="Complete curve coefficient from same configuration/position wind Medium relative factor, or nearest available stage; never enter observed data table",
        aggregation="Equal run weight within condition, position and stage; only wind data",
        selection="Exclude five unsafe conditions, prepare, empty, registry outlier/merge component, control fault/uncommanded landing and interrupted attempts without a hover reserve endpoint",
        uncertainties=["Only one retained complete record per available condition in this extraction; no repeat-based confidence intervals.",
            "Short and few-SOC-level fitted stages retain weak_support flags.",
            "Missing stages are model completions, not measured data; figures omit extension labels at the user's request, provenance tables retain them.",
            "Each drone retains its own hover to 20%; after other drones land, formation membership can change.",
            "The fixed baseline remains the existing calibration candidate, not independently revalidated by this analysis."])
    write_json(out / "manifest.json", manifest)
    write_json(out / "validation.json", validation)
    (out / "README.md").write_text(f"""# 第二步：50厘米风洞独立三阶段处理

独立检查全部{len(run_audit)}个候选CSV；目前{manifest['included_runs']}组完整可用记录、{manifest['plotted_conditions']}个条件，其中Level1/2共{manifest['level1_2_conditions']}个，另外实际Level3单独保留。{len(rates)}条真实观测阶段率、{len(curves)}条曲线阶段系数，其中{manifest['modeled_stage_rows']}条为缺阶段模型补全。这不是声明所有已采集实验都支持完整训练曲线；每份文件及剔除原因在audit。

## 方法

起飞、定位确认和非hover不参与拟合。每机去除最开始SOC一直不变的时间，从首次真实下降之后保留数据；后续整数SOC平台正常保留。每架保留至其自身首次20%，不因另一架先落地而提前截断本机。按每架实际电池自己的High/Medium/Low边界，分别用时间加权OLS拟合真实hover阶段率，恢复/重启块各有截距，不把缺失时间当飞行。

阶段标准化率 = 本机阶段实测率 / 固定九月份本机电池阶段基线率 × 固定参考电池阶段率。先有率，之后参考阶段时间 = 60 × 阶段SOC差 / 标准化率。参考边界仍是100/82/52/20，没有更改或重拟合baseline，没有按照起始电量不同而直接平均飞行时间。

缺少阶段时，曲线系数采用同配置同位置的风洞Medium相对因子（无Medium时用最近可用阶段）延伸；不加入实测率表。图上不放拓展标签，curve_stage_rates.csv的is_modeled和rate_origin、manifest保留来源。High支撑不足的情况会影响曲线的高电量阶段解释。原SOC实测对照图未加入；diamond_50_head_lv2_002数据保留。

## 文件

- all_wind_three_stage_curves.pdf：逐条件五架/位置的三段曲线；figures/含300dpi PNG及PDF/SVG。
- 50cm_wind_overview.png / .pdf：Level1/2全部条件总览；危险及没有完整可用记录的条件分别显示。
- observed_run_stage_rates.csv：仅真实观测阶段独立拟合，不含缺阶段补全。
- condition_position_stage_means.csv：风洞自己的同条件/位置/阶段均值。
- curve_stage_rates.csv / three_stage_curve_knots.csv：三段模型系数和四个节点。
- condition_coverage.csv：哪些阶段来自实测、哪些模型补全、哪些条件没有完整支撑。
- wind_run_audit.csv / wind_drone_window_audit.csv：文件筛选、各机初始平台删去秒数和各自hover窗口。
- manifest.json / validation.json：只读输入哈希和离线数值校验。

本步骤未读入250厘米实验结果，没有进行两类平均；未更新训练数据、模型、控制或数据库。五项危险配置全部排除。无完整可用Level1/2条件：{', '.join(manifest['missing_safe_level1_2_conditions'])}。Vee侧风有Level3实测，未偷偷改成Level2。
""")
    print(json.dumps({k: manifest[k] for k in ("candidate_files", "included_runs", "observed_stage_rate_rows", "plotted_conditions", "modeled_stage_rows", "missing_safe_level1_2_conditions")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["forward", "wind"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default = OUT_ROOT / ("step1_forward_250cm" if args.step == "forward" else "step2_wind_50cm")
    (process_forward if args.step == "forward" else process_wind)((args.output or default).resolve())
