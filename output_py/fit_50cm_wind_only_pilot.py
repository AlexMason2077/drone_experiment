"""Single-condition wind-only pilot. No forward-flight coefficients are read."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from process_50cm_stage_curves import (
    ROOT, MODEL, NAMES, BatteryNormalizer, plt, sha, write_json,
    fit_blocks, observed_hover_frames, calibrated_rate, plot_curve, exclusion_ids,
)

CID = "diamond_50_head_lv2"
EXPERIMENT = "wind_tunnel_diamond_50_head_lv2_002"
SOURCE = ROOT / "database" / EXPERIMENT / f"{EXPERIMENT}_20260928_232343_all_coordination.csv"
OUT = ROOT / "analysis_results/50cm_wind_only_pilot_20260929" / CID
PREVIOUS = ROOT / "analysis_results/wind_tunnel_three_stage_rate_first_corrected_20260929" / EXPERIMENT / "stage_rates.csv"


def main():
    if OUT.exists():
        raise FileExistsError("Preserve prior analysis outputs")
    assert CID not in exclusion_ids()
    normalizer = BatteryNormalizer.load(MODEL)
    inputs = [SOURCE, MODEL, PREVIOUS, ROOT/"battery_normalization.py", ROOT/"TRAINING_EXCLUSIONS.md",
              Path(__file__), ROOT/"output_py/process_50cm_stage_curves.py"]
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    raw = pd.read_csv(SOURCE, low_memory=False)
    for key, value in dict(experiment_id=EXPERIMENT, run_id="20260928_232343", formation="diamond",
                           wind_direction="head wind", wind_speed="Level2", inter_drone_distance_cm=50).items():
        assert set(raw[key]) == {value}
    observed, completed, windows, knots = [], [], [], []
    for drone, group in raw.groupby("drone_name", sort=True):
        batteries = group.battery_id.dropna().astype(str).unique()
        assert len(batteries) == 1
        battery = batteries[0]; own = normalizer.curve_for(battery, drone)
        frame, audit = observed_hover_frames(group)
        assert len(audit) == 1 and audit[0]["own_20_percent_observed"]
        position = int(drone.split("_")[-1])
        windows.append(dict(position=position, battery_id=battery, **audit[0]))
        stages = {}
        for i, (stage, upper, lower) in enumerate(zip(NAMES, own.boundaries, own.boundaries[1:])):
            fit = fit_blocks(frame[frame.raw_soc.between(lower, upper)])
            if fit is None:
                continue
            row = dict(condition_id=CID, protocol="wind_tunnel", experiment_id=EXPERIMENT,
                       run_id="20260928_232343", position=position, drone_name=drone,
                       battery_id=battery, stage=stage, rate_origin="measured_wind_independent_stage_fit",
                       **calibrated_rate(fit, battery, drone, i, normalizer))
            observed.append(row); stages[stage] = row
        assert "Medium" in stages and "Low" in stages
        if "High" not in stages:
            factor = stages["Medium"]["relative_drain_factor"]
            stages["High"] = dict(condition_id=CID, protocol="wind_tunnel", experiment_id=EXPERIMENT,
                run_id="20260928_232343", position=position, drone_name=drone, battery_id=battery,
                stage="High", raw_rate_pp_min=np.nan, normalized_rate_pp_min=factor*normalizer.reference.rates_pp_min[0],
                relative_drain_factor=factor, rate_origin="modeled_missing_high_from_wind_medium_factor",
                completion_anchor_stage="Medium", sample_count=0)
        clock = 0.0
        knots.append(dict(position=position, knot=0, time_s=clock, normalized_soc=100))
        for i, stage in enumerate(NAMES):
            row = stages[stage].copy()
            upper, lower = normalizer.reference.boundaries[i:i+2]
            duration = 60*(upper-lower)/row["normalized_rate_pp_min"]
            row.update(reference_upper_soc=upper, reference_lower_soc=lower, stage_duration_s=duration)
            completed.append(row); clock += duration
            knots.append(dict(position=position, knot=i+1, time_s=clock, normalized_soc=lower))
    observed = pd.DataFrame(observed); completed = pd.DataFrame(completed); knots = pd.DataFrame(knots)
    old = pd.read_csv(PREVIOUS)
    check = completed.merge(old[["position", "stage", "normalized_rate_pp_min"]], on=["position", "stage"],
                            suffixes=("_new", "_old"), validate="one_to_one")
    error = float(np.max(np.abs(check.normalized_rate_pp_min_new-check.normalized_rate_pp_min_old)))
    assert error < 1e-8 and len(check) == 15 and len(observed) == 13
    for position, g in knots.groupby("position"):
        g = g.sort_values("knot")
        assert np.diff(g.time_s).min() > 0 and np.diff(g.normalized_soc).max() < 0
        target = completed[completed.position.eq(position)].set_index("stage").loc[list(NAMES)]
        assert np.allclose(-60*np.diff(g.normalized_soc)/np.diff(g.time_s), target.normalized_rate_pp_min)
    OUT.mkdir(parents=True)
    observed.to_csv(OUT/"observed_stage_rates.csv", index=False)
    completed.to_csv(OUT/"curve_stage_rates.csv", index=False)
    knots.to_csv(OUT/"curve_knots.csv", index=False)
    pd.DataFrame(windows).to_csv(OUT/"wind_window_audit.csv", index=False)
    summary = completed.pivot(index="position", columns="stage", values="normalized_rate_pp_min").reindex(columns=NAMES)
    summary["reference_100_to_20_s"] = completed.groupby("position").stage_duration_s.sum()
    summary.to_csv(OUT/"five_position_summary.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":12, "pdf.fonttype":42, "svg.fonttype":"none"})
    fig, ax = plt.subplots(figsize=(11,6.8)); plot_curve(ax, completed, normalizer)
    for stage, upper, lower in zip(NAMES, normalizer.reference.boundaries, normalizer.reference.boundaries[1:]):
        ax.text(.985, (upper+lower)/2, stage, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=11, color="#64757F", bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
    fig.suptitle("Diamond · 50 cm · Headwind · Level 2", x=.11, y=.966, ha="left", fontsize=17, fontweight="semibold")
    fig.text(.11, .912, "Wind-tunnel data · Independent stage rates on the reference-battery scale", fontsize=10.5, color="#52636C")
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.16), ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.11, right=.96, bottom=.23, top=.855)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT/f"wind_only_three_stage.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    assert all(sha(ROOT/p) == value for p, value in hashes.items())
    write_json(OUT/"manifest.json", dict(status="wind_only_single_condition_pilot_awaiting_review", condition_id=CID,
        source_experiment=EXPERIMENT, input_sha256=hashes, normalization_model_version=normalizer.version,
        reference_boundaries=list(normalizer.reference.boundaries), included_protocols=["wind_tunnel"],
        forward_250cm_results_used=False, averaging_applied=False, bulk_processing_paused=True,
        modeled_rows=2, modeled_high_policy="P3/P4 use corresponding wind-only Medium relative drain factor",
        initial_plateau_policy="Remove initial constant hovering SOC period; retain later integer-SOC plateaus",
        time_definition="Reference 100→82→52→20 stage time computed from fixed normalized rates",
        raw_diagnostic_figure_in_dataset=False, active_training_dataset_modified=False))
    write_json(OUT/"validation.json", dict(only_one_condition=True, no_250cm_coefficients_read=True,
        observed_rows=13, curve_stage_rows=15, original_wind_pilot_max_rate_error=error,
        all_curves_monotone=True, knot_rate_identity_checked=True, raw_inputs_unchanged=True))
    (OUT/"README.md").write_text("""# Diamond / 50cm / 顶风 / Level2：仅风洞试处理

当前结果只使用 wind_tunnel_diamond_50_head_lv2_002 的实测风洞记录和已有电池基线，未读取或合并250cm耗电系数；无双来源平均。其他配置暂不继续，未更新训练数据或在线模型。

各飞机去除hover初始不掉电平台，保留各自到自身首次20%的数据。按各自物理电池的三个阶段独立拟合耗电率，再按对应基线率换算到既有参考电池；阶段时间随后由SOC下降/阶段率计算。参考边界100/82/52/20。

P3/P4没有High实测，因此High采用同位置、仅风洞Medium的相对耗电因子完成。模型系数和13行实测阶段率分开保存；图不标拓展标签。P2/P5 High只含两档SOC，支撑较弱。每机保留其他飞机先落地之后的hover，因此后段未必仍有五架同时飞行。

图为3300×2040 PNG，另有PDF/SVG矢量版。图中时间是统一参考电池100%→20%的计算时间；不是不同起始SOC的原始飞行时长。原始实测SOC对照图没有加入此结果集。旧参考电池模型保留原candidate状态，本次未做独立实机验证。
""")
    receipt = dict(schemaVersion=1, items=[dict(id="wind-only-three-stage-pilot", title="Diamond 50cm顶风Level2：仅风洞三阶段曲线",
        queries=[dict(id="wind-only-rates", source=dict(label="wind_tunnel_diamond_50_head_lv2_002 · 20260928_232343",
            filters=["仅风洞实测与已有电池基线", "配置：Diamond / 50cm / Headwind / Level2", "初始不掉电平台去除；各飞机保留至自身20%"],
            caveats=["P3/P4 High由同位置风洞Medium相对耗电率完成，不是实测阶段。", "P2/P5 High各仅两档SOC。", "横轴为统一参考电池100%到20%的计算时间。", "其他飞机先落地后的本机hover保留，后段未必代表五架同时悬停。"],
            metricDefinitions=[dict(id="normalized-stage-rate", definition="各阶段独立拟合的风洞率先除以本电池对应阶段基线率，再乘参考电池对应阶段率；本图不使用250cm结果。")]),
            rows=completed[["position", "stage", "normalized_rate_pp_min", "stage_duration_s", "rate_origin"]].to_dict("records"),
            columns=[dict(field="position", label="位置"), dict(field="stage", label="阶段"),
                     dict(field="normalized_rate_pp_min", label="标准化耗电率（pp/min）"),
                     dict(field="stage_duration_s", label="参考阶段时间（秒）"), dict(field="rate_origin", label="系数来源")])])])
    write_json(OUT/"answer_sources.json", receipt)
    print(summary.round(6).to_string())
    print(str(OUT/"wind_only_three_stage.png"))


if __name__ == "__main__":
    main()
