"""Independent read-only replay and SOC-span audit of the two separate products.

Writes supplementary provenance only; never averages the two data sources.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from process_50cm_stage_curves import (
    ROOT, MODEL, NAMES, BatteryNormalizer, observed_hover_frames, fit_blocks,
    sha, write_json, exclusion_ids,
)

OUT = ROOT / "analysis_results/separate_baseline_processing_20260929"


def main():
    normalizer = BatteryNormalizer.load(MODEL)
    forward = OUT / "step1_forward_250cm"
    wind = OUT / "step2_wind_50cm"
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (forward, wind)]
    for manifest in manifests:
        assert manifest["normalization_model_sha256"] == normalizer.sha256
        assert not manifest["cross_protocol_averaging_applied"]
        assert all(sha(ROOT / p) == h for p, h in manifest["source_hashes"].items())
    assert manifests[0]["included_protocols"] == ["forward_250cm"]
    assert manifests[1]["included_protocols"] == ["wind_tunnel"]

    rates = pd.read_csv(wind / "observed_run_stage_rates.csv")
    frames, supports = {}, []
    errors = []
    for source, group in rates.groupby("source_file", sort=False):
        raw = pd.read_csv(ROOT / source, low_memory=False)
        for drone, dg in raw.groupby("drone_name", sort=False):
            if drone not in set(group.drone_name):
                continue
            frame, windows = observed_hover_frames(dg, merged="_merged_" in source)
            frames[(source, drone)] = (frame, windows)
        for row in group.itertuples():
            frame, windows = frames[(source, row.drone_name)]
            stage_i = NAMES.index(row.stage)
            own = normalizer.curve_for(row.battery_id, row.drone_name)
            upper, lower = own.boundaries[stage_i:stage_i+2]
            part = frame[frame.raw_soc.between(lower, upper)]
            fit = fit_blocks(part)
            errors.append(abs(fit["raw_rate_pp_min"] - row.raw_rate_pp_min))
            actual_upper, actual_lower = float(part.raw_soc.max()), float(part.raw_soc.min())
            supports.append(dict(condition_id=row.condition_id, experiment_id=row.experiment_id,
                run_id=row.run_id, drone_name=row.drone_name, position=row.position,
                battery_id=row.battery_id, stage=row.stage, own_stage_upper_soc=upper,
                own_stage_lower_soc=lower, observed_upper_soc=actual_upper,
                observed_lower_soc=actual_lower, missing_upper_soc_pp=max(0., upper-actual_upper),
                missing_lower_soc_pp=max(0., actual_lower-lower),
                stage_span_fully_observed=bool(actual_upper >= upper and actual_lower <= lower),
                own_20_endpoint_observed=any(w.get("own_20_percent_observed", False) for w in windows),
                initial_unchanged_SOC_seconds_removed=sum(w.get("initial_plateau_removed_s", 0.) for w in windows),
                fitted_rate_from_real_observations=True,
                full_reference_stage_duration_is_calculated=True))
    support = pd.DataFrame(supports)
    assert len(support) == len(rates) and max(errors) < 1e-8
    support.to_csv(wind / "stage_observation_support.csv", index=False)
    endpoint = support[~support.own_20_endpoint_observed][
        ["experiment_id", "drone_name", "observed_lower_soc"]].groupby(
            ["experiment_id", "drone_name"]).observed_lower_soc.min().reset_index()
    wmanifest = manifests[1]
    wmanifest["observation_support_audit"] = "stage_observation_support.csv"
    wmanifest["partial_stage_policy"] = "Fitted slopes use real samples in the available own-stage SOC span; full reference-stage curves and durations are model calculations, including unobserved edges. Missing entire stages remain separately marked is_modeled."
    wmanifest["drones_without_own_20_endpoint"] = endpoint.to_dict("records")
    wmanifest["independent_raw_stage_replay_max_error_pp_min"] = max(errors)
    wmanifest["supplementary_audit_code"] = dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__))
    write_json(wind / "manifest.json", wmanifest)
    v = json.loads((wind / "validation.json").read_text())
    v["all_293_wind_raw_stage_rates_independently_replayed"] = True
    v["raw_stage_replay_max_error_pp_min"] = max(errors)
    v["partial_SOC_support_traced"] = True
    write_json(wind / "validation.json", v)
    readme = (wind / "README.md").read_text().replace("组完整可用记录", "组经过筛选可用的记录")
    readme = readme.replace("每架保留至其自身首次20%", "有本机20%终点时保留至其首次20%，否则保留至实际hover结束")
    readme += "\n## 实测阶段支撑复核\n\nstage_observation_support.csv记录每个阶段实际覆盖的SOC范围和初始平台删除秒数。wind_tunnel_front_50_tail_lv2_004的D1在38%结束；其低电量段只由实际可用区间拟合，延伸到20%的时间是模型计算。缺整段的22条补全，与有部分真实阶段但把阶段延伸到参考边界的情况分开记录。全部293条风洞原始阶段率已从各自原CSV重新拟合核对；最大误差小于1e-8 pp/min。\n"
    (wind / "README.md").write_text(readme)
    fr = pd.read_csv(forward / "adjusted_run_drone_rates.csv")
    bridges = pd.read_csv(forward / "baseline_stage_transfer_factors.csv")
    expected = fr.observed_raw_rate_pp_min * fr.old_to_new_position_scale
    assert np.allclose(expected, fr.adjusted_raw_rate_on_new_battery_pp_min)
    assert np.allclose(fr.normalized_rate_pp_min,
        fr.observed_raw_rate_pp_min / fr.old_baseline_rate_pp_min * normalizer.reference.rates_pp_min[1])
    for table in (fr, rates):
        assert not set(table.condition_id) & exclusion_ids()
    original_model = manifests[0]["normalization_model_sha256"]
    assert sha(MODEL) == original_model
    check = dict(status="passed_independent_offline_replay", baseline_model_unchanged=True,
        both_products_independent=True, averaging_applied=False,
        wind_observed_stages_replayed=len(support), wind_replay_max_error_pp_min=max(errors),
        wind_partial_stage_spans=int((~support.stage_span_fully_observed).sum()),
        wind_drones_without_own_20_endpoint=endpoint.to_dict("records"),
        forward_transfer_and_normalization_verified=True, active_training_unchanged=True)
    write_json(OUT / "independent_validation.json", check)
    (OUT / "README.md").write_text("""# 两个独立处理步骤：固定九月份baseline

目前完成的是两套独立候选结果；未进行250厘米与风洞的平均，也未写入训练数据或在线模型。

1. step1_forward_250cm/：133组原有250厘米实验，625条Medium率、54条件（50cm 24条件 / 75cm 30条件）。按同无人机旧电池、同SOC阶段的基线，换算到九月份目标及固定参考尺度；实际前进时间不变。40条无对应阶段前进数据保留audit，当前危险配置排除。
2. step2_wind_50cm/：138个候选文件逐一筛选，21组可用、21条件，293条实测阶段率及22条缺整段模型补全。初始不变电量时间去除；各自独立拟合阶段率，然后计算参考100/82/52/20的时间。
3. 后续再平均：本次未执行。将来只有相同条件/位置/阶段且两套都有观测率时，才有两类实测平均依据。目前旧250厘米结果只支持Medium。

## 先看这个例子

- step1_forward_250cm/figures/diamond_50_head_lv2_forward_medium.png：历史前进数据换算后的Medium参考曲线。
- step2_wind_50cm/figures/diamond_50_head_lv2_wind_three_stage.png：同条件风洞三阶段曲线。
- 两套目录分别有逐条件PDF、系数CSV、来源audit与数值校验；曲线时间由耗电率算出，不直接平均实验实际飞行时间。

缺阶段和部分阶段延伸保留在来源表；图没有拓展文字标签。特别是P5旧B15映射到当前B12、日期间负载比例稳定，属于换算假设，不是已经验证的电池老化原因。没有改固定model、实测原件、控制器、训练集，也没有Git push。

复现先运行output_py/process_separate_forward_wind_latest_baseline.py的forward、wind两个独立命令（使用新的输出目录保留既有产物）；再参考output_py/audit_separate_baseline_outputs.py复核原始率和实际SOC支撑。
""")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
