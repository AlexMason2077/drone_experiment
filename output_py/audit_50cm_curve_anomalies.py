"""Read-only investigation of the four questioned wind-tunnel curve exports.

Writes diagnostic evidence only. Does not replace curves, change baselines,
alter raw records, change flight control, or build a training dataset.
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "analysis_results/50cm_selection_corrected_fixed_baseline_20260929"
OUT = ROOT / "analysis_results/50cm_curve_anomaly_review_20260929"
MODEL = ROOT / "analysis_results/battery_normalization_v3_with_b15_20260909/model.json"
CELLS = ["vee_50_tail_lv2", "front_50_tail_lv2", "front_50_side_lv1", "echalon_50_head_lv1"]
STAGES = ["High", "Medium", "Low"]
HASHES = {}


def read_csv(path, **kwargs):
    path = Path(path)
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, **kwargs)


def independently_selected_samples(g, recovered):
    """Reconstruct the existing export's selected samples from original rows."""
    g = g.sort_values("elapsed_time").reset_index(drop=True)
    nominal = g.phase.eq("wind_tunnel_hover")
    if recovered:
        ground = g.phase.isin(["wind_tunnel_uncommanded_landed", "wind_tunnel_landed",
                              "wind_tunnel_landing_20_percent"])
        end_ground = int(ground[ground].index.min()) if ground.any() else len(g)
        fault = g.phase.eq("wind_tunnel_control_fault") & g.mid.isin(range(1, 9))
        fault &= g.z.between(60, 100) & g.tof.between(60, 100)
        eligible = (nominal | fault) & (g.index < end_ground)
        blocks = (~eligible).cumsum() + g.mid.ne(g.mid.shift()).cumsum()
    else:
        eligible = nominal
        blocks = (~eligible).cumsum()
    h = g[eligible].copy()
    drops = h.index[h.battery.diff().lt(0)]
    if not len(drops):
        return h.iloc[:0]
    start = int(drops[0])
    floor = h.index[h.battery.le(20)]
    stop = int(floor[0]) if len(floor) else int(h.index[-1])
    h = h.loc[(h.index >= start) & (h.index <= stop) & h.battery.between(20, 100)].copy()
    h["fit_time_s"] = h.elapsed_time - float(g.loc[start, "elapsed_time"])
    h["block_id"] = blocks.loc[h.index]
    return h


def independent_wls(samples):
    """Full weighted design matrix with one intercept per actual block."""
    blocks = [b.sort_values("fit_time_s") for _, b in samples.groupby("block_id", sort=False) if len(b) >= 3]
    if not blocks or samples.battery.nunique() < 2:
        return None
    design, values = [], []
    for i, b in enumerate(blocks):
        t = b.fit_time_s.to_numpy(float)
        y = b.battery.to_numpy(float)
        dt = np.diff(t)
        assert (dt > 0).all()
        w = np.r_[dt[0], dt[:-1] + dt[1:], dt[-1]] / 2
        x = np.zeros((len(b), len(blocks) + 1))
        x[:, 0] = t - t[0]
        x[:, i + 1] = 1
        design.append(x * np.sqrt(w[:, None]))
        values.append(y * np.sqrt(w))
    coef = np.linalg.lstsq(np.vstack(design), np.concatenate(values), rcond=None)[0]
    rate = float(-60 * coef[0])
    return rate if rate > 0 else None


def trim_terminal_boundary(samples, lower):
    pieces = []
    for _, b in samples.groupby("block_id", sort=False):
        b = b.sort_values("fit_time_s")
        hit = b[b.battery.le(lower)]
        pieces.append(b[b.fit_time_s.le(float(hit.fit_time_s.iloc[0]))] if len(hit) else b)
    return pd.concat(pieces) if pieces else samples


def first_crossing_duration(df, upper, lower):
    h = df.sort_values("elapsed_time")
    start = h[h.battery.le(upper)]
    end = h[h.battery.le(lower)]
    if not len(start) or not len(end) or h.battery.max() < upper:
        return None
    return float(end.elapsed_time.iloc[0] - start.elapsed_time.iloc[0])


def main():
    if OUT.exists():
        raise FileExistsError("Preserve this audit; use a separate date/directory for another run")
    OUT.mkdir(parents=True)
    baseline_hash = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert baseline_hash == "5ea37d52e43b4c283f3ce6f956b2b669c1878030e32afc92fc9749fe66af0fe5"
    HASHES[str(MODEL.relative_to(ROOT))] = baseline_hash
    model = json.loads(MODEL.read_text())
    rates = read_csv(PACKAGE / "wind_tunnel/observed_run_stage_rates.csv")
    support = read_csv(PACKAGE / "wind_tunnel/stage_observation_support.csv")
    curves = read_csv(PACKAGE / "wind_tunnel/curve_stage_rates.csv")
    run_audit = read_csv(PACKAGE / "wind_tunnel/wind_run_audit.csv")
    registry_path = ROOT / "database/experiment_registry.json"
    HASHES[str(registry_path.relative_to(ROOT))] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    registry = json.loads(registry_path.read_text())["experiments"]
    registry = {r["experiment_id"]: r for r in (registry.values() if isinstance(registry, dict) else registry)}
    selected_rates = rates[rates.condition_id.isin(CELLS)].copy()
    cache, frames, details = {}, {}, []
    for row in selected_rates.to_dict("records"):
        file = row["source_file"]
        if file not in cache:
            cache[file] = read_csv(ROOT / file)
        key = (file, row["drone_name"])
        if key not in frames:
            g = cache[file][cache[file].drone_name.eq(row["drone_name"])]
            recovered = row["selection_method"] == "real_hover_and_telemetry_verified_fault_windows"
            frames[key] = independently_selected_samples(g, recovered)
        f = frames[key]
        part = f[f.battery.between(row["own_lower_soc"], row["own_upper_soc"])].copy()
        fit = independent_wls(part)
        assert fit is not None
        boundary_trimmed = trim_terminal_boundary(part, row["own_lower_soc"])
        trimmed_rate = independent_wls(boundary_trimmed)
        span = float(part.battery.max() - part.battery.min())
        ref_width = float(model["reference"]["boundaries_soc"][STAGES.index(row["stage"])] -
                          model["reference"]["boundaries_soc"][STAGES.index(row["stage"]) + 1])
        details.append(dict(condition_id=row["condition_id"], position=row["position"], stage=row["stage"],
            experiment_id=row["experiment_id"], run_id=row["run_id"], source_file=file,
            battery_id=row["battery_id"], observed_upper_soc=float(part.battery.max()),
            observed_lower_soc=float(part.battery.min()), observed_soc_drop_pp=span,
            exported_fit_duration_s=row["duration_s"], raw_rate_pp_min=row["raw_rate_pp_min"],
            independent_raw_rate_pp_min=fit, replay_error_pp_min=abs(fit-row["raw_rate_pp_min"]),
            own_baseline_rate_pp_min=row["own_baseline_rate_pp_min"],
            normalized_rate_pp_min=row["normalized_rate_pp_min"], weak_support=row["weak_support"],
            fault_phase_samples=int(part.phase.eq("wind_tunnel_control_fault").sum()),
            different_pad_samples=int(part.mid.ne(part.target_pad).sum()),
            boundary_trim_only_raw_rate_pp_min=trimmed_rate,
            boundary_trim_only_normalized_rate_pp_min=(trimmed_rate / row["own_baseline_rate_pp_min"] *
                row["reference_baseline_rate_pp_min"]) if trimmed_rate else None,
            terminal_boundary_repeated_samples_removed=len(part) - len(boundary_trimmed),
            reference_stage_width_to_observed_drop_ratio=ref_width / span if span else None))
    trace = pd.DataFrame(details)
    assert trace.replay_error_pp_min.max() < 1e-8
    trace.to_csv(OUT / "exported_stage_source_trace.csv", index=False)
    selected_rates.to_csv(OUT / "questioned_observed_stage_rates_snapshot.csv", index=False)
    curves[curves.condition_id.isin(CELLS)].to_csv(OUT / "questioned_curve_coefficients_snapshot.csv", index=False)
    support[support.condition_id.isin(CELLS)].to_csv(OUT / "questioned_stage_support_snapshot.csv", index=False)

    # Inventory each numbered record, including earlier partial runs that the
    # exporter rejected in their entirety. This is source inspection, not a
    # declaration that these records are valid training observations.
    profiles, overlooked = [], []
    for cid in CELLS:
        for file in sorted((ROOT / "database").glob("wind_tunnel_" + cid + "_0*/*_all_coordination.csv")):
            rel = str(file.relative_to(ROOT))
            if rel not in cache:
                header = pd.read_csv(file, nrows=0).columns
                cols = [c for c in ["drone_name", "battery_id", "elapsed_time", "phase", "battery", "mid",
                                   "target_pad", "z", "tof", "position_error_dist"] if c in header]
                cache[rel] = read_csv(file, usecols=cols)
            raw = cache[rel]
            if raw.empty:
                continue
            prior = run_audit[run_audit.source_file.eq(rel)]
            for dn, g in raw.groupby("drone_name"):
                hover = g[g.phase.eq("wind_tunnel_hover")]
                profiles.append(dict(condition_id=cid, source_file=rel, experiment_id=file.parent.name,
                    drone_name=dn, battery_id=str(g.battery_id.iloc[0]), raw_sample_count=len(g),
                    duration_s=float(g.elapsed_time.max()-g.elapsed_time.min()),
                    raw_start_soc=float(g.battery.iloc[0]), raw_end_soc=float(g.battery.iloc[-1]),
                    hover_sample_count=len(hover), hover_upper_soc=float(hover.battery.max()) if len(hover) else None,
                    hover_lower_soc=float(hover.battery.min()) if len(hover) else None,
                    hover_different_pad_samples=int(hover.mid.ne(hover.target_pad).sum()),
                    hover_position_error_p95_cm=float(hover.position_error_dist.quantile(.95)) if len(hover) else None,
                    fault_sample_count=int(g.phase.eq("wind_tunnel_control_fault").sum()),
                    prior_file_status=str(prior.status.iloc[0]) if len(prior) else "not_in_prior_run_audit",
                    registry_outlier=bool(registry.get(file.parent.name, {}).get("is_outlier", False))))
                if cid == "front_50_tail_lv2" and "20260908_193108" in rel:
                    # Use original nominal-hover rule, without the erroneous
                    # whole-file exclusion triggered by another drone landing.
                    sample = independently_selected_samples(g, recovered=False)
                    battery = model["batteries"][g.battery_id.iloc[0]]
                    for i, stage in enumerate(STAGES):
                        part = sample[sample.battery.between(battery["boundaries_soc"][i+1], battery["boundaries_soc"][i])]
                        rate = independent_wls(part)
                        if rate:
                            overlooked.append(dict(condition_id=cid, position=int(dn.split("_")[-1]), stage=stage,
                                source_file=rel, raw_rate_pp_min=rate,
                                normalized_rate_pp_min=rate/battery["rates_pp_min"][i]*model["reference"]["rates_pp_min"][i],
                                observed_upper_soc=float(part.battery.max()), observed_lower_soc=float(part.battery.min()),
                                use_status="review_candidate_only_not_a_replacement_curve"))
    pd.DataFrame(profiles).to_csv(OUT / "all_numbered_record_profiles.csv", index=False)
    pd.DataFrame(overlooked).to_csv(OUT / "overlooked_front_tail_early_stage_candidates.csv", index=False)

    b13 = read_csv(ROOT / model["batteries"]["B13"]["source"])
    side_row = selected_rates[selected_rates.condition_id.eq("front_50_side_lv1") & selected_rates.position.eq(3) & selected_rates.stage.eq("High")].iloc[0]
    side = cache[side_row.source_file]
    side = side[side.drone_name.eq("drone_3")]
    crossing = {"battery": "B13", "baseline_95_to_81_s": first_crossing_duration(b13, 95, 81),
        "baseline_89_to_81_s": first_crossing_duration(b13, 89, 81),
        "wind_89_to_81_s": first_crossing_duration(side, 89, 81),
        "interpretation": "Raw reported SOC already differs. This does not prove a physical energy cause or validate full-stage extrapolation."}
    (OUT / "b13_same_soc_range_comparison.json").write_text(json.dumps(crossing, indent=2))

    # Diagnostic comparisons only. These measured-SOC plots are not part of
    # a training dataset or a replacement for normalized three-stage curves.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1), dpi=200)
    vee = trace[trace.condition_id.eq("vee_50_tail_lv2") & trace.position.eq(1) & trace.stage.eq("Low")].iloc[0]
    vf = frames[(vee.source_file, "drone_1")]
    vpart = vf[vf.battery.between(50, 51)]
    tt = vpart.fit_time_s - vpart.fit_time_s.iloc[0]
    axes[0].step(tt, vpart.battery, where="post", color="#087ca7", linewidth=2)
    axes[0].set(xlabel="Observed low-stage window (s)", ylabel="Reported SOC (%)", ylim=(49.7, 51.3),
                title="Vee / tail / L2: D1\nOnly one percentage point in 4.078 s")
    for df, label, color in [(b13, "September B13 baseline", "#6b7280"), (side, "Front / side / L1: D3", "#009e73")]:
        h = df[df.battery.between(81, 89)].sort_values("elapsed_time")
        stop = h[h.battery.le(81)].elapsed_time.iloc[0]
        h = h[h.elapsed_time.le(stop)]
        axes[1].step(h.elapsed_time-h.elapsed_time.iloc[0], h.battery, where="post", label=label, color=color, linewidth=2)
    axes[1].set(xlabel="Time since first reported 89% (s)", ylabel="Reported SOC (%)",
                title="B13: same reported 89% to 81%\n15.071 s in baseline; 77.946 s in wind run")
    axes[1].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(OUT / "raw_evidence_two_anomalies.png", dpi=250)
    fig.savefig(OUT / "raw_evidence_two_anomalies.pdf")
    plt.close(fig)

    e = selected_rates[selected_rates.condition_id.eq("echalon_50_head_lv1") & selected_rates.stage.eq("High")]
    erows = e[["position", "raw_rate_pp_min", "own_baseline_rate_pp_min", "normalized_rate_pp_min", "duration_s", "weak_support"]]
    erows.to_csv(OUT / "echelon_high_stage_normalization_factors.csv", index=False)
    totals = curves[curves.condition_id.isin(CELLS)].groupby(["condition_id", "position"], as_index=False).agg(
        exported_full_curve_time_s=("stage_duration_s", "sum"), modeled_stages=("is_modeled", "sum"))
    totals["review_status"] = "needs_revision_not_validated_for_performance_comparison"
    totals.to_csv(OUT / "questioned_curve_review_status.csv", index=False)
    findings = [
        dict(id="vee-short-low", priority="P1", condition="vee_50_tail_lv2", finding="D1 Low rate uses only 51 to 50 percent in 4.078 s; the full reference Low stage is 32 percentage points and 580.629 s. Other Vee positions also include nominal-hover samples with unavailable or different assigned pads.", remedy="Do not certify a full Low-stage duration from this one-step trace. Keep raw evidence; use a genuinely supported same-condition D1 Low record if available, otherwise leave the coefficient unsupported."),
        dict(id="front-tail-selection", priority="P1", condition="front_50_tail_lv2", finding="Earlier numbered 001 has actual High and Medium discharge windows. A whole-run rejection caused by one uncommanded landing excluded all its pre-landing windows, while 004 begins at lower SOC and its missing High stages are filled from partial Medium or Low fits.", remedy="Screen actual per-drone, per-episode windows, retaining valid pre-landing observations. Re-estimate from supported real stages without bridging stops or assuming a late stage rate applies to an unobserved stage."),
        dict(id="front-side-high", priority="P1", condition="front_50_side_lv1", finding="D3 High combines partial-SOC full-stage extrapolation with a B13 baseline High rate of 30.962 pp/min. The reported 89 to 81 percent duration is 77.946 s in the wind record and 15.071 s in the baseline. The exporter also repeats the terminal 81-percent plateau in both High and Medium fits.", remedy="Remove duplicated boundary-plateau time. Preserve and investigate the genuine reported-SOC discrepancy; do not alter the fixed baseline or force the normalized curve to agree with peers."),
        dict(id="echelon-missing-stage", priority="P1", condition="echalon_50_head_lv1", finding="Four of five High fits use fewer than 20 s of fitted blocks. All five have partial High SOC coverage; some use fault-labelled or different-pad observations. Nine of fifteen full-curve stages are completed by borrowing an observed stage factor, often High. Raw High rates are similar, but different baseline divisors create a large relative-factor gap that is repeated in missing stages.", remedy="Do not call fault/displaced short windows validated configuration data. Retain diagnostic observations and show missing stage support explicitly in coefficients. A full curve requires supported per-stage data or a separately justified model completion."),
    ]
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False))
    validation = dict(review_status="needs_revision", review_scope=CELLS, observed_stage_rows_replayed=len(trace),
        replay_max_error_pp_min=float(trace.replay_error_pp_min.max()),
        overlooked_front_tail_stage_candidates=len(overlooked),
        fixed_baseline_unchanged=hashlib.sha256(MODEL.read_bytes()).hexdigest() == baseline_hash,
        every_inspected_source_unchanged=all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in HASHES.items()),
        original_exports_unchanged=True, training_data_changed=False, control_code_changed=False,
        new_full_curve_replacements_created=False,
        limitation="This audit explains and flags unreliable inference. It does not prove a physical cause or validate all 25 conditions.")
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False))
    (OUT / "input_hashes.json").write_text(json.dumps(HASHES, indent=2))
    (OUT / "README.md").write_text("""# 四张 50 cm 风洞曲线核查

结论：这四张曲线需要重算，当前不能作为已验证的配置性能比较。原始记录和固定九月份 baseline 均保持不变；本目录是核查证据，不是训练数据，也没有用人为拉近曲线的方式替代结果。

## 原处理链

先去除每架飞机开头不变的 SOC 时间，按各自电池的 High/Medium/Low 区间，拟合 SOC 对实际时间的下降斜率。再计算：归一化阶段耗电率 = 实测拟合耗电率 / 同电池同阶段基线耗电率 × 参考电池同阶段耗电率。阶段时间 = 60 × 参考阶段 SOC 宽度 / 归一化阶段耗电率。缺阶段曾按同位置另一阶段的相对耗电因子补全。

## 已确认的问题

| 图 | 原始依据及失真来源 | 修正方向 |
| --- | --- | --- |
| Vee / tail / Level2 | D1 Low 仅 51%→50%、4.078 秒，推算完整参考 Low 段得到 580.629 秒。弱支撑标志曾没有阻止整段推算；另有名义hover样本失去或切换目标Pad。 | 保留实际短段；没有可靠 Low 数据时，不把外推结果当已确认的完整耗电曲线。 |
| Front / tail / Level2 | `_001` 的真实 High/Medium 悬停窗口，被整次故障／未到20%的筛选方式排除。`_004` 起始低，缺 High 又被局部 Medium/Low 比例填补。 | 逐架、逐实际窗口核查，将有效早期片段纳入，恢复真实阶段覆盖；跨停止时段不拼接时间。 |
| Front / side / Level1 | B13 同一 89%→81% 区间，原始基线 15.071 秒，风洞 77.946 秒；归一化分母使这项 SOC 差异进一步显现。81% 平台约23秒还同时进入 High 和 Medium。 | 先修正边界重复使用；保留原始 SOC 差异作为待解释证据，不调整 baseline 来让图形接近。 |
| Echelon / head / Level1 | 部分斜率来自故障或不同 Pad 的短记录；中低阶段大量缺失。High 的归一化因子被复制到缺失阶段，放大整个时间差。 | 不把定位失稳后的数据认定为该配置的稳定耗电；短记录可用，但须确实支撑对应阶段，不能据此确认未观察阶段。 |

### 覆盖与限制

范围仅为用户指出的四张图，共20条整曲线；已逐项复核该范围内导出斜率所用原始样本，并检查全部对应编号记录。复核计算相同，只证明之前的算术可复现，不证明其统计方法、实验归属和外推合理。本次没有给出新的完整曲线，更没有将未知阶段填成新的“实测数据”。

### 分析质量核查

| Category | Observed defects | Assessment |
| --- | --- | --- |
| 分析用途与完整性 | 4 / 4 | 四张图均存在不足以支持整曲线解释的阶段估计或数据筛选问题。 |
| 表达清楚程度 | 4 / 4 | 图只显示完整实线，无法从图上判断有限实测段及缺段补全。原系数表有来源标志，但不足以替代解释。 |
| 图形呈现 | 0 / 4 | 图片可读；未用视觉接近程度判断数据正确。 |

| Category | Observed defects | Assessment |
| --- | --- | --- |
| 来源与实验归属 | 3 / 4 | Front tail 整次筛选漏掉实际早期窗口；Echelon及Vee存在故障或失去／切换Pad的样本，稳定编队未确认。 |
| 数值计算 | 1 / 4 | Front side 的分段边界平台重复使用。其他式子可复现，仍不能证明估计可靠。 |
| 图表与数值一致 | 0 / 4 | 图片里的异常时间对应原导出系数，没有证据指向简单绘图错位。 |
| 来源明细 | 0 / 4 | 原文件、斜率、归一化分母及阶段时间已追溯；本次记录各实际 SOC 范围与拟合窗口。 |
| 跨产物一致 | 0 / 4 | 核查输出与原包数字一致；前进250 cm没有纳入本次核查或平均。 |
| 数据质量控制 | 4 / 4 | 短段、部分 SOC 覆盖、边界平台、缺段补全或失稳记录的控制不足。 |
| 结论支持 | 4 / 4 | 当前四张图不能证明无人机之间真实存在图示全部耗尽时间差。 |

计数分母是本次四张图，并非整个50 cm数据集已全部通过验证。未观察到错误不等于实机或统计验证完成。

## 可复核证据

- `exported_stage_source_trace.csv`：每个原导出斜率的实际 SOC 范围、时间、独立重算、故障／Pad标志、边界单项敏感性。边界敏感性只展示该单项处理影响，不是推荐最终结果。
- `all_numbered_record_profiles.csv`：这四个条件全部编号文件逐架概要，包括早期记录，不要求必须到20%。
- `overlooked_front_tail_early_stage_candidates.csv`：此前遗漏的真实早期阶段，只是候选；没有替换整曲线。
- `b13_same_soc_range_comparison.json`：固定基线与风洞同一 SOC 范围的原始时间。
- `echelon_high_stage_normalization_factors.csv`：短 High 阶段的原斜率、基线分母和归一化值。
- `questioned_curve_review_status.csv`：四张图当前需重算的状态；不删除原始记录，不新增危险配置排除。
- `raw_evidence_two_anomalies.png/pdf`：仅核查用的实测图，不进入数据集。
- `validation.json`、`input_hashes.json`：原始来源只读核对和复核边界。
""", encoding="utf-8")
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
