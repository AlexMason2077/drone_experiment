"""Export the approved independent analyses as a complete 50-cm-only package.

Verify source inventories and fingerprints before reusing the reviewed fits.
No cross-protocol averaging, re-calibration, training, aircraft or app actions.
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
    ROOT, MODEL, NAMES, CELL, GRAIN, sha, write_json, exclusion_ids,
    BatteryNormalizer, plt,
)
from process_separate_forward_wind_latest_baseline import forward_figure

SOURCE = ROOT / "analysis_results/separate_baseline_processing_20260929"
DEFAULT_OUT = ROOT / "analysis_results/50cm_processed_fixed_baseline_20260929"


def verify_sources():
    manifests = {}
    source_hashes = {}
    for protocol, sub in [("forward_250cm", "step1_forward_250cm"), ("wind_tunnel", "step2_wind_50cm")]:
        folder = SOURCE / sub
        m = json.loads((folder / "manifest.json").read_text())
        assert m["included_protocols"] == [protocol]
        assert m["normalization_model_sha256"] == sha(MODEL)
        assert not m["cross_protocol_averaging_applied"]
        for relative_path, digest in m["source_hashes"].items():
            if sha(ROOT / relative_path) != digest:
                raise ValueError(f"Input changed; recompute the affected source first: {relative_path}")
            source_hashes[relative_path] = digest
        manifests[protocol] = m
    inventory = {str(p.relative_to(ROOT)) for p in (ROOT / "database").glob("wind_tunnel_*_50_*/*_all_coordination.csv")}
    prior = {p for p in manifests["wind_tunnel"]["source_hashes"]
             if p.startswith("database/wind_tunnel") and p.endswith("_all_coordination.csv")}
    assert inventory == prior, "Wind file inventory changed; extract new files before exporting."
    check = json.loads((SOURCE / "independent_validation.json").read_text())
    assert check["status"] == "passed_independent_offline_replay"
    for sub in ["step1_forward_250cm", "step2_wind_50cm"]:
        folder = SOURCE / sub
        for p in folder.rglob("*"):
            if p.is_file():
                source_hashes[str(p.relative_to(ROOT))] = sha(p)
    source_hashes[str((SOURCE / "independent_validation.json").relative_to(ROOT))] = sha(SOURCE / "independent_validation.json")
    for p in [Path(__file__), ROOT / "output_py/audit_separate_baseline_outputs.py"]:
        source_hashes[str(p.relative_to(ROOT))] = sha(p)
    return manifests, source_hashes, inventory


def main(out):
    if out.exists():
        raise FileExistsError(f"Preserve prior outputs; choose a new folder: {out}")
    manifests, hashes, inventory = verify_sources()
    normalizer = BatteryNormalizer.load(MODEL)
    excluded = exclusion_ids()
    fsource, wsource = SOURCE / "step1_forward_250cm", SOURCE / "step2_wind_50cm"
    fr = pd.read_csv(fsource / "adjusted_run_drone_rates.csv")
    fr = fr[fr.spacing_cm.eq(50)].copy()
    fm = pd.read_csv(fsource / "condition_position_medium_means.csv")
    fm = fm[fm.spacing_cm.eq(50)].copy()
    fa = pd.read_csv(fsource / "forward_record_audit.csv")
    fa = fa[fa.spacing_cm.eq(50)].copy()
    fp = pd.read_csv(fsource / "adjusted_forward_curve_points.csv")
    fp = fp.merge(fr[["experiment_directory", "run_id", "drone_name"]],
        on=["experiment_directory", "run_id", "drone_name"], how="inner", validate="many_to_one")
    wr = pd.read_csv(wsource / "observed_run_stage_rates.csv")
    wm = pd.read_csv(wsource / "condition_position_stage_means.csv")
    wc = pd.read_csv(wsource / "curve_stage_rates.csv")
    wk = pd.read_csv(wsource / "three_stage_curve_knots.csv")
    coverage = pd.read_csv(wsource / "condition_coverage.csv")
    for table in [fr, fm, wr, wm, wc]:
        assert table.spacing_cm.eq(50).all()
        assert not set(table.condition_id) & excluded
    assert set(fr.stage) == {"Medium"}
    assert set(fr.protocol) == {"forward_250cm"}
    assert set(wr.protocol) == {"wind_tunnel"}
    assert not fr.duplicated(["experiment_directory", "run_id", "drone_name", "stage"]).any()
    assert not wr.duplicated(["experiment_id", "run_id", "drone_name", "stage"]).any()
    assert wc.groupby(["condition_id", "position"]).size().eq(3).all()

    # Availability is a join of coverage/counts only; it computes no average.
    available = fm[["condition_id", "position", "stage", "observed_run_count"]].rename(
        columns={"observed_run_count": "forward_observed_run_count"}).merge(
            wm[["condition_id", "position", "stage", "observed_run_count"]].rename(
                columns={"observed_run_count": "wind_observed_run_count"}),
            on=["condition_id", "position", "stage"], how="outer", validate="one_to_one")
    available[["forward_observed_run_count", "wind_observed_run_count"]] = available[
        ["forward_observed_run_count", "wind_observed_run_count"]].fillna(0).astype(int)
    available["both_sources_have_observed_rate"] = available.forward_observed_run_count.gt(0) & available.wind_observed_run_count.gt(0)
    available["average_calculated"] = False

    out.mkdir(parents=True)
    forward, wind = out / "forward_250cm", out / "wind_tunnel"
    for folder in [forward, wind]:
        folder.mkdir(); (folder / "figures").mkdir()
    forward_tables = {
        "adjusted_run_drone_rates.csv": fr, "condition_position_medium_means.csv": fm,
        "adjusted_forward_curve_points.csv": fp, "forward_record_audit.csv": fa,
    }
    for name, frame in forward_tables.items():
        frame.to_csv(forward / name, index=False)
    fm.pivot(index=CELL+["condition_id"], columns="position", values="normalized_rate_pp_min").rename(
        columns=lambda p: f"position_{p}_normalized_pp_min").to_csv(forward / "discharge_rates_wide.csv")
    shutil.copyfile(fsource / "baseline_stage_transfer_factors.csv", forward / "baseline_stage_transfer_factors.csv")
    for name in ["observed_run_stage_rates.csv", "condition_position_stage_means.csv", "curve_stage_rates.csv",
        "three_stage_curve_knots.csv", "wind_run_audit.csv", "wind_drone_window_audit.csv",
        "stage_observation_support.csv", "condition_coverage.csv"]:
        shutil.copyfile(wsource / name, wind / name)
    shutil.copytree(wsource / "figures", wind / "figures", dirs_exist_ok=True)
    for name in ["all_wind_three_stage_curves.pdf", "50cm_wind_overview.png", "50cm_wind_overview.pdf"]:
        shutil.copyfile(wsource / name, wind / name)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "pdf.fonttype": 42, "svg.fonttype": "none"})
    with PdfPages(forward / "all_50cm_forward_medium_curves.pdf") as pdf:
        for cid, group in fm.groupby("condition_id", sort=True):
            fig = forward_figure(group, normalizer)
            pdf.savefig(fig)
            for ext in ["png", "pdf", "svg"]:
                fig.savefig(forward / "figures" / f"{cid}_forward_medium.{ext}", dpi=300, facecolor="white")
            plt.close(fig)
    available.to_csv(out / "source_stage_availability.csv", index=False)
    shutil.copyfile(MODEL, out / "fixed_september_baseline.json")
    assert sha(out / "fixed_september_baseline.json") == normalizer.sha256
    summary = dict(
        wind_candidate_files=len(inventory), wind_retained_runs=int(wr.run_id.nunique()),
        wind_conditions=int(wc.condition_id.nunique()),
        wind_level1_2_conditions=int(wc[wc.wind_level.isin([1,2])].condition_id.nunique()),
        wind_additional_actual_Level3_conditions=int(wc[wc.wind_level.eq(3)].condition_id.nunique()),
        wind_observed_stage_rates=len(wr), wind_curve_model_stage_rates=len(wc),
        wind_missing_whole_stage_completions=int(wc.is_modeled.sum()),
        forward_safe_runs=int(fr.run_id.nunique()), forward_adjusted_Medium_rates=len(fr),
        forward_conditions=int(fm.condition_id.nunique()),
        forward_records_without_Medium=int(fa.status.eq("no_forward_time_in_own_medium").sum()),
        conditions_without_supported_complete_wind=coverage[coverage.status.eq("no_supported_complete_wind_record")].condition_id.tolist(),
        safety_excluded_conditions=sorted(excluded),
        condition_position_stage_cells_with_both_observed_sources=int(available.both_sources_have_observed_rate.sum()),
        cross_protocol_averaging_applied=False)
    for relative_path, digest in hashes.items():
        assert sha(ROOT / relative_path) == digest, f"Input changed during export: {relative_path}"
    validation = dict(status="passed_offline_export_checks", only_50cm=True,
        fixed_baseline_byte_identical=True, all_source_inputs_unchanged=True,
        wind_inventory_complete_and_current=True, five_safety_conditions_absent=True,
        separate_source_grains_unique=True, curve_rows_and_rates_preserved=True,
        forward_actual_time_unchanged=True, averaging_applied=False,
        previous_independent_replay=str((SOURCE / "independent_validation.json").relative_to(ROOT)),
        wind_source_replay_max_error_pp_min=manifests["wind_tunnel"]["independent_raw_stage_replay_max_error_pp_min"],
        baseline_original_sha256=normalizer.sha256,
        active_dataset_or_training_modified=False)
    write_json(out / "validation.json", validation)
    write_json(out / "summary.json", summary)
    manifest = dict(status="completed_50cm_analysis_package_not_activated",
        scope="All available 50cm candidates audited; independent wind and forward results exported",
        normalization_model_version=normalizer.version, normalization_model_sha256=normalizer.sha256,
        source_hashes=hashes, summary=summary,
        cross_protocol_averaging_applied=False, reference_time_rescaled_from_experiments=False,
        active_training_dataset_changed=False, original_raw_records_changed=False,
        paper_control_code_changed=False,
        missing_stage_policy=manifests["wind_tunnel"]["missing_stage_policy"],
        partial_stage_policy=manifests["wind_tunnel"]["partial_stage_policy"],
        protocol_differences="Old forward source supports Medium only. Wind fits three stages where observed; missing stages are curve-model completions, not measured labels.")
    (out / "README.md").write_text(f"""# 50厘米数据处理结果：固定九月份baseline

仅含50cm结果；风洞与旧250cm前进数据分开，没有跨协议平均。原始数据、九月份baseline、控制程序和训练集未改变。原SOC实测对照图未加入。

## 结果

- 风洞：检查{len(inventory)}个CSV，保留{summary['wind_retained_runs']}组、{summary['wind_conditions']}个条件（Level1/2有{summary['wind_level1_2_conditions']}个；实际Level3另保留{summary['wind_additional_actual_Level3_conditions']}个）。{len(wr)}条实测阶段率和{len(wc)}条曲线模型系数；{summary['wind_missing_whole_stage_completions']}条缺整段用同位置风洞相对耗电因子完成，单独标记来源。
- 旧250cm前进：50cm间距范围内{summary['forward_safe_runs']}组实验、{len(fr)}条有效Medium率、{summary['forward_conditions']}个条件；{summary['forward_records_without_Medium']}条无本机Medium前进时间保留audit。75cm未混入本包。
- 五项安全排除条件：{', '.join(sorted(excluded))}。整个条件从系数和曲线中排除，原失败记录仍在原数据库。
- 安全范围内但没有通过完整风洞记录筛选的条件：{', '.join(summary['conditions_without_supported_complete_wind'])}。逐文件原因见wind_tunnel/wind_run_audit.csv，未用其他配置填造数据。

## 方法

旧前进率直接复用已有raw拟合，逐本机/电池/阶段乘“九月份目标基线率÷旧基线同阶段率”；然后换算到固定参考电池阶段率。没有把已标准化率再乘一次比例，实际前进时间不变。P5旧B15映射到当前B12的相对负载保持是一项换算假设。

风洞去除hover起初SOC不变的时间，后续整数SOC平台保留；按本机实际电池的阶段边界独立拟合。每机有20%终点则保留到自身20%，没有该终点则仅使用实际hover窗口。阶段率除本机固定基线率、乘参考率；最后按SOC差/阶段率计算时间。参考边界100/82/52/20保持原样，没有直接平均不同起始SOC飞机的飞行时间。

缺失阶段及部分阶段向参考边界的延伸可从is_modeled、rate_origin及stage_observation_support.csv追溯。图没有额外拓展标签。图中时间是参考模型计算值，不是原始飞行时长。wind_tunnel_front_50_tail_lv2_004的D1实际到38%，其低段延伸到20%是计算延伸。短阶段、少数SOC跳档和原前进零斜率继续保留质量标记。

## 查看和使用

- wind_tunnel/all_wind_three_stage_curves.pdf：风洞全部逐条件三段曲线。
- wind_tunnel/50cm_wind_overview.png / .pdf：Level1/2覆盖总览。
- 两套目录各自figures/：每个条件3300×2040 PNG及PDF/SVG矢量图。
- wind_tunnel/observed_run_stage_rates.csv：293条真实观测阶段率。
- wind_tunnel/curve_stage_rates.csv：315条完整曲线模型系数，含缺段补全。
- forward_250cm/adjusted_run_drone_rates.csv：281条已换算前进Medium率。
- forward_250cm/condition_position_medium_means.csv：各条件五位置均值；仅前进数据。
- source_stage_availability.csv：两来源观测支撑数量，未计算平均。共有{summary['condition_position_stage_cells_with_both_observed_sources']}个同条件/位置/阶段有两类观测率，可用于以后第三步；旧前进只支持Medium。
- fixed_september_baseline.json：与原模型字节相同的只读副本。
- manifest.json / validation.json / summary.json：来源哈希、完整范围及离线校验。

当前为独立分析结果包，未自动加入训练标签。模型补全不算新增实测。后续若平均，应先分开计算各自重复实验的阶段均值，再在相同条件、位置、阶段上平均耗电率，之后算参考时间。
""")
    write_json(out / "manifest.json", manifest)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    write_json(out / "file_checksums.json", {str(p.relative_to(out)):sha(p) for p in files})
    archive = out.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"Preserve existing archive: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, str(Path(out.name) / p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(str(archive))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.output.resolve())
