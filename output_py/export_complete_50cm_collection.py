"""Publish the reviewed 50-cm curve collection without changing coefficients.

Offline packaging only. Source observations, the fixed September baseline,
flight control and active training inputs remain untouched. Figure labels are
neutral; measurement and modeling provenance remains in the accompanying tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

from process_50cm_stage_curves import (
    ROOT, MODEL, NAMES, FORMS, FORM_NAMES, WIND_NAMES, BatteryNormalizer,
    plt, sha, write_json, exclusion_ids, plot_curve,
)
from repair_50cm_wind_curve_estimates import figure as source_figure

SOURCE = ROOT / "analysis_results/50cm_wind_model_refit_20260929"
PREVIOUS = ROOT / "analysis_results/50cm_selection_corrected_fixed_baseline_20260929"
OUTPUT = ROOT / "analysis_results/50cm_complete_fixed_baseline_20260929"



def figure(group, normalizer):
    fig = source_figure(group, normalizer)
    for label in list(fig.texts):
        if "Stage-rate estimates on the fixed September reference-battery scale" in label.get_text():
            label.remove()
    return fig


def render_figures(wind_out, curves, normalizer, excluded):
    figdir = wind_out / "figures"
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    groups = {cid: g for cid, g in curves.groupby("condition_id", sort=True)}
    with PdfPages(wind_out / "all_50cm_curves.pdf") as pdf:
        pdf.infodict()["Title"] = "50 cm three-stage discharge curves"
        for cid, g in groups.items():
            fig = figure(g, normalizer)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(figdir / f"{cid}_three_stage.{ext}", dpi=300, facecolor="white")
            pdf.savefig(fig)
            plt.close(fig)

    fig, axes = plt.subplots(5, 6, figsize=(24, 16))
    max_time = curves.groupby(["condition_id", "position"]).stage_duration_s.sum().max()
    max_time = float(np.ceil(max_time / 100) * 100)
    for i, form in enumerate(FORMS):
        for j, (wind, level) in enumerate((w, l) for w in ("head", "tail", "side") for l in (1, 2)):
            cid = f"{form}_50_{wind}_lv{level}"
            ax = axes[i, j]
            ax.set_title(f"{FORM_NAMES[form]} · {WIND_NAMES[wind]} · L{level}", fontsize=11, loc="left", pad=8)
            if cid in excluded:
                ax.set_facecolor("#F1F2F3")
                ax.text(.5, .5, "Excluded condition", transform=ax.transAxes, ha="center", va="center", color="#757F85", fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[["top", "right"]].set_visible(False)
            else:
                plot_curve(ax, groups[cid], normalizer, small=True)
                ax.set_xlim(0, max_time)
                ax.set_xticks(np.arange(0, max_time + 1, 200))
    handles, labels = next(ax for ax in axes.flat if ax.lines).get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(.5, .018), fontsize=12)
    fig.suptitle("50 cm · Three-stage discharge curves", x=.055, y=.982, ha="left", fontsize=23, fontweight="semibold")
    fig.text(.055, .953, "Fixed September reference-battery scale · 25 included conditions · common time scale", fontsize=13, color="#52636C")
    fig.subplots_adjust(left=.055, right=.985, top=.919, bottom=.078, hspace=.58, wspace=.48)
    for ext in ("png", "pdf"):
        fig.savefig(wind_out / f"50cm_overview.{ext}", dpi=300, facecolor="white")
    plt.close(fig)



def refresh_figures():
    """Apply the authorized wording edit to the existing exported collection."""
    wind_out = OUTPUT / "wind_tunnel"
    normalizer = BatteryNormalizer.load(MODEL)
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    assert normalizer.sha256 == manifest["fixed_baseline_sha256"]
    preserved = {
        p: sha(p) for p in OUTPUT.rglob("*")
        if p.is_file() and p.suffix in (".csv", ".json")
        and p.name not in ("manifest.json", "validation.json", "delivery_validation.json")
    }
    curves = pd.read_csv(wind_out / "curve_stage_rates.csv")
    assert sha(wind_out / "curve_stage_rates.csv") == manifest["source_curve_table_sha256"]
    render_figures(wind_out, curves, normalizer, exclusion_ids())
    dimensions = {p.name: list(Image.open(p).size) for p in (wind_out / "figures").glob("*_three_stage.png") if "_wind_" not in p.name}
    assert len(dimensions) == 25 and all(v == [3300, 2040] for v in dimensions.values())
    for path in (wind_out / "figures").glob("*_three_stage.svg"):
        if "_wind_" not in path.name:
            content = path.read_text().lower()
            assert "wind-tunnel" not in content
            assert "stage-rate estimates on the fixed september reference-battery scale" not in content
    obsolete = list((wind_out / "figures").glob("*_wind_three_stage.*"))
    obsolete += [wind_out / "all_50cm_wind_tunnel_curves.pdf", wind_out / "50cm_wind_tunnel_overview.pdf", wind_out / "50cm_wind_tunnel_overview.png"]
    for path in obsolete:
        if path.exists():
            path.unlink()
    readme = OUTPUT / "README.md"
    readme.write_text(readme.read_text().replace("all_50cm_wind_tunnel_curves", "all_50cm_curves").replace("50cm_wind_tunnel_overview", "50cm_overview"), encoding="utf-8")
    manifest["export_script_sha256"] = sha(Path(__file__))
    manifest["figure_wording"] = "Wind-tunnel removed from figure headings and PDF filenames; individual figure subtitle removed in full; model provenance retained in data tables"
    write_json(OUTPUT / "manifest.json", manifest)
    validation = json.loads((OUTPUT / "validation.json").read_text())
    validation["all_25_png_dimensions"] = dimensions
    validation["figure_labels_and_filenames_updated"] = True
    validation["individual_figure_subtitle_removed"] = True
    write_json(OUTPUT / "validation.json", validation)
    assert all(sha(p) == digest for p, digest in preserved.items())
    assert sha(MODEL) == normalizer.sha256
    with zipfile.ZipFile(OUTPUT.with_suffix(".zip"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(OUTPUT.parent)))
    print(json.dumps({"pdf": str(wind_out / "all_50cm_curves.pdf"), "archive": str(OUTPUT.with_suffix(".zip")), "figures_updated": 25, "data_unchanged": True}, ensure_ascii=False))


def main():
    if OUTPUT.exists():
        raise FileExistsError("Keep existing exports; select a new output directory.")
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (PREVIOUS, SOURCE)]
    input_hashes = {}
    for manifest in manifests:
        for name, digest in manifest["source_hashes"].items():
            assert sha(ROOT / name) == digest, f"Source changed: {name}"
            input_hashes[name] = digest
    source_manifest = manifests[1]
    inventory = pd.read_csv(PREVIOUS / "wind_tunnel/collection_inventory.csv")
    actual_files = {str(p.relative_to(ROOT)) for p in (ROOT / "database").glob("wind_tunnel_*_50_*/*_all_coordination.csv")}
    assert actual_files == set(inventory.source_file)
    normalizer = BatteryNormalizer.load(MODEL)
    assert normalizer.sha256 == source_manifest["fixed_baseline_sha256"]
    excluded = exclusion_ids()
    curves = pd.read_csv(SOURCE / "wind_tunnel/curve_stage_rates.csv")
    cells = sorted(set(curves.condition_id))
    assert len(curves) == 375 and len(cells) == 25
    assert not set(cells) & excluded
    assert curves.spacing_cm.eq(50).all()
    assert not curves.duplicated(["condition_id", "position", "stage"]).any()
    assert curves.groupby(["condition_id", "position"]).stage.apply(set).eq(set(NAMES)).all()
    assert np.isfinite(curves.normalized_rate_pp_min).all() and curves.normalized_rate_pp_min.gt(0).all()

    old_rates = pd.read_csv(PREVIOUS / "wind_tunnel/observed_run_stage_rates.csv")
    updated = set(source_manifest["refitted_conditions"])
    old_rates = old_rates[~old_rates.condition_id.isin(updated)].copy()
    old_rates["analysis_method_source"] = "previous_reviewed_stage_fits_retained"
    new_rates = pd.read_csv(SOURCE / "wind_tunnel/refitted_real_run_stage_rates.csv")
    new_rates["analysis_method_source"] = "actual_SOC_crossing_block_stage_fits"
    real_rates = pd.concat([old_rates, new_rates], ignore_index=True)
    assert not real_rates.duplicated(["condition_id", "source_file", "position", "stage"]).any()
    assert not set(real_rates.condition_id) & excluded
    support = real_rates.groupby(["condition_id", "position", "stage"]).agg(
        fitted_run_rows=("normalized_rate_pp_min", "size"),
        source_file_count=("source_file", "nunique"),
    ).reset_index()
    numbered = inventory.numbered_record.eq(True) & inventory.raw_rows.gt(0)
    coverage = []
    for form in FORMS:
        for wind in ("head", "tail", "side"):
            for level in (1, 2):
                cid = f"{form}_50_{wind}_lv{level}"
                g = curves[curves.condition_id.eq(cid)]
                records = inventory[numbered & inventory.condition_id.eq(cid)]
                observed = support[support.condition_id.eq(cid)]
                coverage.append(dict(
                    formation=form, wind_direction=wind, wind_level=level, spacing_cm=50,
                    condition_id=cid, safety_excluded=cid in excluded,
                    collection_record_found=bool(len(records)), numbered_nonempty_files=len(records),
                    numbered_experiment_ids=records.experiment_id.nunique(),
                    positions_with_curves=g.position.nunique(), stage_coefficient_count=len(g),
                    raw_fit_supported_stage_count=len(observed),
                    direct_fit_stage_count=int((~g.is_modeled).sum()),
                    modeled_stage_count=int(g.is_modeled.sum()),
                    missing_raw_stage_support_count=len(g) - len(observed),
                    analysis_status="excluded_safety_condition" if cid in excluded else "collected_has_stage_curves",
                ))
    coverage = pd.DataFrame(coverage)
    included = coverage[~coverage.safety_excluded]
    assert len(included) == 25 and included.collection_record_found.all()
    assert included.positions_with_curves.eq(5).all() and included.stage_coefficient_count.eq(15).all()

    wind_out = OUTPUT / "wind_tunnel"
    figdir = wind_out / "figures"
    figdir.mkdir(parents=True)
    provenance = OUTPUT / "provenance"
    provenance.mkdir()
    shutil.copytree(SOURCE / "forward_250cm", OUTPUT / "forward_250cm")
    shutil.copy2(MODEL, OUTPUT / "fixed_september_baseline.json")
    for name in ("curve_stage_rates.csv", "three_stage_curve_knots.csv", "cross_speed_factors.json"):
        shutil.copy2(SOURCE / "wind_tunnel" / name, wind_out / name)
    shutil.copy2(PREVIOUS / "wind_tunnel/collection_inventory.csv", wind_out / "collection_inventory.csv")
    real_rates.to_csv(wind_out / "observed_run_stage_rates.csv", index=False)
    support.to_csv(wind_out / "stage_observation_support.csv", index=False)
    coverage.to_csv(wind_out / "condition_coverage.csv", index=False)
    for name in ("actual_SOC_crossing_samples.csv", "actual_window_audit.csv", "refitted_observed_stage_rates.csv", "donor_base_stage_estimates.csv"):
        shutil.copy2(SOURCE / "wind_tunnel" / name, provenance / name)
    for name in ("manifest.json", "validation.json", "independent_validation.json"):
        shutil.copy2(SOURCE / name, provenance / ("source_" + name))

    render_figures(wind_out, curves, normalizer, excluded)

    included_records = inventory[numbered & inventory.condition_id.isin(cells)]
    forward = pd.read_csv(OUTPUT / "forward_250cm/adjusted_run_drone_rates.csv")
    counts = dict(
        condition_grid=30, excluded_conditions=5, included_conditions=25,
        included_numbered_nonempty_source_files=len(included_records),
        included_numbered_experiment_ids=included_records.experiment_id.nunique(),
        included_registry_outlier_source_files=int(included_records.registry_outlier.eq(True).sum()),
        condition_images=25, position_curves=125, stage_coefficients=375,
        actual_run_stage_fit_rows=len(real_rates), actual_fit_supported_unique_stages=len(support),
        direct_fit_stage_coefficients=int((~curves.is_modeled).sum()),
        modeled_stage_coefficients=int(curves.is_modeled.sum()),
        stages_without_actual_fit_support=375-len(support),
        modeled_stages_with_actual_fit_support=len(support)-int((~curves.is_modeled).sum()),
        conditions_by_formation=curves.groupby("formation").condition_id.nunique().to_dict(),
        forward_separate_conditions=forward.condition_id.nunique(),
        forward_separate_runs=forward.run_id.nunique(), forward_separate_run_drone_rows=len(forward),
    )
    pngs = sorted(figdir.glob("*.png"))
    dimensions = {p.name: list(Image.open(p).size) for p in pngs}
    assert len(pngs) == 25 and all(v == [3300, 2040] for v in dimensions.values())
    assert sha(wind_out / "curve_stage_rates.csv") == sha(SOURCE / "wind_tunnel/curve_stage_rates.csv")
    assert sha(MODEL) == normalizer.sha256
    for name, digest in input_hashes.items():
        assert sha(ROOT / name) == digest, f"Source changed during export: {name}"
    forward_same = all(sha(p) == sha(OUTPUT / "forward_250cm" / p.relative_to(SOURCE / "forward_250cm")) for p in (SOURCE / "forward_250cm").rglob("*") if p.is_file())
    assert forward_same
    write_json(OUTPUT / "counts.json", counts)
    write_json(OUTPUT / "validation.json", dict(
        passed=True, complete_collection_coverage=True, coefficients_byte_unchanged=True,
        fixed_baseline_unchanged=True, raw_inputs_unchanged=True, exclusions_absent=True,
        forward_outputs_byte_preserved=forward_same, all_25_png_dimensions=dimensions,
        initial_plateau_policy_preserved=True, hardware_commands_sent=False,
        active_training_changed=False, protocols_averaged=False,
    ))
    write_json(OUTPUT / "manifest.json", dict(
        status="complete_curve_collection_candidate_not_activated", counts=counts,
        fixed_baseline_sha256=normalizer.sha256, excluded_conditions=sorted(excluded),
        source_package=str(SOURCE.relative_to(ROOT)), source_hashes=input_hashes,
        source_curve_table_sha256=sha(SOURCE / "wind_tunnel/curve_stage_rates.csv"),
        export_script_sha256=sha(Path(__file__)),
        coefficient_changes_applied_in_this_export=False,
        plot_labels="neutral condition titles; no repair or extension labels",
        provenance="is_modeled, rate_origin, estimate_kind, observed weights and prior source files retained; 17 conditions retain previously reviewed fits, eight use latest stage estimates",
        uncertainty="Coverage weights and cross-speed scenarios are modeling choices, not confidence intervals or new measurements.",
        active_training_changed=False, raw_records_changed=False, baseline_changed=False,
        protocols_averaged=False,
    ))
    (OUTPUT / "README.md").write_text("""# 50 cm 风洞三阶段曲线图集

25个纳入条件均有编号采集记录和五位置曲线。30种组合中按当前规则排除5种危险条件；Front、Vee、Echelon各6种，Column 3种，Diamond 4种。

## 文件

- `wind_tunnel/all_50cm_curves.pdf`：25页完整图册。
- `wind_tunnel/50cm_overview.png/pdf`：30格总览，5个排除条件显示灰色；所有小图时间尺度一致。
- `wind_tunnel/figures/`：25张独立PNG（3300×2040，300dpi）、PDF、SVG。
- `wind_tunnel/curve_stage_rates.csv`：375个阶段系数，组成125条位置曲线；系数与最新审查版本逐字节一致。
- `wind_tunnel/condition_coverage.csv`：编号采集覆盖、真实拟合支撑及模型阶段计数。
- `wind_tunnel/collection_inventory.csv`：原始候选文件清单；包括失败、prepare和空记录，不等同于有效训练实验数。
- `wind_tunnel/observed_run_stage_rates.csv`、`stage_observation_support.csv`：真实记录拟合及阶段支撑清单，缺段不填假观测。
- `provenance/`、`manifest.json`、`validation.json`：来源、拟合方法、模型先验及离线核对记录。
- `forward_250cm/`：先前按新基准调整的250cm结果原样保留，24条件、60次记录、281个实验/无人机系数；没有与风洞平均。

## 计数和解释

风洞纳入条件共有83个非空编号源文件，属于60个实验编号目录；其中3个源文件在注册表标为异常，采集覆盖计数不代表其全部可训练。338个真实记录阶段拟合支持334个条件/位置/阶段。最终375个曲线系数中308个直接来自阶段拟合，67个包含模型估计：41个没有本阶段实际拟合支撑，26个将实际拟合与另一档风速参考组合。所有相关来源及权重保留。

曲线按固定九月份参考电池的100/82/52/20边界绘制。时间由各阶段耗电率计算，不是不同初始电量的原始实验时长平均。每架无人机开头不变SOC时段已按此前方法排除。图保持中性条件标题，不显示repair或拓展标签。模型补全不作为新的实测记录；本次没有更新训练数据、原始记录、基准或飞行控制。
""", encoding="utf-8")
    archive = OUTPUT.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file():
                z.write(path, str(path.relative_to(OUTPUT.parent)))
    print(json.dumps(dict(output=str(OUTPUT), archive=str(archive), counts=counts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-figures", action="store_true")
    args = parser.parse_args()
    if args.refresh_figures:
        refresh_figures()
    else:
        main()
