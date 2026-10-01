"""Offline single-run pilot; never imports the app or aircraft controllers.

Normalize the observed trajectory continuously with the existing Bideal model,
fit rates on the reference-SOC stages, then integrate three rates into a curve.
Missing High stages in this pilot use the Medium relative-rate factor; their
provenance remains explicit even though the presentation figure is uncluttered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/wind_tunnel_three_stage_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battery_normalization import BatteryNormalizer

EXPERIMENT = "wind_tunnel_diamond_50_head_lv2_002"
SOURCE = ROOT / "database" / EXPERIMENT / f"{EXPERIMENT}_20260928_232343_all_coordination.csv"
MODEL = ROOT / "analysis_results/battery_normalization_v3_with_b15_20260909/model.json"
DEFAULT_OUT = ROOT / "analysis_results/wind_tunnel_three_stage_continuous_trial_20260929" / EXPERIMENT
STAGES = (("High", 100.0, 82.0), ("Medium", 82.0, 52.0), ("Low", 52.0, 20.0))
COLORS = ("#0072B2", "#D55E00", "#009E73", "#7A55A3", "#CC79A7")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_stage(samples, upper, lower):
    part = samples[(samples.normalized_soc <= upper) & (samples.normalized_soc >= lower)]
    if len(part) < 3 or part.normalized_soc.max() - part.normalized_soc.min() < 2:
        return None
    t = part.elapsed_time.to_numpy(float)
    y = part.normalized_soc.to_numpy(float)
    t = t - t[0]
    weights = np.r_[np.diff(t)[0] / 2, (t[2:] - t[:-2]) / 2, np.diff(t)[-1] / 2]
    if np.any(weights <= 0):
        raise ValueError("Nonpositive sampling-time weight")
    slope, intercept = np.polyfit(t, y, 1, w=np.sqrt(weights))
    residual = y - (slope * t + intercept)
    centered = y - np.average(y, weights=weights)
    rate = float(-60 * slope)
    if rate <= 0 or not np.isfinite(rate):
        raise ValueError("Discharge stage does not have a positive finite rate")
    duration = float(t[-1])
    levels = int(part.raw_soc.nunique())
    return {
        "rate_pp_min": rate,
        "status": "fitted_observed_normalized_samples",
        "first_normalized_soc": float(y[0]),
        "last_normalized_soc": float(y[-1]),
        "observed_duration_s": duration,
        "sample_count": len(part),
        "unique_raw_soc_levels": levels,
        "rmse_pp": float(np.sqrt(np.average(residual ** 2, weights=weights))),
        "r_squared": float(1 - np.sum(weights * residual ** 2) / np.sum(weights * centered ** 2)),
        "support_note": "short_or_few_levels" if duration < 20 or levels < 4 else "observed_partial_or_full_stage",
    }


def knots(rates):
    durations = [60 * (upper - lower) / rate for (_, upper, lower), rate in zip(STAGES, rates)]
    return np.r_[0.0, np.cumsum(durations)], np.array([100.0, 82.0, 52.0, 20.0])


def continuous_fit(samples, fitted, reference_rates):
    """Estimate stage slopes together with exact continuity and a fixed anchor.

    The per-stage fits initialize the solver and remain in the evidence table.
    The fit uses observed normalized samples only, never completed model points.
    """
    t = samples.fit_time_s.to_numpy(float)
    y = samples.normalized_soc.to_numpy(float)
    b0 = float(y[0])
    initial = np.array([item["rate_pp_min"] for item in fitted])
    active = [0, 1, 2] if fitted[0]["sample_count"] else [1, 2]
    weights = np.r_[np.diff(t)[0] / 2, (t[2:] - t[:-2]) / 2, np.diff(t)[-1] / 2]

    def predict(log_rates):
        rates = initial.copy()
        rates[active] = np.exp(log_rates)
        kt, kb = knots(rates)
        shift = float(np.interp(b0, kb[::-1], kt[::-1]))
        clock = t + shift
        indices = np.clip(np.searchsorted(kt, clock, side="right") - 1, 0, 2)
        # Linear Low continuation is used only to evaluate optimizer residuals.
        # Normalization retains its strict floor and the exported curve stops20.
        return kb[indices] - (clock - kt[indices]) * rates[indices] / 60

    result = least_squares(lambda values: (predict(values) - y) * np.sqrt(weights),
                           np.log(initial[active]), bounds=(np.log(.1), np.log(100)),
                           xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000)
    if not result.success or np.any(result.active_mask):
        raise ValueError("Continuous fit did not converge to an interior solution")
    rates = initial.copy()
    rates[active] = np.exp(result.x)
    if 0 not in active:
        rates[0] = rates[1] * reference_rates[0] / reference_rates[1]
    predicted = predict(result.x)
    for index, ((stage, upper, lower), part) in enumerate(zip(STAGES, fitted)):
        part["independent_stage_rate_pp_min"] = part["rate_pp_min"]
        if "rmse_pp" in part:
            part["independent_stage_rmse_pp"] = part.pop("rmse_pp")
            part["independent_stage_r_squared"] = part.pop("r_squared")
        part["rate_pp_min"] = float(rates[index])
        part["fit_method"] = "joint_time_weighted_SOC_least_squares_with_continuous_82_52_knots_and_first_drop_anchor"
        mask = (y <= upper) & (y >= lower)
        part["continuous_fit_stage_rmse_pp"] = float(np.sqrt(np.average((predicted[mask] - y[mask]) ** 2, weights=weights[mask]))) if mask.any() else None
    return {"converged": bool(result.success), "function_evaluations": int(result.nfev),
            "observed_normalized_rmse_pp": float(np.sqrt(np.average((predicted - y) ** 2, weights=weights))),
            "initial_reference_soc": b0, "fixed_start_anchor": True}


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=13, pad=13, fontweight="semibold")
    ax.set_ylim(16, 104)
    ax.set_yticks([20, 40, 52, 60, 82, 100])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized SOC (%)")
    ax.grid(axis="both", color="#D9E0E4", linewidth=0.6, alpha=0.7)
    for boundary in (82, 52):
        ax.axhline(boundary, color="#81929B", ls=(0, (4, 4)), lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_axisbelow(True)


def save_figures(out, trajectories, curves, norm):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    style(ax, "Diamond · 50 cm · Headwind · Level 2")
    for color, (drone, item) in zip(COLORS, curves.items()):
        t, b = knots(item["rates_pp_min"])
        ax.plot(t, b, color=color, lw=2.35, marker="o", markersize=4.3, label=f"P{drone[-1]} / D{drone[-1]}")
    t_ref, b_ref = knots(norm.reference.rates_pp_min)
    ax.plot(t_ref, b_ref, color="#4A555D", lw=1.7, ls=(0, (5, 3)), label="Reference battery")
    xmax = max(float(knots(item["rates_pp_min"])[0][-1]) for item in curves.values())
    ax.set_xlim(0, max(xmax, t_ref[-1]) * 1.05)
    for name, upper, lower in STAGES:
        ax.text(0.986, (upper + lower) / 2, name, transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=11, color="#64757F", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=11)
    fig.text(0.11, .905, "Three-stage configuration discharge curves on the existing reference-battery scale", fontsize=11, color="#52636C")
    fig.subplots_adjust(left=.11, right=.96, bottom=.23, top=.85)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"normalized_three_stage.{extension}", dpi=300, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11.8, 10.4), sharex=True, sharey=True)
    for ax, color, (drone, item) in zip(axes.flat, COLORS, curves.items()):
        obs = trajectories[drone]
        t, b = knots(item["rates_pp_min"])
        # Place observations on the model's full-battery clock while retaining
        # their elapsed-time differences; the initial constant-SOC wait is gone.
        shift = float(np.interp(item["reference_start_soc"], b[::-1], t[::-1]))
        ax.step(obs.fit_time_s + shift, obs.normalized_soc, where="post", color="#A1AFB7", lw=1.1, label="Normalized observations")
        ax.plot(t, b, color=color, lw=2.1, marker="o", markersize=3.5, label="Three-stage curve")
        style(ax, f"P{drone[-1]} / D{drone[-1]} · {item['battery_id']}")
    ax = axes.flat[-1]
    style(ax, "Existing reference battery")
    ax.plot(t_ref, b_ref, color="#4A555D", lw=2.0, marker="o", markersize=4)
    for ax in axes.flat:
        ax.set_xlim(0, max(xmax, t_ref[-1]) * 1.05)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .025))
    fig.suptitle("Diamond · 50 cm · Headwind · Level 2", x=.09, y=.982, ha="left", fontsize=17, fontweight="semibold")
    fig.subplots_adjust(left=.09, right=.97, top=.895, bottom=.10, hspace=.42, wspace=.22)
    for extension in ("png", "pdf"):
        fig.savefig(out / f"observations_and_fit.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def run(out):
    if out.exists():
        raise FileExistsError("Choose a new output folder; no previous result is overwritten")
    inputs = [SOURCE, MODEL, ROOT / "battery_normalization.py", ROOT / "TRAINING_EXCLUSIONS.md", ROOT / "database/experiment_registry.json", Path(__file__)]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    norm = BatteryNormalizer.load(MODEL)
    raw = pd.read_csv(SOURCE, low_memory=False)
    if set(raw.experiment_id) != {EXPERIMENT} or set(raw.run_id) != {"20260928_232343"}:
        raise ValueError("Unexpected experiment or run; no file inference from registry")
    expected = {"formation": "diamond", "wind_direction": "head wind", "wind_speed": "Level2", "inter_drone_distance_cm": 50}
    for key, value in expected.items():
        if set(raw[key]) != {value}:
            raise ValueError(f"Condition mismatch: {key}")
    excluded = (ROOT / "TRAINING_EXCLUSIONS.md").read_text()
    if "`diamond_50_head_lv2`" in excluded:
        raise ValueError("Target is excluded from fitting")
    rec = next(r for r in json.loads((ROOT / "database/experiment_registry.json").read_text())["experiments"] if r["experiment_id"] == EXPERIMENT)
    if rec.get("is_outlier", False):
        raise ValueError("Registry marks the target as an outlier")
    required = ["elapsed_time", "battery"]
    if not np.isfinite(raw[required].to_numpy(float)).all():
        raise ValueError("Invalid SOC or timestamp")
    hovering = raw[raw.phase.eq("wind_tunnel_hover")]
    ready = hovering.groupby("drone_name").elapsed_time.min()
    assert len(ready) == 5
    all_ready = float(ready.max())
    first_landing = float(raw[raw.phase.str.contains("landing", case=False, na=False)].elapsed_time.min())
    trajectories, curves, stage_rows, audits, knot_rows = {}, {}, [], [], []
    conservation_errors = []
    for drone, group in raw.groupby("drone_name", sort=True):
        h = group[group.phase.eq("wind_tunnel_hover")].sort_values("elapsed_time").copy()
        assert not h.elapsed_time.duplicated().any()
        if h.battery.diff().gt(0).any():
            raise ValueError("SOC rebound requires explicit treatment")
        first = h[h.battery.diff().lt(0)].iloc[0]
        start = max(float(first.elapsed_time), all_ready)
        selected = h[(h.elapsed_time >= start) & (h.elapsed_time < first_landing)].copy()
        battery = str(selected.battery_id.iloc[0])
        assert set(selected.battery_id) == {battery}
        own = norm.curve_for(battery, drone)
        b0 = float(selected.battery.iloc[0])
        h_equiv = np.array([own.equivalent_seconds(b0, float(b)) for b in selected.battery])
        capacity = norm.reference.equivalent_seconds(b0, 20)
        supported = h_equiv <= capacity + 1e-8
        rejected_floor = int((~supported).sum())
        selected = selected[supported].copy()
        h_equiv = h_equiv[supported]
        normalized = np.array([norm.reference.advance(b0, float(v)) for v in h_equiv])
        errors = [abs(norm.reference.equivalent_seconds(b0, float(b)) - float(v)) for b, v in zip(normalized, h_equiv)]
        conservation_errors.extend(errors)
        samples = pd.DataFrame({"experiment_id": EXPERIMENT, "run_id": "20260928_232343", "drone_id": drone, "battery_id": battery, "elapsed_time": selected.elapsed_time.to_numpy(float), "fit_time_s": selected.elapsed_time.to_numpy(float) - start, "raw_soc": selected.battery.to_numpy(float), "normalized_soc": normalized, "equivalent_hover_seconds": h_equiv, "kind": "derived_Bideal_SOC_from_observed_telemetry"})
        samples["stage"] = np.select([samples.normalized_soc > 82, samples.normalized_soc > 52], ["High", "Medium"], default="Low")
        assert np.all(np.diff(samples.normalized_soc) <= 1e-8)
        fitted = [fit_stage(samples, upper, lower) for _, upper, lower in STAGES]
        if fitted[1] is None or fitted[2] is None:
            raise ValueError("This pilot only defines a missing-High completion, not missing Medium/Low")
        if fitted[0] is None:
            # Explicit pilot assumption: keep Medium's drain factor relative to
            # Bideal when predicting the completely unobserved High stage.
            factor = fitted[1]["rate_pp_min"] / norm.reference.rates_pp_min[1]
            fitted[0] = {"rate_pp_min": factor * norm.reference.rates_pp_min[0], "status": "modeled_missing_high_from_medium_relative_factor", "sample_count": 0, "unique_raw_soc_levels": 0, "observed_duration_s": 0, "support_note": "unobserved_high_requires_review_before_training", "assumption": "High relative drain factor equals this position's fitted Medium relative drain factor"}
        joint_fit = continuous_fit(samples, fitted, norm.reference.rates_pp_min)
        rates = [part["rate_pp_min"] for part in fitted]
        t_knots, b_knots = knots(rates)
        assert len(t_knots) == 4 and np.all(np.diff(t_knots) > 0)
        assert np.array_equal(b_knots, np.array([100., 82., 52., 20.]))
        for index, ((name, upper, lower), part) in enumerate(zip(STAGES, fitted)):
            assert abs(upper - rates[index] * (t_knots[index + 1] - t_knots[index]) / 60 - lower) < 1e-9
            stage_rows.append({"drone_id": drone, "position": int(drone[-1]), "battery_id": battery, "stage": name, "upper_reference_soc": upper, "lower_reference_soc": lower, "model_stage_duration_s": float(t_knots[index + 1] - t_knots[index]), **part})
        own_pad = pd.to_numeric(selected.mid, errors="coerce").eq(pd.to_numeric(selected.mission_pad, errors="coerce"))
        audit = {"drone_id": drone, "battery_id": battery, "initial_takeoff_soc": float(group.battery.iloc[0]), "initial_hover_soc": float(h.battery.iloc[0]), "first_hover_s": float(h.elapsed_time.iloc[0]), "first_drop_s": float(first.elapsed_time), "reference_start_soc": b0, "initial_hover_plateau_removed_s": float(first.elapsed_time) - float(h.elapsed_time.iloc[0]), "excluded_pre_first_drop_s": start, "fit_end_s": float(selected.elapsed_time.iloc[-1]), "last_actual_soc_used": float(samples.raw_soc.iloc[-1]), "last_reference_soc_used": float(normalized[-1]), "sample_count": len(samples), "floor_rejected_rows": rejected_floor, "own_pad_fraction": float(own_pad.mean()), "model_full_100_to_20_s": float(t_knots[-1])}
        audits.append(audit)
        curves[drone] = {"battery_id": battery, "reference_start_soc": b0, "rates_pp_min": rates, "stages": fitted, "continuous_fit": joint_fit, "knots_time_s": t_knots.tolist(), "knots_reference_soc": b_knots.tolist()}
        trajectories[drone] = samples
        knot_rows.extend({"drone_id": drone, "position": int(drone[-1]), "time_s": float(t), "normalized_soc": float(b)} for t, b in zip(t_knots, b_knots))
    for path, expected_sha in hashes.items():
        assert sha(ROOT / path) == expected_sha, f"Source changed: {path}"
    out.mkdir(parents=True)
    pd.concat(trajectories.values(), ignore_index=True).to_csv(out / "normalized_observed_samples.csv", index=False, float_format="%.12g")
    pd.DataFrame(stage_rows).to_csv(out / "stage_rates.csv", index=False, float_format="%.12g")
    pd.DataFrame(audits).to_csv(out / "initial_plateau_and_window_audit.csv", index=False, float_format="%.12g")
    pd.DataFrame(knot_rows).to_csv(out / "three_stage_knots.csv", index=False, float_format="%.12g")
    manifest = {"experiment_id": EXPERIMENT, "run_id": "20260928_232343", "source": str(SOURCE.relative_to(ROOT)), "input_sha256": hashes, "reference_model_version": norm.version, "reference_model_sha256": norm.sha256, "reference_model_status": json.loads(MODEL.read_text())["status"], "reference_frozen": True, "source_rows": len(raw), "analysis_type": "single_run_pilot_not_active_training_data", "experiment_type": "stationary_wind_tunnel_hover_not_forward_flight", "reference_anchor_convention": "At the first retained post-drop sample, reference SOC equals the numerical measured SOC; this is an alignment convention, not a proof of absolute SOC equivalence", "normalization": "Continuous equivalent-hover-time integration using own battery calibration then Bideal; x-axis is actual elapsed hovering time, not equivalent hovering time", "plateau_policy": "Exclude takeoff/centering and initial unchanged-SOC hover plateau; start at the first observed decrease; exclude that first already-spent percentage point; keep subsequent integer-SOC plateaus", "common_five_drone_window": {"all_ready_s": all_ready, "first_landing_s": first_landing, "rule": "Only wind_tunnel_hover samples before the first drone starts landing"}, "fit_method": "Initialize from individual stage regressions; jointly fit a continuous three-stage normalized SOC-time model with fixed 82/52 knots and first-drop SOC anchor. Time-weighted least squares, one trial; no repeated-trial average yet", "missing_high_policy": "For this pilot only, inherit this position's fitted Medium relative drain factor and multiply by the frozen Bideal High rate; no synthetic points enter the fit", "curve_definition": "A derived reference-battery model with three constant rates, continuous at 82 and 52, from 100 to 20. Model time differs from the original recorded flight duration", "plot_extension_labels": False, "training_eligibility": False, "training_limitation": "P3-P5 have no observed High; P1 Low and P2 High have short/few-level support; repeated-trial consistency and missing-stage assumptions remain to be assessed", "audit": audits, "curves": curves}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    validation = {"equivalent_hover_time_max_error_s": float(max(conservation_errors)), "equivalent_hover_time_conserved": bool(max(conservation_errors) < 1e-8), "five_positions": len(curves) == 5, "three_continuous_segments_each": True, "continuous_fit_converged": all(c["continuous_fit"]["converged"] for c in curves.values()), "positive_rates": bool(all(rate > 0 for curve in curves.values() for rate in curve["rates_pp_min"])), "initial_flat_section_removed": bool(all(float(frame.fit_time_s.iloc[0]) == 0 for frame in trajectories.values())), "common_five_drone_window_only": bool(all(frame.elapsed_time.max() < first_landing for frame in trajectories.values())), "missing_high_positions": [int(drone[-1]) for drone, curve in curves.items() if curve["stages"][0]["sample_count"] == 0], "raw_sources_unchanged": all(sha(ROOT / path) == value for path, value in hashes.items()), "hardware_access": False, "training_data_modified": False}
    assert all(validation[key] for key in ["equivalent_hover_time_conserved", "five_positions", "three_continuous_segments_each", "continuous_fit_converged", "positive_rates", "initial_flat_section_removed", "common_five_drone_window_only", "raw_sources_unchanged"])
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    save_figures(out, trajectories, curves, norm)
    (out / "README.md").write_text("# 单组 wind tunnel 三阶段试处理\n\n实验：`" + EXPERIMENT + "`，run `20260928_232343`。\n\n使用已有 Bideal：High 100–82%、Medium 82–52%、Low 52–20%。\n\n先去掉起飞、定位与开头不掉电平台，从有效悬停后的首个掉电样本开始计时；后续整数 SOC 平台保留。仅使用五机全部就位后、首架开始降落前的五机悬停区间。\n\n按照原有等效悬停时间积分进行连续标准化，参考起点与首个实测掉电后的 SOC 数值对齐。先逐参考阶段初拟合，再将 82/52 连接连续性与首个掉电 SOC 锚点纳入联合最小二乘拟合。原始数据未修改。\n\nP3–P5 完全没有本次 High 观测，试图采用同位置 Medium 相对 Bideal 耗电因子延长到 High。此为待确认的建模假设，全部来源与假设见 manifest.json，补出的点不参与拟合。P1 Low 和 P2 High 的有效覆盖很短。该试处理不是已激活的训练数据。\n\n图中没有外推文字标签，原始范围、补齐范围及状态保留于数据文件。normalized_three_stage 为三段模型图，observations_and_fit 为有效观测与模型的对照。模型的 100→20% 总时长不等于这次实际飞行总时长；静态风洞悬停率不冒充 2.5 m 前进率。\n\nstage_rates.csv：三个阶段的速率及来源；initial_plateau_and_window_audit.csv：每机剔除的平台时间及有效窗口；normalized_observed_samples.csv：由实测得到的连续参考 SOC；three_stage_knots.csv：三段连接点。\n")
    for path, value in hashes.items():
        assert sha(ROOT / path) == value, f"Input modified: {path}"
    print(pd.DataFrame(stage_rows)[["drone_id", "stage", "rate_pp_min", "status", "observed_duration_s"]].round(4).to_string(index=False))
    print("Output:", out)
    print("Validation:", json.dumps(validation))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.output.resolve())
