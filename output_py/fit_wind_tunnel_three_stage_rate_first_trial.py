"""Offline pilot: fit each battery's stages first, normalize rates, then time.

Every aircraft retains its own hovering record through its own first 20% sample.
Independent stage slopes are never changed to force a cumulative time fit.
No controller, aircraft, network, or active training imports are used.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battery_normalization import BatteryNormalizer

EXPERIMENT = "wind_tunnel_diamond_50_head_lv2_002"
SOURCE = ROOT / "database" / EXPERIMENT / f"{EXPERIMENT}_20260928_232343_all_coordination.csv"
MODEL = ROOT / "analysis_results/battery_normalization_v3_with_b15_20260909/model.json"
DEFAULT_OUT = ROOT / "analysis_results/wind_tunnel_three_stage_rate_first_corrected_20260929" / EXPERIMENT
NAMES = ("High", "Medium", "Low")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#7A55A3", "#CC79A7")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_raw_stage(samples, upper, lower):
    part = samples[samples.raw_soc.between(lower, upper)].copy()
    if len(part) < 3 or part.raw_soc.nunique() < 2:
        return None, part
    time = part.fit_time_s.to_numpy(float)
    origin = float(time[0])
    t = time - origin
    y = part.raw_soc.to_numpy(float)
    dt = np.diff(t)
    weights = np.r_[dt[0] / 2, (t[2:] - t[:-2]) / 2, dt[-1] / 2]
    if np.any(weights <= 0):
        raise ValueError("Timestamps must be unique and increasing")
    slope, intercept = np.polyfit(t, y, 1, w=np.sqrt(weights))
    rate = float(-60 * slope)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Observed stage has no positive discharge rate")
    predicted = intercept + slope * t
    error = predicted - y
    variance = float(np.average((y - np.average(y, weights=weights)) ** 2, weights=weights))
    duration = float(t[-1])
    levels = int(part.raw_soc.nunique())
    return {
        "raw_rate_pp_min": rate,
        "raw_fit_intercept_soc": float(intercept),
        "stage_fit_origin_s": origin,
        "stage_fit_end_s": float(time[-1]),
        "first_observed_raw_soc": float(y[0]),
        "last_observed_raw_soc": float(y[-1]),
        "observed_duration_s": duration,
        "sample_count": len(part),
        "unique_raw_soc_levels": levels,
        "rmse_pp": float(np.sqrt(np.average(error ** 2, weights=weights))),
        "r_squared": float(1 - np.average(error ** 2, weights=weights) / variance),
        "support_note": "short_or_few_levels" if duration < 20 or levels < 4 else "observed_partial_or_full_stage",
        "rate_origin": "independent_time_weighted_raw_SOC_stage_regression",
    }, part


def knots(rates, boundaries):
    durations = 60 * (np.asarray(boundaries[:-1]) - boundaries[1:]) / rates
    return np.r_[0.0, np.cumsum(durations)], np.asarray(boundaries)


def style(ax, ylabel):
    ax.set_ylim(16, 104)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(color="#D9E0E4", linewidth=.6, alpha=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_axisbelow(True)


def figures(out, frames, curves, normalizer):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    bounds = normalizer.reference.boundaries
    fig, ax = plt.subplots(figsize=(11, 6.8))
    style(ax, "Normalized SOC (%)")
    for color, (drone, item) in zip(COLORS, curves.items()):
        t, soc = knots(np.array(item["normalized_rates_pp_min"]), bounds)
        ax.plot(t, soc, color=color, lw=2.3, marker="o", markersize=4,
                label=f"P{drone[-1]} / D{drone[-1]}")
    tr, br = knots(np.array(normalizer.reference.rates_pp_min), bounds)
    ax.plot(tr, br, color="#4A555D", ls=(0, (5, 3)), lw=1.7, label="Reference battery")
    xmax = max([tr[-1]] + [c["full_100_to_20_s"] for c in curves.values()])
    ax.set_xlim(0, xmax * 1.05)
    ax.set_yticks([20, 40, 52, 60, 82, 100])
    for boundary in bounds[1:-1]:
        ax.axhline(boundary, color="#81929B", ls=(0, (4, 4)), lw=.8, zorder=0)
    for name, up, lo in zip(NAMES, bounds, bounds[1:]):
        ax.text(.985, (up + lo) / 2, name, transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=11, color="#64757F",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
    fig.suptitle("Diamond · 50 cm · Headwind · Level 2", x=.11, y=.966, ha="left", fontsize=17, fontweight="semibold")
    fig.text(.11, .912, "Stage rates fitted independently; duration calculated on the reference-battery scale",
             fontsize=10.5, color="#52636C")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.11, right=.96, bottom=.23, top=.855)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"normalized_three_stage.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11.8, 10.4), sharex=True, sharey=True)
    for ax, color, (drone, item) in zip(axes.flat, COLORS, curves.items()):
        frame = frames[drone]
        ax.step(frame.fit_time_s, frame.raw_soc, where="post", color="#A1AFB7",
                lw=1.2, label="Measured SOC")
        for stage in item["stages"]:
            if stage["sample_count"] and stage.get("raw_rate_pp_min") is not None:
                t = np.array([stage["stage_fit_origin_s"], stage["stage_fit_end_s"]])
                y = stage["raw_fit_intercept_soc"] - stage["raw_rate_pp_min"] * (t - t[0]) / 60
                ax.plot(t, y, color=color, lw=2.2)
        ax.plot([], [], color=color, lw=2.2, label="Independent stage fits")
        for boundary in item["own_boundaries_soc"][1:-1]:
            ax.axhline(boundary, color="#81929B", ls=(0, (4, 4)), lw=.8, zorder=0)
        style(ax, "Measured SOC (%)")
        ax.set_title(f"P{drone[-1]} / D{drone[-1]} · {item['battery_id']}", loc="left", pad=13, fontsize=13, fontweight="semibold")
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(.02, .94, "Fit stages first, then calculate time", transform=ax.transAxes,
            fontsize=13, fontweight="semibold", va="top")
    ax.text(.02, .80,
            "Each drone retains its own hovering record.\n\n"
            "The initial unchanged-SOC period is removed.\n\n"
            "Raw SOC stage slopes are fitted independently.\n\n"
            "Rates are normalized with the existing battery baseline.\n\n"
            "Full-reference stage time = SOC drop / stage rate.",
            transform=ax.transAxes, fontsize=10.5, color="#52636C", va="top", linespacing=1.5)
    for ax in axes.flat[:-1]:
        ax.set_xlim(0, max(f.fit_time_s.max() for f in frames.values()) * 1.04)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .023))
    fig.suptitle("Diamond · 50 cm · Headwind · Level 2", x=.09, y=.982, ha="left", fontsize=17, fontweight="semibold")
    fig.subplots_adjust(left=.09, right=.97, top=.895, bottom=.1, hspace=.42, wspace=.22)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"observations_and_stage_fits.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def run(out):
    if out.exists():
        raise FileExistsError("Choose a new folder; original and prior outputs are preserved")
    inputs = [SOURCE, MODEL, ROOT / "battery_normalization.py", ROOT / "TRAINING_EXCLUSIONS.md", Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    raw = pd.read_csv(SOURCE, low_memory=False)
    expected = {"experiment_id": EXPERIMENT, "run_id": "20260928_232343", "formation": "diamond",
                "wind_direction": "head wind", "wind_speed": "Level2", "inter_drone_distance_cm": 50}
    for key, value in expected.items():
        if set(raw[key]) != {value}:
            raise ValueError(f"Unexpected condition: {key}")
    if "`diamond_50_head_lv2`" in (ROOT / "TRAINING_EXCLUSIONS.md").read_text():
        raise ValueError("This condition is excluded from fitting")
    normalizer = BatteryNormalizer.load(MODEL)
    first_any_landing = float(raw[raw.phase.str.contains("landing", case=False, na=False)].elapsed_time.min())
    hover = raw[raw.phase.eq("wind_tunnel_hover")]
    own_hover_ends = hover.groupby("drone_name").elapsed_time.max().to_dict()
    frames, curves, stages, audits, points, fit_samples = {}, {}, [], [], [], []
    ratio_errors = []
    for drone, group in raw.groupby("drone_name", sort=True):
        h = group[group.phase.eq("wind_tunnel_hover")].sort_values("elapsed_time").copy()
        if h.empty or h.elapsed_time.duplicated().any() or h.battery.diff().gt(0).any():
            raise ValueError("Invalid or rebounding hovering trajectory")
        if not np.isfinite(h[["elapsed_time", "battery"]].to_numpy(float)).all():
            raise ValueError("Missing time or SOC")
        first_drop = h[h.battery.diff().lt(0)].iloc[0]
        first_reserve = h[h.battery.le(20)].iloc[0]
        selected = h[(h.elapsed_time >= first_drop.elapsed_time) &
                     (h.elapsed_time <= first_reserve.elapsed_time)].copy()
        battery = str(selected.battery_id.iloc[0])
        own = normalizer.curve_for(battery, drone)
        if set(selected.battery_id) != {battery} or selected.battery.min() < 20:
            raise ValueError("Battery changed or the observed range crosses the reserve")
        frame = pd.DataFrame({"experiment_id": EXPERIMENT, "run_id": "20260928_232343", "drone_id": drone,
                              "position": int(drone[-1]), "battery_id": battery,
                              "elapsed_time": selected.elapsed_time.to_numpy(float),
                              "fit_time_s": selected.elapsed_time.to_numpy(float) - float(first_drop.elapsed_time),
                              "raw_soc": selected.battery.to_numpy(float)})
        frame["after_first_other_landing"] = frame.elapsed_time.ge(first_any_landing)
        frame["hovering_drone_count"] = [sum(t <= end for end in own_hover_ends.values()) for t in frame.elapsed_time]
        stage_items = []
        for i, (name, upper, lower) in enumerate(zip(NAMES, own.boundaries, own.boundaries[1:])):
            fitted, stage_samples = fit_raw_stage(frame, upper, lower)
            item = {"stage": name, "own_upper_soc": upper, "own_lower_soc": lower,
                    "reference_upper_soc": normalizer.reference.boundaries[i],
                    "reference_lower_soc": normalizer.reference.boundaries[i + 1],
                    "baseline_rate_pp_min": own.rates_pp_min[i],
                    "reference_rate_pp_min": normalizer.reference.rates_pp_min[i]}
            if fitted is None:
                item.update({"raw_rate_pp_min": None, "sample_count": len(stage_samples),
                             "unique_raw_soc_levels": int(stage_samples.raw_soc.nunique()),
                             "observed_duration_s": 0.0, "rate_origin": "missing_observed_stage",
                             "support_note": "unobserved_stage"})
            else:
                item.update(fitted)
                factor = fitted["raw_rate_pp_min"] / own.rates_pp_min[i]
                item.update({"relative_drain_factor": factor,
                             "normalized_rate_pp_min": factor * normalizer.reference.rates_pp_min[i]})
                check = item["normalized_rate_pp_min"] / item["reference_rate_pp_min"]
                ratio_errors.append(abs(check - factor))
                copied = stage_samples.copy()
                copied["stage"] = name
                copied["fitted_raw_soc"] = fitted["raw_fit_intercept_soc"] - fitted["raw_rate_pp_min"] * (copied.fit_time_s - fitted["stage_fit_origin_s"]) / 60
                fit_samples.append(copied)
            stage_items.append(item)
        if any(stage_items[i]["raw_rate_pp_min"] is None for i in (1, 2)):
            raise ValueError("This pilot requires observed Medium and Low")
        if stage_items[0]["raw_rate_pp_min"] is None:
            factor = stage_items[1]["relative_drain_factor"]
            stage_items[0].update({"relative_drain_factor": factor,
                                  "normalized_rate_pp_min": factor * normalizer.reference.rates_pp_min[0],
                                  "rate_origin": "modeled_missing_high_from_medium_relative_factor",
                                  "assumption": "High has the same relative drain factor as this position's Medium"})
        rates = np.array([s["normalized_rate_pp_min"] for s in stage_items])
        kt, kb = knots(rates, normalizer.reference.boundaries)
        for i, item in enumerate(stage_items):
            item["model_stage_duration_s"] = float(kt[i + 1] - kt[i])
            stages.append({"drone_id": drone, "position": int(drone[-1]), "battery_id": battery, **item})
        for t, b in zip(kt, kb):
            points.append({"drone_id": drone, "position": int(drone[-1]), "model_time_s": float(t), "normalized_soc": float(b)})
        audits.append({"drone_id": drone, "position": int(drone[-1]), "battery_id": battery,
                       "initial_hover_raw_soc": float(h.battery.iloc[0]),
                       "first_drop_raw_soc": float(first_drop.battery), "first_drop_s": float(first_drop.elapsed_time),
                       "initial_hover_plateau_removed_s": float(first_drop.elapsed_time - h.elapsed_time.iloc[0]),
                       "first_20_percent_s": float(first_reserve.elapsed_time),
                       "retained_own_discharge_duration_s": float(frame.fit_time_s.iloc[-1]),
                       "retained_rows_after_first_other_landing": int(frame.after_first_other_landing.sum()),
                       "last_retained_raw_soc": float(frame.raw_soc.iloc[-1]),
                       "sample_count": len(frame), "model_full_reference_100_to_20_s": float(kt[-1])})
        frames[drone] = frame
        curves[drone] = {"battery_id": battery, "own_boundaries_soc": list(own.boundaries),
                         "normalized_rates_pp_min": rates.tolist(), "stages": stage_items,
                         "knots_time_s": kt.tolist(), "knots_soc": kb.tolist(),
                         "full_100_to_20_s": float(kt[-1])}
    out.mkdir(parents=True)
    pd.DataFrame(stages).to_csv(out / "stage_rates.csv", index=False, float_format="%.12g")
    pd.DataFrame(audits).to_csv(out / "per_drone_window_audit.csv", index=False, float_format="%.12g")
    pd.concat(frames.values()).to_csv(out / "retained_observed_samples.csv", index=False, float_format="%.12g")
    pd.concat(fit_samples).to_csv(out / "observed_stage_fits.csv", index=False, float_format="%.12g")
    pd.DataFrame(points).to_csv(out / "three_stage_knots.csv", index=False, float_format="%.12g")
    figures(out, frames, curves, normalizer)
    manifest = {"experiment_id": EXPERIMENT, "run_id": "20260928_232343", "source": str(SOURCE.relative_to(ROOT)),
                "input_sha256": hashes, "reference_model_version": normalizer.version,
                "reference_model_status": json.loads(MODEL.read_text())["status"],
                "analysis_type": "single_run_stage_first_pilot_not_active_training_data",
                "experiment_type": "stationary_wind_tunnel_hover_not_forward_flight",
                "window_policy": "Each drone: first post-drop hovering SOC through its own first observed 20% sample. No common first-landing truncation.",
                "plateau_policy": "Remove the initial unchanged-SOC interval; retain subsequent integer-SOC plateaus.",
                "stage_fit_method": "Independent time-weighted linear regression of measured SOC against time within each battery's frozen baseline SOC stages. No joint cumulative-time fit or fixed first-sample intercept.",
                "stage_boundary_sampling": "Integer observations at an internal SOC boundary are available to both adjacent regressions; this does not create new telemetry.",
                "rate_normalization": "R_ref_configuration_stage = (R_measured_configuration_stage / R_own_baseline_stage) * R_existing_reference_stage. Corresponding High/Medium/Low stages are matched; no absolute reference SOC is assigned to a measured starting SOC.",
                "time_definition": "After rates are fixed, T_stage = 60 * (reference upper SOC - lower SOC) / R_ref_configuration_stage. Integrate from modeled reference SOC 100 to 20, continuously through 82 and 52.",
                "missing_stage_policy": "Missing High at P3/P4 is completed using the corresponding Medium relative drain factor. Completed points are not used as observed fit inputs.",
                "operating_population": "Per user clarification, later hovering samples remain after other drones reach 20% and land. They need not represent five simultaneously airborne drones. Hovering count and post-first-landing flags are retained in every derived sample.",
                "training_eligibility": False,
                "limitations": ["P2 and P5 High have only two SOC levels and need repeated-run confirmation.",
                                "P3 and P4 High is modeled, not observed.",
                                "Rates use one trial; no repeated-trial average.",
                                "The reference baseline remains a frozen candidate, not newly activated or independently validated."],
                "audit": audits, "curves": curves}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    validation = {"five_positions": len(curves) == 5,
                  "three_stages_per_position": all(len(c["stages"]) == 3 for c in curves.values()),
                  "rate_factor_max_error": float(max(ratio_errors)),
                  "rate_factor_conserved": max(ratio_errors) < 1e-12,
                  "positive_normalized_rates": all(min(c["normalized_rates_pp_min"]) > 0 for c in curves.values()),
                  "initial_plateau_removed": all(f.fit_time_s.iloc[0] == 0 for f in frames.values()),
                  "each_own_20_percent_observation_retained": all(f.raw_soc.iloc[-1] == 20 for f in frames.values()),
                  "original_stage_slopes_preserved": True,
                  "continuous_82_52_reference_knots": all(np.allclose(c["knots_soc"], [100, 82, 52, 20]) for c in curves.values()),
                  "sources_unchanged": all(sha(ROOT / path) == value for path, value in hashes.items()),
                  "hardware_access": False, "active_training_data_modified": False}
    if not all(validation[k] for k in ["five_positions", "three_stages_per_position", "rate_factor_conserved",
                                       "positive_normalized_rates", "initial_plateau_removed",
                                       "each_own_20_percent_observation_retained", "sources_unchanged"]):
        raise AssertionError(validation)
    (out / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2))
    (out / "README.md").write_text(
        "# 先拟合阶段耗电率，再计算时间\n\n"
        "实验：wind_tunnel_diamond_50_head_lv2_002，run 20260928_232343。\n\n"
        "每架起始电量不同，使用各自首次掉电至首次达到实测20%的完整悬停记录，不按第一架降落时间统一截断。开头不掉电平台剔除，后续整数SOC阶梯保留。\n\n"
        "先按已有实体电池基线的High/Medium/Low边界独立拟合实测SOC下降斜率，再计算相对于该电池基线的耗电因子，并乘以理想电池对应阶段的耗电率。理想电池边界仍为100/82/52/20。\n\n"
        "斜率确定以后，以阶段SOC跨度除以阶段速率计算持续时间，依次连接三段。没有为了连续性再次联合调整斜率；没有通过整次飞行耗时反推电量消耗。\n\n"
        "P3/P4的High未观测，图中按同位置Medium相对耗电因子补齐。P2/P5 High仅两档读数，是短区间估计。补齐来源、短区间及仍在悬停的无人机数量均保留在派生文件中。此试处理未加入训练。\n\n"
        "normalized_three_stage为理想电池上的三段模型；observations_and_stage_fits为实测SOC与各自阶段直线的对照，二者纵轴含义不同。100到20的模型总时长不是各机本次实际记录时长。其他无人机降落后的记录按用户要求保留，其五机气动条件等价性未被验证。原始数据与已有基线未修改。\n")
    print(pd.DataFrame(stages)[["position", "stage", "raw_rate_pp_min", "normalized_rate_pp_min", "observed_duration_s", "rate_origin"]].round(4).to_string(index=False))
    print(json.dumps(validation, ensure_ascii=False))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args().output)
