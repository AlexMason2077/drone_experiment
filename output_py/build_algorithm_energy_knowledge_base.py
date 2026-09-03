"""Build an algorithm-facing swarm energy knowledge base.

The raw Tello battery signal is an integer SOC percentage and discharge is
non-linear in SOC.  This pipeline converts every SOC transition to an
equivalent number of hover-discharge seconds using battery-specific hover
curves.  Stationary time caused by staggered/safety-controlled formation
strategies is then removed on that same energy scale.  All 300 cm runs are
normalized to the first 250 cm-equivalent traversal.

Outputs are descriptive empirical estimates.  B12 is mapped to B15, following
the experiment-owner assumption that the replacement battery has the same
discharge curve.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "swarm_analysis" / "algorithm_energy_knowledge_base"
CHARTS = OUT / "charts"
FORMATIONS = ["front", "vee", "diamond", "echalon", "column"]
DIRECTIONS = ["head", "side", "tail"]
BATTERY_MAP = {"B12": "B15"}
COLORS = {
    "front": "#2878B5",
    "vee": "#D99A22",
    "diamond": "#D9534F",
    "echalon": "#658B38",
    "column": "#A14F86",
}


def effective_battery(battery_id: str) -> str:
    return BATTERY_MAP.get(str(battery_id), str(battery_id))


def crossing_time(frame: pd.DataFrame, soc: float) -> float:
    """Interpolate the first time the monotone envelope reaches ``soc``."""
    x = frame[["elapsed_time", "battery"]].dropna().sort_values("elapsed_time").copy()
    if x.empty:
        return np.nan
    x["battery_monotone"] = x.battery.cummin()
    before = x[x.battery_monotone > soc].tail(1)
    after = x[x.battery_monotone <= soc].head(1)
    if after.empty:
        return np.nan
    if before.empty:
        return float(after.elapsed_time.iloc[0])
    a = before.iloc[0]
    b = after.iloc[0]
    if a.battery_monotone == b.battery_monotone:
        return float(b.elapsed_time)
    weight = (a.battery_monotone - soc) / (a.battery_monotone - b.battery_monotone)
    return float(a.elapsed_time + weight * (b.elapsed_time - a.elapsed_time))


def build_hover_curves() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    summary = pd.read_csv(ROOT / "output_graph" / "hover_battery_runs_summary.csv")
    summary = summary[summary.status.eq("included")].copy()
    batteries = ["B10", "B11", "B13", "B14", "B15"]
    grid = np.arange(10.0, 101.0)
    curve_rows: list[dict] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for battery_id in batteries:
        run_curves = []
        for row in summary[summary.battery_id.eq(battery_id)].itertuples():
            path = Path(row.source_file)
            if not path.is_absolute():
                path = ROOT / path
            try:
                frame = pd.read_csv(path, low_memory=False)
            except Exception:
                continue
            t10 = crossing_time(frame, 10.0)
            if not np.isfinite(t10):
                continue
            remaining = np.array([
                t10 - crossing_time(frame, soc) if np.isfinite(crossing_time(frame, soc)) else np.nan
                for soc in grid
            ])
            run_curves.append(remaining)
        if not run_curves:
            raise RuntimeError(f"No complete hover curve found for {battery_id}")
        median_curve = np.nanmedian(np.vstack(run_curves), axis=0)
        valid = np.isfinite(median_curve)
        median_curve = np.interp(grid, grid[valid], median_curve[valid])
        median_curve = np.maximum.accumulate(median_curve)
        curves[battery_id] = (grid.copy(), median_curve)
        for soc, seconds in zip(grid, median_curve):
            curve_rows.append({
                "battery_id": battery_id,
                "soc": soc,
                "remaining_hover_seconds_to_10pct": seconds,
                "complete_hover_runs": len(run_curves),
            })

    return curves, pd.DataFrame(curve_rows)


def hover_energy(curves: dict[str, tuple[np.ndarray, np.ndarray]], battery_id: str, start: float, end: float) -> float:
    bid = effective_battery(battery_id)
    grid, remaining = curves[bid]
    start = float(np.clip(start, grid.min(), grid.max()))
    end = float(np.clip(end, grid.min(), grid.max()))
    return float(np.interp(start, grid, remaining) - np.interp(end, grid, remaining))


def build_baseline(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(ROOT / "analysis_outputs" / "initial_baseline_quality.csv", low_memory=False)
    base = base[
        base["mode"].eq("head_forward_250")
        & base.battery_id.isin(["B10", "B11", "B13", "B14", "B15"])
        & base.battery_drop.notna()
        & base.battery_hover_start.between(40, 80)
    ].copy()

    # The May-14 block predates the explicit baseline_wind_level field.  Per the
    # experiment owner's description it is the independent forward-flight
    # reference block; later explicit wind-condition runs are not pooled into it.
    base["run_date"] = base.run_id.astype(str).str.slice(0, 8)
    reference = base[base.run_date.eq("20260514")].copy()
    reference["equivalent_hover_seconds"] = [
        hover_energy(curves, bid, start, end)
        for bid, start, end in zip(reference.battery_id, reference.battery_hover_start, reference.battery_hover_end)
    ]
    reference["stationary_control_overhead_sec"] = (reference.node_duration_sec - 25.0).clip(lower=0)
    reference["pure_forward_equivalent_hover_seconds"] = (
        reference.equivalent_hover_seconds - reference.stationary_control_overhead_sec
    )
    reference = reference[
        reference.pure_forward_equivalent_hover_seconds.gt(0)
        & reference.pure_forward_equivalent_hover_seconds.lt(reference.pure_forward_equivalent_hover_seconds.quantile(0.99))
    ].copy()

    models = (
        reference.groupby("battery_id", as_index=False)
        .agg(
            baseline_energy_median=("pure_forward_equivalent_hover_seconds", "median"),
            baseline_energy_q25=("pure_forward_equivalent_hover_seconds", lambda s: s.quantile(.25)),
            baseline_energy_q75=("pure_forward_equivalent_hover_seconds", lambda s: s.quantile(.75)),
            baseline_runs=("run_id", "nunique"),
            baseline_start_soc_min=("battery_hover_start", "min"),
            baseline_start_soc_max=("battery_hover_start", "max"),
        )
    )
    return reference, models


def prepare_swarm_rows(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], baseline_models: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv(ROOT / "swarm_analysis" / "clean_swarm_drone_rows.csv", low_memory=False)
    motion = pd.read_csv(ROOT / "swarm_analysis" / "pure_forward" / "pure_forward_drone_rows.csv", low_memory=False)
    keys = ["experiment_id", "run_id", "drone_name"]
    motion = motion.rename(columns={"run_id": "csv_run_id", "drone_name": "csv_drone_name"})
    data = clean.merge(
        motion[["experiment_id", "csv_run_id", "csv_drone_name", "active_forward_sec", "stationary_wait_sec", "commanded_distance_cm"]],
        on=["experiment_id", "csv_run_id", "csv_drone_name"], how="inner",
    )
    data["effective_battery_id"] = data.csv_battery_id.map(effective_battery)
    data["position"] = data.csv_takeoff_order.astype(int)
    data["observed_energy_hover_seconds"] = [
        hover_energy(curves, bid, start, end)
        for bid, start, end in zip(data.effective_battery_id, data.csv_battery_hover_start, data.csv_battery_hover_end)
    ]

    # A 250/300 cm traversal at the commanded 10 cm/s cannot physically take
    # less than 25/30 s.  The trajectory detector sometimes under-counts active
    # motion when a repeated mission-pad ID is reacquired; enforce this physical
    # lower bound and keep a flag for auditability.
    data["minimum_motion_sec"] = data.commanded_distance_cm / 10.0
    data["motion_floor_applied"] = data.active_forward_sec.lt(data.minimum_motion_sec)
    data["active_forward_sec_adjusted"] = data[["active_forward_sec", "minimum_motion_sec"]].max(axis=1)
    data["active_forward_sec_adjusted"] = data[["active_forward_sec_adjusted", "csv_node_duration_sec"]].min(axis=1)
    data["stationary_wait_sec_adjusted"] = (
        data.csv_node_duration_sec - data.active_forward_sec_adjusted
    ).clip(lower=0)
    data["pure_forward_energy_full_distance"] = (
        data.observed_energy_hover_seconds - data.stationary_wait_sec_adjusted
    )
    data["pure_forward_energy_250cm"] = (
        data.pure_forward_energy_full_distance * 250.0 / data.commanded_distance_cm
    )

    data = data.merge(
        baseline_models[["battery_id", "baseline_energy_median", "baseline_energy_q25", "baseline_energy_q75", "baseline_runs"]],
        left_on="effective_battery_id", right_on="battery_id", how="left", suffixes=("", "_baseline"),
    )
    data["relative_energy_vs_baseline_pct"] = 100 * (
        data.pure_forward_energy_250cm / data.baseline_energy_median - 1
    )
    data["energy_ratio_vs_baseline"] = data.pure_forward_energy_250cm / data.baseline_energy_median

    data["low_energy_quantization_flag"] = data.pure_forward_energy_250cm.le(5)
    data["high_energy_quality_flag"] = data.pure_forward_energy_250cm.gt(180)
    excluded = data[data.baseline_energy_median.isna()].copy()
    if not excluded.empty:
        excluded["exclusion_reason"] = "missing baseline model"
        data = data.drop(excluded.index).copy()

    # Robust within-cell/position outlier screen.  Keep the threshold generous
    # because repeated runs are sparse in some design cells.
    group_cols = ["formation", "distance", "wind_direction_short", "wind_level", "position"]
    med = data.groupby(group_cols).pure_forward_energy_250cm.transform("median")
    mad = (data.pure_forward_energy_250cm - med).abs().groupby([data[c] for c in group_cols]).transform("median")
    robust_z = 0.6745 * (data.pure_forward_energy_250cm - med) / mad.replace(0, np.nan)
    data["energy_robust_z"] = robust_z
    data["statistical_outlier_flag"] = robust_z.abs().gt(4.5).fillna(False)

    # Configuration comparisons require all five positions from the same run.
    # Quantized zero-drop rows are retained rather than deleting the entire run;
    # medians and IQRs across repeats absorb their bounded measurement error.
    complete = data.groupby(["experiment_id", "csv_run_id"]).position.nunique()
    complete_keys = set(complete[complete.eq(5)].index)
    incomplete = data[
        ~data.apply(lambda row: (row.experiment_id, row.csv_run_id) in complete_keys, axis=1)
    ].copy()
    if not incomplete.empty:
        incomplete["exclusion_reason"] = "run lost one or more positions after quality checks"
        excluded = pd.concat([excluded, incomplete], ignore_index=True, sort=False)
    data = data[data.apply(lambda row: (row.experiment_id, row.csv_run_id) in complete_keys, axis=1)].copy()
    return data, excluded


def quantile_summary(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        data.groupby(group_cols, as_index=False)
        .agg(
            energy_median=("pure_forward_energy_250cm", "median"),
            energy_q25=("pure_forward_energy_250cm", lambda s: s.quantile(.25)),
            energy_q75=("pure_forward_energy_250cm", lambda s: s.quantile(.75)),
            relative_vs_baseline_median_pct=("relative_energy_vs_baseline_pct", "median"),
            relative_vs_baseline_q25_pct=("relative_energy_vs_baseline_pct", lambda s: s.quantile(.25)),
            relative_vs_baseline_q75_pct=("relative_energy_vs_baseline_pct", lambda s: s.quantile(.75)),
            run_count=("csv_run_id", "nunique"),
            observation_count=("csv_drone_name", "size"),
            starting_soc_min=("csv_battery_hover_start", "min"),
            starting_soc_max=("csv_battery_hover_start", "max"),
            motion_floor_share=("motion_floor_applied", "mean"),
            low_energy_quantization_share=("low_energy_quantization_flag", "mean"),
            statistical_outlier_share=("statistical_outlier_flag", "mean"),
        )
    )


def build_knowledge_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    position_cols = ["formation", "distance", "wind_direction_short", "wind_level", "position", "effective_battery_id"]
    position = quantile_summary(data, position_cols)

    run = (
        data.groupby(["experiment_id", "csv_run_id", "formation", "distance", "wind_direction_short", "wind_level"], as_index=False)
        .agg(
            total_energy=("pure_forward_energy_250cm", "sum"),
            mean_drone_energy=("pure_forward_energy_250cm", "mean"),
            max_drone_energy=("pure_forward_energy_250cm", "max"),
            min_drone_energy=("pure_forward_energy_250cm", "min"),
            energy_sd=("pure_forward_energy_250cm", "std"),
            mean_relative_vs_baseline_pct=("relative_energy_vs_baseline_pct", "mean"),
            max_relative_vs_baseline_pct=("relative_energy_vs_baseline_pct", "max"),
            stationary_wait_sec=("stationary_wait_sec_adjusted", "mean"),
            duration_sec=("csv_node_duration_sec", "median"),
        )
    )
    run["energy_range"] = run.max_drone_energy - run.min_drone_energy
    run["physically_valid_total_energy"] = run.total_energy.gt(0)

    valid_run = run[run.physically_valid_total_energy].copy()
    config = (
        valid_run.groupby(["formation", "distance", "wind_direction_short", "wind_level"], as_index=False)
        .agg(
            total_energy_median=("total_energy", "median"),
            total_energy_q25=("total_energy", lambda s: s.quantile(.25)),
            total_energy_q75=("total_energy", lambda s: s.quantile(.75)),
            mean_drone_energy_median=("mean_drone_energy", "median"),
            max_drone_energy_median=("max_drone_energy", "median"),
            energy_range_median=("energy_range", "median"),
            energy_sd_median=("energy_sd", "median"),
            relative_vs_baseline_median_pct=("mean_relative_vs_baseline_pct", "median"),
            stationary_wait_sec_median=("stationary_wait_sec", "median"),
            run_count=("csv_run_id", "nunique"),
        )
    )
    config["rank_within_wind_and_distance"] = (
        config.groupby(["wind_direction_short", "wind_level", "distance"]).total_energy_median.rank(method="min")
    ).astype(int)
    config["rank_configuration_within_wind"] = (
        config.groupby(["wind_direction_short", "wind_level"]).total_energy_median.rank(method="min")
    ).astype(int)
    config["evidence_strength"] = pd.cut(
        config.run_count, bins=[0, 2, 4, np.inf], labels=["low", "moderate", "higher"]
    ).astype(str)
    position["evidence_strength"] = pd.cut(
        position.run_count, bins=[0, 2, 4, np.inf], labels=["low", "moderate", "higher"]
    ).astype(str)
    return position, run, config


def build_algorithm_lookup(position: pd.DataFrame, config: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["formation", "distance", "wind_direction_short", "wind_level"]
    energy_wide = position.pivot(index=keys, columns="position", values="energy_median")
    energy_wide.columns = [f"position_{int(col)}_energy_median" for col in energy_wide.columns]
    relative_wide = position.pivot(index=keys, columns="position", values="relative_vs_baseline_median_pct")
    relative_wide.columns = [f"position_{int(col)}_relative_vs_baseline_pct" for col in relative_wide.columns]
    lookup = config.merge(energy_wide.reset_index(), on=keys, how="left").merge(relative_wide.reset_index(), on=keys, how="left")

    extreme_rows = []
    for group_key, group in position.groupby(keys):
        best = group.loc[group.energy_median.idxmin()]
        worst = group.loc[group.energy_median.idxmax()]
        extreme_rows.append({
            **dict(zip(keys, group_key)),
            "lowest_energy_position": int(best.position),
            "lowest_position_energy": best.energy_median,
            "highest_energy_position": int(worst.position),
            "highest_position_energy": worst.energy_median,
            "position_energy_spread": worst.energy_median - best.energy_median,
            "minimum_position_run_count": int(group.run_count.min()),
        })
    extremes = pd.DataFrame(extreme_rows)

    left = config[config.distance.eq(50)].copy()
    right = config[config.distance.eq(75)].copy()
    distance = left.merge(right, on=["formation", "wind_direction_short", "wind_level"], suffixes=("_50", "_75"))
    distance["energy_change_75_minus_50"] = distance.mean_drone_energy_median_75 - distance.mean_drone_energy_median_50
    distance["energy_change_75_vs_50_pct"] = 100 * distance.energy_change_75_minus_50 / distance.mean_drone_energy_median_50
    distance["preferred_distance_by_energy"] = np.where(distance.energy_change_75_minus_50 < 0, 75, 50)
    return lookup, extremes, distance


def bootstrap_rank_stability(run: pd.DataFrame, iterations: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(20260728)
    valid = run[run.physically_valid_total_energy].copy()
    rows = []
    for condition, condition_data in valid.groupby(["wind_direction_short", "wind_level", "distance"]):
        groups = {
            formation: group.mean_drone_energy.to_numpy(float)
            for formation, group in condition_data.groupby("formation")
        }
        formations = [formation for formation in FORMATIONS if formation in groups]
        simulated = np.empty((iterations, len(formations)))
        for column, formation in enumerate(formations):
            values = groups[formation]
            draws = rng.choice(values, size=(iterations, len(values)), replace=True)
            simulated[:, column] = np.median(draws, axis=1)
        ranks = np.argsort(np.argsort(simulated, axis=1), axis=1) + 1
        winners = np.argmin(simulated, axis=1)
        for column, formation in enumerate(formations):
            p_best = float(np.mean(winners == column))
            rows.append({
                "wind_direction_short": condition[0],
                "wind_level": condition[1],
                "distance": condition[2],
                "formation": formation,
                "probability_best": p_best,
                "bootstrap_rank_median": float(np.median(ranks[:, column])),
                "bootstrap_rank_q25": float(np.quantile(ranks[:, column], .25)),
                "bootstrap_rank_q75": float(np.quantile(ranks[:, column], .75)),
                "bootstrap_iterations": iterations,
            })
    result = pd.DataFrame(rows)
    result["ranking_stability"] = pd.cut(
        result.probability_best, bins=[-np.inf, .35, .70, np.inf], labels=["uncertain", "moderate", "strong"]
    ).astype(str)
    return result


def plot_baseline(reference: pd.DataFrame, models: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    order = ["B11", "B10", "B13", "B14", "B15"]
    battery_colors = dict(zip(order, ["#2878B5", "#D99A22", "#D9534F", "#658B38", "#A14F86"]))
    for idx, battery_id in enumerate(order):
        x = reference[reference.battery_id.eq(battery_id)]
        y = models.set_index("battery_id").loc[battery_id]
        jitter = (idx - 2) * .035
        ax.scatter(x.battery_hover_start + jitter, x.pure_forward_equivalent_hover_seconds, s=22, alpha=.28, color=battery_colors[battery_id])
        ax.hlines(y.baseline_energy_median, 39, 81, color=battery_colors[battery_id], lw=2.2, label=battery_id)
    ax.set_xlim(39, 81)
    ax.set_xlabel("Starting SOC (%)")
    ax.set_ylabel("Energy used (equivalent hover seconds / 250 cm)")
    fig.suptitle("Independent single-drone forward-flight baseline", x=.10, y=.98, ha="left", weight="bold", fontsize=15)
    fig.text(.10, .925, "SOC converted with battery-specific hover curves; B12 uses the B15 curve", color="#59636e")
    ax.legend(frameon=False, ncol=5)
    ax.grid(axis="y", color="#e1e5e8")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, .90])
    fig.savefig(CHARTS / "01_single_drone_energy_baseline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_condition_rankings(config: pd.DataFrame) -> None:
    for direction in DIRECTIONS:
        for level in [1, 2]:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), dpi=180, sharey=True)
            subset = config[(config.wind_direction_short.eq(direction)) & (config.wind_level.eq(level))]
            for ax, distance in zip(axes, [50, 75]):
                x = subset[subset.distance.eq(distance)].set_index("formation").reindex(FORMATIONS).dropna(subset=["total_energy_median"])
                y = x.total_energy_median / 5
                lo = y - x.total_energy_q25 / 5
                hi = x.total_energy_q75 / 5 - y
                positions = np.arange(len(x))
                ax.errorbar(positions, y, yerr=[lo, hi], fmt="none", ecolor="#66717b", capsize=4, lw=1.5)
                ax.scatter(positions, y, s=90, c=[COLORS[name] for name in x.index], edgecolor="#30363b", zorder=3)
                baseline = y / (1 + x.relative_vs_baseline_median_pct / 100)
                ax.scatter(positions, baseline, marker="_", s=420, color="#20262b", linewidth=1.5, label="Matched single-drone baseline")
                for pos, value, rank in zip(positions, y, x.rank_within_wind_and_distance):
                    ax.text(pos, value + max(1.0, y.max()*.025), f"#{int(rank)}", ha="center", fontsize=9, weight="bold")
                ax.set_xticks(positions, x.index, rotation=20)
                ax.set_title(f"{distance} cm spacing", loc="left", weight="bold")
                ax.set_ylabel("Median energy per drone\n(equivalent hover seconds / 250 cm)")
                ax.grid(axis="y", color="#e1e5e8")
                ax.spines[["top", "right"]].set_visible(False)
            axes[1].legend(frameon=False, loc="upper right")
            fig.suptitle(f"{direction} wind · level {level}: formation energy ranking", x=.06, ha="left", weight="bold", fontsize=15)
            fig.text(.06, .915, "Dots show pure-forward energy after removing stationary waiting; bars span the run-level IQR", color="#59636e")
            fig.tight_layout(rect=[0, 0, 1, .90])
            fig.savefig(CHARTS / f"02_formation_ranking_{direction}_lv{level}.png", bbox_inches="tight", facecolor="white")
            plt.close(fig)


def plot_position_profiles(position: pd.DataFrame) -> None:
    for formation in FORMATIONS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=170, sharex=True, sharey=True)
        for ax, (direction, level) in zip(axes.flat, [(d, l) for d in DIRECTIONS for l in [1, 2]]):
            x = position[
                position.formation.eq(formation)
                & position.wind_direction_short.eq(direction)
                & position.wind_level.eq(level)
            ]
            for distance, marker, line in [(50, "o", "-"), (75, "s", "--")]:
                z = x[x.distance.eq(distance)].set_index("position").reindex([1, 2, 3, 4, 5])
                ax.plot(z.index, z.relative_vs_baseline_median_pct, marker=marker, ls=line, lw=2, label=f"{distance} cm")
            ax.axhline(0, color="#252a30", lw=1.2)
            ax.set_title(f"{direction} · lv{level}", loc="left", weight="bold")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlabel("Formation position")
            ax.set_ylabel("Energy difference vs position baseline (%)")
            ax.grid(color="#e1e5e8")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, 2].legend(frameon=False)
        fig.suptitle(f"{formation}: position-specific energy under each wind condition", x=.055, ha="left", weight="bold", fontsize=15)
        fig.text(.055, .935, "Each position is compared with its own battery's independent single-drone baseline", color="#59636e")
        fig.tight_layout(rect=[0, 0, 1, .92])
        fig.savefig(CHARTS / f"03_position_profiles_{formation}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_distance_effect(config: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=180, sharey=True)
    for ax, (direction, level) in zip(axes.flat, [(d, l) for d in DIRECTIONS for l in [1, 2]]):
        x = config[(config.wind_direction_short.eq(direction)) & (config.wind_level.eq(level))]
        for formation in FORMATIONS:
            z = x[x.formation.eq(formation)].set_index("distance").reindex([50, 75])
            ax.plot([50, 75], z.mean_drone_energy_median, marker="o", lw=2, color=COLORS[formation], label=formation)
        ax.set_title(f"{direction} · lv{level}", loc="left", weight="bold")
        ax.set_xticks([50, 75])
        ax.set_xlabel("Inter-drone distance (cm)")
        ax.set_ylabel("Median energy per drone")
        ax.grid(color="#e1e5e8")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 2].legend(frameon=False, ncol=2)
    fig.suptitle("Inter-drone distance effect under matched formation and wind", x=.055, ha="left", weight="bold", fontsize=15)
    fig.text(.055, .935, "Missing column 50 cm lv2 conditions are structural safety omissions and are not imputed", color="#59636e")
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(CHARTS / "04_distance_effect_by_wind.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    curves, curve_table = build_hover_curves()
    baseline_reference, baseline_models = build_baseline(curves)
    data, excluded = prepare_swarm_rows(curves, baseline_models)
    position, run, config = build_knowledge_tables(data)
    lookup, position_extremes, distance_effects = build_algorithm_lookup(position, config)
    rank_stability = bootstrap_rank_stability(run)
    config = config.merge(
        rank_stability,
        on=["formation", "distance", "wind_direction_short", "wind_level"],
        how="left",
    )
    lookup = lookup.drop(columns=[col for col in rank_stability.columns if col in lookup.columns and col not in ["formation", "distance", "wind_direction_short", "wind_level"]], errors="ignore").merge(
        rank_stability,
        on=["formation", "distance", "wind_direction_short", "wind_level"],
        how="left",
    )

    curve_table.to_csv(OUT / "battery_hover_energy_curves.csv", index=False)
    baseline_reference.to_csv(OUT / "single_drone_baseline_reference_rows.csv", index=False)
    baseline_models.to_csv(OUT / "single_drone_baseline_energy_models.csv", index=False)
    data.to_csv(OUT / "swarm_drone_energy_rows.csv", index=False)
    excluded.to_csv(OUT / "excluded_energy_rows.csv", index=False)
    position.to_csv(OUT / "position_energy_knowledge_base.csv", index=False)
    run.to_csv(OUT / "swarm_run_energy_profiles.csv", index=False)
    config.to_csv(OUT / "configuration_energy_knowledge_base.csv", index=False)
    lookup.to_csv(OUT / "algorithm_configuration_lookup.csv", index=False)
    position_extremes.to_csv(OUT / "position_energy_extremes.csv", index=False)
    distance_effects.to_csv(OUT / "matched_distance_effects.csv", index=False)
    rank_stability.to_csv(OUT / "formation_rank_stability_bootstrap.csv", index=False)

    plot_baseline(baseline_reference, baseline_models)
    plot_condition_rankings(config)
    plot_position_profiles(position)
    plot_distance_effect(config)

    print("retained drone rows", len(data))
    print("retained complete runs", data[["experiment_id", "csv_run_id"]].drop_duplicates().shape[0])
    print("configuration cells", len(config))
    print("position cells", len(position))
    print("excluded rows", len(excluded))
    print("motion floor applied", int(data.motion_floor_applied.sum()))
    print("charts", len(list(CHARTS.glob("*.png"))))
    print("\nRANKINGS")
    print(config.sort_values(["wind_direction_short", "wind_level", "distance", "rank_within_wind_and_distance"])[[
        "wind_direction_short", "wind_level", "distance", "rank_within_wind_and_distance", "formation",
        "mean_drone_energy_median", "relative_vs_baseline_median_pct", "energy_range_median", "run_count"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
