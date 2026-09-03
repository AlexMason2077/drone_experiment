"""Clean and analyze swarm energy patterns for downstream routing/charging optimization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db_copy_for_cleaning"
INPUT = ROOT / "analysis_outputs" / "initial_swarm_drone_quality.csv"
OUT = ROOT / "swarm_analysis"
CHARTS = OUT / "charts"
FORMATIONS = ["front", "vee", "diamond", "echalon", "column"]
DIRECTIONS = ["head", "side", "tail"]
PALETTE = {"front": "#1f77b4", "vee": "#d28e00", "diamond": "#d55e00", "echalon": "#6f7f22", "column": "#b44c7a"}


def ci95(values: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(x) < 2:
        return (float(x[0]), float(x[0])) if len(x) else (np.nan, np.nan)
    sem = stats.sem(x)
    margin = stats.t.ppf(0.975, len(x) - 1) * sem
    return float(x.mean() - margin), float(x.mean() + margin)


def clean_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = json.loads((DB / "experiment_registry.json").read_text())["experiments"]
    registry_map = {x["experiment_id"]: x for x in registry}
    data = pd.read_csv(INPUT, low_memory=False)
    data["registry_outlier"] = data.experiment_id.map(lambda x: bool(registry_map.get(x, {}).get("is_outlier")))
    data["exclude_reason"] = ""
    data.loc[data.registry_outlier, "exclude_reason"] = "registry is_outlier=true"
    data.loc[data.csv_battery_drop.isna(), "exclude_reason"] = "missing battery summary"
    data.loc[data.csv_node_duration_sec.le(0), "exclude_reason"] = "non-positive duration"
    data.loc[data.csv_battery_drop.lt(0), "exclude_reason"] = "negative battery drop"
    data.loc[(data.csv_battery_hover_start - data.csv_battery_hover_end - data.csv_battery_drop).abs().gt(0.01), "exclude_reason"] = "battery drop mismatch"
    excluded = data[data.exclude_reason.ne("")].copy()
    clean = data[data.exclude_reason.eq("")].copy()
    clean["wind_direction_short"] = clean.wind_direction.str.replace(" wind", "", regex=False)
    clean["battery_drop_rate_pct_per_min"] = clean.csv_battery_drop / clean.csv_node_duration_sec * 60

    run_keys = ["experiment_id", "csv_run_id", "formation", "distance", "wind_direction_short", "wind_level"]
    runs = clean.groupby(run_keys, as_index=False).agg(
        drone_count=("csv_drone_name", "nunique"),
        battery_count=("csv_battery_id", "nunique"),
        mean_battery_drop=("csv_battery_drop", "mean"),
        max_battery_drop=("csv_battery_drop", "max"),
        sd_battery_drop=("csv_battery_drop", "std"),
        mean_drop_rate=("battery_drop_rate_pct_per_min", "mean"),
        max_drop_rate=("battery_drop_rate_pct_per_min", "max"),
        duration_sec=("csv_node_duration_sec", "median"),
        mean_start_battery=("csv_battery_hover_start", "mean"),
        start_battery_range=("csv_battery_hover_start", lambda x: x.max() - x.min()),
    )
    runs = runs[(runs.drone_count == 5) & (runs.battery_count == 5)].copy()
    return clean, runs, excluded


def scenario_summary(runs: pd.DataFrame) -> pd.DataFrame:
    dims = ["formation", "distance", "wind_direction_short", "wind_level"]
    rows = []
    for keys, group in runs.groupby(dims):
        lo, hi = ci95(group.mean_battery_drop)
        rows.append({
            **dict(zip(dims, keys)), "run_count": len(group),
            "mean_battery_drop": group.mean_battery_drop.mean(),
            "ci_low": lo, "ci_high": hi,
            "mean_max_battery_drop": group.max_battery_drop.mean(),
            "mean_duration_sec": group.duration_sec.mean(),
            "mean_drop_rate": group.mean_drop_rate.mean(),
        })
    return pd.DataFrame(rows)


def factor_summary(cells: pd.DataFrame) -> pd.DataFrame:
    # Formation uses only settings observed for all five formations.
    setting_cols = ["distance", "wind_direction_short", "wind_level"]
    common = cells.groupby(setting_cols).formation.nunique()
    common_keys = set(common[common == len(FORMATIONS)].index)
    formation_cells = cells[cells.apply(lambda r: tuple(r[c] for c in setting_cols) in common_keys, axis=1)]
    rows = []
    for factor, source in [
        ("formation", formation_cells), ("distance", cells),
        ("wind_direction_short", cells), ("wind_level", cells),
    ]:
        for level, group in source.groupby(factor):
            lo, hi = ci95(group.mean_battery_drop)
            rows.append({
                "factor": factor, "level": str(level), "cell_count": len(group),
                "mean_battery_drop": group.mean_battery_drop.mean(), "ci_low": lo, "ci_high": hi,
                "mean_max_battery_drop": group.mean_max_battery_drop.mean(),
                "mean_duration_sec": group.mean_duration_sec.mean(),
                "mean_drop_rate": group.mean_drop_rate.mean(),
                "comparison_scope": "common 10 settings" if factor == "formation" else "all observed cells",
            })
    return pd.DataFrame(rows)


def plot_factor_effects(summary: pd.DataFrame, path: Path) -> None:
    configs = [
        ("formation", FORMATIONS, "Formation"),
        ("wind_level", ["1.0", "2.0", "1", "2"], "Wind speed level"),
        ("wind_direction_short", DIRECTIONS, "Wind direction"),
        ("distance", ["50.0", "75.0", "50", "75"], "Inter-drone distance (cm)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    for ax, (factor, order, title) in zip(axes.flat, configs):
        x = summary[summary.factor == factor].copy()
        rank = {v: i for i, v in enumerate(order)}
        x["order"] = x.level.map(lambda v: rank.get(v, 99))
        x = x.sort_values("order")
        positions = np.arange(len(x))
        ax.errorbar(
            positions, x.mean_battery_drop,
            yerr=[x.mean_battery_drop - x.ci_low, x.ci_high - x.mean_battery_drop],
            fmt="o", color="#1f77b4", ecolor="#93a8b8", capsize=4, linewidth=1.6, markersize=7,
        )
        ax.set_xticks(positions, x.level.str.replace(".0", "", regex=False))
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylabel("Mean battery drop per node (% points)")
        ax.grid(axis="y", color="#d9dee3", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Swarm energy main effects", x=0.06, ha="left", fontsize=17, weight="bold")
    fig.text(0.06, 0.94, "Cell-balanced means; formation comparison restricted to settings shared by all five formations", color="#59636e", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_interaction_heatmaps(cells: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=180)
    for row, level in enumerate([1, 2]):
        for col, distance in enumerate([50, 75]):
            ax = axes[row, col]
            subset = cells[(cells.wind_level == level) & (cells.distance == distance)]
            pivot = subset.pivot_table(index="formation", columns="wind_direction_short", values="mean_battery_drop")
            pivot = pivot.reindex(index=FORMATIONS, columns=DIRECTIONS)
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrBr", vmin=4, vmax=14.5, linewidths=.5,
                        cbar=(col == 1), ax=ax, mask=pivot.isna())
            ax.set_title(f"Wind level {level} · spacing {distance} cm", loc="left", weight="bold")
            ax.set_xlabel("Wind direction")
            ax.set_ylabel("Formation")
    fig.suptitle("Mean battery drop by formation and wind condition", x=0.06, ha="left", fontsize=17, weight="bold")
    fig.text(0.06, 0.94, "Values are mean percentage-point drops per 250 cm node traversal; blank cells have no valid non-outlier run", color="#59636e", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distance_interaction(cells: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=180, sharey=True)
    for ax, wind_level in zip(axes, [1, 2]):
        subset = cells[cells.wind_level == wind_level]
        for formation in FORMATIONS:
            formation_data = subset[subset.formation == formation]
            directions_by_distance = {
                distance: set(group.wind_direction_short)
                for distance, group in formation_data.groupby("distance")
            }
            common_directions = set.intersection(*directions_by_distance.values()) if len(directions_by_distance) == 2 else set()
            matched = formation_data[formation_data.wind_direction_short.isin(common_directions)]
            x = matched.groupby("distance").mean_battery_drop.mean()
            ax.plot(x.index, x.values, marker="o", linewidth=2, label=formation, color=PALETTE[formation])
        ax.set_xticks([50, 75])
        ax.set_title(f"Wind level {wind_level}", loc="left", weight="bold")
        ax.set_xlabel("Inter-drone distance (cm)")
        ax.set_ylabel("Mean battery drop per node (% points)")
        ax.grid(axis="y", color="#d9dee3", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle("Spacing effect changes with formation and wind speed", x=0.06, ha="left", fontsize=17, weight="bold")
    fig.text(0.06, 0.91, "Direction-matched scenario-cell means; each line uses only wind directions observed at both spacings", color="#59636e", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_scenario_ranking(cells: pd.DataFrame, path: Path) -> None:
    x = cells.copy()
    x["scenario"] = x.apply(lambda r: f"{r.formation} · {int(r.distance)}cm · {r.wind_direction_short} · lv{int(r.wind_level)}", axis=1)
    x = pd.concat([x.nsmallest(10, "mean_battery_drop"), x.nlargest(10, "mean_battery_drop")]).drop_duplicates("scenario")
    x = x.sort_values("mean_battery_drop")
    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    colors = [PALETTE[f] for f in x.formation]
    ax.barh(x.scenario, x.mean_battery_drop, color=colors, edgecolor="#4f5963", linewidth=.5)
    for idx, row in enumerate(x.itertuples(index=False)):
        ax.text(row.mean_battery_drop + 0.15, idx, f"n={int(row.run_count)}", va="center", fontsize=8, color="#59636e")
    ax.set_xlabel("Mean battery drop per node (% points)")
    ax.set_title("Lowest- and highest-consumption observed scenarios", loc="left", fontsize=16, weight="bold", pad=28)
    ax.text(0, 1.015, "Observed scenario-cell means; labels show the number of retained runs", transform=ax.transAxes, color="#59636e")
    ax.set_xlim(left=0)
    ax.grid(axis="x", color="#d9dee3", linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_duration_relationship(runs: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7), dpi=180)
    for formation in FORMATIONS:
        x = runs[runs.formation == formation]
        ax.scatter(x.duration_sec, x.mean_battery_drop, s=32, alpha=.65, color=PALETTE[formation], label=formation)
    slope, intercept, r, p, _ = stats.linregress(runs.duration_sec, runs.mean_battery_drop)
    grid = np.linspace(runs.duration_sec.min(), runs.duration_sec.max(), 100)
    ax.plot(grid, intercept + slope * grid, color="#252a30", linestyle="--", linewidth=2,
            label=f"Overall fit: r={r:.2f}")
    ax.set_xlabel("Node traversal duration (seconds)")
    ax.set_ylabel("Mean battery drop (% points)")
    ax.set_title("Battery use rises with time spent reaching the node", loc="left", fontsize=16, weight="bold")
    ax.text(0, 1.015, "Each point is one five-drone swarm run", transform=ax.transAxes, color="#59636e")
    ax.legend(frameon=False, ncol=2)
    ax.grid(color="#e1e5e8", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_charging_bottleneck(factors: pd.DataFrame, path: Path) -> None:
    x = factors[factors.factor == "formation"].set_index("level").reindex(FORMATIONS).reset_index()
    positions = np.arange(len(x))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=180)
    ax.bar(positions - width/2, x.mean_battery_drop, width, color="#6ba3c8", label="Average drone")
    ax.bar(positions + width/2, x.mean_max_battery_drop, width, color="#d55e00", label="Most depleted drone")
    ax.set_xticks(positions, x.level)
    ax.set_ylabel("Battery drop per node (% points)")
    ax.set_title("Formation-level charging burden", loc="left", fontsize=16, weight="bold", pad=28)
    ax.text(0, 1.015, "Common-condition means; the most depleted drone is the no-parallel-charging completion bottleneck",
            transform=ax.transAxes, color="#59636e")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#d9dee3", linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_duration_intensity_decomposition(factors: pd.DataFrame, path: Path) -> None:
    x = factors[factors.factor == "formation"].set_index("level").reindex(FORMATIONS)
    baseline = x.loc["front"]
    metrics = pd.DataFrame({
        "Total battery drop": x.mean_battery_drop / baseline.mean_battery_drop * 100,
        "Traversal duration": x.mean_duration_sec / baseline.mean_duration_sec * 100,
        "Drop rate": x.mean_drop_rate / baseline.mean_drop_rate * 100,
    })
    positions = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    for idx, (metric, color) in enumerate(zip(metrics.columns, ["#1f77b4", "#d55e00", "#6f7f22"])):
        ax.plot(positions, metrics[metric], marker="o", linewidth=2.3, markersize=7, label=metric, color=color)
    ax.axhline(100, color="#59636e", linestyle="--", linewidth=1.2)
    ax.set_xticks(positions, metrics.index)
    ax.set_ylabel("Index (front = 100)")
    ax.set_title("Formation energy differences are largely duration-driven", loc="left", fontsize=16, weight="bold", pad=28)
    ax.text(0, 1.015, "Common-condition formation means; indices compare each formation with front", transform=ax.transAxes, color="#59636e")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#d9dee3", linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def role_summary(clean: pd.DataFrame) -> pd.DataFrame:
    role = clean.groupby(["formation", "csv_takeoff_order"], as_index=False).agg(
        rows=("csv_battery_drop", "size"), mean_drop=("csv_battery_drop", "mean"),
        mean_rate=("battery_drop_rate_pct_per_min", "mean"),
    )
    role["formation_mean_drop"] = role.groupby("formation").mean_drop.transform("mean")
    role["drop_vs_formation_mean"] = role.mean_drop - role.formation_mean_drop
    return role


def main() -> None:
    OUT.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)
    clean, runs, excluded = clean_data()
    cells = scenario_summary(runs)
    factors = factor_summary(cells)
    roles = role_summary(clean)
    clean.to_csv(OUT / "clean_swarm_drone_rows.csv", index=False)
    runs.to_csv(OUT / "clean_swarm_runs.csv", index=False)
    excluded.to_csv(OUT / "excluded_swarm_rows.csv", index=False)
    cells.to_csv(OUT / "scenario_cell_summary.csv", index=False)
    factors.to_csv(OUT / "factor_effect_summary.csv", index=False)
    roles.to_csv(OUT / "formation_position_summary.csv", index=False)
    plot_factor_effects(factors, CHARTS / "01_main_effects.png")
    plot_interaction_heatmaps(cells, CHARTS / "02_formation_wind_heatmaps.png")
    plot_distance_interaction(cells, CHARTS / "03_spacing_interactions.png")
    plot_scenario_ranking(cells, CHARTS / "04_scenario_ranking.png")
    plot_duration_relationship(runs, CHARTS / "05_duration_vs_battery_drop.png")
    plot_charging_bottleneck(factors, CHARTS / "06_charging_bottleneck.png")
    plot_duration_intensity_decomposition(factors, CHARTS / "07_duration_intensity_decomposition.png")
    print(f"Clean drone rows: {len(clean)}")
    print(f"Clean swarm runs: {len(runs)}")
    print(f"Observed scenario cells: {len(cells)}")
    print("\nFACTOR SUMMARY")
    print(factors.to_string(index=False))
    print("\nCORRELATION duration vs drop", runs.duration_sec.corr(runs.mean_battery_drop))


if __name__ == "__main__":
    main()
