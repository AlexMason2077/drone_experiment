"""Build meeting-ready condition-by-configuration conclusions and figures.

Condition = wind direction + wind level.
Configuration = formation + position + inter-drone distance.

Primary metric is median motion-only energy relative to the matching independent
single-drone battery baseline. Mission-pad waiting is removed and every flight is
normalized to 250 cm by the upstream energy knowledge-base workflow.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "swarm_analysis" / "algorithm_energy_knowledge_base" / "swarm_drone_energy_rows.csv"
OUT = ROOT / "swarm_analysis" / "meeting_condition_configuration"
CHARTS = OUT / "charts"
WINDS = ["head", "side", "tail"]
LEVELS = [1, 2]
CONDITIONS = [f"{wind.title()} Lv{level}" for wind in WINDS for level in LEVELS]
FORMATION_ORDER = ["front", "vee", "diamond", "echalon", "column"]
FORMATION_COLORS = {
    "front": "#2878B5",
    "vee": "#D9911B",
    "diamond": "#9B6AA6",
    "echalon": "#5B8C3A",
    "column": "#C65A50",
}


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(12000, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.percentile(medians, [2.5, 97.5]))


def load_and_summarize() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(SOURCE, low_memory=False)
    rows["run_key"] = rows["experiment_id"].astype(str) + "|" + rows["csv_run_id"].astype(str)
    rows["condition"] = (
        rows["wind_direction_short"].str.title()
        + " Lv"
        + rows["wind_level"].astype(int).astype(str)
    )

    primary_key = ["experiment_id", "csv_run_id", "csv_drone_name"]
    if rows.duplicated(primary_key).any():
        raise RuntimeError("Duplicate drone-run rows found")
    run_sizes = rows.groupby(["experiment_id", "csv_run_id"]).agg(
        rows=("csv_drone_name", "size"),
        drones=("csv_drone_name", "nunique"),
        positions=("position", "nunique"),
    )
    if not ((run_sizes["rows"] == 5) & (run_sizes["drones"] == 5) & (run_sizes["positions"] == 5)).all():
        raise RuntimeError("Incomplete or duplicated five-drone runs found")

    keys = ["condition", "wind_direction_short", "wind_level", "formation", "distance", "position"]
    summary = (
        rows.groupby(keys, as_index=False)
        .agg(
            n_runs=("run_key", "nunique"),
            median_relative_energy_pct=("relative_energy_vs_baseline_pct", "median"),
            q1_relative_energy_pct=("relative_energy_vs_baseline_pct", lambda s: s.quantile(.25)),
            q3_relative_energy_pct=("relative_energy_vs_baseline_pct", lambda s: s.quantile(.75)),
            median_motion_only_energy_sec=("pure_forward_energy_250cm", "median"),
            median_start_soc=("csv_battery_hover_start", "median"),
            low_quantization_share=("low_energy_quantization_flag", "mean"),
            outlier_share=("statistical_outlier_flag", "mean"),
        )
    )
    summary["supported"] = (
        (summary["n_runs"] >= 3)
        & (summary["low_quantization_share"] <= .25)
        & (summary["outlier_share"] <= .25)
    )
    summary["rank_within_condition"] = summary.groupby("condition")[
        "median_relative_energy_pct"
    ].rank(method="min")
    summary["rank_percentile_within_condition"] = summary.groupby("condition")[
        "median_relative_energy_pct"
    ].rank(method="average", pct=True)
    summary["configuration"] = (
        summary["formation"].str.capitalize()
        + " · P"
        + summary["position"].astype(int).astype(str)
        + " · "
        + summary["distance"].astype(int).astype(str)
        + " cm"
    )
    return rows, summary


def top_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    top = (
        summary[summary["supported"]]
        .sort_values(["condition", "median_relative_energy_pct"])
        .groupby("condition", group_keys=False)
        .head(5)
        .copy()
    )
    top["condition"] = pd.Categorical(top["condition"], CONDITIONS, ordered=True)
    return top.sort_values(["condition", "median_relative_energy_pct"])


def position_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = (
        summary.groupby(["condition", "position"], as_index=False)
        .agg(
            median_relative_energy_pct=("median_relative_energy_pct", "median"),
            configuration_count=("configuration", "size"),
        )
    )
    result["rank_within_condition"] = result.groupby("condition")[
        "median_relative_energy_pct"
    ].rank(method="min")
    return result


def matched_distance_summary(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = summary.pivot_table(
        index=["condition", "wind_direction_short", "wind_level", "formation", "position"],
        columns="distance",
        values="median_relative_energy_pct",
    ).dropna().reset_index()
    paired["delta_75_minus_50_pct_points"] = paired[75.0] - paired[50.0]
    rng = np.random.default_rng(42)
    records = []
    for condition in CONDITIONS:
        values = paired.loc[
            paired["condition"] == condition, "delta_75_minus_50_pct_points"
        ].to_numpy(float)
        ci_low, ci_high = bootstrap_median_ci(values, rng)
        records.append(
            {
                "condition": condition,
                "matched_configuration_count": len(values),
                "median_delta_75_minus_50_pct_points": float(np.median(values)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "share_75cm_lower_energy": float(np.mean(values < 0)),
                "wilcoxon_p_value": float(wilcoxon(values).pvalue),
            }
        )
    return paired, pd.DataFrame(records)


def matched_position_summary(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["condition", "formation", "distance"],
        columns="position",
        values="median_relative_energy_pct",
    ).dropna()
    delta = (pivot[5] - pivot[3]).to_numpy(float)
    rng = np.random.default_rng(7)
    ci_low, ci_high = bootstrap_median_ci(delta, rng)
    return pd.DataFrame(
        [
            {
                "comparison": "Position 5 minus Position 3",
                "matched_cells": len(delta),
                "median_difference_pct_points": float(np.median(delta)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "share_position_5_higher_energy": float(np.mean(delta > 0)),
                "wilcoxon_p_value": float(wilcoxon(delta).pvalue),
            }
        ]
    )


def matched_formation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    records = []
    for formation in ["vee", "diamond", "echalon", "column"]:
        subset = summary[summary["formation"].isin(["front", formation])]
        pivot = subset.pivot_table(
            index=["condition", "distance", "position"],
            columns="formation",
            values="median_relative_energy_pct",
        ).dropna()
        delta = (pivot[formation] - pivot["front"]).to_numpy(float)
        ci_low, ci_high = bootstrap_median_ci(delta, rng)
        records.append(
            {
                "comparison": f"{formation.capitalize()} minus Front",
                "matched_cells": len(delta),
                "median_difference_pct_points": float(np.median(delta)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "share_front_lower_energy": float(np.mean(delta > 0)),
                "wilcoxon_p_value": float(wilcoxon(delta).pvalue),
            }
        )
    return pd.DataFrame(records)


def stable_configuration_summary(summary: pd.DataFrame) -> pd.DataFrame:
    stable = (
        summary.groupby(["formation", "distance", "position"], as_index=False)
        .agg(
            conditions_observed=("condition", "nunique"),
            supported_conditions=("supported", "sum"),
            average_rank_percentile=("rank_percentile_within_condition", "mean"),
            median_relative_energy_across_conditions=("median_relative_energy_pct", "median"),
            worst_condition_relative_energy=("median_relative_energy_pct", "max"),
            median_runs_per_condition=("n_runs", "median"),
        )
    )
    stable["configuration"] = (
        stable["formation"].str.capitalize()
        + " · P"
        + stable["position"].astype(int).astype(str)
        + " · "
        + stable["distance"].astype(int).astype(str)
        + " cm"
    )
    return stable.sort_values(["average_rank_percentile", "median_relative_energy_across_conditions"])


def plot_top_candidates(top: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.2), dpi=240, sharex=True)
    for ax, condition in zip(axes.flat, CONDITIONS):
        data = top[top["condition"] == condition].sort_values(
            "median_relative_energy_pct", ascending=False
        )
        y = np.arange(len(data))
        med = data["median_relative_energy_pct"].to_numpy(float)
        q1 = data["q1_relative_energy_pct"].to_numpy(float)
        q3 = data["q3_relative_energy_pct"].to_numpy(float)
        colors = [FORMATION_COLORS[value] for value in data["formation"]]
        ax.hlines(y, q1, q3, color=colors, linewidth=3.5, alpha=.42)
        ax.scatter(med, y, color=colors, s=62, edgecolor="#30363B", linewidth=.6, zorder=3)
        for yi, value, n_runs in zip(y, med, data["n_runs"]):
            ax.text(value + 2.0, yi, f"{value:+.0f}  (n={int(n_runs)})", va="center", fontsize=8)
        ax.axvline(0, color="#4E555B", linestyle="--", linewidth=1)
        ax.set_yticks(y, data["configuration"])
        ax.set_title(condition, loc="left", weight="bold")
        ax.grid(axis="x", color="#E1E5E8", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Motion-only energy relative to battery baseline (%)\nLower is better; point = median, line = IQR")
    fig.suptitle(
        "Top configuration candidates within each wind condition",
        x=.055,
        y=.995,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        .055,
        .958,
        "Configuration = formation + position + spacing; candidates require ≥3 runs and limited quantization/outlier flags",
        color="#59636E",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, .94], h_pad=2.0, w_pad=1.8)
    fig.savefig(CHARTS / "01_top_configurations_by_condition.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_position_heatmap(position: pd.DataFrame) -> None:
    matrix = position.pivot(index="position", columns="condition", values="median_relative_energy_pct")
    matrix = matrix.reindex(index=[1, 2, 3, 4, 5], columns=CONDITIONS)
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=240)
    image = ax.imshow(matrix.to_numpy(float), cmap="RdYlBu_r", vmin=-10, vmax=80, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            color = "white" if value >= 58 else "#22282D"
            ax.text(col, row, f"{value:+.0f}%", ha="center", va="center", color=color, fontsize=10, weight="bold")
        best_col = int(np.nanargmin(matrix.iloc[row].to_numpy(float)))
        _ = best_col
    for col in range(matrix.shape[1]):
        best_row = int(np.nanargmin(matrix.iloc[:, col].to_numpy(float)))
        ax.add_patch(plt.Rectangle((col - .48, best_row - .48), .96, .96, fill=False, edgecolor="#1D2328", linewidth=2.0))
    ax.set_xticks(range(len(CONDITIONS)), CONDITIONS)
    ax.set_yticks(range(5), [f"Position {position}" for position in range(1, 6)])
    ax.set_title("Position effect across wind conditions", loc="left", weight="bold", fontsize=15, pad=18)
    ax.text(
        0,
        1.03,
        "Median across available formations and spacings; lower relative energy is better; outlined cell is best in each condition",
        transform=ax.transAxes,
        color="#59636E",
        fontsize=9.2,
    )
    cbar = fig.colorbar(image, ax=ax, pad=.02)
    cbar.set_label("Relative motion-only energy (%)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS / "02_position_effect_by_condition.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distance_effect(paired: pd.DataFrame, distance_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=240)
    y_positions = np.arange(len(CONDITIONS))[::-1]
    condition_colors = {
        "Head Lv1": "#2878B5",
        "Head Lv2": "#78A8CC",
        "Side Lv1": "#D9911B",
        "Side Lv2": "#E8B85B",
        "Tail Lv1": "#5B8C3A",
        "Tail Lv2": "#91B777",
    }
    for y, condition in zip(y_positions, CONDITIONS):
        values = paired.loc[
            paired["condition"] == condition, "delta_75_minus_50_pct_points"
        ].to_numpy(float)
        offsets = np.linspace(-.13, .13, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            values,
            y + offsets,
            s=28,
            color=condition_colors[condition],
            alpha=.55,
            edgecolor="none",
        )
        row = distance_summary[distance_summary["condition"] == condition].iloc[0]
        ax.plot(
            [row["bootstrap_ci_low"], row["bootstrap_ci_high"]],
            [y, y],
            color=condition_colors[condition],
            linewidth=5,
            alpha=.85,
        )
        ax.scatter(
            row["median_delta_75_minus_50_pct_points"],
            y,
            s=85,
            marker="D",
            color=condition_colors[condition],
            edgecolor="#30363B",
            linewidth=.7,
            zorder=4,
        )
        ax.text(
            82,
            y,
            f"median {row['median_delta_75_minus_50_pct_points']:+.1f}; "
            f"75 cm better {row['share_75cm_lower_energy']:.0%}",
            va="center",
            fontsize=8.8,
        )
    ax.axvline(0, color="#30363B", linestyle="--", linewidth=1.2)
    ax.set_xlim(-115, 145)
    ax.set_yticks(y_positions, CONDITIONS)
    ax.set_xlabel("Relative-energy difference: 75 cm − 50 cm (percentage points)")
    ax.set_title("Spacing effect changes with wind condition", loc="left", weight="bold", fontsize=15, pad=18)
    ax.text(
        0,
        1.03,
        "Negative values favor 75 cm; dots are matched formation–position comparisons, diamonds are medians, thick lines are bootstrap 95% CIs",
        transform=ax.transAxes,
        color="#59636E",
        fontsize=9.1,
    )
    ax.grid(axis="x", color="#E1E5E8", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS / "03_spacing_effect_by_condition.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    rows, summary = load_and_summarize()
    top = top_candidates(summary)
    positions = position_summary(summary)
    paired_distance, distance_summary = matched_distance_summary(summary)
    position_test = matched_position_summary(summary)
    formation_test = matched_formation_summary(summary)
    stable = stable_configuration_summary(summary)

    summary.to_csv(OUT / "configuration_summary_by_condition.csv", index=False)
    top.to_csv(OUT / "top_configuration_candidates.csv", index=False)
    positions.to_csv(OUT / "position_effect_summary.csv", index=False)
    paired_distance.to_csv(OUT / "matched_spacing_pairs.csv", index=False)
    distance_summary.to_csv(OUT / "spacing_effect_summary.csv", index=False)
    position_test.to_csv(OUT / "position_3_vs_5_test.csv", index=False)
    formation_test.to_csv(OUT / "formation_vs_front_tests.csv", index=False)
    stable.to_csv(OUT / "cross_condition_configuration_stability.csv", index=False)

    plot_top_candidates(top)
    plot_position_heatmap(positions)
    plot_distance_effect(paired_distance, distance_summary)

    print("complete five-drone runs", rows["run_key"].nunique())
    print("configuration-condition cells", len(summary), "supported", int(summary["supported"].sum()))
    print("\nTop candidate per condition")
    print(
        top.sort_values(["condition", "median_relative_energy_pct"])
        .groupby("condition", observed=True)
        .head(1)[["condition", "configuration", "n_runs", "median_relative_energy_pct"]]
        .to_string(index=False)
    )
    print("\nPosition comparison")
    print(position_test.round(4).to_string(index=False))
    print("\nFormation comparisons")
    print(formation_test.round(4).to_string(index=False))
    print("\nDistance comparisons")
    print(distance_summary.round(4).to_string(index=False))
    print("\nMost stable configurations")
    print(stable[stable["conditions_observed"] == 6].head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
