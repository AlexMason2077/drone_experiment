from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "swarm_analysis" / "algorithm_energy_knowledge_base"
OUT = ROOT / "swarm_analysis" / "meeting_formation_comparison"

FORMATIONS = ["front", "vee", "diamond", "echalon", "column"]
FORMATION_LABELS = {
    "front": "Front",
    "vee": "Vee",
    "diamond": "Diamond",
    "echalon": "Echalon",
    "column": "Column",
}
WINDS = ["head", "side", "tail"]
WIND_LABELS = {"head": "Head wind", "side": "Side wind", "tail": "Tail wind"}
LEVEL_COLORS = {1: "#2878B5", 2: "#D9901A"}


def q25(series: pd.Series) -> float:
    return float(series.quantile(0.25))


def q75(series: pd.Series) -> float:
    return float(series.quantile(0.75))


def build_run_level() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(SOURCE / "swarm_drone_energy_rows.csv", low_memory=False)
    runs = pd.read_csv(SOURCE / "swarm_run_energy_profiles.csv", low_memory=False)
    valid_runs = runs[runs["physically_valid_total_energy"]].copy()

    valid_keys = set(
        map(tuple, valid_runs[["experiment_id", "csv_run_id"]].to_numpy())
    )
    rows = rows[
        rows.apply(
            lambda row: (row["experiment_id"], row["csv_run_id"]) in valid_keys,
            axis=1,
        )
    ].copy()

    # Convert the whole-node observed battery energy and the estimated pad-wait
    # component to the same 250 cm basis.  The identity below is audited later:
    # corrected motion energy = observed energy - estimated stationary waiting.
    rows["observed_energy_250cm"] = (
        rows["observed_energy_hover_seconds"] * 250.0 / rows["commanded_distance_cm"]
    )
    rows["waiting_energy_250cm"] = (
        rows["stationary_wait_sec_adjusted"] * 250.0 / rows["commanded_distance_cm"]
    )

    run = (
        rows.groupby(
            [
                "experiment_id",
                "csv_run_id",
                "formation",
                "distance",
                "wind_direction_short",
                "wind_level",
            ],
            as_index=False,
        )
        .agg(
            observed_total_energy_250cm=("observed_energy_250cm", "sum"),
            waiting_energy_removed_250cm=("waiting_energy_250cm", "sum"),
            motion_only_total_energy_250cm=("pure_forward_energy_250cm", "sum"),
            max_drone_motion_energy_250cm=("pure_forward_energy_250cm", "max"),
            median_node_duration_sec=("csv_node_duration_sec", "median"),
            mean_stationary_wait_sec=("stationary_wait_sec_adjusted", "mean"),
            drone_count=("csv_drone_name", "nunique"),
        )
    )
    run["waiting_share_of_observed_pct"] = (
        100.0
        * run["waiting_energy_removed_250cm"]
        / run["observed_total_energy_250cm"]
    )
    run["energy_identity_error"] = (
        run["observed_total_energy_250cm"]
        - run["waiting_energy_removed_250cm"]
        - run["motion_only_total_energy_250cm"]
    )
    return rows, run


def summarize(run: pd.DataFrame) -> pd.DataFrame:
    summary = (
        run.groupby(
            ["formation", "distance", "wind_direction_short", "wind_level"],
            as_index=False,
        )
        .agg(
            median_motion_energy=("motion_only_total_energy_250cm", "median"),
            q25_motion_energy=("motion_only_total_energy_250cm", q25),
            q75_motion_energy=("motion_only_total_energy_250cm", q75),
            median_max_drone_energy=("max_drone_motion_energy_250cm", "median"),
            median_waiting_share_pct=("waiting_share_of_observed_pct", "median"),
            run_count=("csv_run_id", "nunique"),
        )
    )
    summary["rank_within_condition"] = (
        summary.groupby(["distance", "wind_direction_short", "wind_level"])[
            "median_motion_energy"
        ]
        .rank(method="min")
        .astype(int)
    )
    return summary


def bootstrap_rankings(run: pd.DataFrame, iterations: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(20260811)
    records: list[dict[str, float | int | str]] = []
    for condition, group in run.groupby(
        ["distance", "wind_direction_short", "wind_level"]
    ):
        values = {
            formation: sub["motion_only_total_energy_250cm"].to_numpy(float)
            for formation, sub in group.groupby("formation")
        }
        available = [formation for formation in FORMATIONS if formation in values]
        simulated = np.empty((iterations, len(available)))
        for idx, formation in enumerate(available):
            formation_values = values[formation]
            draws = rng.choice(
                formation_values,
                size=(iterations, len(formation_values)),
                replace=True,
            )
            simulated[:, idx] = np.median(draws, axis=1)

        ranks = np.argsort(np.argsort(simulated, axis=1), axis=1) + 1
        winners = np.argmin(simulated, axis=1)
        for idx, formation in enumerate(available):
            records.append(
                {
                    "distance": int(condition[0]),
                    "wind_direction_short": condition[1],
                    "wind_level": int(condition[2]),
                    "formation": formation,
                    "probability_lowest_energy": float(np.mean(winners == idx)),
                    "bootstrap_rank_median": float(np.median(ranks[:, idx])),
                    "bootstrap_rank_q25": float(np.quantile(ranks[:, idx], 0.25)),
                    "bootstrap_rank_q75": float(np.quantile(ranks[:, idx], 0.75)),
                    "bootstrap_iterations": iterations,
                }
            )
    return pd.DataFrame(records)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5F6368")
    ax.spines["bottom"].set_color("#5F6368")
    ax.tick_params(colors="#3C4043")
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.8, zorder=0)


def plot_spacing_comparison(summary: pd.DataFrame, distance: int, filename: str) -> None:
    subset = summary[summary["distance"].eq(distance)].copy()
    max_y = max(400.0, float(subset["q75_motion_energy"].max()) * 1.14)
    x = np.arange(len(FORMATIONS), dtype=float)
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True, dpi=190)
    for ax, wind in zip(axes, WINDS):
        for level, offset, hatch in [(1, -width / 2, ""), (2, width / 2, "//")]:
            cell = subset[
                subset["wind_direction_short"].eq(wind)
                & subset["wind_level"].eq(level)
            ].set_index("formation")
            medians = []
            lower = []
            upper = []
            counts = []
            tested = []
            for formation in FORMATIONS:
                if formation in cell.index:
                    row = cell.loc[formation]
                    medians.append(float(row["median_motion_energy"]))
                    lower.append(float(row["median_motion_energy"] - row["q25_motion_energy"]))
                    upper.append(float(row["q75_motion_energy"] - row["median_motion_energy"]))
                    counts.append(int(row["run_count"]))
                    tested.append(True)
                else:
                    medians.append(0.0)
                    lower.append(0.0)
                    upper.append(0.0)
                    counts.append(0)
                    tested.append(False)

            positions = x + offset
            bars = ax.bar(
                positions,
                medians,
                width=width,
                color=LEVEL_COLORS[level],
                edgecolor="#30343B",
                linewidth=0.7,
                hatch=hatch,
                alpha=0.94,
                label=f"Level {level}",
                zorder=3,
            )
            for idx, (bar, is_tested) in enumerate(zip(bars, tested)):
                if is_tested:
                    ax.errorbar(
                        positions[idx],
                        medians[idx],
                        yerr=np.array([[lower[idx]], [upper[idx]]]),
                        fmt="none",
                        ecolor="#25282D",
                        elinewidth=1.0,
                        capsize=3,
                        zorder=4,
                    )
                    ax.text(
                        positions[idx],
                        medians[idx] + upper[idx] + max_y * 0.018,
                        f"n={counts[idx]}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color="#4D5156",
                    )
                else:
                    bar.set_facecolor("white")
                    bar.set_edgecolor("#B5B9BF")
                    bar.set_hatch("xx")
                    ax.text(
                        positions[idx],
                        max_y * 0.035,
                        "Not\ntested",
                        ha="center",
                        va="bottom",
                        fontsize=7.2,
                        color="#777B82",
                    )

        ax.set_title(WIND_LABELS[wind], fontsize=12, color="#202124", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FORMATION_LABELS[f] for f in FORMATIONS], rotation=22, ha="right"
        )
        ax.set_ylim(0, max_y)
        style_axes(ax)

    axes[0].set_ylabel(
        "Five-drone motion-only energy\n(equivalent hover seconds / 250 cm)",
        color="#202124",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"Formation comparison at {distance} cm spacing",
        fontsize=16,
        color="#202124",
        y=0.975,
    )
    subtitle = (
        "Mission-pad waiting removed; all runs normalized to 250 cm; bars show median and IQR"
    )
    if distance == 50:
        subtitle += "; missing combinations are not imputed"
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=10, color="#5F6368")
    fig.text(
        0.5,
        0.012,
        "Lower is better. Error bars show the interquartile range across repeated five-drone runs.",
        ha="center",
        fontsize=9,
        color="#5F6368",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.79, wspace=0.08)
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def plot_rank_consistency(summary: pd.DataFrame) -> pd.DataFrame:
    subset = summary[summary["distance"].eq(75)].copy()
    subset["condition_best_energy"] = subset.groupby(
        ["wind_direction_short", "wind_level"]
    )["median_motion_energy"].transform("min")
    subset["excess_vs_condition_best_pct"] = 100.0 * (
        subset["median_motion_energy"] / subset["condition_best_energy"] - 1.0
    )
    rank = (
        subset.groupby("formation", as_index=False)
        .agg(
            mean_condition_rank=("rank_within_condition", "mean"),
            median_condition_rank=("rank_within_condition", "median"),
            best_condition_rank=("rank_within_condition", "min"),
            worst_condition_rank=("rank_within_condition", "max"),
            top_two_conditions=("rank_within_condition", lambda s: int((s <= 2).sum())),
            condition_count=("rank_within_condition", "size"),
            mean_excess_vs_condition_best_pct=("excess_vs_condition_best_pct", "mean"),
            median_excess_vs_condition_best_pct=("excess_vs_condition_best_pct", "median"),
            max_excess_vs_condition_best_pct=("excess_vs_condition_best_pct", "max"),
        )
        .sort_values(["mean_condition_rank", "formation"])
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.8), dpi=190)
    y = np.arange(len(rank))
    ax.hlines(
        y,
        rank["best_condition_rank"],
        rank["worst_condition_rank"],
        color="#B8BDC5",
        linewidth=4,
        zorder=1,
    )
    ax.scatter(
        rank["mean_condition_rank"],
        y,
        s=105,
        color="#2878B5",
        edgecolor="#1F3C55",
        linewidth=0.9,
        zorder=3,
    )
    for idx, row in rank.iterrows():
        ax.text(
            row["mean_condition_rank"] + 0.12,
            idx,
            f"mean {row['mean_condition_rank']:.2f}; top-2 in {int(row['top_two_conditions'])}/6",
            va="center",
            fontsize=9,
            color="#30343B",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([FORMATION_LABELS[f] for f in rank["formation"]])
    ax.invert_yaxis()
    ax.set_xlim(0.8, 5.15)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Mean within-condition rank (1 = lowest energy)")
    ax.set_title("Formation rank consistency at 75 cm spacing", fontsize=15, pad=26)
    ax.text(
        0.0,
        1.035,
        "Each of the six wind direction × level conditions receives equal weight",
        transform=ax.transAxes,
        fontsize=10,
        color="#5F6368",
    )
    ax.grid(axis="x", color="#E6E8EB", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#5F6368")
    ax.tick_params(colors="#3C4043")
    fig.text(
        0.5,
        0.015,
        "Dot = mean rank; grey range = best to worst observed condition rank. Descriptive, not a significance test.",
        ha="center",
        fontsize=9,
        color="#5F6368",
    )
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.14, top=0.82)
    fig.savefig(OUT / "03_formation_rank_consistency_75cm.png", bbox_inches="tight")
    plt.close(fig)
    return rank


def plot_waiting_audit(run: pd.DataFrame) -> pd.DataFrame:
    subset = run[run["distance"].eq(75)].copy()
    audit = (
        subset.groupby("formation", as_index=False)
        .agg(
            median_waiting_share_pct=("waiting_share_of_observed_pct", "median"),
            q25_waiting_share_pct=("waiting_share_of_observed_pct", q25),
            q75_waiting_share_pct=("waiting_share_of_observed_pct", q75),
            run_count=("csv_run_id", "nunique"),
        )
        .sort_values("median_waiting_share_pct", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=190)
    y = np.arange(len(audit))
    values = audit["median_waiting_share_pct"].to_numpy(float)
    lower = values - audit["q25_waiting_share_pct"].to_numpy(float)
    upper = audit["q75_waiting_share_pct"].to_numpy(float) - values
    ax.barh(
        y,
        values,
        color="#B9D3E8",
        edgecolor="#2878B5",
        linewidth=0.9,
        zorder=2,
    )
    ax.errorbar(
        values,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#25282D",
        elinewidth=1.1,
        capsize=3,
        zorder=3,
    )
    for idx, row in audit.iterrows():
        ax.text(
            row["median_waiting_share_pct"] + 0.7,
            idx,
            f"{row['median_waiting_share_pct']:.1f}%  (n={int(row['run_count'])})",
            va="center",
            fontsize=9,
            color="#30343B",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([FORMATION_LABELS[f] for f in audit["formation"]])
    ax.set_xlim(0, max(45, float(audit["q75_waiting_share_pct"].max()) + 7))
    ax.set_xlabel("Estimated waiting share of observed energy (%)")
    ax.set_title("Mission-pad waiting correction at 75 cm spacing", fontsize=15, pad=26)
    ax.text(
        0.0,
        1.035,
        "Median and IQR across runs, pooled only for this correction audit",
        transform=ax.transAxes,
        fontsize=10,
        color="#5F6368",
    )
    ax.grid(axis="x", color="#E6E8EB", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#5F6368")
    ax.tick_params(colors="#3C4043")
    fig.text(
        0.5,
        0.015,
        "This is a fairness diagnostic, not a formation-performance ranking.",
        ha="center",
        fontsize=9,
        color="#5F6368",
    )
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.14, top=0.82)
    fig.savefig(OUT / "04_mission_pad_waiting_correction_75cm.png", bbox_inches="tight")
    plt.close(fig)
    return audit


def build_condition_winners(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    winners = (
        summary[summary["distance"].eq(75)]
        .sort_values(
            ["wind_direction_short", "wind_level", "rank_within_condition"]
        )
        .groupby(["wind_direction_short", "wind_level"], as_index=False)
        .first()
    )
    winners = winners.merge(
        bootstrap[
            [
                "distance",
                "wind_direction_short",
                "wind_level",
                "formation",
                "probability_lowest_energy",
            ]
        ],
        on=["distance", "wind_direction_short", "wind_level", "formation"],
        how="left",
    )
    return winners


def build_quantization_sensitivity(rows: pd.DataFrame) -> pd.DataFrame:
    sensitivity = rows[
        [
            "experiment_id",
            "csv_run_id",
            "formation",
            "distance",
            "wind_direction_short",
            "wind_level",
            "pure_forward_energy_250cm",
        ]
    ].copy()
    sensitivity["clipped_drone_energy_250cm"] = sensitivity[
        "pure_forward_energy_250cm"
    ].clip(lower=0)
    run = (
        sensitivity.groupby(
            [
                "experiment_id",
                "csv_run_id",
                "formation",
                "distance",
                "wind_direction_short",
                "wind_level",
            ],
            as_index=False,
        )
        .agg(
            primary_total_energy=("pure_forward_energy_250cm", "sum"),
            clipped_total_energy=("clipped_drone_energy_250cm", "sum"),
        )
    )
    result = (
        run[run["distance"].eq(75)]
        .groupby(["formation", "wind_direction_short", "wind_level"], as_index=False)
        .agg(
            primary_median_energy=("primary_total_energy", "median"),
            clipped_median_energy=("clipped_total_energy", "median"),
            run_count=("csv_run_id", "nunique"),
        )
    )
    result["primary_rank"] = (
        result.groupby(["wind_direction_short", "wind_level"])[
            "primary_median_energy"
        ]
        .rank(method="min")
        .astype(int)
    )
    result["clipped_rank"] = (
        result.groupby(["wind_direction_short", "wind_level"])[
            "clipped_median_energy"
        ]
        .rank(method="min")
        .astype(int)
    )
    return result


def validate(rows: pd.DataFrame, run: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    checks: list[str] = []
    assert rows.groupby(["experiment_id", "csv_run_id"]).size().eq(5).all()
    checks.append("All included runs contain exactly five drone rows.")
    assert run["drone_count"].eq(5).all()
    checks.append("All included run-level records contain five distinct drones.")
    max_identity_error = float(run["energy_identity_error"].abs().max())
    assert max_identity_error < 1e-8
    checks.append(
        f"Energy subtraction identity verified; maximum absolute error = {max_identity_error:.3e}."
    )
    cells75 = summary[summary["distance"].eq(75)].groupby(
        ["wind_direction_short", "wind_level"]
    )["formation"].nunique()
    assert len(cells75) == 6 and cells75.eq(5).all()
    checks.append("75 cm is balanced: all five formations appear in all six conditions.")
    cells50 = summary[summary["distance"].eq(50)]
    expected = {
        (wind, level, formation)
        for wind in WINDS
        for level in [1, 2]
        for formation in FORMATIONS
    }
    observed = set(
        map(
            tuple,
            cells50[["wind_direction_short", "wind_level", "formation"]].to_numpy(),
        )
    )
    missing = sorted(expected - observed)
    assert missing == [
        ("head", 2, "column"),
        ("side", 2, "column"),
        ("side", 2, "diamond"),
    ]
    checks.append(
        "50 cm is incomplete: Head/L2 Column and Side/L2 Column/Diamond were not tested."
    )
    non_positive_components = int((rows["pure_forward_energy_250cm"] <= 0).sum())
    checks.append(
        f"Quantization audit: {non_positive_components}/{len(rows)} included drone components are non-positive after correction; sensitivity results are saved separately."
    )
    floor_share = float(rows["motion_floor_applied"].mean())
    checks.append(
        f"Trajectory audit: the commanded-speed physical motion floor was applied to {floor_share:.1%} of included drone rows."
    )
    return checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, run = build_run_level()
    summary = summarize(run)
    bootstrap = bootstrap_rankings(run)
    rank = plot_rank_consistency(summary)
    audit = plot_waiting_audit(run)
    winners = build_condition_winners(summary, bootstrap)
    quantization = build_quantization_sensitivity(rows)
    checks = validate(rows, run, summary)

    plot_spacing_comparison(summary, 75, "01_formation_comparison_75cm.png")
    plot_spacing_comparison(summary, 50, "02_formation_comparison_50cm_sensitivity.png")

    run.to_csv(OUT / "run_level_fair_energy.csv", index=False)
    summary.to_csv(OUT / "formation_comparison_summary.csv", index=False)
    bootstrap.to_csv(OUT / "formation_bootstrap_rankings.csv", index=False)
    rank.to_csv(OUT / "formation_rank_consistency_75cm.csv", index=False)
    audit.to_csv(OUT / "mission_pad_waiting_correction_75cm.csv", index=False)
    winners.to_csv(OUT / "condition_winners_75cm.csv", index=False)
    quantization.to_csv(OUT / "quantization_sensitivity_75cm.csv", index=False)
    (OUT / "validation_checks.txt").write_text("\n".join(f"PASS: {c}" for c in checks) + "\n")


if __name__ == "__main__":
    main()
