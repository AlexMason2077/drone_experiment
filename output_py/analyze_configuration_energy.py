"""Compare formation + spacing configurations within each wind condition.

The independent analytical unit is a five-drone run.  Drone-level reported SOC
drops are normalized by each physical battery's 75%-40% hover discharge rate,
then conservatively corrected only for trajectory-confirmed stationary waits.
All primary rankings use runs whose five drones remain inside 75%-40%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "db_copy_for_cleaning" / "_cleaning_admin"
TRAJECTORY = ADMIN / "trajectory_qc" / "trajectory_drone_segments.csv"
MASTER = ADMIN / "cleaning_master_run_index.csv"
BASELINES = ROOT / "db_copy_for_cleaning" / "baselines"
OUT = ROOT / "analysis_outputs" / "configuration_energy_analysis"

CONDITION_ORDER = [
    ("head", 1),
    ("head", 2),
    ("side", 1),
    ("side", 2),
    ("tail", 1),
    ("tail", 2),
]
FORMATION_LABEL = {
    "front": "Front",
    "vee": "Vee",
    "diamond": "Diamond",
    "echalon": "Echelon",
    "column": "Column",
}


def hover_calibration() -> pd.DataFrame:
    """Recompute battery-specific 75%-40% hover fits from clean traces."""

    sys.path.insert(0, str(ROOT / "output_py"))
    from generate_hover_battery_charts import (  # noqa: PLC0415
        SELECTED_MEAN_BATTERIES,
        cleaning_reason,
        find_hover_timeseries,
        load_hover_timeseries,
        mean_trace_for_battery,
    )
    from plot_hover_baseline_linear_range import clipped_segment  # noqa: PLC0415

    traces = []
    for path in find_hover_timeseries(BASELINES):
        trace, _ = load_hover_timeseries(path, max_points=2000)
        if trace is not None and not cleaning_reason(trace):
            traces.append(trace)

    rows = []
    for battery_id in SELECTED_MEAN_BATTERIES:
        group = [trace for trace in traces if trace["batteryId"] == battery_id]
        mean_trace = mean_trace_for_battery(battery_id, group, max_points=2000)
        segment = clipped_segment(mean_trace["points"]) if mean_trace else None
        if segment is None:
            continue
        time_min, reported_level = segment
        coefficient = np.polyfit(time_min, reported_level, 1)
        fitted = np.polyval(coefficient, time_min)
        residual = reported_level - fitted
        total = np.sum((reported_level - reported_level.mean()) ** 2)
        rows.append(
            {
                "battery_id": battery_id,
                "clean_hover_trace_count": len(group),
                "duration_75_to_40_min": float(time_min[-1]),
                "hover_discharge_rate_pp_per_min": abs(float(coefficient[0])),
                "linear_fit_r_squared": float(1 - np.sum(residual**2) / total),
                "linear_fit_rmse_pp": float(np.sqrt(np.mean(residual**2))),
            }
        )
    calibration = pd.DataFrame(rows)
    required = {"B10", "B11", "B12", "B13", "B14", "B15"}
    if set(calibration["battery_id"]) != required:
        raise RuntimeError("Missing one or more required physical-battery hover calibrations")
    return calibration


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).astype(float))


def winner_probabilities(
    run_metrics: pd.DataFrame,
    rng: np.random.Generator,
    draws: int = 10000,
) -> dict[tuple[str, int, str], float]:
    probabilities: dict[tuple[str, int, str], float] = {}
    for (wind_direction, wind_level), condition in run_metrics.groupby(
        ["wind_direction", "wind_level"]
    ):
        values = {
            configuration: group["adjusted_hover_equivalent_sec"].to_numpy(float)
            for configuration, group in condition.groupby("configuration")
        }
        wins = {configuration: 0 for configuration in values}
        for _ in range(draws):
            estimates = {
                configuration: float(rng.choice(sample, len(sample), replace=True).mean())
                for configuration, sample in values.items()
            }
            minimum = min(estimates.values())
            tied = [name for name, value in estimates.items() if np.isclose(value, minimum)]
            wins[str(rng.choice(tied))] += 1
        for configuration, count in wins.items():
            probabilities[(str(wind_direction), int(wind_level), configuration)] = count / draws
    return probabilities


def prepare_metrics(calibration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    drone = pd.read_csv(TRAJECTORY, dtype={"run_id": "string"}, low_memory=False)
    master = pd.read_csv(MASTER, dtype={"run_id": "string"}, low_memory=False)
    primary_keys = master[master["primary_analysis_status"].eq("eligible_primary_75_to_40")][
        ["experiment_directory", "run_id"]
    ]
    drone = drone.merge(primary_keys, on=["experiment_directory", "run_id"], how="inner")
    if len(drone) != len(primary_keys) * 5:
        raise RuntimeError("Primary run/drone grain is not exactly five rows per run")

    rates = calibration.set_index("battery_id")["hover_discharge_rate_pp_per_min"]
    drone["hover_rate_pp_per_min"] = drone["battery_id"].map(rates)
    if drone["hover_rate_pp_per_min"].isna().any():
        raise RuntimeError("A primary drone row lacks a hover calibration")
    drone["reported_drop_pp"] = drone["reported_battery_drop_pct_points"].astype(float)
    drone["hover_equivalent_sec"] = 60.0 * drone["reported_drop_pp"] / drone["hover_rate_pp_per_min"]
    drone["adjusted_hover_equivalent_sec"] = (
        drone["hover_equivalent_sec"] - drone["confirmed_stationary_wait_sec"]
    ).clip(lower=0.0)
    drone["configuration"] = (
        drone["formation"].map(FORMATION_LABEL)
        + " · "
        + drone["inter_drone_spacing_cm"].astype(int).astype(str)
        + " cm"
    )
    drone["condition"] = (
        drone["wind_direction"].str.title()
        + " wind · Level "
        + drone["wind_level"].astype(int).astype(str)
    )

    run_keys = [
        "experiment_directory",
        "run_id",
        "formation",
        "inter_drone_spacing_cm",
        "wind_direction",
        "wind_level",
        "configuration",
        "condition",
    ]
    run = (
        drone.groupby(run_keys, as_index=False)
        .agg(
            drone_count=("drone_name", "nunique"),
            mean_reported_drop_pp=("reported_drop_pp", "mean"),
            mean_hover_equivalent_sec=("hover_equivalent_sec", "mean"),
            adjusted_hover_equivalent_sec=("adjusted_hover_equivalent_sec", "mean"),
            mean_confirmed_wait_sec=("confirmed_stationary_wait_sec", "mean"),
            mean_selected_window_sec=("selected_window_sec", "mean"),
        )
    )
    no_drone_5 = (
        drone[drone["drone_name"].ne("drone_5")]
        .groupby(["experiment_directory", "run_id"], as_index=False)
        .agg(adjusted_hover_equivalent_sec_without_drone_5=("adjusted_hover_equivalent_sec", "mean"))
    )
    run = run.merge(no_drone_5, on=["experiment_directory", "run_id"], how="left")
    if not run["drone_count"].eq(5).all():
        raise RuntimeError("A primary run does not contain five unique drones")
    return drone, run


def summarize_configurations(run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260812)
    probabilities = winner_probabilities(run, rng)
    rows = []
    for keys, group in run.groupby(
        ["wind_direction", "wind_level", "condition", "formation", "inter_drone_spacing_cm", "configuration"]
    ):
        values = group["adjusted_hover_equivalent_sec"].to_numpy(float)
        ci_low, ci_high = bootstrap_ci(values, rng)
        rows.append(
            {
                "wind_direction": keys[0],
                "wind_level": int(keys[1]),
                "condition": keys[2],
                "formation": keys[3],
                "inter_drone_spacing_cm": int(keys[4]),
                "configuration": keys[5],
                "run_count": len(group),
                "adjusted_mean_sec": float(np.mean(values)),
                "adjusted_median_sec": float(np.median(values)),
                "adjusted_sd_sec": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "adjusted_ci95_low_sec": ci_low,
                "adjusted_ci95_high_sec": ci_high,
                "raw_mean_drop_pp": float(group["mean_reported_drop_pp"].mean()),
                "uncorrected_mean_hover_equivalent_sec": float(group["mean_hover_equivalent_sec"].mean()),
                "mean_confirmed_wait_sec": float(group["mean_confirmed_wait_sec"].mean()),
                "adjusted_mean_without_drone_5_sec": float(
                    group["adjusted_hover_equivalent_sec_without_drone_5"].mean()
                ),
                "bootstrap_winner_probability": probabilities[(str(keys[0]), int(keys[1]), str(keys[5]))],
            }
        )
    ranking = pd.DataFrame(rows)
    ranking["adjusted_mean_rank"] = ranking.groupby(["wind_direction", "wind_level"])[
        "adjusted_mean_sec"
    ].rank(method="min", ascending=True).astype(int)
    ranking["adjusted_median_rank"] = ranking.groupby(["wind_direction", "wind_level"])[
        "adjusted_median_sec"
    ].rank(method="min", ascending=True).astype(int)
    ranking["raw_rank"] = ranking.groupby(["wind_direction", "wind_level"])["raw_mean_drop_pp"].rank(
        method="min", ascending=True
    ).astype(int)
    ranking["uncorrected_rank"] = ranking.groupby(["wind_direction", "wind_level"])[
        "uncorrected_mean_hover_equivalent_sec"
    ].rank(method="min", ascending=True).astype(int)
    ranking["without_drone_5_rank"] = ranking.groupby(["wind_direction", "wind_level"])[
        "adjusted_mean_without_drone_5_sec"
    ].rank(method="min", ascending=True).astype(int)

    winner_rows = []
    for (wind_direction, wind_level), group in ranking.groupby(["wind_direction", "wind_level"]):
        ordered = group.sort_values("adjusted_mean_sec").reset_index(drop=True)
        winner = ordered.iloc[0]
        runner_up = ordered.iloc[1]

        def metric_winner(field: str) -> str:
            return str(group.loc[group[field].idxmin(), "configuration"])

        variant_winners = {
            "adjusted_mean": str(winner["configuration"]),
            "adjusted_median": metric_winner("adjusted_median_sec"),
            "raw_mean_drop": metric_winner("raw_mean_drop_pp"),
            "uncorrected_normalized": metric_winner("uncorrected_mean_hover_equivalent_sec"),
            "without_drone_5": metric_winner("adjusted_mean_without_drone_5_sec"),
        }
        agreement = sum(value == winner["configuration"] for value in variant_winners.values())
        gap = float(runner_up["adjusted_mean_sec"] - winner["adjusted_mean_sec"])
        gap_pct = gap / float(runner_up["adjusted_mean_sec"]) * 100.0
        if int(winner["run_count"]) < 2:
            confidence = "Insufficient replication"
        elif agreement == 5 and gap_pct >= 15 and float(winner["bootstrap_winner_probability"]) >= 0.60:
            confidence = "Most stable leader"
        elif agreement >= 4:
            confidence = "Provisional; close alternatives"
        else:
            confidence = "Metric-sensitive; no clear winner"
        winner_rows.append(
            {
                "wind_direction": wind_direction,
                "wind_level": int(wind_level),
                "condition": str(winner["condition"]),
                "leading_configuration": str(winner["configuration"]),
                "leader_adjusted_mean_sec": float(winner["adjusted_mean_sec"]),
                "leader_run_count": int(winner["run_count"]),
                "runner_up_configuration": str(runner_up["configuration"]),
                "runner_up_adjusted_mean_sec": float(runner_up["adjusted_mean_sec"]),
                "gap_to_runner_up_sec": gap,
                "gap_to_runner_up_pct": gap_pct,
                "bootstrap_winner_probability": float(winner["bootstrap_winner_probability"]),
                "variant_agreement_count_of_5": agreement,
                "adjusted_median_winner": variant_winners["adjusted_median"],
                "raw_drop_winner": variant_winners["raw_mean_drop"],
                "uncorrected_normalized_winner": variant_winners["uncorrected_normalized"],
                "without_drone_5_winner": variant_winners["without_drone_5"],
                "evidence_label": confidence,
                "configurations_observed": len(group),
            }
        )
    winners = pd.DataFrame(winner_rows)
    condition_cat = pd.CategoricalDtype(
        [f"{direction.title()} wind · Level {level}" for direction, level in CONDITION_ORDER],
        ordered=True,
    )
    winners["condition"] = winners["condition"].astype(condition_cat)
    winners = winners.sort_values("condition").reset_index(drop=True)
    ranking["condition"] = ranking["condition"].astype(condition_cat)
    ranking = ranking.sort_values(["condition", "adjusted_mean_rank"]).reset_index(drop=True)
    return ranking, winners


def plot_rankings(ranking: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 15.8), dpi=180)
    for axis, (direction, level) in zip(axes.flat, CONDITION_ORDER):
        data = ranking[
            ranking["wind_direction"].eq(direction) & ranking["wind_level"].eq(level)
        ].sort_values("adjusted_mean_sec", ascending=True)
        positions = np.arange(len(data))
        colors = ["#2A6F97"] + ["#B8C5CE"] * max(0, len(data) - 1)
        axis.barh(positions, data["adjusted_mean_sec"], color=colors, edgecolor="#40505C", lw=0.6)
        axis.set_yticks(positions, data["configuration"])
        axis.invert_yaxis()
        axis.set_xlim(left=0)
        axis.set_xlabel("Adjusted hover-equivalent seconds per drone / 250 cm")
        axis.set_title(f"{direction.title()} wind · Level {level}", loc="left", weight="bold")
        axis.grid(axis="x", color="#E1E6EA", lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        for position, row in enumerate(data.itertuples()):
            axis.text(
                row.adjusted_mean_sec + 1,
                position,
                f"{row.adjusted_mean_sec:.1f}  (n={row.run_count})",
                va="center",
                fontsize=8,
                color="#26333B",
            )
    fig.suptitle(
        "Configuration energy ranking within each wind condition",
        x=0.08,
        y=0.992,
        ha="left",
        fontsize=17,
    )
    fig.text(
        0.08,
        0.968,
        "Primary 75%-40% runs; lower is better. Battery-normalized and conservatively corrected for confirmed stationary waits.",
        ha="left",
        fontsize=10,
        color="#52606A",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.938], h_pad=2.2)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = hover_calibration()
    drone, run = prepare_metrics(calibration)
    ranking, winners = summarize_configurations(run)

    calibration.to_csv(OUT / "hover_battery_calibration_75_40.csv", index=False)
    drone.to_csv(OUT / "primary_drone_metrics.csv", index=False)
    run.to_csv(OUT / "primary_run_metrics.csv", index=False)
    ranking.to_csv(OUT / "configuration_ranking_by_condition.csv", index=False)
    winners.to_csv(OUT / "condition_configuration_leaders.csv", index=False)
    plot_rankings(ranking, OUT / "configuration_rankings_by_condition.png")

    coverage = (
        ranking.pivot_table(
            index="configuration",
            columns="condition",
            values="run_count",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reset_index()
    )
    coverage.to_csv(OUT / "configuration_condition_coverage.csv", index=False)

    summary = {
        "analysis_unit": "five-drone run",
        "primary_run_count": int(run[["experiment_directory", "run_id"]].drop_duplicates().shape[0]),
        "primary_drone_row_count": int(len(drone)),
        "condition_count": int(winners.shape[0]),
        "configuration_cell_count": int(len(ranking)),
        "configuration_cell_run_count_min": int(ranking["run_count"].min()),
        "configuration_cell_run_count_max": int(ranking["run_count"].max()),
        "reported_drop_zero_row_count": int(drone["reported_drop_pp"].eq(0).sum()),
        "wait_correction_clipped_to_zero_count": int(drone["adjusted_hover_equivalent_sec"].eq(0).sum()),
        "leaders": winners.astype({"condition": "string"})[
            ["condition", "leading_configuration", "evidence_label"]
        ].to_dict(orient="records"),
        "metric_definition": (
            "For each drone: 60 * reported SOC drop / battery-specific 75%-40% hover rate, "
            "minus trajectory-confirmed stationary-wait seconds, clipped at zero. "
            "Average five drones to one run, then average runs within condition and configuration."
        ),
        "important_caveats": [
            "Reported battery level is integer-valued and sometimes unchanged over an entire selected segment.",
            "Confirmed stationary waiting is a conservative lower bound.",
            (
                "Most condition-configuration cells contain only 2-5 runs; "
                f"{int(ranking['run_count'].eq(1).sum())} cells contain one run."
            ),
            "Bootstrap winner probabilities are descriptive and degenerate for a one-run cell.",
            "Configuration effects are descriptive, not causal, because runs were not randomized as a balanced blocked experiment.",
        ],
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
