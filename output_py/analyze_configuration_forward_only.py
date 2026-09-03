"""Rank configurations using energy attributed only to forward-moving flight.

The first trajectory-observed 250 cm remains the common distance.  All in-flight
time without detected forward progress is removed at the physical battery's hover
baseline rate before the five drones are averaged to one run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_configuration_energy import (
    CONDITION_ORDER,
    FORMATION_LABEL,
    bootstrap_ci,
    hover_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "db_copy_for_cleaning" / "_cleaning_admin"
TRAJECTORY = ADMIN / "trajectory_qc" / "trajectory_drone_segments.csv"
FORWARD_MOTION = ADMIN / "trajectory_qc" / "forward_motion_drone_segments.csv"
FORWARD_SENSITIVITY = ADMIN / "trajectory_qc" / "forward_motion_threshold_sensitivity.csv"
MASTER = ADMIN / "cleaning_master_run_index.csv"
OUT = ROOT / "analysis_outputs" / "configuration_energy_analysis"


def prepare_metrics(calibration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectory = pd.read_csv(TRAJECTORY, dtype={"run_id": "string"}, low_memory=False)
    master = pd.read_csv(MASTER, dtype={"run_id": "string"}, low_memory=False)
    primary_keys = master[master["primary_analysis_status"].eq("eligible_primary_75_to_40")][
        ["experiment_directory", "run_id"]
    ]
    drone = trajectory.merge(
        primary_keys,
        on=["experiment_directory", "run_id"],
        how="inner",
        validate="many_to_one",
    )
    if len(drone) != len(primary_keys) * 5:
        raise RuntimeError("Primary grain is not exactly five drone rows per run")

    forward = pd.read_csv(FORWARD_MOTION, dtype={"run_id": "string"}, low_memory=False)
    forward_columns = [
        "experiment_directory",
        "run_id",
        "drone_name",
        "forward_movement_sec",
        "in_flight_nonforward_sec",
        "forward_movement_fraction",
        "detected_forward_distance_cm",
        "mean_detected_forward_speed_cm_s",
        "reported_drop_during_forward_events_pp",
        "reported_drop_during_nonforward_events_pp",
        "forward_segmentation_issue_codes",
    ]
    drone = drone.merge(
        forward[forward_columns],
        on=["experiment_directory", "run_id", "drone_name"],
        how="left",
        validate="one_to_one",
    )
    if drone["forward_movement_sec"].isna().any():
        raise RuntimeError("Forward-motion segmentation is missing for a primary drone row")

    sensitivity = pd.read_csv(FORWARD_SENSITIVITY, dtype={"run_id": "string"})
    sensitivity = sensitivity[sensitivity["threshold_cm_s"].isin([1.0, 3.0, 4.0])]
    sensitivity = sensitivity.pivot_table(
        index=["experiment_directory", "run_id", "drone_name"],
        columns="threshold_cm_s",
        values="in_flight_nonforward_sec",
        aggfunc="first",
    )
    sensitivity.columns = [
        f"in_flight_nonforward_sec_threshold_{int(threshold)}"
        for threshold in sensitivity.columns
    ]
    drone = drone.merge(
        sensitivity.reset_index(),
        on=["experiment_directory", "run_id", "drone_name"],
        how="left",
        validate="one_to_one",
    )

    rates = calibration.set_index("battery_id")["hover_discharge_rate_pp_per_min"]
    drone["hover_rate_pp_per_min"] = drone["battery_id"].map(rates)
    if drone["hover_rate_pp_per_min"].isna().any():
        raise RuntimeError("A primary drone row lacks a physical-battery hover calibration")

    drone["reported_drop_pp"] = drone["reported_battery_drop_pct_points"].astype(float)
    drone["total_hover_equivalent_sec"] = (
        60.0 * drone["reported_drop_pp"] / drone["hover_rate_pp_per_min"]
    )
    drone["forward_only_hover_equivalent_sec"] = (
        drone["total_hover_equivalent_sec"] - drone["in_flight_nonforward_sec"]
    ).clip(lower=0.0)
    drone["confirmed_wait_only_hover_equivalent_sec"] = (
        drone["total_hover_equivalent_sec"] - drone["confirmed_stationary_wait_sec"]
    ).clip(lower=0.0)
    for threshold in (1, 3, 4):
        drone[f"forward_only_threshold_{threshold}_sec"] = (
            drone["total_hover_equivalent_sec"]
            - drone[f"in_flight_nonforward_sec_threshold_{threshold}"]
        ).clip(lower=0.0)
    drone["direct_event_forward_hover_equivalent_sec"] = (
        60.0
        * drone["reported_drop_during_forward_events_pp"]
        / drone["hover_rate_pp_per_min"]
    )
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
    run = drone.groupby(run_keys, as_index=False).agg(
        drone_count=("drone_name", "nunique"),
        mean_reported_drop_pp=("reported_drop_pp", "mean"),
        total_hover_equivalent_sec=("total_hover_equivalent_sec", "mean"),
        forward_only_hover_equivalent_sec=("forward_only_hover_equivalent_sec", "mean"),
        confirmed_wait_only_hover_equivalent_sec=(
            "confirmed_wait_only_hover_equivalent_sec",
            "mean",
        ),
        forward_only_threshold_1_sec=("forward_only_threshold_1_sec", "mean"),
        forward_only_threshold_3_sec=("forward_only_threshold_3_sec", "mean"),
        forward_only_threshold_4_sec=("forward_only_threshold_4_sec", "mean"),
        direct_event_forward_hover_equivalent_sec=(
            "direct_event_forward_hover_equivalent_sec",
            "mean",
        ),
        mean_forward_movement_sec=("forward_movement_sec", "mean"),
        mean_in_flight_nonforward_sec=("in_flight_nonforward_sec", "mean"),
        mean_forward_movement_fraction=("forward_movement_fraction", "mean"),
        mean_detected_forward_distance_cm=("detected_forward_distance_cm", "mean"),
        flagged_drone_count=(
            "forward_segmentation_issue_codes",
            lambda values: int(values.fillna("").ne("").sum()),
        ),
    )
    no_drone_5 = (
        drone[drone["drone_name"].ne("drone_5")]
        .groupby(["experiment_directory", "run_id"], as_index=False)
        .agg(
            forward_only_without_drone_5_sec=(
                "forward_only_hover_equivalent_sec",
                "mean",
            )
        )
    )
    run = run.merge(
        no_drone_5,
        on=["experiment_directory", "run_id"],
        how="left",
        validate="one_to_one",
    )
    if not run["drone_count"].eq(5).all():
        raise RuntimeError("A primary run does not contain five unique drones")
    return drone, run


def winner_probabilities(
    run: pd.DataFrame,
    rng: np.random.Generator,
    draws: int = 10000,
) -> dict[tuple[str, int, str], float]:
    output = {}
    for (direction, level), condition in run.groupby(["wind_direction", "wind_level"]):
        values = {
            configuration: group["forward_only_hover_equivalent_sec"].to_numpy(float)
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
            output[(str(direction), int(level), configuration)] = count / draws
    return output


def summarize(run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260813)
    probabilities = winner_probabilities(run, rng)
    rows = []
    group_fields = [
        "wind_direction",
        "wind_level",
        "condition",
        "formation",
        "inter_drone_spacing_cm",
        "configuration",
    ]
    for keys, group in run.groupby(group_fields):
        values = group["forward_only_hover_equivalent_sec"].to_numpy(float)
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
                "forward_only_mean_sec": float(np.mean(values)),
                "forward_only_median_sec": float(np.median(values)),
                "forward_only_sd_sec": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "forward_only_ci95_low_sec": ci_low,
                "forward_only_ci95_high_sec": ci_high,
                "raw_mean_drop_pp": float(group["mean_reported_drop_pp"].mean()),
                "total_mean_hover_equivalent_sec": float(group["total_hover_equivalent_sec"].mean()),
                "mean_in_flight_nonforward_sec": float(group["mean_in_flight_nonforward_sec"].mean()),
                "mean_forward_movement_sec": float(group["mean_forward_movement_sec"].mean()),
                "mean_forward_movement_fraction": float(group["mean_forward_movement_fraction"].mean()),
                "confirmed_wait_only_mean_sec": float(
                    group["confirmed_wait_only_hover_equivalent_sec"].mean()
                ),
                "threshold_1_mean_sec": float(group["forward_only_threshold_1_sec"].mean()),
                "threshold_3_mean_sec": float(group["forward_only_threshold_3_sec"].mean()),
                "threshold_4_mean_sec": float(group["forward_only_threshold_4_sec"].mean()),
                "direct_event_forward_mean_sec": float(
                    group["direct_event_forward_hover_equivalent_sec"].mean()
                ),
                "without_drone_5_mean_sec": float(group["forward_only_without_drone_5_sec"].mean()),
                "flagged_drone_rows": int(group["flagged_drone_count"].sum()),
                "bootstrap_winner_probability": probabilities[(str(keys[0]), int(keys[1]), str(keys[5]))],
            }
        )
    ranking = pd.DataFrame(rows)
    rank_fields = {
        "forward_only_mean_rank": "forward_only_mean_sec",
        "forward_only_median_rank": "forward_only_median_sec",
        "threshold_1_rank": "threshold_1_mean_sec",
        "threshold_3_rank": "threshold_3_mean_sec",
        "threshold_4_rank": "threshold_4_mean_sec",
        "without_drone_5_rank": "without_drone_5_mean_sec",
        "confirmed_wait_only_rank": "confirmed_wait_only_mean_sec",
        "direct_event_forward_rank": "direct_event_forward_mean_sec",
        "raw_rank": "raw_mean_drop_pp",
    }
    for rank_name, value_name in rank_fields.items():
        ranking[rank_name] = ranking.groupby(["wind_direction", "wind_level"])[value_name].rank(
            method="min", ascending=True
        ).astype(int)

    winner_rows = []
    for (direction, level), group in ranking.groupby(["wind_direction", "wind_level"]):
        ordered = group.sort_values("forward_only_mean_sec").reset_index(drop=True)
        winner, runner_up = ordered.iloc[0], ordered.iloc[1]

        def metric_winner(field: str) -> str:
            return str(group.loc[group[field].idxmin(), "configuration"])

        robust_winners = {
            "mean": str(winner["configuration"]),
            "median": metric_winner("forward_only_median_sec"),
            "threshold_1": metric_winner("threshold_1_mean_sec"),
            "threshold_4": metric_winner("threshold_4_mean_sec"),
            "without_drone_5": metric_winner("without_drone_5_mean_sec"),
        }
        agreement = sum(value == winner["configuration"] for value in robust_winners.values())
        gap = float(runner_up["forward_only_mean_sec"] - winner["forward_only_mean_sec"])
        gap_pct = gap / float(runner_up["forward_only_mean_sec"]) * 100.0
        if int(winner["run_count"]) < 2:
            evidence = "Insufficient replication"
        elif (
            int(winner["run_count"]) >= 3
            and agreement == 5
            and gap_pct >= 15
            and float(winner["bootstrap_winner_probability"]) >= 0.60
        ):
            evidence = "Most stable leader"
        elif agreement >= 4:
            evidence = "Provisional; close alternatives"
        else:
            evidence = "Metric-sensitive; no clear winner"
        winner_rows.append(
            {
                "wind_direction": direction,
                "wind_level": int(level),
                "condition": str(winner["condition"]),
                "leading_configuration": str(winner["configuration"]),
                "leader_forward_only_mean_sec": float(winner["forward_only_mean_sec"]),
                "leader_run_count": int(winner["run_count"]),
                "runner_up_configuration": str(runner_up["configuration"]),
                "runner_up_forward_only_mean_sec": float(runner_up["forward_only_mean_sec"]),
                "gap_to_runner_up_sec": gap,
                "gap_to_runner_up_pct": gap_pct,
                "bootstrap_winner_probability": float(winner["bootstrap_winner_probability"]),
                "variant_agreement_count_of_5": agreement,
                "forward_only_median_winner": robust_winners["median"],
                "threshold_1_winner": robust_winners["threshold_1"],
                "threshold_4_winner": robust_winners["threshold_4"],
                "without_drone_5_winner": robust_winners["without_drone_5"],
                "confirmed_wait_only_winner": metric_winner("confirmed_wait_only_mean_sec"),
                "direct_event_forward_winner": metric_winner("direct_event_forward_mean_sec"),
                "raw_drop_winner": metric_winner("raw_mean_drop_pp"),
                "evidence_label": evidence,
                "configurations_observed": len(group),
            }
        )
    winners = pd.DataFrame(winner_rows)
    condition_type = pd.CategoricalDtype(
        [f"{direction.title()} wind · Level {level}" for direction, level in CONDITION_ORDER],
        ordered=True,
    )
    winners["condition"] = winners["condition"].astype(condition_type)
    winners = winners.sort_values("condition").reset_index(drop=True)
    ranking["condition"] = ranking["condition"].astype(condition_type)
    ranking = ranking.sort_values(["condition", "forward_only_mean_rank"]).reset_index(drop=True)
    return ranking, winners


def plot_rankings(ranking: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 15.8), dpi=180)
    for axis, (direction, level) in zip(axes.flat, CONDITION_ORDER):
        data = ranking[
            ranking["wind_direction"].eq(direction) & ranking["wind_level"].eq(level)
        ].sort_values("forward_only_mean_sec")
        positions = np.arange(len(data))
        colors = ["#2A6F97"] + ["#B8C5CE"] * max(0, len(data) - 1)
        axis.barh(
            positions,
            data["forward_only_mean_sec"],
            color=colors,
            edgecolor="#40505C",
            lw=0.6,
        )
        axis.set_yticks(positions, data["configuration"])
        axis.invert_yaxis()
        axis.set_xlim(left=0)
        axis.set_xlabel("Forward-only hover-equivalent seconds per drone / 250 cm")
        axis.set_title(f"{direction.title()} wind · Level {level}", loc="left", weight="bold")
        axis.grid(axis="x", color="#E1E6EA", lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        for position, row in enumerate(data.itertuples()):
            axis.text(
                row.forward_only_mean_sec + 1,
                position,
                f"{row.forward_only_mean_sec:.1f}  (n={row.run_count})",
                va="center",
                fontsize=8,
                color="#26333B",
            )
    fig.suptitle(
        "Forward-moving energy ranking within each wind condition",
        x=0.08,
        y=0.992,
        ha="left",
        fontsize=17,
    )
    fig.text(
        0.08,
        0.968,
        "Primary 75%-40% runs; lower is better. All detected in-flight non-forward time is removed at the battery-specific hover rate.",
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
    ranking, winners = summarize(run)

    calibration.to_csv(OUT / "forward_only_hover_battery_calibration_75_40.csv", index=False)
    drone.to_csv(OUT / "forward_only_primary_drone_metrics.csv", index=False)
    run.to_csv(OUT / "forward_only_primary_run_metrics.csv", index=False)
    ranking.to_csv(OUT / "forward_only_configuration_ranking_by_condition.csv", index=False)
    winners.to_csv(OUT / "forward_only_condition_configuration_leaders.csv", index=False)
    plot_rankings(ranking, OUT / "forward_only_configuration_rankings_by_condition.png")

    prior_path = OUT / "condition_configuration_leaders.csv"
    comparison = winners.copy()
    if prior_path.exists():
        prior = pd.read_csv(prior_path)[["condition", "leading_configuration"]].rename(
            columns={"leading_configuration": "previous_wait_corrected_leader"}
        )
        comparison["condition"] = comparison["condition"].astype("string")
        comparison = comparison.merge(prior, on="condition", how="left", validate="one_to_one")
        comparison["leader_changed_after_full_hover_removal"] = comparison[
            "leading_configuration"
        ].ne(comparison["previous_wait_corrected_leader"])
    comparison.to_csv(OUT / "forward_only_leader_change_comparison.csv", index=False)

    summary = {
        "analysis_unit": "five-drone run",
        "primary_run_count": int(len(run)),
        "primary_drone_row_count": int(len(drone)),
        "median_forward_movement_sec": float(drone["forward_movement_sec"].median()),
        "median_removed_nonforward_sec": float(drone["in_flight_nonforward_sec"].median()),
        "forward_only_metric_clipped_to_zero_count": int(
            drone["forward_only_hover_equivalent_sec"].eq(0).sum()
        ),
        "segmentation_flagged_drone_rows": int(
            drone["forward_segmentation_issue_codes"].fillna("").ne("").sum()
        ),
        "leaders": winners.astype({"condition": "string"})[
            ["condition", "leading_configuration", "evidence_label"]
        ].to_dict(orient="records"),
        "metric_definition": (
            "For each drone, convert the total selected-window SOC drop to battery-specific "
            "75%-40% hover-equivalent seconds, subtract every trajectory-detected in-flight "
            "non-forward second, clip at zero, average five drones to one run, then average "
            "runs within condition and configuration."
        ),
        "important_caveats": [
            "The subtraction assumes non-forward flight consumes energy at the battery-specific hover baseline rate.",
            "Mission-pad coordinates are integer-quantized; movement state is not observable at sub-second precision.",
            "Reported battery level is integer-valued, so short segments can be clipped to zero after hover removal.",
            "Configuration effects remain descriptive rather than causal because the design is unbalanced and not randomized as a blocked experiment.",
        ],
    }
    (OUT / "forward_only_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
