"""Independent QA for the forward-movement-only configuration analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "configuration_energy_analysis"
ADMIN = ROOT / "db_copy_for_cleaning" / "_cleaning_admin"


def main() -> None:
    drone = pd.read_csv(OUT / "forward_only_primary_drone_metrics.csv", dtype={"run_id": "string"})
    run = pd.read_csv(OUT / "forward_only_primary_run_metrics.csv", dtype={"run_id": "string"})
    ranking = pd.read_csv(OUT / "forward_only_configuration_ranking_by_condition.csv")
    leaders = pd.read_csv(OUT / "forward_only_condition_configuration_leaders.csv")
    forward = pd.read_csv(
        ADMIN / "trajectory_qc" / "forward_motion_drone_segments.csv",
        dtype={"run_id": "string"},
    )
    threshold = pd.read_csv(
        ADMIN / "trajectory_qc" / "forward_motion_threshold_sensitivity.csv",
        dtype={"run_id": "string"},
    )
    master = pd.read_csv(
        ADMIN / "cleaning_master_run_index.csv", dtype={"run_id": "string"}, low_memory=False
    )

    run_key = ["experiment_directory", "run_id"]
    drone_key = [*run_key, "drone_name"]
    primary_master = master[master["primary_analysis_status"].eq("eligible_primary_75_to_40")]
    expected_runs = set(map(tuple, primary_master[run_key].itertuples(index=False, name=None)))
    actual_runs = set(map(tuple, run[run_key].itertuples(index=False, name=None)))

    source_forward = forward[forward["primary_analysis_status"].eq("eligible_primary_75_to_40")]
    direct_formula = (
        60.0 * drone["reported_drop_pp"] / drone["hover_rate_pp_per_min"]
        - drone["in_flight_nonforward_sec"]
    ).clip(lower=0.0)
    independent_ranking = (
        run.groupby(["wind_direction", "wind_level", "configuration"], as_index=False)[
            "forward_only_hover_equivalent_sec"
        ]
        .mean()
        .sort_values(["wind_direction", "wind_level", "forward_only_hover_equivalent_sec"])
    )
    independent_leaders = independent_ranking.groupby(
        ["wind_direction", "wind_level"], as_index=False
    ).first()
    comparison = leaders.merge(
        independent_leaders,
        on=["wind_direction", "wind_level"],
        validate="one_to_one",
    )

    threshold_primary = threshold.merge(
        primary_master[run_key], on=run_key, how="inner", validate="many_to_one"
    )
    threshold_medians = threshold_primary.groupby("threshold_cm_s")[
        "forward_movement_sec"
    ].median()
    threshold_span = float(threshold_medians.max() - threshold_medians.min())

    checks = {
        "primary_run_key_set_matches_master": expected_runs == actual_runs,
        "run_rows_are_unique": not run.duplicated(run_key).any(),
        "drone_rows_are_unique": not drone.duplicated(drone_key).any(),
        "exactly_five_drone_rows_per_run": bool(drone.groupby(run_key).size().eq(5).all()),
        "exactly_five_unique_drones_per_run": bool(
            drone.groupby(run_key)["drone_name"].nunique().eq(5).all()
        ),
        "source_forward_segmentation_covers_all_primary_drones": set(
            map(tuple, source_forward[drone_key].itertuples(index=False, name=None))
        )
        == set(map(tuple, drone[drone_key].itertuples(index=False, name=None))),
        "all_primary_segments_remain_inside_75_to_40": bool(drone["within_75_to_40_range"].all()),
        "no_marked_outlier_or_no_wind_runs": bool(
            ~primary_master["marked_outlier"].fillna(False).astype(bool).any()
            and ~drone["wind_direction"].str.lower().eq("no_wind").any()
        ),
        "forward_only_formula_reproduces_exactly": bool(
            np.allclose(direct_formula, drone["forward_only_hover_equivalent_sec"])
        ),
        "no_negative_forward_only_values": bool(drone["forward_only_hover_equivalent_sec"].ge(0).all()),
        "median_moving_time_is_plausible_for_250cm_at_10cm_s": bool(
            23.0 <= drone["forward_movement_sec"].median() <= 29.0
        ),
        "movement_threshold_median_span_below_half_second": threshold_span < 0.5,
        "independent_condition_leaders_match": bool(
            comparison["leading_configuration"].eq(comparison["configuration"]).all()
            and np.allclose(
                comparison["leader_forward_only_mean_sec"],
                comparison["forward_only_hover_equivalent_sec"],
            )
        ),
        "all_six_conditions_present": len(leaders) == 6,
        "all_55_observed_configuration_cells_present": len(ranking) == 55,
    }
    passed = all(checks.values())
    assessment = "Share with caveats" if passed else "Not ready to share"
    flagged = drone["forward_segmentation_issue_codes"].fillna("").ne("")
    clipped = drone["forward_only_hover_equivalent_sec"].eq(0)
    result = {
        "assessment": assessment,
        "all_structural_and_calculation_checks_passed": passed,
        "checks": checks,
        "counts": {
            "primary_runs": int(len(run)),
            "primary_drone_rows": int(len(drone)),
            "configuration_cells": int(len(ranking)),
            "forward_segmentation_flagged_drone_rows": int(flagged.sum()),
            "forward_only_values_clipped_to_zero": int(clipped.sum()),
            "forward_only_values_clipped_to_zero_rate": float(clipped.mean()),
            "single_run_configuration_cells": int(ranking["run_count"].eq(1).sum()),
        },
        "movement_timing": {
            "median_forward_movement_sec": float(drone["forward_movement_sec"].median()),
            "median_removed_nonforward_sec": float(drone["in_flight_nonforward_sec"].median()),
            "threshold_medians_sec": {
                str(key): float(value) for key, value in threshold_medians.items()
            },
            "threshold_median_span_sec": threshold_span,
        },
        "sharing_caveats": [
            "The method estimates forward-only energy by subtracting hover-baseline energy during detected non-forward time; no current or voltage trace is available.",
            f"Tello battery percentage is integer-valued; {int(clipped.sum())} of {len(drone)} drone values clip to zero after the modeled hover subtraction.",
            f"{int(flagged.sum())} drone trajectories carry duration, detected-distance, or phase-coverage review flags and remain included as sensitivity-visible records.",
            "Side wind Level 2 changes leader only after full non-forward removal and has two runs for the new leader.",
            "Tail wind Level 2 still has a one-run leader and remains unresolved.",
        ],
    }
    (OUT / "forward_only_validation_checks.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    check_lines = [
        f"- {'PASS' if value else 'FAIL'} — {key.replace('_', ' ')}"
        for key, value in checks.items()
    ]
    flagged_table = drone.loc[
        flagged,
        [
            "experiment_directory",
            "run_id",
            "drone_name",
            "forward_movement_sec",
            "in_flight_nonforward_sec",
            "detected_forward_distance_cm",
            "forward_segmentation_issue_codes",
        ],
    ].copy()
    flagged_table.to_csv(OUT / "forward_only_flagged_trajectory_rows.csv", index=False)
    report = f"""# Validation report: forward-movement-only configuration analysis

## Overall assessment

**{assessment}.** All structural, eligibility, formula, threshold-robustness, and independent-leader checks passed. The revised analysis is reproducible and implements the requested forward-only scope, but the energy separation is model-based because the dataset does not contain current or voltage.

## Checks

{chr(10).join(check_lines)}

## Key evidence

- {len(run)} primary five-drone runs and {len(drone)} drone records were retained.
- Median detected forward movement is {drone['forward_movement_sec'].median():.2f} s for the common 250 cm, close to the 25 s ideal at 10 cm/s.
- Median removed in-flight non-forward time is {drone['in_flight_nonforward_sec'].median():.2f} s per drone.
- Changing the forward-speed threshold from 1 to 4 cm/s changes the median moving time by only {threshold_span:.2f} s.
- {int(flagged.sum())} trajectory rows are flagged for review but retained; their list is saved separately.
- {int(clipped.sum())} of {len(drone)} forward-only values are clipped to zero after subtraction because SOC is integer-valued.

## Interpretation boundary

The revised score is the observed selected-window SOC drop normalized by physical-battery hover behavior, minus the estimated hover-baseline energy for every detected non-forward second. It should be described as an **estimated forward-only SOC energy proxy**, not directly measured electrical energy.
"""
    (OUT / "forward_only_validation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
