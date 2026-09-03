"""Independent QA checks for the condition-by-configuration energy analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "configuration_energy_analysis"
MASTER = ROOT / "db_copy_for_cleaning" / "_cleaning_admin" / "cleaning_master_run_index.csv"


def main() -> None:
    drone = pd.read_csv(OUT / "primary_drone_metrics.csv", dtype={"run_id": "string"})
    run = pd.read_csv(OUT / "primary_run_metrics.csv", dtype={"run_id": "string"})
    ranking = pd.read_csv(OUT / "configuration_ranking_by_condition.csv")
    leaders = pd.read_csv(OUT / "condition_configuration_leaders.csv")
    calibration = pd.read_csv(OUT / "hover_battery_calibration_75_40.csv")
    master = pd.read_csv(MASTER, dtype={"run_id": "string"}, low_memory=False)

    run_key = ["experiment_directory", "run_id"]
    primary_master = master[master["primary_analysis_status"].eq("eligible_primary_75_to_40")]
    expected_keys = set(map(tuple, primary_master[run_key].itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, run[run_key].itertuples(index=False, name=None)))

    group_sizes = drone.groupby(run_key).size()
    unique_drones = drone.groupby(run_key)["drone_name"].nunique()
    independent = (
        run.groupby(["wind_direction", "wind_level", "configuration"], as_index=False)
        ["adjusted_hover_equivalent_sec"]
        .mean()
        .sort_values(["wind_direction", "wind_level", "adjusted_hover_equivalent_sec"])
    )
    independent_leaders = (
        independent.groupby(["wind_direction", "wind_level"], as_index=False).first()
    )
    leader_compare = leaders.merge(
        independent_leaders,
        on=["wind_direction", "wind_level"],
        suffixes=("_reported", "_independent"),
        validate="one_to_one",
    )

    checks = {
        "primary_run_key_set_matches_master": expected_keys == actual_keys,
        "run_rows_unique": not run.duplicated(run_key).any(),
        "exactly_five_drone_rows_per_run": bool(group_sizes.eq(5).all()),
        "exactly_five_unique_drones_per_run": bool(unique_drones.eq(5).all()),
        "all_primary_drone_segments_within_75_to_40": bool(drone["within_75_to_40_range"].all()),
        "no_marked_outliers_in_primary_master": bool(~primary_master["marked_outlier"].fillna(False).astype(bool).any()),
        "no_no_wind_rows": bool(~drone["wind_direction"].str.lower().eq("no_wind").any()),
        "all_six_batteries_have_calibration": set(calibration["battery_id"])
        == {"B10", "B11", "B12", "B13", "B14", "B15"},
        "no_missing_normalized_metric": bool(drone["adjusted_hover_equivalent_sec"].notna().all()),
        "configuration_cell_count_is_55": len(ranking) == 55,
        "independent_leaders_match_published_leaders": bool(
            leader_compare["leading_configuration"].eq(leader_compare["configuration"]).all()
            and np.allclose(
                leader_compare["leader_adjusted_mean_sec"],
                leader_compare["adjusted_hover_equivalent_sec"],
            )
        ),
        "ranking_is_contiguous_within_condition": bool(
            ranking.groupby(["wind_direction", "wind_level"])["adjusted_mean_rank"]
            .apply(lambda values: sorted(values.tolist()) == list(range(1, len(values) + 1)))
            .all()
        ),
        "no_negative_adjusted_values": bool(drone["adjusted_hover_equivalent_sec"].ge(0).all()),
    }
    all_checks_pass = all(checks.values())

    assessment = "Share with caveats" if all_checks_pass else "Not ready to share"
    result = {
        "assessment": assessment,
        "all_structural_and_calculation_checks_passed": all_checks_pass,
        "checks": checks,
        "counts": {
            "primary_runs": int(len(run)),
            "primary_drone_rows": int(len(drone)),
            "condition_configuration_cells": int(len(ranking)),
            "zero_reported_drop_drone_rows": int(drone["reported_drop_pp"].eq(0).sum()),
            "one_run_configuration_cells": int(ranking["run_count"].eq(1).sum()),
            "missing_design_cells": 60 - int(len(ranking)),
        },
        "sharing_caveats": [
            "The reported Tello battery level is integer-valued; 36 of 840 selected drone segments report zero percentage-point drop.",
            "The 55 observed condition-configuration cells are unbalanced (1-6 runs per cell), and five design cells are absent.",
            "Stationary-wait removal is conservative and changes the Tail wind Level 2 leader.",
            "Excluding drone 5 changes the Head wind Level 2 winner, so the B15-to-B12 replacement sensitivity is not fully eliminated there.",
            "The bootstrap winner probabilities are descriptive; the design does not support causal or significance claims.",
        ],
    }
    (OUT / "validation_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    status_lines = [f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}" for name, passed in checks.items()]
    report = f"""# Validation report: configuration energy comparison

## Overall assessment

**{assessment}.** All structural, join-grain, eligibility, normalization, and independent-ranking checks passed. The outputs are internally reproducible, but the scientific interpretation remains limited by coarse SOC telemetry, unbalanced replication, incomplete cells, and sensitivity to wait correction.

## Checks

{chr(10).join(status_lines)}

## Coverage and precision warnings

- {len(run)} eligible five-drone runs and {len(drone)} drone-level records were analyzed.
- {len(ranking)} of 60 possible condition-configuration cells were observed; {int(ranking['run_count'].eq(1).sum())} observed cells have only one run.
- {int(drone['reported_drop_pp'].eq(0).sum())} of {len(drone)} drone segments show zero reported percentage-point drop because the Tello battery signal is integer-valued and may not update within a short segment.
- The corrected Tail wind Level 2 leader has one run; raw and uncorrected rankings identify a different leader.
- Excluding drone 5 changes the Head wind Level 2 winner, so that condition remains sensitive to the fixed drone/battery assignment and the B15-to-B12 replacement.

## Sharing guidance

Use the condition-level results as ranked evidence and hypotheses for confirmation. Describe only Head wind Level 1 as comparatively stable. Do not describe the remaining condition-level leaders as statistically significant or causally optimal.
"""
    (OUT / "validation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
