"""Expand Oracle states into position-aware structure candidate rows.

For every independent state, this dataset stores the exact best position for
every available formation/spacing structure, not only the globally selected
structure. It is the supervised source for a future structured configuration
policy whose output includes formation, spacing, and drone-to-slot position.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

from ml_policy.oracle_optimizer import EmpiricalRateTable, OracleState, SafetyTier, solve_oracle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_SOURCE = DEFAULT_DIR / "oracle_training_states_0p25_25m.csv"
DEFAULT_OUTPUT = DEFAULT_DIR / "position_aware_training_candidates.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_from_row(row: dict[str, str], manifest: dict[str, object]) -> OracleState:
    return OracleState(
        wind_direction=row["wind_direction"],
        wind_level=int(row["wind_level"]),
        charging_pad_count=int(row["charging_pad_count"]),
        current_soc=tuple(float(row[f"soc_d{index}"]) for index in range(1, 6)),
        remaining_distance_m=float(row["remaining_distance_m"]),
        forward_speed_m_per_s=float(manifest["forward_speed_m_per_s"]),
        fully_charged_soc=float(manifest["fully_charged_soc"]),
        zero_to_fully_charged_minutes=float(
            manifest["zero_to_fully_charged_minutes"]
        ),
        minimum_arrival_soc=float(manifest["minimum_arrival_soc"]),
    )


def build_position_aware_dataset(source_csv: Path, output_csv: Path) -> dict[str, object]:
    source_manifest_path = source_csv.with_suffix(".manifest.json")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    rate_table_path = Path(str(source_manifest["source_rate_table"]))
    rate_table = EmpiricalRateTable.from_csv(rate_table_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_id", "scenario_seed", "wind_direction", "wind_level",
        "charging_pad_count", "remaining_distance_m",
        "soc_d1", "soc_d2", "soc_d3", "soc_d4", "soc_d5",
        "structure", "formation", "inter_drone_spacing_cm",
        "safety_tier", "eligible_for_selection", "selected_run_count",
        "slot_1_id", "slot_1_rate_pp_per_min",
        "slot_2_id", "slot_2_rate_pp_per_min",
        "slot_3_id", "slot_3_rate_pp_per_min",
        "slot_4_id", "slot_4_rate_pp_per_min",
        "slot_5_id", "slot_5_rate_pp_per_min",
        "assigned_slot_d1", "assigned_slot_d2", "assigned_slot_d3",
        "assigned_slot_d4", "assigned_slot_d5",
        "assigned_slot_index_d1", "assigned_slot_index_d2",
        "assigned_slot_index_d3", "assigned_slot_index_d4",
        "assigned_slot_index_d5", "position_json",
        "slot_1_assigned_drone", "slot_2_assigned_drone",
        "slot_3_assigned_drone", "slot_4_assigned_drone",
        "slot_5_assigned_drone",
        "predicted_drop_d1", "predicted_drop_d2", "predicted_drop_d3",
        "predicted_drop_d4", "predicted_drop_d5",
        "arrival_soc_d1", "arrival_soc_d2", "arrival_soc_d3",
        "arrival_soc_d4", "arrival_soc_d5",
        "charging_job_minutes_d1", "charging_job_minutes_d2",
        "charging_job_minutes_d3", "charging_job_minutes_d4",
        "charging_job_minutes_d5", "charging_schedule_json",
        "charging_makespan_minutes", "remaining_flight_minutes",
        "total_completion_minutes", "global_oracle_structure",
        "global_oracle_position_json", "global_oracle_total_minutes",
        "objective_regret_minutes", "is_global_time_optimal",
        "is_exact_global_configuration",
    ]

    scenario_count = 0
    candidate_count = 0
    eligible_count = 0
    exact_global_count = 0
    objective_optimal_count = 0
    with source_csv.open(newline="", encoding="utf-8") as source_handle, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(source_handle)
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()

        for source_row in reader:
            state = _state_from_row(source_row, source_manifest)
            result = solve_oracle(state, rate_table)
            rates_by_label = {
                rates.structure.label: rates
                for rates in rate_table.structures_for(
                    state.wind_direction,
                    state.wind_level,
                    expected_drone_count=len(state.drone_ids),
                )
            }
            has_safe = any(
                evaluation.safety_tier == SafetyTier.SAFE
                for evaluation in result.ranked_structures
            )
            selected_position = result.selected.position_mapping(state.drone_ids)

            for evaluation in result.ranked_structures:
                rates = rates_by_label[evaluation.structure.label]
                eligible = (
                    evaluation.safety_tier == SafetyTier.SAFE
                    if has_safe
                    else evaluation.safety_tier == SafetyTier.BACKUP_ONLY
                )
                regret = (
                    evaluation.total_completion_minutes
                    - result.selected.total_completion_minutes
                    if eligible
                    else None
                )
                time_optimal = eligible and abs(float(regret)) <= 1e-9
                position = evaluation.position_mapping(state.drone_ids)
                exact_global = (
                    evaluation.structure == result.selected.structure
                    and position == selected_position
                )
                slot_index = {
                    slot.slot_id: index
                    for index, slot in enumerate(rates.slots, start=1)
                }
                drone_by_slot = {
                    slot_id: drone_id
                    for drone_id, slot_id in position.items()
                }
                schedule_drone_ids = tuple(
                    tuple(state.drone_ids[index] for index in group)
                    for group in evaluation.charging_schedule.drone_indices_by_pad
                )

                row: dict[str, object] = {
                    "scenario_id": source_row["scenario_id"],
                    "scenario_seed": source_row["scenario_seed"],
                    "wind_direction": state.wind_direction,
                    "wind_level": state.wind_level,
                    "charging_pad_count": state.charging_pad_count,
                    "remaining_distance_m": state.remaining_distance_m,
                    "structure": evaluation.structure.label,
                    "formation": evaluation.structure.formation,
                    "inter_drone_spacing_cm": evaluation.structure.distance_cm,
                    "safety_tier": evaluation.safety_tier.name.lower(),
                    "eligible_for_selection": int(eligible),
                    "selected_run_count": evaluation.selected_run_count,
                    "position_json": json.dumps(position, sort_keys=True),
                    "charging_schedule_json": json.dumps(schedule_drone_ids),
                    "charging_makespan_minutes": (
                        evaluation.charging_schedule.makespan_minutes
                    ),
                    "remaining_flight_minutes": evaluation.remaining_flight_minutes,
                    "total_completion_minutes": evaluation.total_completion_minutes,
                    "global_oracle_structure": result.selected.structure.label,
                    "global_oracle_position_json": json.dumps(
                        selected_position, sort_keys=True
                    ),
                    "global_oracle_total_minutes": result.selected.total_completion_minutes,
                    "objective_regret_minutes": regret,
                    "is_global_time_optimal": int(time_optimal),
                    "is_exact_global_configuration": int(exact_global),
                }
                for index, soc in enumerate(state.current_soc, start=1):
                    row[f"soc_d{index}"] = soc
                    row[f"assigned_slot_d{index}"] = evaluation.slot_by_drone[index - 1]
                    row[f"assigned_slot_index_d{index}"] = slot_index[
                        evaluation.slot_by_drone[index - 1]
                    ]
                    row[f"predicted_drop_d{index}"] = evaluation.predicted_drop_pp[
                        index - 1
                    ]
                    row[f"arrival_soc_d{index}"] = evaluation.projected_arrival_soc[
                        index - 1
                    ]
                    row[f"charging_job_minutes_d{index}"] = (
                        evaluation.charging_job_minutes[index - 1]
                    )
                for index, slot in enumerate(rates.slots, start=1):
                    row[f"slot_{index}_id"] = slot.slot_id
                    row[f"slot_{index}_rate_pp_per_min"] = slot.rate_pp_per_min
                    row[f"slot_{index}_assigned_drone"] = drone_by_slot[slot.slot_id]

                writer.writerow(row)
                candidate_count += 1
                eligible_count += int(eligible)
                exact_global_count += int(exact_global)
                objective_optimal_count += int(time_optimal)

            scenario_count += 1
            if scenario_count % 250 == 0:
                print(f"Expanded {scenario_count} states")

    manifest = {
        "dataset_type": "position_aware_structure_candidates",
        "source_state_csv": str(source_csv.resolve()),
        "source_state_csv_sha256": _sha256(source_csv),
        "source_state_manifest": str(source_manifest_path.resolve()),
        "source_rate_table": str(rate_table_path.resolve()),
        "scenario_count": scenario_count,
        "candidate_row_count": candidate_count,
        "eligible_candidate_row_count": eligible_count,
        "exact_global_configuration_rows": exact_global_count,
        "objective_optimal_rows_including_ties": objective_optimal_count,
        "drone_count": 5,
        "position_representation": (
            "Five drone-to-slot indices plus a JSON identity mapping; each structure row "
            "contains its exact best position under the Oracle objective."
        ),
        "objective": (
            "remaining flight time plus exact K-pad exponential-charging makespan"
        ),
        "fully_charged_soc": source_manifest["fully_charged_soc"],
        "zero_to_fully_charged_minutes": source_manifest[
            "zero_to_fully_charged_minutes"
        ],
        "decision_interval_seconds": source_manifest["decision_interval_seconds"],
        "minimum_arrival_soc": source_manifest["minimum_arrival_soc"],
        "tie_tolerance_minutes": 1e-9,
    }
    output_csv.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_position_aware_dataset(args.source_csv, args.output_csv)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
