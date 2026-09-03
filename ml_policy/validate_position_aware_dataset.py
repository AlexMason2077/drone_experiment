"""Independently re-solve and audit every position-aware candidate row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
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
DEFAULT_DATASET = DEFAULT_DIR / "position_aware_training_candidates.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance


def validate_position_aware_dataset(
    dataset_csv: Path,
    output_json: Path,
    *,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    dataset_manifest_path = dataset_csv.with_suffix(".manifest.json")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    source_manifest_path = Path(dataset_manifest["source_state_manifest"])
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    rate_table_path = Path(dataset_manifest["source_rate_table"])
    rate_table = EmpiricalRateTable.from_csv(rate_table_path)

    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    with dataset_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[int(row["scenario_id"])].append(row)

    checked_rows = 0
    position_matches = 0
    objective_matches = 0
    schedule_matches = 0
    structure_sets_match = 0
    permutation_valid_rows = 0
    global_labels_match = 0
    mismatch_examples: list[dict[str, object]] = []
    started = time.perf_counter()

    for group_index, (scenario_id, rows) in enumerate(sorted(groups.items()), start=1):
        first = rows[0]
        state = OracleState(
            wind_direction=first["wind_direction"],
            wind_level=int(first["wind_level"]),
            charging_pad_count=int(first["charging_pad_count"]),
            current_soc=tuple(float(first[f"soc_d{index}"]) for index in range(1, 6)),
            remaining_distance_m=float(first["remaining_distance_m"]),
            forward_speed_m_per_s=float(source_manifest["forward_speed_m_per_s"]),
            fully_charged_soc=float(source_manifest["fully_charged_soc"]),
            zero_to_fully_charged_minutes=float(
                source_manifest["zero_to_fully_charged_minutes"]
            ),
            minimum_arrival_soc=float(source_manifest["minimum_arrival_soc"]),
        )
        result = solve_oracle(state, rate_table)
        recalculated = {
            evaluation.structure.label: evaluation
            for evaluation in result.ranked_structures
        }
        stored_labels = {row["structure"] for row in rows}
        structure_ok = stored_labels == set(recalculated)
        structure_sets_match += int(structure_ok)
        selected_position = result.selected.position_mapping(state.drone_ids)

        for row in rows:
            checked_rows += 1
            evaluation = recalculated.get(row["structure"])
            if evaluation is None:
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {"scenario_id": scenario_id, "structure": row["structure"],
                         "reason": "stored structure missing from fresh Oracle solve"}
                    )
                continue

            stored_position = json.loads(row["position_json"])
            recalculated_position = evaluation.position_mapping(state.drone_ids)
            position_ok = stored_position == recalculated_position
            objective_ok = (
                _close(
                    float(row["total_completion_minutes"]),
                    evaluation.total_completion_minutes,
                    tolerance,
                )
                and _close(
                    float(row["charging_makespan_minutes"]),
                    evaluation.charging_schedule.makespan_minutes,
                    tolerance,
                )
            )
            stored_schedule = tuple(
                tuple(group) for group in json.loads(row["charging_schedule_json"])
            )
            recalculated_schedule = tuple(
                tuple(state.drone_ids[index] for index in group)
                for group in evaluation.charging_schedule.drone_indices_by_pad
            )
            schedule_ok = stored_schedule == recalculated_schedule
            permutation_ok = sorted(
                int(row[f"assigned_slot_index_d{index}"]) for index in range(1, 6)
            ) == [1, 2, 3, 4, 5]

            has_safe = any(
                item.safety_tier == SafetyTier.SAFE
                for item in result.ranked_structures
            )
            eligible = (
                evaluation.safety_tier == SafetyTier.SAFE
                if has_safe
                else evaluation.safety_tier == SafetyTier.BACKUP_ONLY
            )
            exact_global = (
                evaluation.structure == result.selected.structure
                and recalculated_position == selected_position
            )
            regret = (
                evaluation.total_completion_minutes
                - result.selected.total_completion_minutes
                if eligible
                else None
            )
            global_ok = (
                row["global_oracle_structure"] == result.selected.structure.label
                and json.loads(row["global_oracle_position_json"])
                == selected_position
                and _close(
                    float(row["global_oracle_total_minutes"]),
                    result.selected.total_completion_minutes,
                    tolerance,
                )
                and int(row["eligible_for_selection"]) == int(eligible)
                and int(row["is_exact_global_configuration"]) == int(exact_global)
                and (
                    (regret is None and row["objective_regret_minutes"] == "")
                    or (
                        regret is not None
                        and _close(
                            float(row["objective_regret_minutes"]),
                            regret,
                            tolerance,
                        )
                    )
                )
            )

            position_matches += int(position_ok)
            objective_matches += int(objective_ok)
            schedule_matches += int(schedule_ok)
            permutation_valid_rows += int(permutation_ok)
            global_labels_match += int(global_ok)
            if not all((position_ok, objective_ok, schedule_ok, permutation_ok, global_ok)):
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "scenario_id": scenario_id,
                            "structure": row["structure"],
                            "position_ok": position_ok,
                            "objective_ok": objective_ok,
                            "schedule_ok": schedule_ok,
                            "permutation_ok": permutation_ok,
                            "global_ok": global_ok,
                        }
                    )

        if group_index % 250 == 0:
            print(f"Validated {group_index} states")

    scenario_count = len(groups)
    all_rows_match = min(
        position_matches,
        objective_matches,
        schedule_matches,
        permutation_valid_rows,
        global_labels_match,
    ) == checked_rows
    passed = all_rows_match and structure_sets_match == scenario_count
    report: dict[str, object] = {
        "status": "pass" if passed else "fail",
        "dataset_csv": str(dataset_csv.resolve()),
        "dataset_csv_sha256": _sha256(dataset_csv),
        "dataset_manifest": str(dataset_manifest_path.resolve()),
        "scenario_count": scenario_count,
        "checked_candidate_rows": checked_rows,
        "structure_sets_match": structure_sets_match,
        "position_matches": position_matches,
        "objective_matches": objective_matches,
        "charging_schedule_matches": schedule_matches,
        "permutation_valid_rows": permutation_valid_rows,
        "global_labels_match": global_labels_match,
        "mismatch_count": 0 if passed else checked_rows - min(
            position_matches,
            objective_matches,
            schedule_matches,
            permutation_valid_rows,
            global_labels_match,
        ),
        "mismatch_examples": mismatch_examples,
        "elapsed_seconds": time.perf_counter() - started,
        "definition": (
            "Every stored structure-specific best position, completion time, charging "
            "schedule, eligibility flag, and global label must match a fresh exact "
            "Oracle solve; every position must be a one-to-one slot permutation."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = args.output_json or args.dataset_csv.with_suffix(".validation.json")
    report = validate_position_aware_dataset(args.dataset_csv, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
