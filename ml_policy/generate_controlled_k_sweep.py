"""Generate controlled charging-pad-count sweeps with exact Oracle labels.

For every selected base state, wind direction, wind level, remaining distance,
and all five current SOC values are held fixed.  Only the charging-pad count K
is varied from 1 through 5.  Keeping these five rows together makes the effect
of charging-pad availability identifiable and prevents train/validation
leakage when the data are split by base_state_id.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from ml_policy.oracle_optimizer import (
    DEFAULT_RATE_TABLE_PATH,
    EmpiricalRateTable,
    OracleState,
    SafetyTier,
    solve_oracle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
    / "oracle_training_states_0p25_25m.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "controlled_k_sweep"
    / "oracle_controlled_k_sweep.csv"
)


def _charging_schedule_json(result, drone_ids: Sequence[str]) -> str:
    schedule = [
        [drone_ids[index] for index in group]
        for group in result.selected.charging_schedule.drone_indices_by_pad
    ]
    return json.dumps(schedule, separators=(",", ":"))


def generate_controlled_k_sweep(
    *,
    source_csv: Path,
    output_csv: Path,
    base_state_count: int,
    base_state_offset: int,
    rate_table_path: Path,
    minimum_arrival_soc: float,
) -> None:
    table = EmpiricalRateTable.from_csv(rate_table_path)
    with source_csv.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    selected_rows = source_rows[
        base_state_offset : base_state_offset + base_state_count
    ]
    if len(selected_rows) != base_state_count:
        raise ValueError(
            f"Requested {base_state_count} base states at offset {base_state_offset}, "
            f"but only {len(selected_rows)} are available"
        )

    output_rows: list[dict[str, object]] = []
    for local_index, source in enumerate(selected_rows):
        base_state_id = local_index
        current_soc = tuple(float(source[f"soc_d{index}"]) for index in range(1, 6))
        direction = source["wind_direction"].strip().lower()
        level = int(source["wind_level"])
        distance = float(source["remaining_distance_m"])

        group_rows: list[dict[str, object]] = []
        for k in range(1, 6):
            state = OracleState(
                wind_direction=direction,
                wind_level=level,
                charging_pad_count=k,
                current_soc=current_soc,
                remaining_distance_m=distance,
                minimum_arrival_soc=minimum_arrival_soc,
            )
            result = solve_oracle(state, table)
            safe_ranked = [
                evaluation
                for evaluation in result.ranked_structures
                if evaluation.safety_tier == SafetyTier.SAFE
            ]
            ranking = safe_ranked or list(result.ranked_structures)
            second_best = ranking[1] if len(ranking) > 1 else ranking[0]
            group_rows.append(
                {
                    "base_state_id": base_state_id,
                    "source_scenario_id": source.get("scenario_id", ""),
                    "wind_direction": direction,
                    "wind_level": level,
                    "remaining_distance_m": distance,
                    "soc_d1": current_soc[0],
                    "soc_d2": current_soc[1],
                    "soc_d3": current_soc[2],
                    "soc_d4": current_soc[3],
                    "soc_d5": current_soc[4],
                    "charging_pad_count": k,
                    "oracle_structure": result.selected.structure.label,
                    "oracle_position_json": json.dumps(
                        result.selected.position_mapping(state.drone_ids),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "oracle_total_minutes": result.selected.total_completion_minutes,
                    "oracle_charging_minutes": (
                        result.selected.charging_schedule.makespan_minutes
                    ),
                    "oracle_arrival_soc_json": json.dumps(
                        result.selected.projected_arrival_soc,
                        separators=(",", ":"),
                    ),
                    "oracle_charging_schedule_json": _charging_schedule_json(
                        result, state.drone_ids
                    ),
                    "oracle_second_structure": second_best.structure.label,
                    "oracle_second_total_minutes": second_best.total_completion_minutes,
                    "oracle_margin_minutes": (
                        second_best.total_completion_minutes
                        - result.selected.total_completion_minutes
                    ),
                    "safe_structure_count": len(safe_ranked),
                }
            )

        structures = {str(row["oracle_structure"]) for row in group_rows}
        positions = {str(row["oracle_position_json"]) for row in group_rows}
        structure_changes = len(structures) > 1
        complete_configuration_changes = len(
            {
                (str(row["oracle_structure"]), str(row["oracle_position_json"]))
                for row in group_rows
            }
        ) > 1
        for row in group_rows:
            row["structure_changes_across_k"] = int(structure_changes)
            row["position_changes_across_k"] = int(len(positions) > 1)
            row["complete_configuration_changes_across_k"] = int(
                complete_configuration_changes
            )
            output_rows.append(row)

    fields = list(output_rows[0])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    base_groups = {
        int(row["base_state_id"]): row for row in output_rows if row["charging_pad_count"] == 1
    }
    manifest = {
        "source_csv": str(source_csv),
        "source_rate_table": str(rate_table_path),
        "base_state_count": base_state_count,
        "rows": len(output_rows),
        "k_values": [1, 2, 3, 4, 5],
        "minimum_arrival_soc": minimum_arrival_soc,
        "controlled_variables": [
            "wind_direction",
            "wind_level",
            "remaining_distance_m",
            "soc_d1",
            "soc_d2",
            "soc_d3",
            "soc_d4",
            "soc_d5",
        ],
        "varied_variable": "charging_pad_count",
        "split_rule": "Split by base_state_id; never split individual K rows.",
        "groups_with_structure_change": sum(
            int(row["structure_changes_across_k"]) for row in base_groups.values()
        ),
        "groups_with_complete_configuration_change": sum(
            int(row["complete_configuration_changes_across_k"])
            for row in base_groups.values()
        ),
    }
    output_csv.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-state-count", type=int, default=200)
    parser.add_argument("--base-state-offset", type=int, default=0)
    parser.add_argument("--rate-table", type=Path, default=DEFAULT_RATE_TABLE_PATH)
    parser.add_argument("--minimum-arrival-soc", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    generate_controlled_k_sweep(
        source_csv=args.source_csv,
        output_csv=args.output_csv,
        base_state_count=args.base_state_count,
        base_state_offset=args.base_state_offset,
        rate_table_path=args.rate_table,
        minimum_arrival_soc=args.minimum_arrival_soc,
    )
    print(f"Wrote controlled K sweep to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
