"""Build complete-configuration cost matrices for controlled-K state groups."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ml_policy.charging_model import FULLY_CHARGED_SOC, ZERO_TO_FULLY_CHARGED_MINUTES
from ml_policy.oracle_optimizer import (
    DEFAULT_RATE_TABLE_PATH,
    EmpiricalRateTable,
    OracleState,
    SafetyTier,
    _evaluate_fixed_position,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASS_TABLE = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
    / "joint_full_configuration_ranker"
    / "complete_configuration_classes.csv"
)


def build_cost_matrix(
    *,
    states_path: Path,
    class_table_path: Path,
    rate_table_path: Path,
    output_path: Path,
) -> dict[str, object]:
    states = pd.read_csv(states_path)
    classes = pd.read_csv(class_table_path)
    rate_table = EmpiricalRateTable.from_csv(rate_table_path)
    source_manifest = json.loads(states_path.with_suffix(".manifest.json").read_text())
    minimum_arrival_soc = float(source_manifest["minimum_arrival_soc"])

    required = {
        "base_state_id",
        "wind_direction",
        "wind_level",
        "charging_pad_count",
        "remaining_distance_m",
        "oracle_total_minutes",
        *(f"soc_d{index}" for index in range(1, 6)),
    }
    missing = required.difference(states.columns)
    if missing:
        raise ValueError(f"State table is missing columns: {sorted(missing)}")

    class_groups: dict[str, list[tuple[int, tuple[int, ...]]]] = {}
    for _, row in classes.iterrows():
        permutation = tuple(
            int(row[f"slot_index_d{index}"]) - 1 for index in range(1, 6)
        )
        class_groups.setdefault(str(row["structure"]), []).append(
            (int(row["class_index"]), permutation)
        )

    costs = np.full((len(states), len(classes)), np.inf, dtype=np.float32)
    mismatch_count = 0
    started = time.perf_counter()
    rate_cache: dict[tuple[str, int], dict[str, object]] = {}

    for row_number, (_, row) in enumerate(states.iterrows()):
        state = OracleState(
            wind_direction=str(row["wind_direction"]),
            wind_level=int(row["wind_level"]),
            charging_pad_count=int(row["charging_pad_count"]),
            current_soc=tuple(float(row[f"soc_d{index}"]) for index in range(1, 6)),
            remaining_distance_m=float(row["remaining_distance_m"]),
            forward_speed_m_per_s=0.10,
            fully_charged_soc=FULLY_CHARGED_SOC,
            zero_to_fully_charged_minutes=ZERO_TO_FULLY_CHARGED_MINUTES,
            minimum_arrival_soc=minimum_arrival_soc,
        )
        condition_key = (state.wind_direction, state.wind_level)
        if condition_key not in rate_cache:
            cells = rate_table.structures_for(
                state.wind_direction, state.wind_level, expected_drone_count=5
            )
            rate_cache[condition_key] = {
                cell.structure.label: cell for cell in cells
            }

        safety_tier_by_structure: dict[str, SafetyTier] = {}
        for structure, rates in rate_cache[condition_key].items():
            safety_tier_by_structure[structure] = rates.safety_tier
            slot_ids = tuple(slot.slot_id for slot in rates.slots)
            for class_index, permutation in class_groups[structure]:
                evaluation = _evaluate_fixed_position(
                    state,
                    rates,
                    tuple(slot_ids[index] for index in permutation),
                )
                if evaluation is not None:
                    costs[row_number, class_index] = evaluation.total_completion_minutes

        # Match solve_oracle exactly: prefer SAFE structures only when at least
        # one SAFE complete configuration is feasible for this particular
        # state.  If all SAFE structures violate minimum arrival SOC, allow
        # feasible BACKUP_ONLY structures instead.
        safe_class_indices = [
            class_index
            for structure, entries in class_groups.items()
            if safety_tier_by_structure.get(structure) == SafetyTier.SAFE
            for class_index, _ in entries
        ]
        has_feasible_safe = bool(
            safe_class_indices
            and np.isfinite(costs[row_number, safe_class_indices]).any()
        )
        if has_feasible_safe:
            for structure, entries in class_groups.items():
                if safety_tier_by_structure.get(structure) == SafetyTier.BACKUP_ONLY:
                    costs[row_number, [class_index for class_index, _ in entries]] = np.inf

        finite = costs[row_number, np.isfinite(costs[row_number])]
        if not len(finite):
            raise RuntimeError(
                f"No feasible configuration for base_state_id={row['base_state_id']}, "
                f"K={row['charging_pad_count']}"
            )
        if not np.isclose(
            float(finite.min()), float(row["oracle_total_minutes"]), atol=2e-5
        ):
            mismatch_count += 1

        if (row_number + 1) % 500 == 0 or row_number + 1 == len(states):
            print(
                f"cost-matrix {row_number + 1}/{len(states)} "
                f"elapsed={time.perf_counter() - started:.1f}s "
                f"mismatches={mismatch_count}",
                flush=True,
            )

    if mismatch_count:
        raise RuntimeError(f"{mismatch_count} rows did not reproduce the global optimum")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        costs=costs,
        base_state_ids=states["base_state_id"].to_numpy(dtype=np.int64),
        charging_pad_counts=states["charging_pad_count"].to_numpy(dtype=np.int8),
    )
    finite_counts = np.isfinite(costs).sum(axis=1)
    report = {
        "status": "pass",
        "states": len(states),
        "base_state_groups": int(states["base_state_id"].nunique()),
        "classes": len(classes),
        "structures": int(classes["structure"].nunique()),
        "positions_per_structure": 120,
        "oracle_minimum_mismatches": mismatch_count,
        "finite_candidates_per_state": {
            "minimum": int(finite_counts.min()),
            "mean": float(finite_counts.mean()),
            "maximum": int(finite_counts.max()),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "states_csv": str(states_path.resolve()),
        "source_rate_table": str(rate_table_path.resolve()),
        "class_table": str(class_table_path.resolve()),
        "cost_matrix": str(output_path.resolve()),
        "grouping_rule": "All K=1..5 rows sharing base_state_id remain in one split.",
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-table", type=Path, default=DEFAULT_CLASS_TABLE)
    parser.add_argument("--rate-table", type=Path, default=DEFAULT_RATE_TABLE_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_cost_matrix(
        states_path=args.states,
        class_table_path=args.class_table,
        rate_table_path=args.rate_table,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
