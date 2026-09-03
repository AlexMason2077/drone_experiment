"""Build exact Oracle cost matrices over every complete configuration class."""

from __future__ import annotations

import argparse
import json
import time
from itertools import permutations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ml_policy.oracle_optimizer import EmpiricalRateTable, OracleState, _evaluate_fixed_position


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_OUTPUT_DIR = DEFAULT_DIR / "joint_full_configuration_ranker"


def slot_ids_by_structure(*candidate_frames: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for frame in candidate_frames:
        for _, row in frame.drop_duplicates("structure").iterrows():
            structure = str(row["structure"])
            slots = tuple(str(row[f"slot_{index}_id"]) for index in range(1, 6))
            if structure in result and result[structure] != slots:
                raise ValueError(f"Inconsistent slot ordering for {structure}")
            result[structure] = slots
    return dict(sorted(result.items()))


def configuration_classes(
    slots: dict[str, tuple[str, ...]],
) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (structure, tuple(int(value) for value in permutation))
        for structure in sorted(slots)
        for permutation in permutations(range(5))
    ]


def build_cost_matrix(
    states_path: Path,
    candidates_path: Path,
    all_candidates_path: Path,
    output_path: Path,
) -> dict[str, object]:
    states = pd.read_csv(states_path)
    candidates = pd.read_csv(candidates_path)
    all_candidates = pd.read_csv(all_candidates_path)
    slots_by_structure = slot_ids_by_structure(candidates, all_candidates)
    classes = configuration_classes(slots_by_structure)
    class_indices_by_structure: dict[str, list[tuple[int, tuple[int, ...]]]] = {}
    for index, (structure, permutation) in enumerate(classes):
        class_indices_by_structure.setdefault(structure, []).append((index, permutation))

    candidate_groups = {
        int(scenario_id): group[group["eligible_for_selection"].eq(1)]
        for scenario_id, group in candidates.groupby("scenario_id", sort=False)
    }
    manifest = json.loads(candidates_path.with_suffix(".manifest.json").read_text())
    source_manifest = json.loads(Path(manifest["source_state_manifest"]).read_text())
    rate_table = EmpiricalRateTable.from_csv(Path(manifest["source_rate_table"]))
    costs = np.full((len(states), len(classes)), np.inf, dtype=np.float32)
    rate_cache: dict[tuple[str, int, str], object] = {}
    mismatch_count = 0
    started = time.perf_counter()

    for row_number, (_, row) in enumerate(states.iterrows()):
        state = OracleState(
            wind_direction=row["wind_direction"],
            wind_level=int(row["wind_level"]),
            charging_pad_count=int(row["charging_pad_count"]),
            current_soc=tuple(float(row[f"soc_d{index}"]) for index in range(1, 6)),
            remaining_distance_m=float(row["remaining_distance_m"]),
            forward_speed_m_per_s=float(source_manifest["forward_speed_m_per_s"]),
            fully_charged_soc=float(source_manifest["fully_charged_soc"]),
            zero_to_fully_charged_minutes=float(source_manifest["zero_to_fully_charged_minutes"]),
            minimum_arrival_soc=float(source_manifest["minimum_arrival_soc"]),
        )
        eligible = candidate_groups[int(row["scenario_id"])]
        for structure in eligible["structure"].astype(str):
            cache_key = (state.wind_direction, state.wind_level, structure)
            if cache_key not in rate_cache:
                rate_cache[cache_key] = next(
                    rates
                    for rates in rate_table.structures_for(
                        state.wind_direction, state.wind_level, expected_drone_count=5
                    )
                    if rates.structure.label == structure
                )
            rates = rate_cache[cache_key]
            slot_ids = slots_by_structure[structure]
            for class_index, permutation in class_indices_by_structure[structure]:
                evaluation = _evaluate_fixed_position(
                    state,
                    rates,
                    tuple(slot_ids[index] for index in permutation),
                )
                if evaluation is not None:
                    costs[row_number, class_index] = evaluation.total_completion_minutes

        finite = costs[row_number, np.isfinite(costs[row_number])]
        if not len(finite):
            raise RuntimeError(f"No feasible configuration for scenario {row['scenario_id']}")
        if not np.isclose(float(finite.min()), float(row["oracle_total_minutes"]), atol=2e-5):
            mismatch_count += 1
        if (row_number + 1) % 250 == 0 or row_number + 1 == len(states):
            print(
                f"cost-matrix {row_number + 1}/{len(states)} "
                f"elapsed={time.perf_counter() - started:.1f}s mismatches={mismatch_count}",
                flush=True,
            )

    if mismatch_count:
        raise RuntimeError(f"{mismatch_count} rows did not reproduce the Oracle optimum")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        costs=costs,
        scenario_ids=states["scenario_id"].to_numpy(dtype=np.int64),
    )
    class_frame = pd.DataFrame(
        [
            {
                "class_index": index,
                "structure": structure,
                **{
                    f"slot_index_d{drone}": permutation[drone - 1] + 1
                    for drone in range(1, 6)
                },
            }
            for index, (structure, permutation) in enumerate(classes)
        ]
    )
    class_path = output_path.parent / "complete_configuration_classes.csv"
    class_frame.to_csv(class_path, index=False)
    finite_counts = np.isfinite(costs).sum(axis=1)
    report = {
        "status": "pass",
        "states": len(states),
        "classes": len(classes),
        "structures": len(slots_by_structure),
        "positions_per_structure": 120,
        "oracle_minimum_mismatches": mismatch_count,
        "finite_candidates_per_state": {
            "minimum": int(finite_counts.min()),
            "mean": float(finite_counts.mean()),
            "maximum": int(finite_counts.max()),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "states_csv": str(states_path.resolve()),
        "candidate_csv": str(candidates_path.resolve()),
        "cost_matrix": str(output_path.resolve()),
        "class_table": str(class_path.resolve()),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--all-candidates",
        type=Path,
        default=DEFAULT_DIR / "position_aware_training_candidates.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_cost_matrix(
        args.states, args.candidates, args.all_candidates, args.output
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
