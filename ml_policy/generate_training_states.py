"""Generate reproducible Oracle-labelled states for policy training.

The current prototype samples remaining distances from 0.25--25 m.  Each row
is an independent decision-epoch scenario:
the five batteries begin at 100%, accumulate a random but reachable history
in fixed 30-second (3 m at 0.1 m/s) steps using safe empirical rate cells, and
are then labelled by the exact offline Oracle at one observed decision state.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from ml_policy.charging_model import (
    DECISION_INTERVAL_SECONDS,
    FULLY_CHARGED_SOC,
    ZERO_TO_FULLY_CHARGED_MINUTES,
    charging_time_constant_minutes,
)
from ml_policy.oracle_optimizer import (
    DEFAULT_RATE_TABLE_PATH,
    EmpiricalRateTable,
    OracleState,
    SafetyTier,
    solve_oracle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
CONDITIONS = tuple(
    (direction, level)
    for direction in ("head", "side", "tail")
    for level in (1, 2)
)


def _simulate_reachable_soc(
    rng: random.Random,
    rate_table: EmpiricalRateTable,
    *,
    history_step_min_m: float,
    history_step_max_m: float,
    maximum_history_steps: int,
    minimum_history_soc: float,
) -> tuple[tuple[float, ...], int, float]:
    soc = [100.0] * 5
    completed_steps = 0
    total_history_distance = 0.0
    requested_steps = rng.randint(0, maximum_history_steps)

    for _ in range(requested_steps):
        direction, level = rng.choice(CONDITIONS)
        cells = [
            cell
            for cell in rate_table.structures_for(direction, level)
            if cell.safety_tier == SafetyTier.SAFE
        ]
        cell = rng.choice(cells)
        segment_distance_m = rng.uniform(history_step_min_m, history_step_max_m)
        segment_minutes = segment_distance_m / 0.10 / 60.0
        rates = [slot.rate_pp_per_min for slot in cell.slots]
        rng.shuffle(rates)
        projected_soc = [
            current - rate * segment_minutes
            for current, rate in zip(soc, rates)
        ]
        if min(projected_soc) < minimum_history_soc:
            break
        soc = projected_soc
        completed_steps += 1
        total_history_distance += segment_distance_m

    return tuple(soc), completed_steps, total_history_distance


@lru_cache(maxsize=4)
def _load_rate_table(path: str) -> EmpiricalRateTable:
    return EmpiricalRateTable.from_csv(path)


def _generate_one(
    scenario_id: int,
    seed: int,
    rate_table_path: str,
    remaining_distance_min_m: float,
    remaining_distance_max_m: float,
    history_step_min_m: float,
    history_step_max_m: float,
    maximum_history_steps: int,
    minimum_history_soc: float,
    minimum_arrival_soc: float,
) -> dict[str, object]:
    rng = random.Random(seed)
    table = _load_rate_table(rate_table_path)

    for _ in range(100):
        current_soc, history_steps, history_distance_m = _simulate_reachable_soc(
            rng,
            table,
            history_step_min_m=history_step_min_m,
            history_step_max_m=history_step_max_m,
            maximum_history_steps=maximum_history_steps,
            minimum_history_soc=minimum_history_soc,
        )
        wind_direction, wind_level = rng.choice(CONDITIONS)
        charging_pad_count = rng.randint(1, 5)
        remaining_distance_m = rng.uniform(
            remaining_distance_min_m,
            remaining_distance_max_m,
        )
        state = OracleState(
            wind_direction=wind_direction,
            wind_level=wind_level,
            charging_pad_count=charging_pad_count,
            current_soc=current_soc,
            remaining_distance_m=remaining_distance_m,
            minimum_arrival_soc=minimum_arrival_soc,
        )
        started = time.perf_counter()
        try:
            result = solve_oracle(state, table)
        except RuntimeError:
            continue
        oracle_runtime_ms = (time.perf_counter() - started) * 1000.0
        break
    else:
        raise RuntimeError(f"Could not generate a feasible state for scenario {scenario_id}")

    sorted_soc = sorted(current_soc)
    safe_ranked = [
        evaluation
        for evaluation in result.ranked_structures
        if evaluation.safety_tier == SafetyTier.SAFE
    ]
    ranking = safe_ranked or list(result.ranked_structures)
    second_best = ranking[1] if len(ranking) > 1 else ranking[0]

    row: dict[str, object] = {
        "scenario_id": scenario_id,
        "scenario_seed": seed,
        "wind_direction": wind_direction,
        "wind_level": wind_level,
        "charging_pad_count": charging_pad_count,
        "remaining_distance_m": remaining_distance_m,
        "history_steps": history_steps,
        "history_distance_m": history_distance_m,
        "soc_d1": current_soc[0],
        "soc_d2": current_soc[1],
        "soc_d3": current_soc[2],
        "soc_d4": current_soc[3],
        "soc_d5": current_soc[4],
        "soc_lowest": sorted_soc[0],
        "soc_second_lowest": sorted_soc[1],
        "soc_middle": sorted_soc[2],
        "soc_second_highest": sorted_soc[3],
        "soc_highest": sorted_soc[4],
        "soc_range": sorted_soc[4] - sorted_soc[0],
        "oracle_structure": result.selected.structure.label,
        "oracle_position_json": json.dumps(
            result.selected.position_mapping(state.drone_ids),
            sort_keys=True,
        ),
        "oracle_total_minutes": result.selected.total_completion_minutes,
        "oracle_charging_minutes": result.selected.charging_schedule.makespan_minutes,
        "oracle_second_structure": second_best.structure.label,
        "oracle_second_total_minutes": second_best.total_completion_minutes,
        "oracle_margin_minutes": (
            second_best.total_completion_minutes
            - result.selected.total_completion_minutes
        ),
        "oracle_runtime_ms": oracle_runtime_ms,
        "safe_structure_count": len(safe_ranked),
    }
    for evaluation in result.ranked_structures:
        row[f"time__{evaluation.structure.label}"] = evaluation.total_completion_minutes
        row[f"tier__{evaluation.structure.label}"] = evaluation.safety_tier.name.lower()
    return row


def _worker(payload: tuple[object, ...]) -> dict[str, object]:
    return _generate_one(*payload)  # type: ignore[arg-type]


def generate_dataset(
    *,
    scenario_count: int,
    random_seed: int,
    output_csv: Path,
    rate_table_path: Path,
    remaining_distance_min_m: float,
    remaining_distance_max_m: float,
    history_step_min_m: float,
    history_step_max_m: float,
    maximum_history_steps: int,
    minimum_history_soc: float,
    minimum_arrival_soc: float,
    worker_count: int,
) -> None:
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive")
    if remaining_distance_min_m <= 0:
        raise ValueError("remaining_distance_min_m must be positive")
    if remaining_distance_max_m < remaining_distance_min_m:
        raise ValueError("remaining distance maximum cannot be below minimum")

    master_rng = random.Random(random_seed)
    payloads = [
        (
            scenario_id,
            master_rng.randrange(0, 2**63),
            str(rate_table_path),
            remaining_distance_min_m,
            remaining_distance_max_m,
            history_step_min_m,
            history_step_max_m,
            maximum_history_steps,
            minimum_history_soc,
            minimum_arrival_soc,
        )
        for scenario_id in range(scenario_count)
    ]

    if worker_count == 1:
        rows = [_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(_worker, payloads, chunksize=8))

    all_fields = sorted({field for row in rows for field in row})
    preferred_fields = [
        "scenario_id",
        "scenario_seed",
        "wind_direction",
        "wind_level",
        "charging_pad_count",
        "remaining_distance_m",
        "history_steps",
        "history_distance_m",
        "soc_d1",
        "soc_d2",
        "soc_d3",
        "soc_d4",
        "soc_d5",
        "soc_lowest",
        "soc_second_lowest",
        "soc_middle",
        "soc_second_highest",
        "soc_highest",
        "soc_range",
        "oracle_structure",
        "oracle_position_json",
        "oracle_total_minutes",
        "oracle_charging_minutes",
        "oracle_second_structure",
        "oracle_second_total_minutes",
        "oracle_margin_minutes",
        "oracle_runtime_ms",
        "safe_structure_count",
    ]
    remaining_fields = [field for field in all_fields if field not in preferred_fields]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred_fields + remaining_fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "scenario_count": scenario_count,
        "random_seed": random_seed,
        "source_rate_table": str(rate_table_path),
        "remaining_distance_min_m": remaining_distance_min_m,
        "remaining_distance_max_m": remaining_distance_max_m,
        "history_step_min_m": history_step_min_m,
        "history_step_max_m": history_step_max_m,
        "maximum_history_steps": maximum_history_steps,
        "minimum_history_soc": minimum_history_soc,
        "minimum_arrival_soc": minimum_arrival_soc,
        "worker_count": worker_count,
        "charging_model": "exponential_to_fully_charged",
        "fully_charged_soc": FULLY_CHARGED_SOC,
        "zero_to_fully_charged_minutes": ZERO_TO_FULLY_CHARGED_MINUTES,
        "charging_time_constant_minutes": charging_time_constant_minutes(),
        "decision_interval_seconds": DECISION_INTERVAL_SECONDS,
        "refresh_at_each_interval": [
            "wind_direction",
            "wind_level",
            "charging_pad_count",
            "current_soc",
            "remaining_distance_m",
        ],
        "decision_interval_distance_m_at_nominal_speed": 3.0,
        "future_wind_known": False,
        "future_charging_pad_availability_known": False,
        "forward_speed_m_per_s": 0.10,
        "safety_rule": "2+ selected runs safe; 1 run backup-only; repeated-collision cells prohibited",
    }
    output_csv.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-count", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260817)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "oracle_training_states_0p25_25m.csv",
    )
    parser.add_argument("--rate-table", type=Path, default=DEFAULT_RATE_TABLE_PATH)
    parser.add_argument("--remaining-distance-min-m", type=float, default=0.25)
    parser.add_argument("--remaining-distance-max-m", type=float, default=25.0)
    parser.add_argument("--history-step-min-m", type=float, default=3.0)
    parser.add_argument("--history-step-max-m", type=float, default=3.0)
    parser.add_argument("--maximum-history-steps", type=int, default=20)
    parser.add_argument("--minimum-history-soc", type=float, default=35.0)
    parser.add_argument("--minimum-arrival-soc", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    generate_dataset(
        scenario_count=args.scenario_count,
        random_seed=args.random_seed,
        output_csv=args.output_csv,
        rate_table_path=args.rate_table,
        remaining_distance_min_m=args.remaining_distance_min_m,
        remaining_distance_max_m=args.remaining_distance_max_m,
        history_step_min_m=args.history_step_min_m,
        history_step_max_m=args.history_step_max_m,
        maximum_history_steps=args.maximum_history_steps,
        minimum_history_soc=args.minimum_history_soc,
        minimum_arrival_soc=args.minimum_arrival_soc,
        worker_count=args.workers,
    )
    print(f"Wrote {args.scenario_count} Oracle-labelled states to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
