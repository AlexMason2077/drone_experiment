"""Exact offline reference solver for configuration-policy training.

The solver uses the pooled, Bideal-normalized, forward-only discharge rates
already produced by the data-cleaning pipeline.  For a single observed state
it evaluates every *safe* formation/spacing structure, finds that structure's
best drone-to-slot position assignment, and finds the optimal allocation of
the resulting charging jobs to a fixed number of identical charging pads.

This module is intentionally an offline reference implementation.  It is used
to create and validate labels for a faster learned online policy; it is not
the learned policy itself.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from enum import IntEnum
from itertools import permutations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ml_policy.charging_model import (
    FULLY_CHARGED_SOC,
    ZERO_TO_FULLY_CHARGED_MINUTES,
    exponential_charging_minutes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATE_TABLE_PATH = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "configuration_condition_rate_bar_charts"
    / "pooled_configuration_drone_Bideal_forward_rates.csv"
)

DEFAULT_DRONE_IDS = ("D1", "D2", "D3", "D4", "D5")

# These cells were declared unsafe because of repeated collisions.  The
# current rate table omits them; the explicit mask prevents a later table
# update from accidentally returning them to the feasible set.
UNSAFE_STRUCTURES_BY_CONDITION: Mapping[
    tuple[str, int], frozenset[tuple[str, int]]
] = {
    ("head", 2): frozenset({("column", 50)}),
    ("side", 2): frozenset({("column", 50), ("diamond", 50)}),
    ("tail", 2): frozenset({("diamond", 50)}),
}


def _canonical_formation(value: str) -> str:
    formation = value.strip().lower()
    # The source database uses "echalon"; accept the standard spelling at
    # interfaces while retaining the database key internally.
    if formation == "echelon":
        return "echalon"
    return formation


def _slot_sort_key(slot_id: str) -> tuple[str, int, str]:
    prefix, separator, suffix = slot_id.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, int(suffix), slot_id
    return slot_id, 0, slot_id


@dataclass(frozen=True, order=True)
class StructureKey:
    formation: str
    distance_cm: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "formation", _canonical_formation(self.formation))
        if self.distance_cm <= 0:
            raise ValueError("distance_cm must be positive")

    @property
    def label(self) -> str:
        display = "echelon" if self.formation == "echalon" else self.formation
        return f"{display}_{self.distance_cm}"


@dataclass(frozen=True)
class SlotRate:
    slot_id: str
    rate_pp_per_min: float
    selected_run_count: int


class SafetyTier(IntEnum):
    """Condition-specific eligibility tier used before time optimisation."""

    SAFE = 0
    BACKUP_ONLY = 1


@dataclass(frozen=True)
class StructureRates:
    wind_direction: str
    wind_level: int
    structure: StructureKey
    slots: tuple[SlotRate, ...]

    @property
    def selected_run_count(self) -> int:
        counts = {slot.selected_run_count for slot in self.slots}
        if len(counts) != 1:
            raise ValueError(
                f"Inconsistent selected_run_count values for {self.structure.label}: "
                f"{sorted(counts)}"
            )
        return next(iter(counts))

    @property
    def safety_tier(self) -> SafetyTier:
        # User-defined evidence rule: one or fewer selected runs has a safety
        # concern and is retained only as a fallback.  Cells with at least two
        # selected runs are eligible for ordinary optimisation.
        if self.selected_run_count <= 1:
            return SafetyTier.BACKUP_ONLY
        return SafetyTier.SAFE


class EmpiricalRateTable:
    """Validated access to the processed pooled forward-rate table."""

    def __init__(self, cells: Mapping[tuple[str, int, StructureKey], StructureRates]):
        self._cells = dict(cells)

    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_RATE_TABLE_PATH) -> "EmpiricalRateTable":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Rate table not found: {source}")

        grouped: dict[
            tuple[str, int, StructureKey], list[SlotRate]
        ] = {}
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "formation",
                "inter_drone_spacing_cm",
                "wind_direction",
                "wind_level",
                "slot_id",
                "selected_run_count",
                "pooled_Bideal_forward_rate_pp_per_min",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    "Rate table is missing required columns: "
                    + ", ".join(sorted(missing))
                )

            for row in reader:
                direction = row["wind_direction"].strip().lower()
                level = int(row["wind_level"])
                structure = StructureKey(
                    formation=row["formation"],
                    distance_cm=int(float(row["inter_drone_spacing_cm"])),
                )
                rate = float(row["pooled_Bideal_forward_rate_pp_per_min"])
                if rate < 0:
                    raise ValueError(
                        f"Negative forward rate for {direction} level {level}, "
                        f"{structure.label}, {row['slot_id']}: {rate}"
                    )
                key = (direction, level, structure)
                grouped.setdefault(key, []).append(
                    SlotRate(
                        slot_id=row["slot_id"].strip(),
                        rate_pp_per_min=rate,
                        selected_run_count=int(row["selected_run_count"]),
                    )
                )

        cells: dict[tuple[str, int, StructureKey], StructureRates] = {}
        for key, slots in grouped.items():
            direction, level, structure = key
            ordered_slots = tuple(sorted(slots, key=lambda item: _slot_sort_key(item.slot_id)))
            slot_ids = [slot.slot_id for slot in ordered_slots]
            if len(slot_ids) != len(set(slot_ids)):
                raise ValueError(
                    f"Duplicate slot IDs for {direction} level {level}, {structure.label}"
                )
            cells[key] = StructureRates(
                wind_direction=direction,
                wind_level=level,
                structure=structure,
                slots=ordered_slots,
            )
        return cls(cells)

    def structures_for(
        self,
        wind_direction: str,
        wind_level: int,
        *,
        expected_drone_count: int | None = 5,
    ) -> tuple[StructureRates, ...]:
        direction = wind_direction.strip().lower()
        unsafe = UNSAFE_STRUCTURES_BY_CONDITION.get((direction, wind_level), frozenset())
        cells = [
            cell
            for (cell_direction, cell_level, structure), cell in self._cells.items()
            if cell_direction == direction
            and cell_level == wind_level
            and (structure.formation, structure.distance_cm) not in unsafe
        ]
        cells.sort(key=lambda cell: cell.structure)
        if not cells:
            raise KeyError(f"No rate data for {direction} wind, level {wind_level}")
        if expected_drone_count is not None:
            wrong = [
                cell.structure.label
                for cell in cells
                if len(cell.slots) != expected_drone_count
            ]
            if wrong:
                raise ValueError(
                    f"Expected {expected_drone_count} slots but found a different count for: "
                    + ", ".join(wrong)
                )
        return tuple(cells)


@dataclass(frozen=True)
class OracleState:
    wind_direction: str
    wind_level: int
    charging_pad_count: int
    current_soc: tuple[float, ...]
    remaining_distance_m: float
    drone_ids: tuple[str, ...] = DEFAULT_DRONE_IDS
    forward_speed_m_per_s: float = 0.10
    fully_charged_soc: float = FULLY_CHARGED_SOC
    zero_to_fully_charged_minutes: float = ZERO_TO_FULLY_CHARGED_MINUTES
    minimum_arrival_soc: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "wind_direction", self.wind_direction.strip().lower())
        object.__setattr__(self, "current_soc", tuple(float(value) for value in self.current_soc))
        object.__setattr__(self, "drone_ids", tuple(self.drone_ids))
        if self.wind_level <= 0:
            raise ValueError("wind_level must be positive")
        if self.charging_pad_count <= 0:
            raise ValueError("charging_pad_count must be positive")
        if len(self.current_soc) != len(self.drone_ids):
            raise ValueError("current_soc and drone_ids must have the same length")
        if len(set(self.drone_ids)) != len(self.drone_ids):
            raise ValueError("drone_ids must be unique")
        if any(value < 0 or value > 100 for value in self.current_soc):
            raise ValueError("Every current SOC value must be between 0 and 100")
        if self.remaining_distance_m < 0:
            raise ValueError("remaining_distance_m cannot be negative")
        if self.forward_speed_m_per_s <= 0:
            raise ValueError("forward_speed_m_per_s must be positive")
        if not 0 < self.fully_charged_soc < 100:
            raise ValueError("fully_charged_soc must be strictly between 0 and 100")
        if self.zero_to_fully_charged_minutes <= 0:
            raise ValueError("zero_to_fully_charged_minutes must be positive")
        if self.minimum_arrival_soc < 0 or self.minimum_arrival_soc > 100:
            raise ValueError("minimum_arrival_soc must be between 0 and 100")

    @property
    def remaining_forward_minutes(self) -> float:
        return self.remaining_distance_m / self.forward_speed_m_per_s / 60.0


@dataclass(frozen=True)
class ChargingSchedule:
    makespan_minutes: float
    drone_indices_by_pad: tuple[tuple[int, ...], ...]
    pad_loads_minutes: tuple[float, ...]


@dataclass(frozen=True)
class StructureEvaluation:
    structure: StructureKey
    slot_by_drone: tuple[str, ...]
    predicted_drop_pp: tuple[float, ...]
    projected_arrival_soc: tuple[float, ...]
    charging_job_minutes: tuple[float, ...]
    charging_schedule: ChargingSchedule
    remaining_flight_minutes: float
    total_completion_minutes: float
    safety_tier: SafetyTier
    selected_run_count: int

    def position_mapping(self, drone_ids: Sequence[str]) -> dict[str, str]:
        return dict(zip(drone_ids, self.slot_by_drone))


@dataclass(frozen=True)
class OracleResult:
    state: OracleState
    selected: StructureEvaluation
    ranked_structures: tuple[StructureEvaluation, ...]


def _optimal_parallel_charging_schedule(
    charging_jobs_minutes: Sequence[float],
    pad_count: int,
) -> ChargingSchedule:
    """Return the exact minimum-makespan assignment to identical pads."""

    if pad_count <= 0:
        raise ValueError("pad_count must be positive")
    jobs = tuple(float(value) for value in charging_jobs_minutes)
    if any(value < 0 for value in jobs):
        raise ValueError("charging jobs cannot be negative")
    if not jobs:
        return ChargingSchedule(0.0, (), ())

    effective_pad_count = min(pad_count, len(jobs))
    if effective_pad_count == 1:
        return ChargingSchedule(
            makespan_minutes=sum(jobs),
            drone_indices_by_pad=(tuple(range(len(jobs))),),
            pad_loads_minutes=(sum(jobs),),
        )
    if effective_pad_count == len(jobs):
        order = tuple(sorted(range(len(jobs)), key=lambda index: (-jobs[index], index)))
        return ChargingSchedule(
            makespan_minutes=max(jobs),
            drone_indices_by_pad=tuple((index,) for index in order),
            pad_loads_minutes=tuple(jobs[index] for index in order),
        )

    order = sorted(range(len(jobs)), key=lambda index: (-jobs[index], index))
    loads = [0.0] * effective_pad_count
    groups: list[list[int]] = [[] for _ in range(effective_pad_count)]
    best_makespan = float("inf")
    best_groups: tuple[tuple[int, ...], ...] | None = None
    best_loads: tuple[float, ...] | None = None

    def canonical_solution() -> tuple[tuple[float, tuple[int, ...]], ...]:
        return tuple(
            sorted(
                (
                    (round(loads[index], 12), tuple(sorted(groups[index])))
                    for index in range(effective_pad_count)
                ),
                key=lambda item: (-item[0], item[1]),
            )
        )

    def search(job_position: int) -> None:
        nonlocal best_makespan, best_groups, best_loads
        if job_position == len(order):
            solution = canonical_solution()
            makespan = solution[0][0]
            candidate_loads = tuple(item[0] for item in solution)
            candidate_groups = tuple(item[1] for item in solution)
            if (
                makespan < best_makespan - 1e-12
                or (
                    abs(makespan - best_makespan) <= 1e-12
                    and (best_groups is None or candidate_groups < best_groups)
                )
            ):
                best_makespan = makespan
                best_groups = candidate_groups
                best_loads = candidate_loads
            return

        drone_index = order[job_position]
        duration = jobs[drone_index]
        tried_loads: set[float] = set()
        for pad_index, current_load in enumerate(loads):
            rounded_load = round(current_load, 12)
            if rounded_load in tried_loads:
                continue
            tried_loads.add(rounded_load)
            new_load = current_load + duration
            if new_load > best_makespan + 1e-12:
                continue
            loads[pad_index] = new_load
            groups[pad_index].append(drone_index)
            search(job_position + 1)
            groups[pad_index].pop()
            loads[pad_index] = current_load

    search(0)
    assert best_groups is not None and best_loads is not None
    return ChargingSchedule(
        makespan_minutes=best_makespan,
        drone_indices_by_pad=best_groups,
        pad_loads_minutes=best_loads,
    )


def _evaluate_structure(
    state: OracleState,
    rates: StructureRates,
) -> StructureEvaluation | None:
    if len(rates.slots) != len(state.drone_ids):
        raise ValueError(
            f"{rates.structure.label} has {len(rates.slots)} slots for "
            f"{len(state.drone_ids)} drones"
        )

    rate_by_slot = {slot.slot_id: slot.rate_pp_per_min for slot in rates.slots}
    slot_ids = tuple(rate_by_slot)
    best: StructureEvaluation | None = None

    for slot_by_drone in permutations(slot_ids):
        candidate = _evaluate_fixed_position(state, rates, slot_by_drone)
        if candidate is None:
            continue
        if best is None:
            best = candidate
            continue
        candidate_key = (
            round(candidate.total_completion_minutes, 12),
            candidate.slot_by_drone,
            candidate.charging_schedule.drone_indices_by_pad,
        )
        best_key = (
            round(best.total_completion_minutes, 12),
            best.slot_by_drone,
            best.charging_schedule.drone_indices_by_pad,
        )
        if candidate_key < best_key:
            best = candidate

    return best


def _evaluate_fixed_position(
    state: OracleState,
    rates: StructureRates,
    slot_by_drone: Sequence[str],
) -> StructureEvaluation | None:
    """Evaluate one complete (formation, spacing, position) configuration."""

    assignment = tuple(slot_by_drone)
    valid_slots = tuple(slot.slot_id for slot in rates.slots)
    if len(assignment) != len(state.drone_ids):
        raise ValueError("slot_by_drone must contain one slot for every drone")
    if sorted(assignment) != sorted(valid_slots):
        raise ValueError("slot_by_drone must be a one-to-one permutation of structure slots")

    rate_by_slot = {slot.slot_id: slot.rate_pp_per_min for slot in rates.slots}
    predicted_drop = tuple(
        rate_by_slot[slot_id] * state.remaining_forward_minutes
        for slot_id in assignment
    )
    arrival_soc = tuple(
        soc - drop
        for soc, drop in zip(state.current_soc, predicted_drop)
    )
    if min(arrival_soc) < state.minimum_arrival_soc - 1e-12:
        return None
    charging_jobs = tuple(
        exponential_charging_minutes(
            soc,
            fully_charged_soc=state.fully_charged_soc,
            zero_to_fully_charged_minutes=state.zero_to_fully_charged_minutes,
        )
        for soc in arrival_soc
    )
    schedule = _optimal_parallel_charging_schedule(
        charging_jobs,
        state.charging_pad_count,
    )
    return StructureEvaluation(
        structure=rates.structure,
        slot_by_drone=assignment,
        predicted_drop_pp=predicted_drop,
        projected_arrival_soc=arrival_soc,
        charging_job_minutes=charging_jobs,
        charging_schedule=schedule,
        remaining_flight_minutes=state.remaining_forward_minutes,
        total_completion_minutes=(
            state.remaining_forward_minutes + schedule.makespan_minutes
        ),
        safety_tier=rates.safety_tier,
        selected_run_count=rates.selected_run_count,
    )


def solve_oracle(
    state: OracleState,
    rate_table: EmpiricalRateTable | None = None,
) -> OracleResult:
    """Solve one state exactly within the calibrated empirical model."""

    table = rate_table or EmpiricalRateTable.from_csv()
    structures = table.structures_for(
        state.wind_direction,
        state.wind_level,
        expected_drone_count=len(state.drone_ids),
    )
    evaluations = [
        evaluation
        for rates in structures
        if (evaluation := _evaluate_structure(state, rates)) is not None
    ]
    if not evaluations:
        raise RuntimeError(
            "No safe structure can satisfy the minimum arrival SOC constraint"
        )
    safe_evaluations = [
        evaluation
        for evaluation in evaluations
        if evaluation.safety_tier == SafetyTier.SAFE
    ]
    selectable = safe_evaluations or evaluations
    selectable.sort(
        key=lambda item: (
            round(item.total_completion_minutes, 12),
            item.structure,
            item.slot_by_drone,
        )
    )
    evaluations.sort(
        key=lambda item: (
            item.safety_tier,
            round(item.total_completion_minutes, 12),
            item.structure,
            item.slot_by_drone,
        )
    )
    return OracleResult(
        state=state,
        selected=selectable[0],
        ranked_structures=tuple(evaluations),
    )


def _format_list(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-direction", required=True, choices=("head", "side", "tail"))
    parser.add_argument("--wind-level", required=True, type=int, choices=(1, 2))
    parser.add_argument("--k", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--remaining-distance-m", required=True, type=float)
    parser.add_argument(
        "--soc",
        required=True,
        nargs=5,
        type=float,
        metavar=("D1", "D2", "D3", "D4", "D5"),
    )
    parser.add_argument("--minimum-arrival-soc", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    state = OracleState(
        wind_direction=args.wind_direction,
        wind_level=args.wind_level,
        charging_pad_count=args.k,
        current_soc=tuple(args.soc),
        remaining_distance_m=args.remaining_distance_m,
        minimum_arrival_soc=args.minimum_arrival_soc,
    )
    result = solve_oracle(state)
    print(
        "structure,safety_tier,runs,total_min,charge_min,arrival_soc,best_slot_by_drone"
    )
    for evaluation in result.ranked_structures:
        print(
            f"{evaluation.structure.label},"
            f"{evaluation.safety_tier.name.lower()},"
            f"{evaluation.selected_run_count},"
            f"{evaluation.total_completion_minutes:.4f},"
            f"{evaluation.charging_schedule.makespan_minutes:.4f},"
            f'"{_format_list(evaluation.projected_arrival_soc)}",'
            f'"{list(evaluation.slot_by_drone)}"'
        )
    print(f"\nselected={result.selected.structure.label}")
    print(f"position={result.selected.position_mapping(state.drone_ids)}")
    print(
        "charging_pad_drone_ids="
        + str(
            tuple(
                tuple(state.drone_ids[index] for index in group)
                for group in result.selected.charging_schedule.drone_indices_by_pad
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
