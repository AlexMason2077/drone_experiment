"""Online complete-configuration policy with adaptive neural shortlisting."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import time
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/drone-matplotlib-cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from ml_policy.oracle_optimizer import (
    DEFAULT_RATE_TABLE_PATH,
    EmpiricalRateTable,
    OracleState,
    SafetyTier,
    _evaluate_fixed_position,
)
from ml_policy.train_controlled_k_candidate_cost_ranker import (
    DEFAULT_CLASSES,
    DEFAULT_DIR,
    _candidate_metadata,
    _features_for_state,
    _rate_groups,
    _scale,
)


DEFAULT_EXPERIMENT_DIR = DEFAULT_DIR / "regret_aware_pairwise_experiment"
DEFAULT_MODEL = DEFAULT_EXPERIMENT_DIR / "pairwise_finetuned_candidate_ranker.keras"
DEFAULT_SCALER = DEFAULT_EXPERIMENT_DIR / "feature_scaler.json"
# Fixed after five-fold grouped cross-validation.  These are verification
# budgets, not hard-coded configuration choices: the learned ranker still
# determines which complete configurations enter each shortlist.
DEFAULT_TOP_N_BY_K = {1: 1, 2: 3, 3: 36, 4: 25, 5: 1}


@lru_cache(maxsize=4)
def _load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path, compile=False)


@lru_cache(maxsize=4)
def _load_scaler(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load_classes(path: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    table = pd.read_csv(path).sort_values("class_index").reset_index(drop=True)
    return table, _candidate_metadata(table)


@lru_cache(maxsize=4)
def _load_rate_resources(
    path: str,
) -> tuple[EmpiricalRateTable, dict[tuple[str, int], dict[str, np.ndarray]]]:
    source = Path(path)
    return EmpiricalRateTable.from_csv(source), _rate_groups(source)


def adaptive_top_n(charging_pad_count: int) -> int:
    try:
        return DEFAULT_TOP_N_BY_K[int(charging_pad_count)]
    except KeyError as error:
        raise ValueError("charging_pad_count must be between 1 and 5") from error


def _state_row(state: OracleState) -> pd.Series:
    values: dict[str, object] = {
        "wind_direction": state.wind_direction,
        "wind_level": state.wind_level,
        "charging_pad_count": state.charging_pad_count,
        "remaining_distance_m": state.remaining_distance_m,
    }
    values.update(
        {f"soc_d{index}": value for index, value in enumerate(state.current_soc, 1)}
    )
    return pd.Series(values)


def _eligible_classes(
    state: OracleState,
    metadata: dict[str, np.ndarray],
    rate_table: EmpiricalRateTable,
) -> tuple[np.ndarray, dict[str, object]]:
    cells = rate_table.structures_for(
        state.wind_direction,
        state.wind_level,
        expected_drone_count=len(state.drone_ids),
    )
    cells_by_label = {cell.structure.label: cell for cell in cells}
    safe: list[int] = []
    backup: list[int] = []
    flight_minutes = state.remaining_forward_minutes

    for class_index, structure in enumerate(metadata["structure"]):
        cell = cells_by_label.get(str(structure))
        if cell is None:
            continue
        slot_rates = np.asarray(
            [slot.rate_pp_per_min for slot in cell.slots], dtype=np.float64
        )
        assigned_rates = slot_rates[metadata["permutation"][class_index]]
        arrival = np.asarray(state.current_soc) - assigned_rates * flight_minutes
        if float(arrival.min()) < state.minimum_arrival_soc - 1e-12:
            continue
        target = safe if cell.safety_tier == SafetyTier.SAFE else backup
        target.append(class_index)

    eligible = safe or backup
    if not eligible:
        raise RuntimeError("No feasible complete configuration for the observed state")
    return np.asarray(eligible, dtype=np.int32), cells_by_label


def predict_adaptive_configuration(
    state: OracleState,
    *,
    model_path: Path = DEFAULT_MODEL,
    scaler_path: Path = DEFAULT_SCALER,
    class_table_path: Path = DEFAULT_CLASSES,
    rate_table_path: Path = DEFAULT_RATE_TABLE_PATH,
    top_n: int | None = None,
) -> dict[str, object]:
    """Return the exact best configuration inside a learned adaptive shortlist."""

    started = time.perf_counter()
    model = _load_model(str(model_path.resolve()))
    scaler = _load_scaler(str(scaler_path.resolve()))
    _, metadata = _load_classes(str(class_table_path.resolve()))
    rate_table, rates = _load_rate_resources(str(rate_table_path.resolve()))
    eligible, cells_by_label = _eligible_classes(state, metadata, rate_table)
    shortlist_size = adaptive_top_n(state.charging_pad_count) if top_n is None else top_n
    if shortlist_size <= 0:
        raise ValueError("top_n must be positive")

    ranking_started = time.perf_counter()
    row = _state_row(state)
    features, lower_bound = _features_for_state(
        row,
        eligible,
        metadata,
        rates[(state.wind_direction, state.wind_level)],
    )
    residual_log = model(_scale(features, scaler), training=False).numpy().reshape(-1)
    predicted_total = lower_bound + np.expm1(np.maximum(residual_log, 0.0))
    order = np.argsort(predicted_total, kind="stable")
    shortlist_positions = order[: min(shortlist_size, len(order))]
    ranking_ms = (time.perf_counter() - ranking_started) * 1000.0

    exact_started = time.perf_counter()
    candidate_records: list[dict[str, object]] = []
    exact_candidates = []
    for neural_rank, position in enumerate(shortlist_positions, start=1):
        class_index = int(eligible[int(position)])
        structure = str(metadata["structure"][class_index])
        cell = cells_by_label[structure]
        slot_ids = tuple(slot.slot_id for slot in cell.slots)
        slot_by_drone = tuple(
            slot_ids[int(slot_index)]
            for slot_index in metadata["permutation"][class_index]
        )
        evaluation = _evaluate_fixed_position(state, cell, slot_by_drone)
        if evaluation is None:
            continue
        exact_candidates.append((evaluation.total_completion_minutes, class_index, evaluation))
        candidate_records.append(
            {
                "neural_rank": neural_rank,
                "class_index": class_index,
                "structure": structure,
                "position": evaluation.position_mapping(state.drone_ids),
                "model_predicted_total_minutes": float(predicted_total[int(position)]),
                "exact_charging_minutes": evaluation.charging_schedule.makespan_minutes,
                "exact_total_completion_minutes": evaluation.total_completion_minutes,
            }
        )
    if not exact_candidates:
        raise RuntimeError("Adaptive shortlist contained no feasible configuration")
    exact_candidates.sort(key=lambda item: (round(item[0], 12), item[1]))
    _, selected_class, selected = exact_candidates[0]
    exact_ms = (time.perf_counter() - exact_started) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0

    structure = str(metadata["structure"][selected_class])
    formation, spacing = structure.rsplit("_", 1)
    selected_record = next(
        record for record in candidate_records if record["class_index"] == selected_class
    )
    return {
        "input": {
            "wind_direction": state.wind_direction,
            "wind_level": state.wind_level,
            "charging_pad_count": state.charging_pad_count,
            "remaining_distance_m": state.remaining_distance_m,
            "current_soc": list(state.current_soc),
            "minimum_arrival_soc": state.minimum_arrival_soc,
        },
        "policy": {
            "method": "candidate-aware neural ranking plus adaptive exact reranking",
            "eligible_configuration_count": len(eligible),
            "adaptive_top_n": shortlist_size,
            "exactly_evaluated_configuration_count": len(candidate_records),
        },
        "timing_ms": {
            "neural_ranking": ranking_ms,
            "exact_shortlist_reranking": exact_ms,
            "total_including_cached_resource_access": total_ms,
        },
        "candidates": candidate_records,
        "selected_configuration": {
            "class_index": selected_class,
            "formation": "echelon" if formation == "echalon" else formation,
            "inter_drone_spacing_cm": int(spacing),
            "position": selected.position_mapping(state.drone_ids),
            "predicted_arrival_soc": list(selected.projected_arrival_soc),
            "charging_minutes": selected.charging_schedule.makespan_minutes,
            "total_completion_minutes": selected.total_completion_minutes,
            "neural_rank_before_exact_reranking": selected_record["neural_rank"],
            "safety_tier": selected.safety_tier.name.lower(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-direction", required=True, choices=("head", "side", "tail"))
    parser.add_argument("--wind-level", required=True, type=int, choices=(1, 2))
    parser.add_argument("--k", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--remaining-distance-m", required=True, type=float)
    parser.add_argument("--soc", required=True, nargs=5, type=float)
    parser.add_argument("--minimum-arrival-soc", type=float, default=30.0)
    parser.add_argument("--top-n", type=int)
    args = parser.parse_args(argv)
    state = OracleState(
        wind_direction=args.wind_direction,
        wind_level=args.wind_level,
        charging_pad_count=args.k,
        current_soc=tuple(args.soc),
        remaining_distance_m=args.remaining_distance_m,
        minimum_arrival_soc=args.minimum_arrival_soc,
    )
    result = predict_adaptive_configuration(state, top_n=args.top_n)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
