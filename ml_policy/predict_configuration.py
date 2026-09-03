"""Run the trained policy for one observed online state."""

from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import permutations
import json
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd

from ml_policy.oracle_optimizer import (
    DEFAULT_RATE_TABLE_PATH,
    EmpiricalRateTable,
    OracleState,
    SafetyTier,
    _evaluate_structure,
)
from ml_policy.train_gradient_boosted_policy import FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
    / "gradient_boosted_policy.joblib"
)


@lru_cache(maxsize=4)
def _load_model(path: str):
    return joblib.load(path)


@lru_cache(maxsize=4)
def _load_rate_table(path: str) -> EmpiricalRateTable:
    return EmpiricalRateTable.from_csv(path)


def _features(state: OracleState) -> pd.DataFrame:
    ordered_soc = sorted(state.current_soc)
    row = {
        "wind_direction": state.wind_direction,
        "wind_level": state.wind_level,
        "charging_pad_count": state.charging_pad_count,
        "remaining_distance_m": state.remaining_distance_m,
        "soc_lowest": ordered_soc[0],
        "soc_second_lowest": ordered_soc[1],
        "soc_middle": ordered_soc[2],
        "soc_second_highest": ordered_soc[3],
        "soc_highest": ordered_soc[4],
        "soc_range": ordered_soc[4] - ordered_soc[0],
    }
    return pd.DataFrame([row], columns=FEATURES)


def _has_feasible_position(state: OracleState, rates) -> bool:
    remaining_minutes = state.remaining_forward_minutes
    slot_rates = [slot.rate_pp_per_min for slot in rates.slots]
    for assigned_rates in permutations(slot_rates):
        if all(
            soc - rate * remaining_minutes >= state.minimum_arrival_soc - 1e-12
            for soc, rate in zip(state.current_soc, assigned_rates)
        ):
            return True
    return False


def predict_configuration(
    state: OracleState,
    *,
    model_path: Path = DEFAULT_MODEL,
    rate_table_path: Path = DEFAULT_RATE_TABLE_PATH,
    top_k: int = 2,
) -> dict[str, object]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    model = _load_model(str(model_path.resolve()))
    probabilities = model.predict_proba(_features(state))[0]
    classes = model.named_steps["classifier"].classes_
    probability_by_label = {
        label: float(probability)
        for label, probability in zip(classes, probabilities)
    }

    table = _load_rate_table(str(rate_table_path.resolve()))
    condition_rates = table.structures_for(state.wind_direction, state.wind_level)
    rates_by_label = {rates.structure.label: rates for rates in condition_rates}
    safe_feasible_exists = any(
        rates.safety_tier == SafetyTier.SAFE and _has_feasible_position(state, rates)
        for rates in condition_rates
    )
    allowed_tier = SafetyTier.SAFE if safe_feasible_exists else SafetyTier.BACKUP_ONLY

    ranked_labels = sorted(
        probability_by_label,
        key=lambda label: (-probability_by_label[label], label),
    )
    candidates = []
    for label in ranked_labels:
        rates = rates_by_label.get(label)
        if rates is None or rates.safety_tier != allowed_tier:
            continue
        evaluation = _evaluate_structure(state, rates)
        if evaluation is None:
            continue
        candidates.append((label, probability_by_label[label], evaluation))
        if len(candidates) >= top_k:
            break
    if not candidates:
        raise RuntimeError("The learned policy produced no feasible candidate")

    selected_label, selected_probability, selected = min(
        candidates,
        key=lambda item: (item[2].total_completion_minutes, item[0]),
    )
    candidate_records = [
        {
            "structure": label,
            "model_probability": probability,
            "position": evaluation.position_mapping(state.drone_ids),
            "predicted_arrival_soc": list(evaluation.projected_arrival_soc),
            "charging_minutes": evaluation.charging_schedule.makespan_minutes,
            "total_completion_minutes": evaluation.total_completion_minutes,
            "safety_tier": evaluation.safety_tier.name.lower(),
        }
        for label, probability, evaluation in candidates
    ]
    return {
        "input": {
            "wind_direction": state.wind_direction,
            "wind_level": state.wind_level,
            "charging_pad_count": state.charging_pad_count,
            "remaining_distance_m": state.remaining_distance_m,
            "current_soc": list(state.current_soc),
            "minimum_arrival_soc": state.minimum_arrival_soc,
            "charging_model": "90-minute exponential to fully charged (99%)",
        },
        "candidate_count": len(candidate_records),
        "candidates": candidate_records,
        "selected_configuration": {
            "structure": selected_label,
            "formation": selected.structure.formation,
            "inter_drone_spacing_cm": selected.structure.distance_cm,
            "position": selected.position_mapping(state.drone_ids),
            "model_probability": selected_probability,
            "predicted_arrival_soc": list(selected.projected_arrival_soc),
            "charging_minutes": selected.charging_schedule.makespan_minutes,
            "total_completion_minutes": selected.total_completion_minutes,
            "safety_tier": selected.safety_tier.name.lower(),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-direction", required=True, choices=("head", "side", "tail"))
    parser.add_argument("--wind-level", required=True, type=int, choices=(1, 2))
    parser.add_argument("--k", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--remaining-distance-m", required=True, type=float)
    parser.add_argument("--soc", required=True, nargs=5, type=float, metavar=("D1", "D2", "D3", "D4", "D5"))
    parser.add_argument("--minimum-arrival-soc", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
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
    result = predict_configuration(state, model_path=args.model, top_k=args.top_k)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
