"""Evaluate the hierarchical ML policy that outputs formation, spacing, and position."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ml_policy.oracle_optimizer import EmpiricalRateTable, OracleState, _evaluate_fixed_position
from ml_policy.pairwise_position_policy import FEATURES as POSITION_FEATURES
from ml_policy.pairwise_position_policy import expand_candidate_rows
from ml_policy.train_gradient_boosted_policy import FEATURES as STRUCTURE_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_STRUCTURE_MODEL = DEFAULT_DIR / "gradient_boosted_policy.joblib"
DEFAULT_POSITION_MODEL = DEFAULT_DIR / "pairwise_position_policy.joblib"
DEFAULT_VALIDATION = DEFAULT_DIR / "position_aware_validation_candidates.csv"


def _structure_features(row: pd.Series) -> pd.DataFrame:
    soc = sorted(float(row[f"soc_d{index}"]) for index in range(1, 6))
    values = {
        "wind_direction": row["wind_direction"],
        "wind_level": int(row["wind_level"]),
        "charging_pad_count": int(row["charging_pad_count"]),
        "remaining_distance_m": float(row["remaining_distance_m"]),
        "soc_lowest": soc[0],
        "soc_second_lowest": soc[1],
        "soc_middle": soc[2],
        "soc_second_highest": soc[3],
        "soc_highest": soc[4],
        "soc_range": soc[4] - soc[0],
    }
    return pd.DataFrame([values], columns=STRUCTURE_FEATURES)


def _decode_pair_scores(
    probability: np.ndarray,
    projected_arrival_soc: np.ndarray | None = None,
    minimum_arrival_soc: float = 30.0,
) -> tuple[int, ...]:
    scores = np.asarray(probability, dtype=float).reshape(5, 5)
    if projected_arrival_soc is not None:
        arrival = np.asarray(projected_arrival_soc, dtype=float).reshape(5, 5)
        scores = scores.copy()
        scores[arrival < minimum_arrival_soc - 1e-12] = -1e12
    drones, slots = linear_sum_assignment(-scores)
    assignment = np.empty(5, dtype=int)
    assignment[drones] = slots
    return tuple(int(value) for value in assignment)


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "maximum": float(finite.max()),
    }


def evaluate_full_configuration_ml(
    structure_model_path: Path,
    position_model_path: Path,
    validation_csv: Path,
    output_dir: Path,
) -> dict[str, object]:
    structure_model = joblib.load(structure_model_path)
    position_model = joblib.load(position_model_path)
    frame = pd.read_csv(validation_csv)
    frame = frame[frame["eligible_for_selection"].eq(1)].copy().reset_index(drop=True)
    manifest = json.loads(
        validation_csv.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        Path(manifest["source_state_manifest"]).read_text(encoding="utf-8")
    )
    rate_table = EmpiricalRateTable.from_csv(Path(manifest["source_rate_table"]))

    pair_frame = expand_candidate_rows(frame, include_target=True)
    pair_probability = position_model.predict_proba(pair_frame[POSITION_FEATURES])[:, 1]
    pair_frame["probability"] = pair_probability
    decoded_by_candidate: dict[int, tuple[int, ...]] = {}
    for candidate_index, group in pair_frame.groupby("candidate_index", sort=False):
        decoded_by_candidate[int(candidate_index)] = _decode_pair_scores(
            group["probability"].to_numpy(),
            group["projected_arrival_soc"].to_numpy(),
        )

    candidate_position_exact: list[bool] = []
    candidate_position_optimal: list[bool] = []
    candidate_position_regret: list[float] = []
    fixed_evaluation_by_candidate: dict[int, object] = {}
    rate_cells: dict[tuple[str, int, str], object] = {}
    for row_index, row in frame.iterrows():
        state = OracleState(
            wind_direction=row["wind_direction"],
            wind_level=int(row["wind_level"]),
            charging_pad_count=int(row["charging_pad_count"]),
            current_soc=tuple(float(row[f"soc_d{index}"]) for index in range(1, 6)),
            remaining_distance_m=float(row["remaining_distance_m"]),
            forward_speed_m_per_s=float(source_manifest["forward_speed_m_per_s"]),
            fully_charged_soc=float(source_manifest["fully_charged_soc"]),
            zero_to_fully_charged_minutes=float(
                source_manifest["zero_to_fully_charged_minutes"]
            ),
            minimum_arrival_soc=float(source_manifest["minimum_arrival_soc"]),
        )
        rate_key = (state.wind_direction, state.wind_level, row["structure"])
        if rate_key not in rate_cells:
            rate_cells[rate_key] = next(
                rates
                for rates in rate_table.structures_for(
                    state.wind_direction, state.wind_level, expected_drone_count=5
                )
                if rates.structure.label == row["structure"]
            )
        rates = rate_cells[rate_key]
        ordered_slots = tuple(slot.slot_id for slot in rates.slots)
        decoded = decoded_by_candidate[row_index]
        fixed = _evaluate_fixed_position(
            state,
            rates,
            tuple(ordered_slots[index] for index in decoded),
        )
        fixed_evaluation_by_candidate[row_index] = fixed
        truth = tuple(
            int(row[f"assigned_slot_index_d{index}"]) - 1 for index in range(1, 6)
        )
        actual_time = fixed.total_completion_minutes if fixed is not None else float("inf")
        regret = actual_time - float(row["total_completion_minutes"])
        candidate_position_exact.append(decoded == truth)
        candidate_position_optimal.append(np.isclose(regret, 0.0, atol=1e-9))
        candidate_position_regret.append(regret)

    frame["position_exact_match"] = candidate_position_exact
    frame["position_objective_optimal"] = candidate_position_optimal
    frame["position_regret_minutes"] = candidate_position_regret

    classes = structure_model.named_steps["classifier"].classes_
    records: list[dict[str, object]] = []
    benchmark_groups: list[pd.DataFrame] = []
    for scenario_id, group in frame.groupby("scenario_id", sort=True):
        benchmark_groups.append(group)
        first = group.iloc[0]
        probabilities = structure_model.predict_proba(_structure_features(first))[0]
        probability_by_label = dict(zip(classes, probabilities))
        selected_index = min(
            group.index,
            key=lambda index: (
                -float(probability_by_label.get(frame.loc[index, "structure"], 0.0)),
                frame.loc[index, "structure"],
            ),
        )
        selected = frame.loc[selected_index]
        fixed = fixed_evaluation_by_candidate[selected_index]
        actual_time = fixed.total_completion_minutes if fixed is not None else float("inf")
        global_time = float(selected["global_oracle_total_minutes"])
        regret = actual_time - global_time
        predicted_position = (
            fixed.position_mapping(("D1", "D2", "D3", "D4", "D5"))
            if fixed is not None
            else {}
        )
        oracle_position = json.loads(selected["global_oracle_position_json"])
        records.append(
            {
                "scenario_id": int(scenario_id),
                "predicted_structure": selected["structure"],
                "oracle_structure": selected["global_oracle_structure"],
                "structure_match": selected["structure"] == selected["global_oracle_structure"],
                "predicted_position_json": json.dumps(predicted_position, sort_keys=True),
                "oracle_position_json": json.dumps(oracle_position, sort_keys=True),
                "exact_position_match": predicted_position == oracle_position,
                "actual_completion_minutes": actual_time,
                "oracle_completion_minutes": global_time,
                "regret_minutes": regret,
                "objective_global_optimal": np.isclose(regret, 0.0, atol=1e-9),
                "feasible": fixed is not None,
            }
        )
    states = pd.DataFrame(records)

    online_started = time.perf_counter()
    for group in benchmark_groups[:100]:
        first = group.iloc[0]
        probabilities = structure_model.predict_proba(_structure_features(first))[0]
        probability_by_label = dict(zip(classes, probabilities))
        selected = group.loc[min(
            group.index,
            key=lambda index: (
                -float(probability_by_label.get(frame.loc[index, "structure"], 0.0)),
                frame.loc[index, "structure"],
            ),
        )]
        one_pair_frame = expand_candidate_rows(
            pd.DataFrame([selected]).reset_index(drop=True), include_target=False
        )
        one_probability = position_model.predict_proba(
            one_pair_frame[POSITION_FEATURES]
        )[:, 1]
        _decode_pair_scores(
            one_probability,
            one_pair_frame["projected_arrival_soc"].to_numpy(),
        )
    online_ms = (time.perf_counter() - online_started) * 10.0

    position_regret = np.asarray(candidate_position_regret, dtype=float)
    full_regret = states["regret_minutes"].to_numpy(dtype=float)
    report: dict[str, object] = {
        "status": "pass",
        "method": "GradientBoosting structure head plus learned pairwise position head and Hungarian decoder",
        "independent_scenarios": len(states),
        "eligible_candidate_rows": len(frame),
        "candidate_exact_position_rate": float(np.mean(candidate_position_exact)),
        "candidate_objective_optimal_position_rate": float(np.mean(candidate_position_optimal)),
        "candidate_position_regret_minutes": _summary(position_regret),
        "global_structure_match_rate": float(states["structure_match"].mean()),
        "global_exact_position_match_rate": float(states["exact_position_match"].mean()),
        "full_configuration_global_optimal_rate": float(
            states["objective_global_optimal"].mean()
        ),
        "full_configuration_within_1_second_rate": float(
            np.mean(full_regret <= (1.0 / 60.0) + 1e-12)
        ),
        "full_configuration_within_0p1_minute_rate": float(
            np.mean(full_regret <= 0.1 + 1e-12)
        ),
        "full_configuration_within_0p5_minute_rate": float(
            np.mean(full_regret <= 0.5 + 1e-12)
        ),
        "full_configuration_feasible_rate": float(states["feasible"].mean()),
        "full_configuration_regret_minutes": _summary(full_regret),
        "warm_online_decision_ms_per_state": online_ms,
        "online_benchmark_states": 100,
        "structure_model": str(structure_model_path.resolve()),
        "position_model": str(position_model_path.resolve()),
        "validation_csv": str(validation_csv.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "pairwise_candidate_position_predictions.csv", index=False)
    states.to_csv(output_dir / "full_configuration_ml_predictions.csv", index=False)
    (output_dir / "full_configuration_ml_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-model", type=Path, default=DEFAULT_STRUCTURE_MODEL)
    parser.add_argument("--position-model", type=Path, default=DEFAULT_POSITION_MODEL)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_full_configuration_ml(
        args.structure_model,
        args.position_model,
        args.validation_csv,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
