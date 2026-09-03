"""Evaluate the learned full configuration, including decoded position regret."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/drone-matplotlib-cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from ml_policy.oracle_optimizer import (
    EmpiricalRateTable,
    OracleState,
    _evaluate_fixed_position,
)
from ml_policy.structured_configuration_policy import (
    assignment_targets,
    decode_assignment,
    inverse_time,
    load_preprocessor,
    transform_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_MODEL = DEFAULT_DIR / "structured_configuration_policy.keras"
DEFAULT_PREPROCESSOR = DEFAULT_DIR / "structured_configuration_preprocessor.json"
DEFAULT_VALIDATION = DEFAULT_DIR / "position_aware_validation_candidates.csv"


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": float("inf"), "median": float("inf"),
                "p95": float("inf"), "maximum": float("inf")}
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "maximum": float(finite.max()),
    }


def evaluate_structured_policy(
    model_path: Path,
    preprocessor_path: Path,
    validation_csv: Path,
    output_dir: Path,
) -> dict[str, object]:
    frame = pd.read_csv(validation_csv)
    frame = frame[frame["eligible_for_selection"].eq(1)].copy().reset_index(drop=True)
    manifest = json.loads(
        validation_csv.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        Path(manifest["source_state_manifest"]).read_text(encoding="utf-8")
    )
    rate_table = EmpiricalRateTable.from_csv(Path(manifest["source_rate_table"]))
    preprocessor = load_preprocessor(preprocessor_path)
    model = tf.keras.models.load_model(model_path)

    features = transform_features(frame, preprocessor)
    inference_started = time.perf_counter()
    prediction = model.predict(features, batch_size=512, verbose=0)
    batch_inference_ms = (time.perf_counter() - inference_started) * 1000.0
    predicted_times = inverse_time(
        prediction["time_z"].reshape(-1), preprocessor
    )
    assignment_logits = prediction["assignment_logits"]
    true_assignments = assignment_targets(frame)
    decoded_assignments = np.asarray(
        [decode_assignment(logits) for logits in assignment_logits],
        dtype=np.int32,
    )
    frame["predicted_completion_minutes"] = predicted_times
    frame["decoded_position_indices"] = [
        "|".join(str(value + 1) for value in assignment)
        for assignment in decoded_assignments
    ]

    candidate_exact_position: list[bool] = []
    candidate_actual_time: list[float] = []
    candidate_position_regret: list[float] = []
    candidate_feasible: list[bool] = []
    evaluation_by_row: dict[int, object] = {}

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
        rates = next(
            item
            for item in rate_table.structures_for(
                state.wind_direction,
                state.wind_level,
                expected_drone_count=5,
            )
            if item.structure.label == row["structure"]
        )
        ordered_slots = tuple(slot.slot_id for slot in rates.slots)
        assignment = decoded_assignments[row_index]
        slot_by_drone = tuple(ordered_slots[index] for index in assignment)
        fixed = _evaluate_fixed_position(state, rates, slot_by_drone)
        evaluation_by_row[row_index] = fixed
        feasible = fixed is not None
        actual_time = (
            fixed.total_completion_minutes if fixed is not None else float("inf")
        )
        best_time = float(row["total_completion_minutes"])
        candidate_exact_position.append(
            bool(np.array_equal(assignment, true_assignments[row_index]))
        )
        candidate_feasible.append(feasible)
        candidate_actual_time.append(actual_time)
        candidate_position_regret.append(actual_time - best_time)

    frame["decoded_position_feasible"] = candidate_feasible
    frame["decoded_actual_completion_minutes"] = candidate_actual_time
    frame["decoded_position_regret_minutes"] = candidate_position_regret
    frame["decoded_exact_position_match"] = candidate_exact_position
    frame["decoded_objective_optimal_for_structure"] = np.isclose(
        frame["decoded_position_regret_minutes"].to_numpy(dtype=float),
        0.0,
        atol=1e-9,
    )

    state_records: list[dict[str, object]] = []
    for scenario_id, group in frame.groupby("scenario_id", sort=True):
        feasible_group = group[group["decoded_position_feasible"]]
        if feasible_group.empty:
            selected_index = int(group["predicted_completion_minutes"].idxmin())
        else:
            selected_index = int(
                feasible_group["predicted_completion_minutes"].idxmin()
            )
        selected_row = frame.loc[selected_index]
        selected_evaluation = evaluation_by_row[selected_index]
        actual_time = (
            selected_evaluation.total_completion_minutes
            if selected_evaluation is not None
            else float("inf")
        )
        global_time = float(selected_row["global_oracle_total_minutes"])
        regret = actual_time - global_time
        oracle_position = json.loads(selected_row["global_oracle_position_json"])
        decoded_position = (
            selected_evaluation.position_mapping(("D1", "D2", "D3", "D4", "D5"))
            if selected_evaluation is not None
            else {}
        )
        state_records.append(
            {
                "scenario_id": int(scenario_id),
                "predicted_structure": selected_row["structure"],
                "oracle_structure": selected_row["global_oracle_structure"],
                "structure_match": (
                    selected_row["structure"] == selected_row["global_oracle_structure"]
                ),
                "predicted_position_json": json.dumps(decoded_position, sort_keys=True),
                "oracle_position_json": json.dumps(oracle_position, sort_keys=True),
                "exact_position_match": decoded_position == oracle_position,
                "predicted_actual_completion_minutes": actual_time,
                "oracle_total_minutes": global_time,
                "full_configuration_regret_minutes": regret,
                "objective_global_optimal": np.isclose(regret, 0.0, atol=1e-9),
                "feasible": selected_evaluation is not None,
            }
        )
    states = pd.DataFrame(state_records)

    benchmark_groups = [group for _, group in frame.groupby("scenario_id", sort=True)][:100]
    model.predict(
        transform_features(benchmark_groups[0], preprocessor), verbose=0
    )
    online_started = time.perf_counter()
    for group in benchmark_groups:
        group_prediction = model.predict(
            transform_features(group, preprocessor), verbose=0
        )
        group_times = inverse_time(
            group_prediction["time_z"].reshape(-1), preprocessor
        )
        group_assignments = [
            decode_assignment(logits)
            for logits in group_prediction["assignment_logits"]
        ]
        _ = group_assignments[int(np.argmin(group_times))]
    online_ms_per_state = (
        (time.perf_counter() - online_started) * 1000.0 / len(benchmark_groups)
    )

    time_error = np.abs(
        predicted_times - frame["total_completion_minutes"].to_numpy(dtype=float)
    )
    candidate_regret = frame["decoded_position_regret_minutes"].to_numpy(dtype=float)
    state_regret = states["full_configuration_regret_minutes"].to_numpy(dtype=float)
    report: dict[str, object] = {
        "status": "pass",
        "model": str(model_path.resolve()),
        "validation_csv": str(validation_csv.resolve()),
        "independent_scenario_count": len(states),
        "eligible_candidate_rows": len(frame),
        "candidate_time_mae_minutes": float(time_error.mean()),
        "candidate_time_p95_absolute_error_minutes": float(
            np.quantile(time_error, 0.95)
        ),
        "candidate_exact_position_rate": float(np.mean(candidate_exact_position)),
        "candidate_objective_optimal_position_rate": float(
            frame["decoded_objective_optimal_for_structure"].mean()
        ),
        "candidate_feasible_position_rate": float(np.mean(candidate_feasible)),
        "candidate_position_regret_minutes": _quantiles(candidate_regret),
        "global_structure_match_rate": float(states["structure_match"].mean()),
        "global_exact_position_match_rate": float(states["exact_position_match"].mean()),
        "full_configuration_global_optimal_rate": float(
            states["objective_global_optimal"].mean()
        ),
        "full_configuration_feasible_rate": float(states["feasible"].mean()),
        "full_configuration_regret_minutes": _quantiles(state_regret),
        "batch_candidate_inference_ms": batch_inference_ms,
        "warm_online_neural_decision_ms_per_state": online_ms_per_state,
        "online_benchmark_states": len(benchmark_groups),
        "tie_policy": (
            "A different position is counted as globally optimal when its exact "
            "completion time equals the Oracle minimum within 1e-9 minutes."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "structured_candidate_validation_predictions.csv", index=False)
    states.to_csv(output_dir / "structured_full_configuration_predictions.csv", index=False)
    (output_dir / "structured_configuration_independent_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--preprocessor", type=Path, default=DEFAULT_PREPROCESSOR)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_structured_policy(
        args.model,
        args.preprocessor,
        args.validation_csv,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
