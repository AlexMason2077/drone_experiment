"""Train one neural classifier over the complete configuration C=(f,p,d).

Unlike the hierarchical prototype, this model has no formation-first or
position-second decision rule.  Each output neuron represents one complete
formation/spacing plus five-drone position permutation.  Safety and minimum
arrival-SOC masks are applied before the highest-scoring class is selected.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from itertools import permutations
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from ml_policy.oracle_optimizer import (
    EmpiricalRateTable,
    OracleState,
    _evaluate_fixed_position,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_TRAINING_STATES = DEFAULT_DIR / "oracle_training_states_0p25_25m.csv"
DEFAULT_VALIDATION_STATES = DEFAULT_DIR / "oracle_validation_states_1000.csv"
DEFAULT_TRAINING_CANDIDATES = DEFAULT_DIR / "position_aware_training_candidates.csv"
DEFAULT_VALIDATION_CANDIDATES = DEFAULT_DIR / "position_aware_validation_candidates.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_DIR / "joint_full_configuration_neural"

NUMERIC_FEATURES = (
    "wind_level",
    "charging_pad_count",
    "remaining_distance_m",
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
)
WIND_DIRECTIONS = ("head", "side", "tail")


def _slot_ids_by_structure(candidate_frames: Sequence[pd.DataFrame]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for frame in candidate_frames:
        for _, row in frame.drop_duplicates("structure").iterrows():
            structure = str(row["structure"])
            slots = tuple(str(row[f"slot_{index}_id"]) for index in range(1, 6))
            previous = result.setdefault(structure, slots)
            if previous != slots:
                raise ValueError(f"Inconsistent slot order for {structure}: {previous} vs {slots}")
    return dict(sorted(result.items()))


def _build_class_universe(
    slots_by_structure: dict[str, tuple[str, ...]],
) -> tuple[list[tuple[str, tuple[int, ...]]], dict[tuple[str, tuple[int, ...]], int]]:
    classes = [
        (structure, tuple(int(value) for value in permutation))
        for structure in sorted(slots_by_structure)
        for permutation in permutations(range(5))
    ]
    return classes, {value: index for index, value in enumerate(classes)}


def _label_for_row(
    row: pd.Series,
    slots_by_structure: dict[str, tuple[str, ...]],
    class_to_index: dict[tuple[str, tuple[int, ...]], int],
) -> int:
    structure = str(row["oracle_structure"])
    slots = slots_by_structure[structure]
    slot_to_index = {slot: index for index, slot in enumerate(slots)}
    mapping = json.loads(row["oracle_position_json"])
    permutation = tuple(slot_to_index[str(mapping[f"D{index}"])] for index in range(1, 6))
    return class_to_index[(structure, permutation)]


def _fit_preprocessor(frame: pd.DataFrame) -> dict[str, object]:
    numeric = frame.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float64)
    mean = numeric.mean(axis=0)
    scale = numeric.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return {
        "numeric_features": list(NUMERIC_FEATURES),
        "numeric_mean": mean.tolist(),
        "numeric_scale": scale.tolist(),
        "wind_directions": list(WIND_DIRECTIONS),
    }


def _transform_features(frame: pd.DataFrame, preprocessor: dict[str, object]) -> np.ndarray:
    numeric = frame.loc[:, list(preprocessor["numeric_features"])].to_numpy(dtype=np.float32)
    numeric = (
        numeric - np.asarray(preprocessor["numeric_mean"], dtype=np.float32)
    ) / np.asarray(preprocessor["numeric_scale"], dtype=np.float32)
    wind_values = frame["wind_direction"].astype(str).to_numpy()
    wind = np.column_stack(
        [wind_values == direction for direction in preprocessor["wind_directions"]]
    ).astype(np.float32)
    return np.concatenate([numeric, wind], axis=1)


def _build_model(input_width: int, class_count: int, random_seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(random_seed)
    inputs = tf.keras.Input(shape=(input_width,), name="observed_state")
    hidden = tf.keras.layers.Dense(512, activation=tf.nn.gelu)(inputs)
    hidden = tf.keras.layers.LayerNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(0.12)(hidden)
    hidden = tf.keras.layers.Dense(512, activation=tf.nn.gelu)(hidden)
    residual = tf.keras.layers.Dense(512)(hidden)
    hidden = tf.keras.layers.Add()([hidden, residual])
    hidden = tf.keras.layers.LayerNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(0.10)(hidden)
    hidden = tf.keras.layers.Dense(384, activation=tf.nn.gelu)(hidden)
    logits = tf.keras.layers.Dense(class_count, name="complete_configuration_logits")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=logits, name="joint_full_configuration_policy")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="exact_class_accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model


def _candidate_rows_by_scenario(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(scenario_id): group.set_index("structure", drop=False)
        for scenario_id, group in frame.groupby("scenario_id", sort=False)
    }


def _feasible_class_mask(
    state_row: pd.Series,
    candidates: pd.DataFrame,
    classes: Sequence[tuple[str, tuple[int, ...]]],
    *,
    minimum_arrival_soc: float,
    forward_speed_m_per_s: float,
) -> np.ndarray:
    mask = np.zeros(len(classes), dtype=bool)
    soc = np.asarray([float(state_row[f"soc_d{index}"]) for index in range(1, 6)])
    flight_minutes = float(state_row["remaining_distance_m"]) / forward_speed_m_per_s / 60.0
    eligible = candidates[candidates["eligible_for_selection"].eq(1)]
    rates_by_structure = {
        str(row["structure"]): np.asarray(
            [float(row[f"slot_{index}_rate_pp_per_min"]) for index in range(1, 6)]
        )
        for _, row in eligible.iterrows()
    }
    for class_index, (structure, permutation) in enumerate(classes):
        rates = rates_by_structure.get(structure)
        if rates is None:
            continue
        arrival = soc - rates[np.asarray(permutation)] * flight_minutes
        mask[class_index] = bool(np.min(arrival) >= minimum_arrival_soc - 1e-12)
    return mask


def train_and_evaluate(
    training_states_path: Path,
    validation_states_path: Path,
    training_candidates_path: Path,
    validation_candidates_path: Path,
    output_dir: Path,
    *,
    random_seed: int,
    epochs: int,
) -> dict[str, object]:
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    training_states = pd.read_csv(training_states_path)
    validation_states = pd.read_csv(validation_states_path)
    training_candidates = pd.read_csv(training_candidates_path)
    validation_candidates = pd.read_csv(validation_candidates_path)
    slots_by_structure = _slot_ids_by_structure((training_candidates, validation_candidates))
    classes, class_to_index = _build_class_universe(slots_by_structure)
    labels = np.asarray(
        [
            _label_for_row(row, slots_by_structure, class_to_index)
            for _, row in training_states.iterrows()
        ],
        dtype=np.int32,
    )

    scenario_indices = np.arange(len(training_states))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(scenario_indices)
    internal_count = max(1, int(round(0.10 * len(scenario_indices))))
    internal_indices = scenario_indices[:internal_count]
    fit_indices = scenario_indices[internal_count:]
    fit_frame = training_states.iloc[fit_indices].copy()
    internal_frame = training_states.iloc[internal_indices].copy()
    preprocessor = _fit_preprocessor(fit_frame)
    x_fit = _transform_features(fit_frame, preprocessor)
    x_internal = _transform_features(internal_frame, preprocessor)
    y_fit = labels[fit_indices]
    y_internal = labels[internal_indices]

    counts = np.bincount(y_fit, minlength=len(classes)).astype(np.float64)
    sample_weights = np.asarray(
        [min(5.0, 1.0 / np.sqrt(max(counts[value], 1.0))) for value in y_fit],
        dtype=np.float32,
    )
    sample_weights /= float(sample_weights.mean())

    model = _build_model(x_fit.shape[1], len(classes), random_seed)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=18, min_delta=1e-4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5
        ),
    ]
    started = time.perf_counter()
    history = model.fit(
        x_fit,
        y_fit,
        sample_weight=sample_weights,
        validation_data=(x_internal, y_internal),
        epochs=epochs,
        batch_size=128,
        verbose=2,
        callbacks=callbacks,
    )
    training_seconds = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "joint_full_configuration_policy.keras"
    model.save(model_path)
    (output_dir / "joint_full_configuration_preprocessor.json").write_text(
        json.dumps(preprocessor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    class_records = [
        {
            "class_index": index,
            "structure": structure,
            **{f"slot_index_d{drone}": permutation[drone - 1] + 1 for drone in range(1, 6)},
        }
        for index, (structure, permutation) in enumerate(classes)
    ]
    pd.DataFrame(class_records).to_csv(output_dir / "joint_configuration_classes.csv", index=False)
    pd.DataFrame(history.history).to_csv(output_dir / "joint_training_history.csv", index=False)

    validation_manifest = json.loads(
        validation_candidates_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        Path(validation_manifest["source_state_manifest"]).read_text(encoding="utf-8")
    )
    rate_table = EmpiricalRateTable.from_csv(Path(validation_manifest["source_rate_table"]))
    candidate_groups = _candidate_rows_by_scenario(validation_candidates)
    x_validation = _transform_features(validation_states, preprocessor)
    logits = model.predict(x_validation, batch_size=256, verbose=0)

    class_structure = [value[0] for value in classes]
    records: list[dict[str, object]] = []
    rate_cache: dict[tuple[str, int, str], object] = {}
    for row_number, (_, row) in enumerate(validation_states.iterrows()):
        scenario_id = int(row["scenario_id"])
        candidates = candidate_groups[scenario_id]
        mask = _feasible_class_mask(
            row,
            candidates,
            classes,
            minimum_arrival_soc=float(source_manifest["minimum_arrival_soc"]),
            forward_speed_m_per_s=float(source_manifest["forward_speed_m_per_s"]),
        )
        if not np.any(mask):
            raise RuntimeError(f"No feasible complete class for scenario {scenario_id}")
        masked_logits = np.where(mask, logits[row_number], -np.inf)
        selected_index = int(np.argmax(masked_logits))
        structure, permutation = classes[selected_index]
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
        slots = slots_by_structure[structure]
        selected_slots = tuple(slots[index] for index in permutation)
        evaluation = _evaluate_fixed_position(state, rates, selected_slots)
        if evaluation is None:
            raise RuntimeError(f"Mask admitted an infeasible class for scenario {scenario_id}")
        oracle_time = float(row["oracle_total_minutes"])
        regret = float(evaluation.total_completion_minutes - oracle_time)
        oracle_label = _label_for_row(row, slots_by_structure, class_to_index)
        records.append(
            {
                "scenario_id": scenario_id,
                "predicted_class": selected_index,
                "oracle_class": oracle_label,
                "predicted_structure": structure,
                "oracle_structure": row["oracle_structure"],
                "exact_full_label": selected_index == oracle_label,
                "structure_match": structure == row["oracle_structure"],
                "objective_global_optimal": bool(np.isclose(regret, 0.0, atol=1e-9)),
                "regret_minutes": regret,
                "predicted_total_minutes": evaluation.total_completion_minutes,
                "oracle_total_minutes": oracle_time,
                "predicted_position_json": json.dumps(
                    evaluation.position_mapping(("D1", "D2", "D3", "D4", "D5")),
                    sort_keys=True,
                ),
                "oracle_position_json": row["oracle_position_json"],
            }
        )

    prediction_frame = pd.DataFrame(records)
    prediction_frame.to_csv(output_dir / "joint_independent_predictions.csv", index=False)
    regret = prediction_frame["regret_minutes"].to_numpy(dtype=float)

    benchmark_count = min(100, len(validation_states))
    _ = model(x_validation[:1], training=False).numpy()
    benchmark_started = time.perf_counter()
    for row_number in range(benchmark_count):
        row = validation_states.iloc[row_number]
        one_logits = model(x_validation[row_number : row_number + 1], training=False).numpy()[0]
        mask = _feasible_class_mask(
            row,
            candidate_groups[int(row["scenario_id"])],
            classes,
            minimum_arrival_soc=float(source_manifest["minimum_arrival_soc"]),
            forward_speed_m_per_s=float(source_manifest["forward_speed_m_per_s"]),
        )
        _ = int(np.argmax(np.where(mask, one_logits, -np.inf)))
    online_ms = (time.perf_counter() - benchmark_started) * 1000.0 / benchmark_count

    training_label_set = set(int(value) for value in labels)
    independent_labels = np.asarray(
        [
            _label_for_row(row, slots_by_structure, class_to_index)
            for _, row in validation_states.iterrows()
        ],
        dtype=np.int32,
    )
    metrics: dict[str, object] = {
        "status": "pass",
        "method": "single joint dense neural classifier over complete C=(formation, spacing, position)",
        "class_count": len(classes),
        "structure_count": len(slots_by_structure),
        "position_per_structure": 120,
        "training_states": len(training_states),
        "fit_states": len(fit_frame),
        "internal_validation_states": len(internal_frame),
        "independent_states": len(validation_states),
        "observed_training_classes": len(training_label_set),
        "independent_rows_with_label_unseen_in_all_training": int(
            np.sum([int(value) not in training_label_set for value in independent_labels])
        ),
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "training_seconds": training_seconds,
        "independent_exact_full_label_rate": float(prediction_frame["exact_full_label"].mean()),
        "independent_structure_match_rate": float(prediction_frame["structure_match"].mean()),
        "independent_strict_global_optimal_rate": float(
            prediction_frame["objective_global_optimal"].mean()
        ),
        "independent_within_0p1_minute_rate": float(np.mean(regret <= 0.1 + 1e-12)),
        "independent_within_0p5_minute_rate": float(np.mean(regret <= 0.5 + 1e-12)),
        "regret_minutes": {
            "mean": float(regret.mean()),
            "median": float(np.median(regret)),
            "p95": float(np.quantile(regret, 0.95)),
            "maximum": float(regret.max()),
        },
        "warm_online_ms_per_state": online_ms,
        "model": str(model_path.resolve()),
        "training_states_csv": str(training_states_path.resolve()),
        "validation_states_csv": str(validation_states_path.resolve()),
        "label_definition": "one class equals one complete (formation, spacing, D1-D5 slot permutation)",
        "factor_priority": "none imposed; all factors are contained in the same output label",
    }
    (output_dir / "joint_full_configuration_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-states", type=Path, default=DEFAULT_TRAINING_STATES)
    parser.add_argument("--validation-states", type=Path, default=DEFAULT_VALIDATION_STATES)
    parser.add_argument("--training-candidates", type=Path, default=DEFAULT_TRAINING_CANDIDATES)
    parser.add_argument("--validation-candidates", type=Path, default=DEFAULT_VALIDATION_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=20260819)
    parser.add_argument("--epochs", type=int, default=160)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metrics = train_and_evaluate(
        args.training_states,
        args.validation_states,
        args.training_candidates,
        args.validation_candidates,
        args.output_dir,
        random_seed=args.random_seed,
        epochs=args.epochs,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
