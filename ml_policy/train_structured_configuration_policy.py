"""Train a neural policy that predicts structure value and optimal position."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/drone-matplotlib-cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from ml_policy.structured_configuration_policy import (
    assignment_targets,
    fit_preprocessor,
    inverse_time,
    save_preprocessor,
    transform_features,
    transform_time,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_TRAINING_CSV = DEFAULT_DIR / "position_aware_training_candidates.csv"


def _build_model(input_width: int, random_seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(random_seed)
    inputs = tf.keras.Input(shape=(input_width,), name="state_structure_features")
    hidden = tf.keras.layers.Dense(192, activation="relu")(inputs)
    hidden = tf.keras.layers.LayerNormalization()(hidden)
    hidden = tf.keras.layers.Dense(192, activation="relu")(hidden)
    hidden = tf.keras.layers.Dropout(0.08)(hidden)
    hidden = tf.keras.layers.Dense(128, activation="relu")(hidden)

    time_hidden = tf.keras.layers.Dense(64, activation="relu")(hidden)
    time_output = tf.keras.layers.Dense(1, name="time_z")(time_hidden)

    assignment_hidden = tf.keras.layers.Dense(128, activation="relu")(hidden)
    assignment_flat = tf.keras.layers.Dense(25)(assignment_hidden)
    assignment_output = tf.keras.layers.Reshape(
        (5, 5), name="assignment_logits"
    )(assignment_flat)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={"time_z": time_output, "assignment_logits": assignment_output},
        name="structured_configuration_policy",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss={
            "time_z": tf.keras.losses.Huber(delta=0.5),
            "assignment_logits": tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
        },
        loss_weights={"time_z": 1.0, "assignment_logits": 1.0},
        metrics={
            "time_z": [tf.keras.metrics.MeanAbsoluteError(name="mae_z")],
            "assignment_logits": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="slot_accuracy")
            ],
        },
    )
    return model


def train_structured_policy(
    training_csv: Path,
    output_dir: Path,
    *,
    random_seed: int,
    epochs: int,
) -> dict[str, object]:
    frame = pd.read_csv(training_csv)
    frame = frame[frame["eligible_for_selection"].eq(1)].copy()
    scenario_ids = np.asarray(sorted(frame["scenario_id"].unique()))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(scenario_ids)
    validation_count = max(1, int(round(0.20 * len(scenario_ids))))
    validation_ids = set(int(value) for value in scenario_ids[:validation_count])
    train_frame = frame[~frame["scenario_id"].isin(validation_ids)].copy()
    validation_frame = frame[frame["scenario_id"].isin(validation_ids)].copy()

    preprocessor = fit_preprocessor(train_frame)
    x_train = transform_features(train_frame, preprocessor)
    x_validation = transform_features(validation_frame, preprocessor)
    y_train_time = transform_time(
        train_frame["total_completion_minutes"].to_numpy(), preprocessor
    ).reshape(-1, 1)
    y_validation_time = transform_time(
        validation_frame["total_completion_minutes"].to_numpy(), preprocessor
    ).reshape(-1, 1)
    y_train_assignment = assignment_targets(train_frame)
    y_validation_assignment = assignment_targets(validation_frame)

    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    model = _build_model(x_train.shape[1], random_seed)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            min_delta=1e-4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        ),
    ]
    started = time.perf_counter()
    history = model.fit(
        x_train,
        {"time_z": y_train_time, "assignment_logits": y_train_assignment},
        validation_data=(
            x_validation,
            {
                "time_z": y_validation_time,
                "assignment_logits": y_validation_assignment,
            },
        ),
        epochs=epochs,
        batch_size=256,
        verbose=2,
        callbacks=callbacks,
    )
    elapsed_seconds = time.perf_counter() - started

    validation_prediction = model.predict(x_validation, batch_size=512, verbose=0)
    predicted_time = inverse_time(
        validation_prediction["time_z"].reshape(-1), preprocessor
    )
    true_time = validation_frame["total_completion_minutes"].to_numpy(dtype=float)
    time_error = np.abs(predicted_time - true_time)
    raw_assignment = np.argmax(
        validation_prediction["assignment_logits"], axis=2
    )
    assignment_exact_without_decoder = np.all(
        raw_assignment == y_validation_assignment, axis=1
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "structured_configuration_policy.keras"
    model.save(model_path)
    preprocessor_path = output_dir / "structured_configuration_preprocessor.json"
    save_preprocessor(preprocessor, preprocessor_path)
    history_frame = pd.DataFrame(history.history)
    history_frame.index.name = "epoch"
    history_frame.to_csv(output_dir / "structured_configuration_training_history.csv")

    metrics: dict[str, object] = {
        "model_type": "multi_task_dense_neural_network_with_hungarian_decoder",
        "training_csv": str(training_csv.resolve()),
        "eligible_candidate_rows": len(frame),
        "training_scenarios": len(scenario_ids) - validation_count,
        "internal_validation_scenarios": validation_count,
        "training_candidate_rows": len(train_frame),
        "internal_validation_candidate_rows": len(validation_frame),
        "random_seed": random_seed,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "training_elapsed_seconds": elapsed_seconds,
        "input_feature_count": int(x_train.shape[1]),
        "validation_candidate_time_mae_minutes": float(time_error.mean()),
        "validation_candidate_time_p95_absolute_error_minutes": float(
            np.quantile(time_error, 0.95)
        ),
        "validation_raw_exact_position_rate_before_hungarian": float(
            assignment_exact_without_decoder.mean()
        ),
        "outputs": {
            "time_z": "predicted structure-specific optimal completion time",
            "assignment_logits": "5x5 drone-to-slot score matrix",
        },
        "decoder": "Hungarian one-to-one assignment",
        "scope": {
            "drone_count": 5,
            "fully_charged_soc": 99.0,
            "zero_to_fully_charged_minutes": 90.0,
            "decision_interval_seconds": 30.0,
            "switching_cost": 0.0,
        },
    }
    (output_dir / "structured_configuration_training_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--random-seed", type=int, default=20260819)
    parser.add_argument("--epochs", type=int, default=120)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metrics = train_structured_policy(
        args.training_csv,
        args.output_dir,
        random_seed=args.random_seed,
        epochs=args.epochs,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
