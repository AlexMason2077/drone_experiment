"""Tune a joint listwise neural ranker over complete configuration costs."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
    / "joint_full_configuration_ranker"
)
SOURCE_DIR = DEFAULT_DIR.parent
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


TRIALS = (
    {
        "name": "compact_t005",
        "hidden": (256, 256),
        "dropout": 0.05,
        "learning_rate": 1e-3,
        "temperature_minutes": 0.05,
    },
    {
        "name": "medium_t005",
        "hidden": (512, 512, 256),
        "dropout": 0.08,
        "learning_rate": 7e-4,
        "temperature_minutes": 0.05,
    },
    {
        "name": "medium_t015",
        "hidden": (512, 512, 256),
        "dropout": 0.08,
        "learning_rate": 7e-4,
        "temperature_minutes": 0.15,
    },
    {
        "name": "wide_t015",
        "hidden": (768, 512, 384),
        "dropout": 0.10,
        "learning_rate": 5e-4,
        "temperature_minutes": 0.15,
    },
    {
        "name": "medium_t040",
        "hidden": (512, 384, 256),
        "dropout": 0.03,
        "learning_rate": 5e-4,
        "temperature_minutes": 0.40,
    },
    {
        "name": "wide_t008",
        "hidden": (768, 768, 384),
        "dropout": 0.12,
        "learning_rate": 3e-4,
        "temperature_minutes": 0.08,
    },
)


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


def _transform(frame: pd.DataFrame, preprocessor: dict[str, object]) -> np.ndarray:
    numeric = frame.loc[:, list(preprocessor["numeric_features"])].to_numpy(dtype=np.float32)
    numeric = (
        numeric - np.asarray(preprocessor["numeric_mean"], dtype=np.float32)
    ) / np.asarray(preprocessor["numeric_scale"], dtype=np.float32)
    wind_values = frame["wind_direction"].astype(str).to_numpy()
    wind = np.column_stack(
        [wind_values == value for value in preprocessor["wind_directions"]]
    ).astype(np.float32)
    return np.concatenate([numeric, wind], axis=1)


def _soft_oracle_targets(costs: np.ndarray, temperature: float) -> np.ndarray:
    finite = np.isfinite(costs)
    minimum = np.min(np.where(finite, costs, np.inf), axis=1, keepdims=True)
    logits = np.where(finite, -(costs - minimum) / temperature, -1e9)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits, dtype=np.float64) * finite
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


def _build_model(
    input_width: int,
    class_count: int,
    hidden: Sequence[int],
    dropout: float,
    learning_rate: float,
    random_seed: int,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(random_seed)
    inputs = tf.keras.Input(shape=(input_width,), name="observed_state")
    x = inputs
    for layer_index, width in enumerate(hidden):
        x = tf.keras.layers.Dense(width, activation=tf.nn.gelu, name=f"dense_{layer_index+1}")(x)
        x = tf.keras.layers.LayerNormalization(name=f"norm_{layer_index+1}")(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout, name=f"dropout_{layer_index+1}")(x)
    logits = tf.keras.layers.Dense(class_count, name="complete_configuration_logits")(x)
    model = tf.keras.Model(inputs=inputs, outputs=logits, name="joint_listwise_configuration_ranker")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )
    return model


def _ranking_metrics(logits: np.ndarray, costs: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(costs)
    selected = np.argmax(np.where(finite, logits, -np.inf), axis=1)
    row = np.arange(len(costs))
    selected_cost = costs[row, selected]
    optimum = np.min(np.where(finite, costs, np.inf), axis=1)
    regret = selected_cost - optimum
    return {
        "strict_global_optimal_rate": float(np.mean(np.isclose(regret, 0.0, atol=2e-5))),
        "within_0p1_minute_rate": float(np.mean(regret <= 0.1 + 2e-5)),
        "within_0p5_minute_rate": float(np.mean(regret <= 0.5 + 2e-5)),
        "mean_regret_minutes": float(regret.mean()),
        "median_regret_minutes": float(np.median(regret)),
        "p95_regret_minutes": float(np.quantile(regret, 0.95)),
        "maximum_regret_minutes": float(regret.max()),
    }


def tune_and_evaluate(
    training_states_path: Path,
    independent_states_path: Path,
    training_costs_path: Path,
    independent_costs_path: Path,
    class_table_path: Path,
    output_dir: Path,
    *,
    random_seed: int,
    max_epochs: int,
) -> dict[str, object]:
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    states = pd.read_csv(training_states_path)
    independent_states = pd.read_csv(independent_states_path)
    training_costs = np.load(training_costs_path)["costs"]
    independent_costs = np.load(independent_costs_path)["costs"]
    classes = pd.read_csv(class_table_path)
    if training_costs.shape != (len(states), len(classes)):
        raise ValueError("Training cost matrix shape does not match states/classes")
    if independent_costs.shape != (len(independent_states), len(classes)):
        raise ValueError("Independent cost matrix shape does not match states/classes")

    indices = np.arange(len(states))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)
    internal_count = max(1, int(round(0.10 * len(indices))))
    internal_indices = indices[:internal_count]
    fit_indices = indices[internal_count:]
    fit_frame = states.iloc[fit_indices]
    internal_frame = states.iloc[internal_indices]
    preprocessor = _fit_preprocessor(fit_frame)
    x_fit = _transform(fit_frame, preprocessor)
    x_internal = _transform(internal_frame, preprocessor)
    fit_costs = training_costs[fit_indices]
    internal_costs = training_costs[internal_indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = output_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_records: list[dict[str, object]] = []
    best_key: tuple[float, float, float] | None = None
    best_spec: dict[str, object] | None = None
    best_epochs = 0

    for trial_number, raw_spec in enumerate(TRIALS, start=1):
        spec = dict(raw_spec)
        print(f"\n=== Trial {trial_number}/{len(TRIALS)}: {spec['name']} ===", flush=True)
        y_fit = _soft_oracle_targets(fit_costs, float(spec["temperature_minutes"]))
        y_internal = _soft_oracle_targets(
            internal_costs, float(spec["temperature_minutes"])
        )
        model = _build_model(
            x_fit.shape[1],
            len(classes),
            spec["hidden"],
            float(spec["dropout"]),
            float(spec["learning_rate"]),
            random_seed + trial_number,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=14,
                min_delta=1e-4,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
            ),
        ]
        started = time.perf_counter()
        history = model.fit(
            x_fit,
            y_fit,
            validation_data=(x_internal, y_internal),
            epochs=max_epochs,
            batch_size=128,
            verbose=0,
            callbacks=callbacks,
        )
        elapsed = time.perf_counter() - started
        internal_logits = model.predict(x_internal, batch_size=256, verbose=0)
        metrics = _ranking_metrics(internal_logits, internal_costs)
        completed_epochs = len(history.history["loss"])
        record: dict[str, object] = {
            **spec,
            "hidden": list(spec["hidden"]),
            "epochs_completed": completed_epochs,
            "best_val_loss": float(min(history.history["val_loss"])),
            "elapsed_seconds": elapsed,
            **metrics,
        }
        trial_records.append(record)
        pd.DataFrame(history.history).to_csv(
            trial_dir / f"{spec['name']}_history.csv", index=False
        )
        (trial_dir / f"{spec['name']}_metrics.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        key = (
            float(metrics["strict_global_optimal_rate"]),
            -float(metrics["mean_regret_minutes"]),
            -float(metrics["p95_regret_minutes"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_spec = spec
            best_epochs = completed_epochs
        tf.keras.backend.clear_session()

    assert best_spec is not None
    pd.DataFrame(trial_records).to_csv(output_dir / "hyperparameter_trials.csv", index=False)

    # Retrain only after the internal split has selected the specification.
    final_preprocessor = _fit_preprocessor(states)
    x_all = _transform(states, final_preprocessor)
    y_all = _soft_oracle_targets(training_costs, float(best_spec["temperature_minutes"]))
    final_model = _build_model(
        x_all.shape[1],
        len(classes),
        best_spec["hidden"],
        float(best_spec["dropout"]),
        float(best_spec["learning_rate"]),
        random_seed + 100,
    )
    final_started = time.perf_counter()
    final_history = final_model.fit(
        x_all,
        y_all,
        epochs=best_epochs,
        batch_size=128,
        verbose=0,
    )
    final_training_seconds = time.perf_counter() - final_started
    model_path = output_dir / "joint_listwise_full_configuration_policy.keras"
    final_model.save(model_path)
    (output_dir / "joint_listwise_preprocessor.json").write_text(
        json.dumps(final_preprocessor, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(final_history.history).to_csv(
        output_dir / "final_training_history.csv", index=False
    )

    x_independent = _transform(independent_states, final_preprocessor)
    independent_logits = final_model.predict(x_independent, batch_size=256, verbose=0)
    independent_metrics = _ranking_metrics(independent_logits, independent_costs)
    finite = np.isfinite(independent_costs)
    selected = np.argmax(np.where(finite, independent_logits, -np.inf), axis=1)
    optimum = np.min(np.where(finite, independent_costs, np.inf), axis=1)
    selected_cost = independent_costs[np.arange(len(independent_costs)), selected]
    prediction = independent_states[
        [
            "scenario_id",
            "wind_direction",
            "wind_level",
            "charging_pad_count",
            "remaining_distance_m",
        ]
    ].copy()
    prediction["selected_class_index"] = selected
    prediction["selected_structure"] = classes.iloc[selected]["structure"].to_numpy()
    for drone in range(1, 6):
        prediction[f"selected_slot_index_d{drone}"] = classes.iloc[selected][
            f"slot_index_d{drone}"
        ].to_numpy()
    prediction["selected_total_minutes"] = selected_cost
    prediction["oracle_total_minutes"] = optimum
    prediction["regret_minutes"] = selected_cost - optimum
    prediction.to_csv(output_dir / "independent_predictions.csv", index=False)

    _ = final_model(x_independent[:1], training=False).numpy()
    benchmark_count = min(100, len(independent_states))
    benchmark_started = time.perf_counter()
    for row_number in range(benchmark_count):
        one_logits = final_model(
            x_independent[row_number : row_number + 1], training=False
        ).numpy()[0]
        _ = int(np.argmax(np.where(finite[row_number], one_logits, -np.inf)))
    online_ms = (time.perf_counter() - benchmark_started) * 1000.0 / benchmark_count

    report: dict[str, object] = {
        "status": "pass",
        "method": "joint listwise neural ranking over exact costs of complete C=(f,p,d)",
        "factor_priority": "none imposed",
        "training_states": len(states),
        "internal_validation_states": len(internal_indices),
        "independent_states": len(independent_states),
        "configuration_classes": len(classes),
        "hyperparameter_trials": trial_records,
        "selected_hyperparameters": {
            **best_spec,
            "hidden": list(best_spec["hidden"]),
            "selected_epochs": best_epochs,
        },
        "final_training_seconds": final_training_seconds,
        "independent_metrics": independent_metrics,
        "warm_online_ms_per_state": online_ms,
        "model": str(model_path.resolve()),
        "training_costs": str(training_costs_path.resolve()),
        "independent_costs": str(independent_costs_path.resolve()),
        "selection_protocol": (
            "Six trials selected only on a fixed 500-state internal split; "
            "the independent 1,000-state set was evaluated once after final retraining."
        ),
    }
    (output_dir / "joint_listwise_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-states",
        type=Path,
        default=SOURCE_DIR / "oracle_training_states_0p25_25m.csv",
    )
    parser.add_argument(
        "--independent-states",
        type=Path,
        default=SOURCE_DIR / "oracle_validation_states_1000.csv",
    )
    parser.add_argument("--training-costs", type=Path, default=DEFAULT_DIR / "training_costs.npz")
    parser.add_argument(
        "--independent-costs", type=Path, default=DEFAULT_DIR / "independent_costs.npz"
    )
    parser.add_argument(
        "--class-table", type=Path, default=DEFAULT_DIR / "complete_configuration_classes.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--random-seed", type=int, default=20260820)
    parser.add_argument("--max-epochs", type=int, default=120)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = tune_and_evaluate(
        args.training_states,
        args.independent_states,
        args.training_costs,
        args.independent_costs,
        args.class_table,
        args.output_dir,
        random_seed=args.random_seed,
        max_epochs=args.max_epochs,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
