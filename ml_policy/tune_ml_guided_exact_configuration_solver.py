"""Train a residual cost network and use it to guide certified exact search."""

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

from ml_policy.tune_joint_candidate_cost_network import (
    SOURCE_DIR,
    RANKER_DIR,
    _candidate_groups,
    _candidate_metadata,
    _features_for_state,
    _sample_class_indices,
)


DEFAULT_DIR = SOURCE_DIR / "ml_guided_exact_solver"
TRIALS = (
    {"name": "residual_small", "hidden": (128, 128), "dropout": 0.0, "learning_rate": 1e-3},
    {"name": "residual_medium", "hidden": (256, 256), "dropout": 0.02, "learning_rate": 6e-4},
    {"name": "residual_deep", "hidden": (256, 256, 128), "dropout": 0.02, "learning_rate": 4e-4},
    {"name": "residual_wide", "hidden": (512, 256), "dropout": 0.04, "learning_rate": 4e-4},
)


def _build_rows(
    states: pd.DataFrame,
    costs: np.ndarray,
    state_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    groups: dict[int, dict[str, np.ndarray]],
    *,
    random_seed: int,
    hard_count: int,
    random_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    x_parts = []
    y_parts = []
    for state_index in state_indices:
        row = states.iloc[int(state_index)]
        class_indices = _sample_class_indices(
            costs[int(state_index)], rng, hard_count, random_count
        )
        features = _features_for_state(
            row,
            class_indices,
            metadata,
            groups[int(row["scenario_id"])],
        )
        flight_minutes = float(row["remaining_distance_m"]) / 0.10 / 60.0
        lower_bound = flight_minutes + features[:, -1]
        gap = np.maximum(0.0, costs[int(state_index), class_indices] - lower_bound)
        x_parts.append(features)
        y_parts.append(np.log1p(gap).astype(np.float32))
    return np.concatenate(x_parts), np.concatenate(y_parts)


def _fit_x_scaler(x: np.ndarray) -> dict[str, object]:
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return {"x_mean": mean.tolist(), "x_scale": scale.tolist()}


def _scale_x(x: np.ndarray, scaler: dict[str, object]) -> np.ndarray:
    return (
        (x - np.asarray(scaler["x_mean"], dtype=np.float32))
        / np.asarray(scaler["x_scale"], dtype=np.float32)
    ).astype(np.float32)


def _build_model(
    input_width: int,
    hidden: Sequence[int],
    dropout: float,
    learning_rate: float,
    random_seed: int,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(random_seed)
    inputs = tf.keras.Input(shape=(input_width,), name="state_and_complete_configuration")
    x = inputs
    for width in hidden:
        x = tf.keras.layers.Dense(width, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)
    log_gap = tf.keras.layers.Dense(1, name="log1p_scheduling_gap")(x)
    model = tf.keras.Model(inputs, log_gap, name="joint_residual_configuration_cost_network")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae_log_gap")],
    )
    return model


def _certified_search(
    exact_costs: np.ndarray,
    lower_bounds: np.ndarray,
    predicted_costs: np.ndarray,
) -> tuple[float, int, int]:
    order = np.argsort(predicted_costs, kind="stable")
    visited = np.zeros(len(order), dtype=bool)
    best_cost = float("inf")
    best_index = -1
    evaluations = 0
    for candidate_index in order:
        candidate_index = int(candidate_index)
        visited[candidate_index] = True
        evaluations += 1
        cost = float(exact_costs[candidate_index])
        if cost < best_cost:
            best_cost = cost
            best_index = candidate_index
        if np.all(visited):
            break
        remaining_lower_bound = float(np.min(lower_bounds[~visited]))
        if best_cost <= remaining_lower_bound + 2e-5:
            break
    return best_cost, best_index, evaluations


def _evaluate(
    model: tf.keras.Model,
    scaler: dict[str, object],
    states: pd.DataFrame,
    costs: np.ndarray,
    state_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    groups: dict[int, dict[str, np.ndarray]],
) -> tuple[dict[str, float], pd.DataFrame]:
    records = []
    for state_index in state_indices:
        row = states.iloc[int(state_index)]
        class_indices = np.flatnonzero(np.isfinite(costs[int(state_index)])).astype(np.int32)
        features = _features_for_state(
            row,
            class_indices,
            metadata,
            groups[int(row["scenario_id"])],
        )
        flight_minutes = float(row["remaining_distance_m"]) / 0.10 / 60.0
        # The analytical value is an admissible lower bound.  Subtract a small
        # numerical guard because the cached exact costs use float32 storage;
        # this prevents rounding from ever turning the bound into an overestimate.
        lower_bounds = flight_minutes + features[:, -1] - 1e-3
        if int(row["charging_pad_count"]) in (1, 5):
            predicted_gap = np.zeros(len(class_indices), dtype=np.float32)
        else:
            predicted_log_gap = model.predict(
                _scale_x(features, scaler), batch_size=2048, verbose=0
            ).reshape(-1)
            predicted_gap = np.maximum(0.0, np.expm1(predicted_log_gap))
        predicted_costs = lower_bounds + predicted_gap
        local_costs = costs[int(state_index), class_indices]
        best_cost, best_local_index, evaluations = _certified_search(
            local_costs, lower_bounds, predicted_costs
        )
        optimum = float(np.min(local_costs))
        selected_class = int(class_indices[best_local_index])
        records.append(
            {
                "state_index": int(state_index),
                "scenario_id": int(row["scenario_id"]),
                "charging_pad_count": int(row["charging_pad_count"]),
                "selected_class_index": selected_class,
                "selected_structure": str(metadata["structure"][selected_class]),
                "selected_total_minutes": best_cost,
                "oracle_total_minutes": optimum,
                "regret_minutes": best_cost - optimum,
                "exact_candidate_evaluations": evaluations,
                "feasible_candidate_count": len(class_indices),
                "evaluated_fraction": evaluations / len(class_indices),
            }
        )
    frame = pd.DataFrame(records)
    regret = frame["regret_minutes"].to_numpy(dtype=float)
    evaluations = frame["exact_candidate_evaluations"].to_numpy(dtype=float)
    fraction = frame["evaluated_fraction"].to_numpy(dtype=float)
    metrics = {
        "certified_global_optimal_rate": float(np.mean(np.isclose(regret, 0.0, atol=2e-5))),
        "mean_exact_candidate_evaluations": float(evaluations.mean()),
        "median_exact_candidate_evaluations": float(np.median(evaluations)),
        "p95_exact_candidate_evaluations": float(np.quantile(evaluations, 0.95)),
        "maximum_exact_candidate_evaluations": float(evaluations.max()),
        "mean_evaluated_candidate_fraction": float(fraction.mean()),
        "mean_pruned_candidate_fraction": float(1.0 - fraction.mean()),
    }
    return metrics, frame


def tune(
    training_states_path: Path,
    independent_states_path: Path,
    training_candidates_path: Path,
    independent_candidates_path: Path,
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
    training_candidates = pd.read_csv(training_candidates_path)
    independent_candidates = pd.read_csv(independent_candidates_path)
    training_costs = np.load(training_costs_path)["costs"]
    independent_costs = np.load(independent_costs_path)["costs"]
    metadata = _candidate_metadata(pd.read_csv(class_table_path))
    training_groups = _candidate_groups(training_candidates)
    independent_groups = _candidate_groups(independent_candidates)

    indices = np.arange(len(states))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)
    internal_count = int(round(0.10 * len(indices)))
    internal_indices = indices[:internal_count]
    fit_indices = indices[internal_count:]
    x_fit, y_fit = _build_rows(
        states,
        training_costs,
        fit_indices,
        metadata,
        training_groups,
        random_seed=random_seed,
        hard_count=80,
        random_count=80,
    )
    x_internal, y_internal = _build_rows(
        states,
        training_costs,
        internal_indices,
        metadata,
        training_groups,
        random_seed=random_seed + 1,
        hard_count=100,
        random_count=60,
    )
    scaler = _fit_x_scaler(x_fit)
    sx_fit, sx_internal = _scale_x(x_fit, scaler), _scale_x(x_internal, scaler)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    best_key = None
    best_spec = None
    best_epochs = None

    for trial_index, raw_spec in enumerate(TRIALS, start=1):
        spec = dict(raw_spec)
        print(f"\n=== Certified-search trial {trial_index}/{len(TRIALS)}: {spec['name']} ===", flush=True)
        model = _build_model(
            sx_fit.shape[1],
            spec["hidden"],
            float(spec["dropout"]),
            float(spec["learning_rate"]),
            random_seed + trial_index,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, min_delta=1e-5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5
            ),
        ]
        started = time.perf_counter()
        history = model.fit(
            sx_fit,
            y_fit,
            validation_data=(sx_internal, y_internal),
            epochs=max_epochs,
            batch_size=1024,
            verbose=0,
            callbacks=callbacks,
        )
        elapsed = time.perf_counter() - started
        metrics, _ = _evaluate(
            model,
            scaler,
            states,
            training_costs,
            internal_indices,
            metadata,
            training_groups,
        )
        record = {
            **spec,
            "hidden": list(spec["hidden"]),
            "epochs_completed": len(history.history["loss"]),
            "best_val_loss": float(min(history.history["val_loss"])),
            "elapsed_seconds": elapsed,
            **metrics,
        }
        trials.append(record)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        key = (
            -float(metrics["mean_exact_candidate_evaluations"]),
            -float(metrics["p95_exact_candidate_evaluations"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_spec = spec
            best_epochs = len(history.history["loss"])
        tf.keras.backend.clear_session()

    assert best_spec is not None and best_epochs is not None
    pd.DataFrame(trials).to_csv(output_dir / "hyperparameter_trials.csv", index=False)
    x_all, y_all = _build_rows(
        states,
        training_costs,
        np.arange(len(states)),
        metadata,
        training_groups,
        random_seed=random_seed + 100,
        hard_count=100,
        random_count=100,
    )
    final_scaler = _fit_x_scaler(x_all)
    final_model = _build_model(
        x_all.shape[1],
        best_spec["hidden"],
        float(best_spec["dropout"]),
        float(best_spec["learning_rate"]),
        random_seed + 100,
    )
    final_started = time.perf_counter()
    final_model.fit(
        _scale_x(x_all, final_scaler),
        y_all,
        epochs=best_epochs,
        batch_size=1024,
        verbose=0,
    )
    final_seconds = time.perf_counter() - final_started
    model_path = output_dir / "joint_residual_search_guidance.keras"
    final_model.save(model_path)
    (output_dir / "residual_feature_scaler.json").write_text(
        json.dumps(final_scaler, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    independent_metrics, predictions = _evaluate(
        final_model,
        final_scaler,
        independent_states,
        independent_costs,
        np.arange(len(independent_states)),
        metadata,
        independent_groups,
    )
    predictions.to_csv(output_dir / "independent_certified_predictions.csv", index=False)
    report = {
        "status": "pass",
        "method": "joint neural residual cost guidance plus admissible-bound exact certification",
        "global_optimality": "guaranteed by exact lower-bound termination, independent of ML errors",
        "factor_priority": "none imposed",
        "training_states": len(states),
        "sampled_training_candidate_rows": len(x_all),
        "configuration_classes": training_costs.shape[1],
        "hyperparameter_trials": trials,
        "selected_hyperparameters": {
            **best_spec,
            "hidden": list(best_spec["hidden"]),
            "selected_epochs": best_epochs,
        },
        "final_training_seconds": final_seconds,
        "independent_metrics": independent_metrics,
        "model": str(model_path.resolve()),
        "selection_protocol": (
            "Four guidance networks selected only by exact evaluations on a 500-state "
            "internal split; independent 1,000 states evaluated once after final fitting."
        ),
    }
    (output_dir / "ml_guided_exact_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-states", type=Path, default=SOURCE_DIR / "oracle_training_states_0p25_25m.csv")
    parser.add_argument("--independent-states", type=Path, default=SOURCE_DIR / "oracle_validation_states_1000.csv")
    parser.add_argument("--training-candidates", type=Path, default=SOURCE_DIR / "position_aware_training_candidates.csv")
    parser.add_argument("--independent-candidates", type=Path, default=SOURCE_DIR / "position_aware_validation_candidates.csv")
    parser.add_argument("--training-costs", type=Path, default=RANKER_DIR / "training_costs.npz")
    parser.add_argument("--independent-costs", type=Path, default=RANKER_DIR / "independent_costs.npz")
    parser.add_argument("--class-table", type=Path, default=RANKER_DIR / "complete_configuration_classes.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--random-seed", type=int, default=20260822)
    parser.add_argument("--max-epochs", type=int, default=70)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = tune(
        args.training_states,
        args.independent_states,
        args.training_candidates,
        args.independent_candidates,
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
