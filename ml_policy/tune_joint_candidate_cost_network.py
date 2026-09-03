"""Tune a shared neural cost model over complete candidate configurations."""

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

from ml_policy.charging_model import exponential_charging_minutes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_DIR = SOURCE_DIR / "joint_candidate_cost_network"
RANKER_DIR = SOURCE_DIR / "joint_full_configuration_ranker"
FORMATIONS = ("column", "diamond", "echelon", "front", "vee")
WIND_DIRECTIONS = ("head", "side", "tail")

TRIALS = (
    {"name": "small", "hidden": (128, 128), "dropout": 0.02, "learning_rate": 1e-3},
    {"name": "medium", "hidden": (256, 256, 128), "dropout": 0.04, "learning_rate": 7e-4},
    {"name": "wide", "hidden": (512, 256, 128), "dropout": 0.05, "learning_rate": 5e-4},
    {"name": "deep", "hidden": (256, 256, 256, 128), "dropout": 0.02, "learning_rate": 4e-4},
)


def _canonical_formation(structure: str) -> str:
    value = structure.rsplit("_", 1)[0]
    return "echelon" if value == "echalon" else value


def _candidate_metadata(class_table: pd.DataFrame) -> dict[str, np.ndarray]:
    structures = class_table["structure"].astype(str).to_numpy()
    formation = np.column_stack(
        [np.asarray([_canonical_formation(value) == item for value in structures]) for item in FORMATIONS]
    ).astype(np.float32)
    spacing = np.asarray([float(value.rsplit("_", 1)[1]) for value in structures], dtype=np.float32)
    permutation = np.column_stack(
        [class_table[f"slot_index_d{index}"].to_numpy(dtype=np.int32) - 1 for index in range(1, 6)]
    )
    return {
        "structure": structures,
        "formation": formation,
        "spacing": spacing,
        "permutation": permutation,
    }


def _candidate_groups(frame: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for scenario_id, group in frame.groupby("scenario_id", sort=False):
        rate_by_structure = {}
        for _, row in group.iterrows():
            rate_by_structure[str(row["structure"])] = np.asarray(
                [float(row[f"slot_{index}_rate_pp_per_min"]) for index in range(1, 6)],
                dtype=np.float32,
            )
        result[int(scenario_id)] = rate_by_structure
    return result


def _charging_jobs(arrival: np.ndarray) -> np.ndarray:
    flat = arrival.reshape(-1)
    result = np.asarray(
        [exponential_charging_minutes(float(value)) for value in flat], dtype=np.float32
    )
    return result.reshape(arrival.shape)


def _features_for_state(
    row: pd.Series,
    class_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    rates_by_structure: dict[str, np.ndarray],
) -> np.ndarray:
    count = len(class_indices)
    wind = np.asarray(
        [str(row["wind_direction"]) == value for value in WIND_DIRECTIONS], dtype=np.float32
    )
    wind = np.repeat(wind[None, :], count, axis=0)
    scalar = np.repeat(
        np.asarray(
            [[float(row["wind_level"]), float(row["charging_pad_count"]), float(row["remaining_distance_m"])]],
            dtype=np.float32,
        ),
        count,
        axis=0,
    )
    soc = np.asarray([float(row[f"soc_d{index}"]) for index in range(1, 6)], dtype=np.float32)
    soc_matrix = np.repeat(soc[None, :], count, axis=0)
    sorted_soc = np.repeat(np.sort(soc)[None, :], count, axis=0)
    soc_range = np.full((count, 1), float(np.ptp(soc)), dtype=np.float32)
    structures = metadata["structure"][class_indices]
    base_rates = np.stack([rates_by_structure[str(value)] for value in structures])
    permutation = metadata["permutation"][class_indices]
    assigned_rates = np.take_along_axis(base_rates, permutation, axis=1)
    flight_minutes = float(row["remaining_distance_m"]) / 0.10 / 60.0
    arrival = soc_matrix - assigned_rates * flight_minutes
    jobs = _charging_jobs(arrival)
    sorted_jobs = np.sort(jobs, axis=1)
    job_sum = jobs.sum(axis=1, keepdims=True)
    job_max = jobs.max(axis=1, keepdims=True)
    job_mean = jobs.mean(axis=1, keepdims=True)
    job_std = jobs.std(axis=1, keepdims=True)
    k = max(1.0, float(row["charging_pad_count"]))
    schedule_lower_bound = np.maximum(job_max, job_sum / k)
    return np.concatenate(
        [
            wind,
            scalar,
            soc_matrix,
            sorted_soc,
            soc_range,
            metadata["formation"][class_indices],
            (metadata["spacing"][class_indices] / 75.0)[:, None],
            assigned_rates,
            arrival,
            jobs,
            sorted_jobs,
            job_sum,
            job_max,
            job_mean,
            job_std,
            schedule_lower_bound,
        ],
        axis=1,
    ).astype(np.float32)


def _sample_class_indices(
    costs: np.ndarray,
    rng: np.random.Generator,
    hard_count: int,
    random_count: int,
) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(costs))
    order = finite[np.argsort(costs[finite], kind="stable")]
    hard = order[: min(hard_count, len(order))]
    remaining = order[len(hard) :]
    if len(remaining) > random_count:
        random_part = rng.choice(remaining, size=random_count, replace=False)
    else:
        random_part = remaining
    return np.unique(np.concatenate([hard, random_part])).astype(np.int32)


def _build_training_rows(
    states: pd.DataFrame,
    costs: np.ndarray,
    state_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    groups: dict[int, dict[str, np.ndarray]],
    *,
    random_seed: int,
    hard_count: int = 64,
    random_count: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for position, state_index in enumerate(state_indices):
        row = states.iloc[int(state_index)]
        indices = _sample_class_indices(costs[int(state_index)], rng, hard_count, random_count)
        features.append(
            _features_for_state(
                row, indices, metadata, groups[int(row["scenario_id"])]
            )
        )
        targets.append(costs[int(state_index), indices].astype(np.float32))
        if (position + 1) % 1000 == 0:
            print(f"feature rows built for {position + 1}/{len(state_indices)} states", flush=True)
    return np.concatenate(features), np.concatenate(targets)


def _fit_scaler(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    x_mean = x.mean(axis=0, dtype=np.float64)
    x_scale = x.std(axis=0, dtype=np.float64)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = float(y.mean())
    y_scale = float(y.std()) or 1.0
    return {
        "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(),
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def _scale_x(x: np.ndarray, scaler: dict[str, object]) -> np.ndarray:
    return (
        (x - np.asarray(scaler["x_mean"], dtype=np.float32))
        / np.asarray(scaler["x_scale"], dtype=np.float32)
    ).astype(np.float32)


def _scale_y(y: np.ndarray, scaler: dict[str, object]) -> np.ndarray:
    return ((y - float(scaler["y_mean"])) / float(scaler["y_scale"])).astype(np.float32)


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
    for index, width in enumerate(hidden):
        x = tf.keras.layers.Dense(width, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)
    output = tf.keras.layers.Dense(1, name="predicted_total_completion_time_z")(x)
    model = tf.keras.Model(inputs, output, name="joint_complete_configuration_cost_network")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=0.25),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae_z")],
    )
    return model


def _evaluate_states(
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
        indices = np.flatnonzero(np.isfinite(costs[int(state_index)])).astype(np.int32)
        features = _features_for_state(
            row, indices, metadata, groups[int(row["scenario_id"])]
        )
        prediction_z = model.predict(_scale_x(features, scaler), batch_size=2048, verbose=0).reshape(-1)
        selected_local = int(np.argmin(prediction_z))
        selected_class = int(indices[selected_local])
        optimum = float(np.min(costs[int(state_index), indices]))
        selected_cost = float(costs[int(state_index), selected_class])
        records.append(
            {
                "state_index": int(state_index),
                "scenario_id": int(row["scenario_id"]),
                "selected_class_index": selected_class,
                "selected_structure": str(metadata["structure"][selected_class]),
                "selected_total_minutes": selected_cost,
                "oracle_total_minutes": optimum,
                "regret_minutes": selected_cost - optimum,
            }
        )
    frame = pd.DataFrame(records)
    regret = frame["regret_minutes"].to_numpy(dtype=float)
    metrics = {
        "strict_global_optimal_rate": float(np.mean(np.isclose(regret, 0.0, atol=2e-5))),
        "within_0p1_minute_rate": float(np.mean(regret <= 0.1 + 2e-5)),
        "within_0p5_minute_rate": float(np.mean(regret <= 0.5 + 2e-5)),
        "mean_regret_minutes": float(regret.mean()),
        "median_regret_minutes": float(np.median(regret)),
        "p95_regret_minutes": float(np.quantile(regret, 0.95)),
        "maximum_regret_minutes": float(regret.max()),
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
    class_table = pd.read_csv(class_table_path)
    metadata = _candidate_metadata(class_table)
    training_groups = _candidate_groups(training_candidates)
    independent_groups = _candidate_groups(independent_candidates)

    indices = np.arange(len(states))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)
    internal_count = max(1, int(round(0.10 * len(indices))))
    internal_indices = indices[:internal_count]
    fit_indices = indices[internal_count:]
    x_fit, y_fit = _build_training_rows(
        states,
        training_costs,
        fit_indices,
        metadata,
        training_groups,
        random_seed=random_seed,
    )
    x_internal, y_internal = _build_training_rows(
        states,
        training_costs,
        internal_indices,
        metadata,
        training_groups,
        random_seed=random_seed + 1,
        hard_count=96,
        random_count=32,
    )
    scaler = _fit_scaler(x_fit, y_fit)
    sx_fit, sy_fit = _scale_x(x_fit, scaler), _scale_y(y_fit, scaler)
    sx_internal, sy_internal = _scale_x(x_internal, scaler), _scale_y(y_internal, scaler)

    output_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    best_key = None
    best_spec = None
    best_epochs = None
    for trial_index, raw_spec in enumerate(TRIALS, start=1):
        spec = dict(raw_spec)
        print(f"\n=== Candidate-cost trial {trial_index}/{len(TRIALS)}: {spec['name']} ===", flush=True)
        model = _build_model(
            sx_fit.shape[1],
            spec["hidden"],
            float(spec["dropout"]),
            float(spec["learning_rate"]),
            random_seed + trial_index,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, min_delta=1e-4, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5
            ),
        ]
        started = time.perf_counter()
        history = model.fit(
            sx_fit,
            sy_fit,
            validation_data=(sx_internal, sy_internal),
            epochs=max_epochs,
            batch_size=1024,
            verbose=0,
            callbacks=callbacks,
        )
        elapsed = time.perf_counter() - started
        metrics, _ = _evaluate_states(
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
            "elapsed_seconds": elapsed,
            "best_val_loss": float(min(history.history["val_loss"])),
            **metrics,
        }
        trials.append(record)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        key = (
            float(metrics["strict_global_optimal_rate"]),
            -float(metrics["mean_regret_minutes"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_spec = spec
            best_epochs = len(history.history["loss"])
        pd.DataFrame(history.history).to_csv(
            output_dir / f"{spec['name']}_history.csv", index=False
        )
        tf.keras.backend.clear_session()

    assert best_spec is not None and best_epochs is not None
    pd.DataFrame(trials).to_csv(output_dir / "hyperparameter_trials.csv", index=False)

    # Final fit uses all training states with a new deterministic sample.
    all_indices = np.arange(len(states))
    x_all, y_all = _build_training_rows(
        states,
        training_costs,
        all_indices,
        metadata,
        training_groups,
        random_seed=random_seed + 100,
        hard_count=80,
        random_count=80,
    )
    final_scaler = _fit_scaler(x_all, y_all)
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
        _scale_y(y_all, final_scaler),
        epochs=best_epochs,
        batch_size=1024,
        verbose=0,
    )
    final_seconds = time.perf_counter() - final_started
    model_path = output_dir / "joint_candidate_cost_policy.keras"
    final_model.save(model_path)
    (output_dir / "candidate_cost_scaler.json").write_text(
        json.dumps(final_scaler, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    independent_metrics, predictions = _evaluate_states(
        final_model,
        final_scaler,
        independent_states,
        independent_costs,
        np.arange(len(independent_states)),
        metadata,
        independent_groups,
    )
    predictions.to_csv(output_dir / "independent_predictions.csv", index=False)
    report = {
        "status": "pass",
        "method": "shared neural cost model over complete candidate C=(f,p,d)",
        "factor_priority": "none imposed",
        "training_states": len(states),
        "sampled_training_candidate_rows": len(x_all),
        "configuration_classes": len(class_table),
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
            "Four architectures selected on a 500-state internal split; independent "
            "1,000 states evaluated once after final fitting."
        ),
    }
    (output_dir / "joint_candidate_cost_metrics.json").write_text(
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
    parser.add_argument("--random-seed", type=int, default=20260821)
    parser.add_argument("--max-epochs", type=int, default=80)
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
