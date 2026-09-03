"""Train a shared candidate-aware neural residual-cost ranker.

Each complete C=(formation, spacing, position) is represented by empirical
slot rates, assigned drone rates, projected arrival SOC, and charging-job
features.  One shared network learns the residual above an admissible charging
lower bound, so rare permutations can share statistical strength.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/drone-matplotlib-cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from ml_policy.charging_model import (
    FULLY_CHARGED_SOC,
    ZERO_TO_FULLY_CHARGED_MINUTES,
    charging_time_constant_minutes,
)
from ml_policy.oracle_optimizer import DEFAULT_RATE_TABLE_PATH, EmpiricalRateTable
from ml_policy.train_controlled_k_cost_ranker import _metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "controlled_k_sweep"
    / "cost_aware_ranker"
)
DEFAULT_CLASSES = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
    / "joint_full_configuration_ranker"
    / "complete_configuration_classes.csv"
)
FORMATIONS = ("column", "diamond", "echelon", "front", "vee")
WIND_DIRECTIONS = ("head", "side", "tail")


def _canonical_formation(structure: str) -> str:
    value = structure.rsplit("_", 1)[0]
    return "echelon" if value == "echalon" else value


def _candidate_metadata(class_table: pd.DataFrame) -> dict[str, np.ndarray]:
    ordered = class_table.sort_values("class_index")
    structures = ordered["structure"].astype(str).to_numpy()
    formation = np.column_stack(
        [
            np.asarray(
                [_canonical_formation(value) == item for value in structures]
            )
            for item in FORMATIONS
        ]
    ).astype(np.float32)
    spacing = np.asarray(
        [float(value.rsplit("_", 1)[1]) for value in structures], dtype=np.float32
    )
    permutation = np.column_stack(
        [
            ordered[f"slot_index_d{index}"].to_numpy(dtype=np.int32) - 1
            for index in range(1, 6)
        ]
    )
    return {
        "structure": structures,
        "formation": formation,
        "spacing": spacing,
        "permutation": permutation,
    }


def _rate_groups(
    rate_table_path: Path,
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    table = EmpiricalRateTable.from_csv(rate_table_path)
    result: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for direction in WIND_DIRECTIONS:
        for level in (1, 2):
            result[(direction, level)] = {
                cell.structure.label: np.asarray(
                    [slot.rate_pp_per_min for slot in cell.slots], dtype=np.float32
                )
                for cell in table.structures_for(direction, level)
            }
    return result


def _charging_jobs(arrival_soc: np.ndarray) -> np.ndarray:
    tau = charging_time_constant_minutes(
        fully_charged_soc=FULLY_CHARGED_SOC,
        zero_to_fully_charged_minutes=ZERO_TO_FULLY_CHARGED_MINUTES,
    )
    clipped = np.clip(arrival_soc.astype(np.float64), 0.0, 100.0)
    target_remaining = 1.0 - FULLY_CHARGED_SOC / 100.0
    jobs = tau * np.log((1.0 - clipped / 100.0) / target_remaining)
    jobs = np.where(clipped >= FULLY_CHARGED_SOC, 0.0, jobs)
    return np.maximum(jobs, 0.0).astype(np.float32)


def _features_for_state(
    row: pd.Series,
    class_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    rates_for_condition: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(class_indices)
    direction = str(row["wind_direction"])
    wind = np.repeat(
        np.asarray([[direction == value for value in WIND_DIRECTIONS]], dtype=np.float32),
        count,
        axis=0,
    )
    level = float(row["wind_level"])
    k = float(row["charging_pad_count"])
    distance = float(row["remaining_distance_m"])
    scalar = np.repeat(
        np.asarray([[level, k / 5.0, distance / 25.0]], dtype=np.float32),
        count,
        axis=0,
    )
    soc = np.asarray(
        [float(row[f"soc_d{index}"]) for index in range(1, 6)], dtype=np.float32
    )
    soc_matrix = np.repeat((soc / 100.0)[None, :], count, axis=0)
    sorted_soc = np.repeat((np.sort(soc) / 100.0)[None, :], count, axis=0)
    soc_range = np.full((count, 1), float(np.ptp(soc) / 100.0), dtype=np.float32)
    structures = metadata["structure"][class_indices]
    base_rates = np.stack(
        [rates_for_condition[str(structure)] for structure in structures]
    )
    permutation = metadata["permutation"][class_indices]
    assigned_rates = np.take_along_axis(base_rates, permutation, axis=1)
    flight_minutes = distance / 0.10 / 60.0
    arrival = soc[None, :] - assigned_rates * flight_minutes
    jobs = _charging_jobs(arrival)
    sorted_jobs = np.sort(jobs, axis=1)
    job_sum = jobs.sum(axis=1, keepdims=True)
    job_max = jobs.max(axis=1, keepdims=True)
    job_mean = jobs.mean(axis=1, keepdims=True)
    job_std = jobs.std(axis=1, keepdims=True)
    schedule_lower_bound = np.maximum(job_max, job_sum / max(k, 1.0))
    total_lower_bound = flight_minutes + schedule_lower_bound.reshape(-1)
    features = np.concatenate(
        [
            wind,
            scalar,
            soc_matrix,
            sorted_soc,
            soc_range,
            metadata["formation"][class_indices],
            (metadata["spacing"][class_indices] / 75.0)[:, None],
            assigned_rates / 20.0,
            arrival / 100.0,
            jobs / 100.0,
            sorted_jobs / 100.0,
            job_sum / 500.0,
            job_max / 100.0,
            job_mean / 100.0,
            job_std / 100.0,
            schedule_lower_bound / 100.0,
        ],
        axis=1,
    ).astype(np.float32)
    return features, total_lower_bound.astype(np.float32)


def _sample_indices(
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


def _build_sampled_rows(
    states: pd.DataFrame,
    costs: np.ndarray,
    metadata: dict[str, np.ndarray],
    rate_groups: dict[tuple[str, int], dict[str, np.ndarray]],
    *,
    random_seed: int,
    hard_count: int,
    random_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    state_numbers: list[np.ndarray] = []
    for state_number, (_, row) in enumerate(states.iterrows()):
        indices = _sample_indices(
            costs[state_number], rng, hard_count, random_count
        )
        state_features, lower_bound = _features_for_state(
            row,
            indices,
            metadata,
            rate_groups[(str(row["wind_direction"]), int(row["wind_level"]))],
        )
        exact = costs[state_number, indices].astype(np.float32)
        residual = np.maximum(exact - lower_bound, 0.0)
        target = np.log1p(residual).astype(np.float32)
        optimum = float(np.min(costs[state_number, np.isfinite(costs[state_number])]))
        regret = exact - optimum
        sample_weight = (1.0 + 4.0 * np.exp(-regret / 0.5)).astype(np.float32)
        features.append(state_features)
        targets.append(target)
        weights.append(sample_weight)
        state_numbers.append(np.full(len(indices), state_number, dtype=np.int32))
        if (state_number + 1) % 1000 == 0:
            print(
                f"sampled candidate features {state_number + 1}/{len(states)}",
                flush=True,
            )
    return (
        np.concatenate(features),
        np.concatenate(targets),
        np.concatenate(weights),
        np.concatenate(state_numbers),
    )


def _fit_scaler(x: np.ndarray) -> dict[str, object]:
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def _scale(x: np.ndarray, scaler: dict[str, object]) -> np.ndarray:
    return (
        (x - np.asarray(scaler["mean"], dtype=np.float32))
        / np.asarray(scaler["scale"], dtype=np.float32)
    ).astype(np.float32)


def _build_model(input_width: int, random_seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(random_seed)
    inputs = tf.keras.Input(shape=(input_width,), name="state_and_candidate")
    x = inputs
    for index, width in enumerate((256, 256, 128), start=1):
        x = tf.keras.layers.Dense(width, activation=tf.nn.gelu, name=f"dense_{index}")(x)
        x = tf.keras.layers.LayerNormalization(name=f"norm_{index}")(x)
        x = tf.keras.layers.Dropout(0.08, name=f"dropout_{index}")(x)
    residual_log = tf.keras.layers.Dense(
        1, activation="softplus", name="predicted_log1p_residual"
    )(x)
    model = tf.keras.Model(inputs, residual_log, name="candidate_aware_residual_ranker")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=7e-4, weight_decay=1e-4
        ),
        loss=tf.keras.losses.Huber(delta=0.10),
    )
    return model


def _evaluate(
    model: tf.keras.Model,
    scaler: dict[str, object],
    states: pd.DataFrame,
    costs: np.ndarray,
    state_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    rate_groups: dict[tuple[str, int], dict[str, np.ndarray]],
) -> tuple[dict[str, float], pd.DataFrame]:
    records: list[dict[str, object]] = []
    for position, state_index in enumerate(state_indices):
        row = states.iloc[int(state_index)]
        indices = np.flatnonzero(np.isfinite(costs[int(state_index)])).astype(np.int32)
        features, lower_bound = _features_for_state(
            row,
            indices,
            metadata,
            rate_groups[(str(row["wind_direction"]), int(row["wind_level"]))],
        )
        predicted_log = model(
            _scale(features, scaler), training=False
        ).numpy().reshape(-1)
        predicted_total = lower_bound + np.expm1(np.maximum(predicted_log, 0.0))
        selected_class = int(indices[int(np.argmin(predicted_total))])
        exact = costs[int(state_index)]
        optimum = float(np.min(exact[np.isfinite(exact)]))
        selected_cost = float(exact[selected_class])
        records.append(
            {
                "state_index": int(state_index),
                "base_state_id": int(row["base_state_id"]),
                "wind_direction": str(row["wind_direction"]),
                "wind_level": int(row["wind_level"]),
                "charging_pad_count": int(row["charging_pad_count"]),
                "selected_class_index": selected_class,
                "selected_structure": str(metadata["structure"][selected_class]),
                "selected_total_minutes": selected_cost,
                "global_optimum_minutes": optimum,
                "regret_minutes": selected_cost - optimum,
            }
        )
        if (position + 1) % 1000 == 0:
            print(f"evaluated {position + 1}/{len(state_indices)} states", flush=True)
    frame = pd.DataFrame(records)
    regret = frame["regret_minutes"].to_numpy(dtype=float)
    metrics = {
        "states": len(frame),
        "strict_global_optimal_rate": float(
            np.mean(np.isclose(regret, 0.0, atol=2e-5))
        ),
        "within_0p1_minute_rate": float(np.mean(regret <= 0.1 + 2e-5)),
        "within_0p5_minute_rate": float(np.mean(regret <= 0.5 + 2e-5)),
        "mean_regret_minutes": float(regret.mean()),
        "median_regret_minutes": float(np.median(regret)),
        "p95_regret_minutes": float(np.quantile(regret, 0.95)),
        "maximum_regret_minutes": float(regret.max()),
        "unsafe_or_infeasible_selection_count": 0,
    }
    return metrics, frame


def train(
    *,
    training_states_path: Path,
    independent_states_path: Path,
    training_costs_path: Path,
    independent_costs_path: Path,
    class_table_path: Path,
    rate_table_path: Path,
    output_dir: Path,
    folds: int,
    max_epochs: int,
    patience: int,
    random_seed: int,
) -> dict[str, object]:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    states = pd.read_csv(training_states_path)
    independent = pd.read_csv(independent_states_path)
    training_costs = np.load(training_costs_path)["costs"]
    independent_costs = np.load(independent_costs_path)["costs"]
    class_table = pd.read_csv(class_table_path)
    metadata = _candidate_metadata(class_table)
    rates = _rate_groups(rate_table_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, sample_weight, sample_state = _build_sampled_rows(
        states,
        training_costs,
        metadata,
        rates,
        random_seed=random_seed,
        hard_count=48,
        random_count=32,
    )
    groups = states["base_state_id"].to_numpy()
    strata = (
        states["wind_direction"].astype(str)
        + "_lv"
        + states["wind_level"].astype(str)
    ).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=random_seed
    )
    splits = list(splitter.split(states, strata, groups))
    cv_records: list[dict[str, object]] = []
    histories = output_dir / "cross_validation_histories"
    histories.mkdir(parents=True, exist_ok=True)
    epochs_completed: list[int] = []
    cv_started = time.perf_counter()

    for fold_number, (fit_states, validation_states) in enumerate(splits, start=1):
        if set(groups[fit_states]).intersection(set(groups[validation_states])):
            raise RuntimeError("base_state_id leakage")
        fit_mask = np.isin(sample_state, fit_states)
        validation_mask = np.isin(sample_state, validation_states)
        scaler = _fit_scaler(x[fit_mask])
        model = _build_model(x.shape[1], random_seed + fold_number)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=1e-5,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
            ),
        ]
        print(f"=== candidate-aware fold {fold_number}/{folds} ===", flush=True)
        started = time.perf_counter()
        history = model.fit(
            _scale(x[fit_mask], scaler),
            y[fit_mask],
            sample_weight=sample_weight[fit_mask],
            validation_data=(
                _scale(x[validation_mask], scaler),
                y[validation_mask],
                sample_weight[validation_mask],
            ),
            epochs=max_epochs,
            batch_size=2048,
            verbose=0,
            callbacks=callbacks,
        )
        metrics, _ = _evaluate(
            model,
            scaler,
            states,
            training_costs,
            validation_states,
            metadata,
            rates,
        )
        epochs = len(history.history["loss"])
        epochs_completed.append(epochs)
        record: dict[str, object] = {
            "fold": fold_number,
            "fit_groups": len(set(groups[fit_states])),
            "validation_groups": len(set(groups[validation_states])),
            "fit_candidate_rows": int(fit_mask.sum()),
            "validation_candidate_rows": int(validation_mask.sum()),
            "epochs_completed": epochs,
            "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
            "best_val_loss": float(min(history.history["val_loss"])),
            "elapsed_seconds": time.perf_counter() - started,
            **metrics,
        }
        cv_records.append(record)
        pd.DataFrame(history.history).to_csv(
            histories / f"fold_{fold_number}.csv", index=False
        )
        print(json.dumps(record, sort_keys=True), flush=True)
        tf.keras.backend.clear_session()

    pd.DataFrame(cv_records).to_csv(
        output_dir / "grouped_cross_validation_metrics.csv", index=False
    )
    final_epochs = max(1, int(round(float(np.median(epochs_completed)))))
    final_scaler = _fit_scaler(x)
    final_model = _build_model(x.shape[1], random_seed + 999)
    final_started = time.perf_counter()
    final_history = final_model.fit(
        _scale(x, final_scaler),
        y,
        sample_weight=sample_weight,
        epochs=final_epochs,
        batch_size=2048,
        verbose=0,
    )
    final_training_seconds = time.perf_counter() - final_started
    model_path = output_dir / "candidate_aware_residual_ranker.keras"
    final_model.save(model_path)
    (output_dir / "feature_scaler.json").write_text(
        json.dumps(final_scaler, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(final_history.history).to_csv(
        output_dir / "final_training_history.csv", index=False
    )
    independent_started = time.perf_counter()
    independent_metrics, independent_predictions = _evaluate(
        final_model,
        final_scaler,
        independent,
        independent_costs,
        np.arange(len(independent)),
        metadata,
        rates,
    )
    independent_seconds = time.perf_counter() - independent_started
    independent_predictions.to_csv(
        output_dir / "independent_predictions.csv", index=False
    )
    subgroup_metrics: dict[str, dict[str, float]] = {}
    for (direction, level, k), group in independent_predictions.groupby(
        ["wind_direction", "wind_level", "charging_pad_count"], sort=True
    ):
        regret = group["regret_minutes"].to_numpy(dtype=float)
        subgroup_metrics[f"{direction}_lv{int(level)}_k{int(k)}"] = {
            "states": len(group),
            "strict_global_optimal_rate": float(
                np.mean(np.isclose(regret, 0.0, atol=2e-5))
            ),
            "within_0p5_minute_rate": float(np.mean(regret <= 0.5 + 2e-5)),
            "mean_regret_minutes": float(regret.mean()),
            "p95_regret_minutes": float(np.quantile(regret, 0.95)),
        }
    cv_frame = pd.DataFrame(cv_records)
    summary_columns = (
        "strict_global_optimal_rate",
        "within_0p1_minute_rate",
        "within_0p5_minute_rate",
        "mean_regret_minutes",
        "p95_regret_minutes",
        "maximum_regret_minutes",
    )
    report: dict[str, object] = {
        "status": "pass",
        "method": "candidate-aware shared neural log-residual cost ranking",
        "factor_priority": "none imposed",
        "training_states": len(states),
        "training_base_state_groups": int(states["base_state_id"].nunique()),
        "sampled_candidate_rows": len(x),
        "configuration_classes": len(class_table),
        "folds": folds,
        "group_leakage_count": 0,
        "cross_validation_summary": {
            column: {
                "mean": float(cv_frame[column].mean()),
                "standard_deviation": float(cv_frame[column].std(ddof=1)),
                "minimum": float(cv_frame[column].min()),
                "maximum": float(cv_frame[column].max()),
            }
            for column in summary_columns
        },
        "cross_validation_seconds": time.perf_counter() - cv_started,
        "selected_final_epochs": final_epochs,
        "final_training_seconds": final_training_seconds,
        "independent_states": len(independent),
        "independent_base_state_groups": int(independent["base_state_id"].nunique()),
        "independent_metrics": independent_metrics,
        "independent_subgroup_metrics": subgroup_metrics,
        "independent_evaluation_seconds": independent_seconds,
        "mean_online_ms_per_state_including_all_candidate_features_and_scoring": (
            independent_seconds * 1000.0 / len(independent)
        ),
        "model": str(model_path.resolve()),
    }
    (output_dir / "candidate_aware_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-states",
        type=Path,
        default=DEFAULT_DIR.parent / "oracle_training_controlled_k_5000x5.csv",
    )
    parser.add_argument(
        "--independent-states",
        type=Path,
        default=DEFAULT_DIR.parent / "oracle_independent_controlled_k_1000x5.csv",
    )
    parser.add_argument("--training-costs", type=Path, default=DEFAULT_DIR / "training_costs.npz")
    parser.add_argument(
        "--independent-costs", type=Path, default=DEFAULT_DIR / "independent_costs.npz"
    )
    parser.add_argument("--class-table", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--rate-table", type=Path, default=DEFAULT_RATE_TABLE_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DIR / "candidate_aware_training"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260820)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = train(
        training_states_path=args.training_states,
        independent_states_path=args.independent_states,
        training_costs_path=args.training_costs,
        independent_costs_path=args.independent_costs,
        class_table_path=args.class_table,
        rate_table_path=args.rate_table,
        output_dir=args.output_dir,
        folds=args.folds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
