"""Train and evaluate a grouped cost-sensitive neural configuration ranker."""

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
from sklearn.model_selection import StratifiedGroupKFold


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
WIND_DIRECTIONS = ("head", "side", "tail")
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


def _add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    soc = result[[f"soc_d{index}" for index in range(1, 6)]].to_numpy(dtype=float)
    ordered = np.sort(soc, axis=1)
    names = (
        "soc_lowest",
        "soc_second_lowest",
        "soc_middle",
        "soc_second_highest",
        "soc_highest",
    )
    for index, name in enumerate(names):
        result[name] = ordered[:, index]
    result["soc_range"] = ordered[:, -1] - ordered[:, 0]
    return result


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
    numeric = frame.loc[:, list(preprocessor["numeric_features"])].to_numpy(
        dtype=np.float32
    )
    numeric = (
        numeric - np.asarray(preprocessor["numeric_mean"], dtype=np.float32)
    ) / np.asarray(preprocessor["numeric_scale"], dtype=np.float32)
    wind_values = frame["wind_direction"].astype(str).to_numpy()
    wind = np.column_stack(
        [wind_values == value for value in preprocessor["wind_directions"]]
    ).astype(np.float32)
    return np.concatenate([numeric, wind], axis=1)


def _soft_cost_targets(costs: np.ndarray, temperature_minutes: float) -> np.ndarray:
    finite = np.isfinite(costs)
    optimum = np.min(np.where(finite, costs, np.inf), axis=1, keepdims=True)
    logits = np.where(finite, -(costs - optimum) / temperature_minutes, -1e9)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits, dtype=np.float64) * finite
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


def _build_model(
    *,
    input_width: int,
    class_count: int,
    random_seed: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(random_seed)
    inputs = tf.keras.Input(shape=(input_width,), name="observed_state")
    x = inputs
    for layer_index, width in enumerate((512, 512, 256), start=1):
        x = tf.keras.layers.Dense(
            width, activation=tf.nn.gelu, name=f"dense_{layer_index}"
        )(x)
        x = tf.keras.layers.LayerNormalization(name=f"norm_{layer_index}")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"dropout_{layer_index}")(x)
    logits = tf.keras.layers.Dense(
        class_count, name="complete_configuration_logits"
    )(x)
    model = tf.keras.Model(inputs, logits, name="controlled_k_cost_ranker")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )
    return model


def _select(logits: np.ndarray, costs: np.ndarray) -> np.ndarray:
    return np.argmax(np.where(np.isfinite(costs), logits, -np.inf), axis=1)


def _metrics(selected: np.ndarray, costs: np.ndarray) -> dict[str, float]:
    row = np.arange(len(costs))
    optimum = np.min(np.where(np.isfinite(costs), costs, np.inf), axis=1)
    selected_cost = costs[row, selected]
    regret = selected_cost - optimum
    return {
        "states": int(len(costs)),
        "strict_global_optimal_rate": float(
            np.mean(np.isclose(regret, 0.0, atol=2e-5))
        ),
        "within_0p1_minute_rate": float(np.mean(regret <= 0.1 + 2e-5)),
        "within_0p5_minute_rate": float(np.mean(regret <= 0.5 + 2e-5)),
        "mean_regret_minutes": float(regret.mean()),
        "median_regret_minutes": float(np.median(regret)),
        "p95_regret_minutes": float(np.quantile(regret, 0.95)),
        "maximum_regret_minutes": float(regret.max()),
        "unsafe_or_infeasible_selection_count": int(
            np.sum(~np.isfinite(selected_cost))
        ),
    }


def _subgroup_metrics(
    frame: pd.DataFrame, selected: np.ndarray, costs: np.ndarray
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for (direction, level, k), indices in frame.groupby(
        ["wind_direction", "wind_level", "charging_pad_count"], sort=True
    ).indices.items():
        index = np.asarray(indices, dtype=int)
        result[f"{direction}_lv{int(level)}_k{int(k)}"] = _metrics(
            selected[index], costs[index]
        )
    return result


def train_grouped_ranker(
    *,
    training_states_path: Path,
    independent_states_path: Path,
    training_costs_path: Path,
    independent_costs_path: Path,
    class_table_path: Path,
    output_dir: Path,
    folds: int,
    seeds: Sequence[int],
    max_epochs: int,
    patience: int,
    temperature_minutes: float,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    batch_size: int,
) -> dict[str, object]:
    random.seed(seeds[0])
    np.random.seed(seeds[0])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    states = _add_derived_features(pd.read_csv(training_states_path))
    independent = _add_derived_features(pd.read_csv(independent_states_path))
    training_costs = np.load(training_costs_path)["costs"]
    independent_costs = np.load(independent_costs_path)["costs"]
    classes = pd.read_csv(class_table_path).sort_values("class_index")
    if training_costs.shape != (len(states), len(classes)):
        raise ValueError("Training cost matrix shape does not match states/classes")
    if independent_costs.shape != (len(independent), len(classes)):
        raise ValueError("Independent cost matrix shape does not match states/classes")

    output_dir.mkdir(parents=True, exist_ok=True)
    histories_dir = output_dir / "cross_validation_histories"
    histories_dir.mkdir(parents=True, exist_ok=True)
    groups = states["base_state_id"].to_numpy()
    strata = (
        states["wind_direction"].astype(str)
        + "_lv"
        + states["wind_level"].astype(str)
    ).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seeds[0]
    )
    splits = list(splitter.split(states, strata, groups))
    fold_records: list[dict[str, object]] = []
    cv_prediction_rows: list[pd.DataFrame] = []
    completed_epochs: list[int] = []
    cv_started = time.perf_counter()

    for fold_number, (fit_indices, validation_indices) in enumerate(splits, start=1):
        fit_groups = set(groups[fit_indices])
        validation_groups = set(groups[validation_indices])
        if fit_groups.intersection(validation_groups):
            raise RuntimeError("base_state_id leakage between fit and validation")
        preprocessor = _fit_preprocessor(states.iloc[fit_indices])
        x_fit = _transform(states.iloc[fit_indices], preprocessor)
        x_validation = _transform(states.iloc[validation_indices], preprocessor)
        fit_costs = training_costs[fit_indices]
        validation_costs = training_costs[validation_indices]
        y_fit = _soft_cost_targets(fit_costs, temperature_minutes)
        y_validation = _soft_cost_targets(validation_costs, temperature_minutes)

        for seed_number, seed in enumerate(seeds, start=1):
            print(
                f"=== fold {fold_number}/{folds}, seed {seed_number}/{len(seeds)} "
                f"({seed}) ===",
                flush=True,
            )
            model = _build_model(
                input_width=x_fit.shape[1],
                class_count=len(classes),
                random_seed=seed + fold_number * 100,
                learning_rate=learning_rate,
                dropout=dropout,
                weight_decay=weight_decay,
            )
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
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
                validation_data=(x_validation, y_validation),
                epochs=max_epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
            )
            elapsed = time.perf_counter() - started
            logits = model.predict(x_validation, batch_size=512, verbose=0)
            selected = _select(logits, validation_costs)
            run_metrics = _metrics(selected, validation_costs)
            epochs = len(history.history["loss"])
            completed_epochs.append(epochs)
            record: dict[str, object] = {
                "fold": fold_number,
                "seed": seed,
                "fit_groups": len(fit_groups),
                "validation_groups": len(validation_groups),
                "fit_states": len(fit_indices),
                "validation_states": len(validation_indices),
                "epochs_completed": epochs,
                "best_val_loss": float(min(history.history["val_loss"])),
                "elapsed_seconds": elapsed,
                **run_metrics,
            }
            fold_records.append(record)
            pd.DataFrame(history.history).to_csv(
                histories_dir / f"fold_{fold_number}_seed_{seed}.csv", index=False
            )
            prediction = states.iloc[validation_indices][
                [
                    "base_state_id",
                    "wind_direction",
                    "wind_level",
                    "charging_pad_count",
                    "remaining_distance_m",
                ]
            ].copy()
            prediction["fold"] = fold_number
            prediction["seed"] = seed
            prediction["selected_class_index"] = selected
            optimum = np.min(
                np.where(np.isfinite(validation_costs), validation_costs, np.inf),
                axis=1,
            )
            selected_cost = validation_costs[
                np.arange(len(validation_indices)), selected
            ]
            prediction["selected_total_minutes"] = selected_cost
            prediction["global_optimum_minutes"] = optimum
            prediction["regret_minutes"] = selected_cost - optimum
            cv_prediction_rows.append(prediction)
            print(json.dumps(record, sort_keys=True), flush=True)
            tf.keras.backend.clear_session()

    pd.DataFrame(fold_records).to_csv(
        output_dir / "grouped_cross_validation_metrics.csv", index=False
    )
    pd.concat(cv_prediction_rows, ignore_index=True).to_csv(
        output_dir / "grouped_cross_validation_predictions.csv", index=False
    )

    # Train one deployable model after cross-validation has fixed the epoch count.
    final_epochs = max(1, int(round(float(np.median(completed_epochs)))))
    final_preprocessor = _fit_preprocessor(states)
    x_all = _transform(states, final_preprocessor)
    y_all = _soft_cost_targets(training_costs, temperature_minutes)
    final_model = _build_model(
        input_width=x_all.shape[1],
        class_count=len(classes),
        random_seed=seeds[0] + 9999,
        learning_rate=learning_rate,
        dropout=dropout,
        weight_decay=weight_decay,
    )
    final_started = time.perf_counter()
    final_history = final_model.fit(
        x_all,
        y_all,
        epochs=final_epochs,
        batch_size=batch_size,
        verbose=0,
    )
    final_training_seconds = time.perf_counter() - final_started
    model_path = output_dir / "controlled_k_cost_ranker.keras"
    final_model.save(model_path)
    (output_dir / "preprocessor.json").write_text(
        json.dumps(final_preprocessor, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(final_history.history).to_csv(
        output_dir / "final_training_history.csv", index=False
    )

    x_independent = _transform(independent, final_preprocessor)
    independent_logits = final_model.predict(x_independent, batch_size=512, verbose=0)
    independent_selected = _select(independent_logits, independent_costs)
    independent_metrics = _metrics(independent_selected, independent_costs)
    subgroup_metrics = _subgroup_metrics(
        independent.reset_index(drop=True), independent_selected, independent_costs
    )
    row = np.arange(len(independent))
    optimum = np.min(
        np.where(np.isfinite(independent_costs), independent_costs, np.inf), axis=1
    )
    selected_cost = independent_costs[row, independent_selected]
    prediction = independent[
        [
            "base_state_id",
            "wind_direction",
            "wind_level",
            "charging_pad_count",
            "remaining_distance_m",
        ]
    ].copy()
    prediction["selected_class_index"] = independent_selected
    prediction["selected_structure"] = classes.iloc[independent_selected][
        "structure"
    ].to_numpy()
    for drone in range(1, 6):
        prediction[f"selected_slot_index_d{drone}"] = classes.iloc[
            independent_selected
        ][f"slot_index_d{drone}"].to_numpy()
    prediction["selected_total_minutes"] = selected_cost
    prediction["global_optimum_minutes"] = optimum
    prediction["regret_minutes"] = selected_cost - optimum
    prediction.to_csv(output_dir / "independent_predictions.csv", index=False)

    _ = final_model(x_independent[:1], training=False).numpy()
    benchmark_count = min(500, len(independent))
    benchmark_started = time.perf_counter()
    for index in range(benchmark_count):
        one_logits = final_model(
            x_independent[index : index + 1], training=False
        ).numpy()[0]
        _ = int(
            np.argmax(
                np.where(np.isfinite(independent_costs[index]), one_logits, -np.inf)
            )
        )
    online_ms = (
        (time.perf_counter() - benchmark_started) * 1000.0 / benchmark_count
    )

    metric_frame = pd.DataFrame(fold_records)
    summary_columns = [
        "strict_global_optimal_rate",
        "within_0p1_minute_rate",
        "within_0p5_minute_rate",
        "mean_regret_minutes",
        "p95_regret_minutes",
        "maximum_regret_minutes",
    ]
    cv_summary = {
        column: {
            "mean": float(metric_frame[column].mean()),
            "standard_deviation": float(metric_frame[column].std(ddof=1)),
            "minimum": float(metric_frame[column].min()),
            "maximum": float(metric_frame[column].max()),
        }
        for column in summary_columns
    }
    report: dict[str, object] = {
        "status": "pass",
        "method": "grouped cost-sensitive listwise neural ranking of complete C=(f,d,p)",
        "training_states": len(states),
        "training_base_state_groups": int(states["base_state_id"].nunique()),
        "independent_states": len(independent),
        "independent_base_state_groups": int(independent["base_state_id"].nunique()),
        "configuration_classes": len(classes),
        "folds": folds,
        "seeds": list(seeds),
        "training_runs": folds * len(seeds),
        "group_leakage_count": 0,
        "hyperparameters": {
            "hidden_layers": [512, 512, 256],
            "dropout": dropout,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "temperature_minutes": temperature_minutes,
            "maximum_epochs": max_epochs,
            "early_stopping_patience": patience,
            "batch_size": batch_size,
        },
        "cross_validation_summary": cv_summary,
        "cross_validation_seconds": time.perf_counter() - cv_started,
        "selected_final_epochs": final_epochs,
        "final_training_seconds": final_training_seconds,
        "independent_metrics": independent_metrics,
        "independent_subgroup_metrics": subgroup_metrics,
        "warm_online_ms_per_state": online_ms,
        "model": str(model_path.resolve()),
        "training_costs": str(training_costs_path.resolve()),
        "independent_costs": str(independent_costs_path.resolve()),
    }
    (output_dir / "controlled_k_cost_ranker_metrics.json").write_text(
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR / "grouped_training")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(20260820, 20260821, 20260822)
    )
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--temperature-minutes", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = train_grouped_ranker(
        training_states_path=args.training_states,
        independent_states_path=args.independent_states,
        training_costs_path=args.training_costs,
        independent_costs_path=args.independent_costs,
        class_table_path=args.class_table,
        output_dir=args.output_dir,
        folds=args.folds,
        seeds=tuple(args.seeds),
        max_epochs=args.max_epochs,
        patience=args.patience,
        temperature_minutes=args.temperature_minutes,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
