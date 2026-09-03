"""Controlled pairwise fine-tuning experiment for the candidate-aware ranker."""

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
from sklearn.model_selection import StratifiedGroupKFold

from ml_policy.oracle_optimizer import DEFAULT_RATE_TABLE_PATH
from ml_policy.train_controlled_k_candidate_cost_ranker import (
    DEFAULT_CLASSES,
    DEFAULT_DIR,
    _build_model,
    _build_sampled_rows,
    _candidate_metadata,
    _evaluate,
    _features_for_state,
    _fit_scaler,
    _rate_groups,
    _scale,
)


def _build_pairs(
    states: pd.DataFrame,
    costs: np.ndarray,
    state_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    rates: dict[tuple[str, int], dict[str, np.ndarray]],
    *,
    nearest_negative_count: int,
    far_negative_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    x_better: list[np.ndarray] = []
    x_worse: list[np.ndarray] = []
    lb_better: list[np.ndarray] = []
    lb_worse: list[np.ndarray] = []
    target_better: list[np.ndarray] = []
    target_worse: list[np.ndarray] = []
    exact_gaps: list[np.ndarray] = []

    for position, state_index in enumerate(state_indices):
        row = states.iloc[int(state_index)]
        finite = np.flatnonzero(np.isfinite(costs[int(state_index)]))
        ordered = finite[np.argsort(costs[int(state_index), finite], kind="stable")]
        optimum = float(costs[int(state_index), ordered[0]])
        nonoptimal = ordered[costs[int(state_index), ordered] > optimum + 2e-5]
        if not len(nonoptimal):
            continue
        better_indices: list[int] = []
        worse_indices: list[int] = []

        # Focus directly on mistakes that can change the selected first place.
        for negative in nonoptimal[:nearest_negative_count]:
            better_indices.append(int(ordered[0]))
            worse_indices.append(int(negative))

        # Add adjacent hard pairs so the local top of the list remains ordered.
        distinct = [int(ordered[0])]
        previous_cost = optimum
        for candidate in nonoptimal:
            candidate_cost = float(costs[int(state_index), candidate])
            if candidate_cost > previous_cost + 2e-5:
                distinct.append(int(candidate))
                previous_cost = candidate_cost
            if len(distinct) >= 7:
                break
        for better, worse in zip(distinct[:-1], distinct[1:]):
            better_indices.append(better)
            worse_indices.append(worse)

        # Protect the scorer against promoting clearly inferior candidates while
        # it learns the fine ordering around the optimum.
        far_pool = nonoptimal[nearest_negative_count:]
        if far_negative_count and len(far_pool):
            far_positions = np.linspace(
                0,
                len(far_pool) - 1,
                num=min(far_negative_count, len(far_pool)),
                dtype=int,
            )
            for far_position in far_positions:
                better_indices.append(int(ordered[0]))
                worse_indices.append(int(far_pool[far_position]))

        unique_pairs = list(dict.fromkeys(zip(better_indices, worse_indices)))
        better_indices = [pair[0] for pair in unique_pairs]
        worse_indices = [pair[1] for pair in unique_pairs]

        better_array = np.asarray(better_indices, dtype=np.int32)
        worse_array = np.asarray(worse_indices, dtype=np.int32)
        condition_rates = rates[
            (str(row["wind_direction"]), int(row["wind_level"]))
        ]
        better_features, better_lb = _features_for_state(
            row, better_array, metadata, condition_rates
        )
        worse_features, worse_lb = _features_for_state(
            row, worse_array, metadata, condition_rates
        )
        better_exact = costs[int(state_index), better_array]
        worse_exact = costs[int(state_index), worse_array]
        x_better.append(better_features)
        x_worse.append(worse_features)
        lb_better.append(better_lb[:, None])
        lb_worse.append(worse_lb[:, None])
        target_better.append(
            np.log1p(np.maximum(better_exact - better_lb, 0.0))[:, None].astype(
                np.float32
            )
        )
        target_worse.append(
            np.log1p(np.maximum(worse_exact - worse_lb, 0.0))[:, None].astype(
                np.float32
            )
        )
        exact_gaps.append((worse_exact - better_exact)[:, None].astype(np.float32))
        if (position + 1) % 1000 == 0:
            print(f"pair features {position + 1}/{len(state_indices)} states", flush=True)

    return tuple(
        np.concatenate(values).astype(np.float32)
        for values in (
            x_better,
            x_worse,
            lb_better,
            lb_worse,
            target_better,
            target_worse,
            exact_gaps,
        )
    )  # type: ignore[return-value]


def _build_pair_model(
    scorer: tf.keras.Model,
    *,
    input_width: int,
    learning_rate: float,
    rank_scale_minutes: float,
    rank_loss_weight: float,
    residual_loss_weight: float,
) -> tf.keras.Model:
    better_x = tf.keras.Input(shape=(input_width,), name="better_candidate")
    worse_x = tf.keras.Input(shape=(input_width,), name="worse_candidate")
    better_lb = tf.keras.Input(shape=(1,), name="better_lower_bound")
    worse_lb = tf.keras.Input(shape=(1,), name="worse_lower_bound")
    better_residual = scorer(better_x)
    worse_residual = scorer(worse_x)

    def rank_logit(values):
        b_residual, w_residual, b_lb, w_lb = values
        b_total = b_lb + tf.math.expm1(tf.maximum(b_residual, 0.0))
        w_total = w_lb + tf.math.expm1(tf.maximum(w_residual, 0.0))
        return (w_total - b_total) / rank_scale_minutes

    ranking_logit = tf.keras.layers.Lambda(
        rank_logit, name="better_than_worse_logit"
    )([better_residual, worse_residual, better_lb, worse_lb])
    better_output = tf.keras.layers.Activation(
        "linear", name="better_residual"
    )(better_residual)
    worse_output = tf.keras.layers.Activation("linear", name="worse_residual")(
        worse_residual
    )
    model = tf.keras.Model(
        [better_x, worse_x, better_lb, worse_lb],
        {
            "better_than_worse_logit": ranking_logit,
            "better_residual": better_output,
            "worse_residual": worse_output,
        },
        name="candidate_aware_pairwise_finetuner",
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-4
        ),
        loss={
            "better_than_worse_logit": tf.keras.losses.BinaryCrossentropy(
                from_logits=True
            ),
            "better_residual": tf.keras.losses.Huber(delta=0.10),
            "worse_residual": tf.keras.losses.Huber(delta=0.10),
        },
        loss_weights={
            "better_than_worse_logit": rank_loss_weight,
            "better_residual": residual_loss_weight,
            "worse_residual": residual_loss_weight,
        },
    )
    return model


def experiment(
    *,
    states_path: Path,
    costs_path: Path,
    class_table_path: Path,
    rate_table_path: Path,
    output_dir: Path,
    regression_epochs: int,
    pairwise_epochs: int,
    random_seed: int,
    far_negative_count: int,
    gap_weight_cap_minutes: float,
    pairwise_learning_rate: float,
    rank_loss_weight: float,
    residual_loss_weight: float,
) -> dict[str, object]:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    states = pd.read_csv(states_path)
    costs = np.load(costs_path)["costs"]
    metadata = _candidate_metadata(pd.read_csv(class_table_path))
    rates = _rate_groups(rate_table_path)
    groups = states["base_state_id"].to_numpy()
    strata = (
        states["wind_direction"].astype(str)
        + "_lv"
        + states["wind_level"].astype(str)
    ).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_seed
    )
    fit_states, validation_states = next(splitter.split(states, strata, groups))
    if set(groups[fit_states]).intersection(set(groups[validation_states])):
        raise RuntimeError("base_state_id leakage")

    x, y, sample_weight, sample_state = _build_sampled_rows(
        states,
        costs,
        metadata,
        rates,
        random_seed=random_seed,
        hard_count=48,
        random_count=32,
    )
    fit_mask = np.isin(sample_state, fit_states)
    validation_mask = np.isin(sample_state, validation_states)
    scaler = _fit_scaler(x[fit_mask])
    scorer = _build_model(x.shape[1], random_seed)
    regression_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, min_delta=1e-5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]
    regression_started = time.perf_counter()
    regression_history = scorer.fit(
        _scale(x[fit_mask], scaler),
        y[fit_mask],
        sample_weight=sample_weight[fit_mask],
        validation_data=(
            _scale(x[validation_mask], scaler),
            y[validation_mask],
            sample_weight[validation_mask],
        ),
        epochs=regression_epochs,
        batch_size=2048,
        verbose=0,
        callbacks=regression_callbacks,
    )
    regression_seconds = time.perf_counter() - regression_started
    before_metrics, before_predictions = _evaluate(
        scorer,
        scaler,
        states,
        costs,
        validation_states,
        metadata,
        rates,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer.save(output_dir / "regression_baseline_candidate_ranker.keras")

    pair_data = _build_pairs(
        states,
        costs,
        fit_states,
        metadata,
        rates,
        nearest_negative_count=8,
        far_negative_count=far_negative_count,
    )
    validation_pair_data = _build_pairs(
        states,
        costs,
        validation_states,
        metadata,
        rates,
        nearest_negative_count=8,
        far_negative_count=far_negative_count,
    )
    bx, wx, blb, wlb, by, wy, gap = pair_data
    vbx, vwx, vblb, vwlb, vby, vwy, vgap = validation_pair_data
    pair_model = _build_pair_model(
        scorer,
        input_width=x.shape[1],
        learning_rate=pairwise_learning_rate,
        rank_scale_minutes=0.10,
        rank_loss_weight=rank_loss_weight,
        residual_loss_weight=residual_loss_weight,
    )
    ones = np.ones((len(bx), 1), dtype=np.float32)
    validation_ones = np.ones((len(vbx), 1), dtype=np.float32)
    if gap_weight_cap_minutes > 0:
        rank_sample_weight = 0.25 + 1.75 * np.sqrt(
            np.clip(gap / gap_weight_cap_minutes, 0.0, 1.0)
        )
        validation_rank_sample_weight = 0.25 + 1.75 * np.sqrt(
            np.clip(vgap / gap_weight_cap_minutes, 0.0, 1.0)
        )
        rank_sample_weight /= float(rank_sample_weight.mean())
        validation_rank_sample_weight /= float(
            validation_rank_sample_weight.mean()
        )
    else:
        rank_sample_weight = np.ones_like(gap, dtype=np.float32)
        validation_rank_sample_weight = np.ones_like(vgap, dtype=np.float32)
    pair_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, min_delta=1e-5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    pair_started = time.perf_counter()
    pair_history = pair_model.fit(
        [
            _scale(bx, scaler),
            _scale(wx, scaler),
            blb,
            wlb,
        ],
        {
            "better_than_worse_logit": ones,
            "better_residual": by,
            "worse_residual": wy,
        },
        sample_weight={
            "better_than_worse_logit": rank_sample_weight,
            "better_residual": np.ones_like(by, dtype=np.float32),
            "worse_residual": np.ones_like(wy, dtype=np.float32),
        },
        validation_data=(
            [
                _scale(vbx, scaler),
                _scale(vwx, scaler),
                vblb,
                vwlb,
            ],
            {
                "better_than_worse_logit": validation_ones,
                "better_residual": vby,
                "worse_residual": vwy,
            },
            {
                "better_than_worse_logit": validation_rank_sample_weight,
                "better_residual": np.ones_like(vby, dtype=np.float32),
                "worse_residual": np.ones_like(vwy, dtype=np.float32),
            },
        ),
        epochs=pairwise_epochs,
        batch_size=1024,
        verbose=0,
        callbacks=pair_callbacks,
    )
    pairwise_seconds = time.perf_counter() - pair_started
    after_metrics, after_predictions = _evaluate(
        scorer,
        scaler,
        states,
        costs,
        validation_states,
        metadata,
        rates,
    )

    scorer.save(output_dir / "pairwise_finetuned_candidate_ranker.keras")
    (output_dir / "feature_scaler.json").write_text(
        json.dumps(scaler, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(regression_history.history).to_csv(
        output_dir / "regression_history.csv", index=False
    )
    pd.DataFrame(pair_history.history).to_csv(
        output_dir / "pairwise_history.csv", index=False
    )
    before_predictions.to_csv(output_dir / "before_predictions.csv", index=False)
    after_predictions.to_csv(output_dir / "after_predictions.csv", index=False)
    report = {
        "status": "pass",
        "experiment": "same grouped fold before versus after pairwise fine-tuning",
        "fit_groups": len(set(groups[fit_states])),
        "validation_groups": len(set(groups[validation_states])),
        "group_leakage_count": 0,
        "regression_candidate_rows": int(fit_mask.sum()),
        "pairwise_training_pairs": len(bx),
        "pairwise_validation_pairs": len(vbx),
        "far_negative_count_per_state": far_negative_count,
        "gap_weight_cap_minutes": gap_weight_cap_minutes,
        "pairwise_learning_rate": pairwise_learning_rate,
        "rank_loss_weight": rank_loss_weight,
        "residual_loss_weight_each": residual_loss_weight,
        "training_gap_minutes": {
            "median": float(np.median(gap)),
            "p95": float(np.quantile(gap, 0.95)),
            "maximum": float(np.max(gap)),
        },
        "ranking_sample_weight": {
            "minimum": float(np.min(rank_sample_weight)),
            "median": float(np.median(rank_sample_weight)),
            "maximum": float(np.max(rank_sample_weight)),
        },
        "regression_epochs_completed": len(regression_history.history["loss"]),
        "pairwise_epochs_completed": len(pair_history.history["loss"]),
        "regression_seconds": regression_seconds,
        "pairwise_seconds": pairwise_seconds,
        "before_pairwise": before_metrics,
        "after_pairwise": after_metrics,
        "change": {
            key: float(after_metrics[key]) - float(before_metrics[key])
            for key in (
                "strict_global_optimal_rate",
                "within_0p1_minute_rate",
                "within_0p5_minute_rate",
                "mean_regret_minutes",
                "p95_regret_minutes",
                "maximum_regret_minutes",
            )
        },
    }
    (output_dir / "pairwise_experiment_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        type=Path,
        default=DEFAULT_DIR.parent / "oracle_training_controlled_k_5000x5.csv",
    )
    parser.add_argument("--costs", type=Path, default=DEFAULT_DIR / "training_costs.npz")
    parser.add_argument("--class-table", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--rate-table", type=Path, default=DEFAULT_RATE_TABLE_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DIR / "pairwise_experiment"
    )
    parser.add_argument("--regression-epochs", type=int, default=60)
    parser.add_argument("--pairwise-epochs", type=int, default=15)
    parser.add_argument("--random-seed", type=int, default=20260820)
    parser.add_argument("--far-negative-count", type=int, default=0)
    parser.add_argument("--gap-weight-cap-minutes", type=float, default=0.0)
    parser.add_argument("--pairwise-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--residual-loss-weight", type=float, default=0.15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = experiment(
        states_path=args.states,
        costs_path=args.costs,
        class_table_path=args.class_table,
        rate_table_path=args.rate_table,
        output_dir=args.output_dir,
        regression_epochs=args.regression_epochs,
        pairwise_epochs=args.pairwise_epochs,
        random_seed=args.random_seed,
        far_negative_count=args.far_negative_count,
        gap_weight_cap_minutes=args.gap_weight_cap_minutes,
        pairwise_learning_rate=args.pairwise_learning_rate,
        rank_loss_weight=args.rank_loss_weight,
        residual_loss_weight=args.residual_loss_weight,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
