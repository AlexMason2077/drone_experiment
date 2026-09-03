"""Grouped cross-validation for regret-aware adaptive Top-N configuration policy."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/drone-matplotlib-cache")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from ml_policy.adaptive_topn_policy import DEFAULT_TOP_N_BY_K
from ml_policy.oracle_optimizer import DEFAULT_RATE_TABLE_PATH
from ml_policy.train_controlled_k_candidate_cost_ranker import (
    DEFAULT_CLASSES,
    DEFAULT_DIR,
    _build_model,
    _build_sampled_rows,
    _candidate_metadata,
    _features_for_state,
    _fit_scaler,
    _rate_groups,
    _scale,
)
from ml_policy.train_controlled_k_pairwise_ranker import (
    _build_pair_model,
    _build_pairs,
)


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    regret = frame["adaptive_regret_minutes"].to_numpy(dtype=float)
    return {
        "states": len(frame),
        "top1_global_optimal_rate": float(frame["top1_is_global_optimum"].mean()),
        "adaptive_global_optimal_rate": float(frame["adaptive_is_global_optimum"].mean()),
        "adaptive_mean_exact_candidates": float(frame["adaptive_top_n"].mean()),
        "adaptive_mean_regret_minutes": float(regret.mean()),
        "adaptive_p95_regret_minutes": float(np.quantile(regret, 0.95)),
        "adaptive_maximum_regret_minutes": float(regret.max()),
    }


def _evaluate_adaptive(
    model: tf.keras.Model,
    scaler: dict[str, object],
    states: pd.DataFrame,
    costs: np.ndarray,
    state_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    rates: dict[tuple[str, int], dict[str, np.ndarray]],
    *,
    fold_number: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    records: list[dict[str, object]] = []
    for position, state_index in enumerate(state_indices):
        row = states.iloc[int(state_index)]
        finite = np.flatnonzero(np.isfinite(costs[int(state_index)])).astype(np.int32)
        features, lower_bound = _features_for_state(
            row,
            finite,
            metadata,
            rates[(str(row["wind_direction"]), int(row["wind_level"]))],
        )
        predicted_log = model(_scale(features, scaler), training=False).numpy().reshape(-1)
        predicted_total = lower_bound + np.expm1(np.maximum(predicted_log, 0.0))
        ordered = finite[np.argsort(predicted_total, kind="stable")]
        exact = costs[int(state_index)]
        optimum = float(np.min(exact[finite]))
        top1_regret = float(exact[int(ordered[0])] - optimum)
        adaptive_n = DEFAULT_TOP_N_BY_K[int(row["charging_pad_count"])]
        shortlist = ordered[:adaptive_n]
        selected_class = int(shortlist[int(np.argmin(exact[shortlist]))])
        adaptive_regret = float(exact[selected_class] - optimum)
        records.append(
            {
                "fold": fold_number,
                "state_index": int(state_index),
                "base_state_id": int(row["base_state_id"]),
                "wind_direction": str(row["wind_direction"]),
                "wind_level": int(row["wind_level"]),
                "charging_pad_count": int(row["charging_pad_count"]),
                "top1_class_index": int(ordered[0]),
                "top1_is_global_optimum": bool(top1_regret <= 2e-5),
                "top1_regret_minutes": top1_regret,
                "adaptive_top_n": adaptive_n,
                "adaptive_selected_class_index": selected_class,
                "adaptive_is_global_optimum": bool(adaptive_regret <= 2e-5),
                "adaptive_regret_minutes": adaptive_regret,
            }
        )
        if (position + 1) % 1000 == 0:
            print(
                f"fold {fold_number}: evaluated {position + 1}/{len(state_indices)} states",
                flush=True,
            )
    frame = pd.DataFrame(records)
    return _metrics(frame), frame


def run_cross_validation(
    *,
    states_path: Path,
    costs_path: Path,
    class_table_path: Path,
    rate_table_path: Path,
    output_dir: Path,
    folds: int,
    max_folds: int | None,
    regression_epochs: int,
    pairwise_epochs: int,
    random_seed: int,
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
        n_splits=folds, shuffle=True, random_state=random_seed
    )
    splits = list(splitter.split(states, strata, groups))
    if max_folds is not None:
        splits = splits[:max_folds]

    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, sample_weight, sample_state = _build_sampled_rows(
        states,
        costs,
        metadata,
        rates,
        random_seed=random_seed,
        hard_count=48,
        random_count=32,
    )
    fold_records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    started = time.perf_counter()

    for fold_number, (fit_states, validation_states) in enumerate(splits, start=1):
        fold_started = time.perf_counter()
        if set(groups[fit_states]).intersection(set(groups[validation_states])):
            raise RuntimeError(f"base_state_id leakage in fold {fold_number}")
        print(f"=== adaptive regret-aware fold {fold_number}/{len(splits)} ===", flush=True)
        fold_dir = output_dir / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fit_mask = np.isin(sample_state, fit_states)
        validation_mask = np.isin(sample_state, validation_states)
        scaler = _fit_scaler(x[fit_mask])
        scorer = _build_model(x.shape[1], random_seed + fold_number)
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
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    min_delta=1e-5,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
                ),
            ],
        )

        pair_data = _build_pairs(
            states,
            costs,
            fit_states,
            metadata,
            rates,
            nearest_negative_count=8,
            far_negative_count=4,
        )
        validation_pair_data = _build_pairs(
            states,
            costs,
            validation_states,
            metadata,
            rates,
            nearest_negative_count=8,
            far_negative_count=4,
        )
        bx, wx, blb, wlb, by, wy, gap = pair_data
        vbx, vwx, vblb, vwlb, vby, vwy, vgap = validation_pair_data
        rank_weight = 0.25 + 1.75 * np.sqrt(np.clip(gap / 0.5, 0.0, 1.0))
        validation_rank_weight = 0.25 + 1.75 * np.sqrt(
            np.clip(vgap / 0.5, 0.0, 1.0)
        )
        rank_weight /= float(rank_weight.mean())
        validation_rank_weight /= float(validation_rank_weight.mean())
        pair_model = _build_pair_model(
            scorer,
            input_width=x.shape[1],
            learning_rate=5e-5,
            rank_scale_minutes=0.10,
            rank_loss_weight=0.5,
            residual_loss_weight=0.5,
        )
        pair_history = pair_model.fit(
            [_scale(bx, scaler), _scale(wx, scaler), blb, wlb],
            {
                "better_than_worse_logit": np.ones_like(gap, dtype=np.float32),
                "better_residual": by,
                "worse_residual": wy,
            },
            sample_weight={
                "better_than_worse_logit": rank_weight,
                "better_residual": np.ones_like(by, dtype=np.float32),
                "worse_residual": np.ones_like(wy, dtype=np.float32),
            },
            validation_data=(
                [_scale(vbx, scaler), _scale(vwx, scaler), vblb, vwlb],
                {
                    "better_than_worse_logit": np.ones_like(vgap, dtype=np.float32),
                    "better_residual": vby,
                    "worse_residual": vwy,
                },
                {
                    "better_than_worse_logit": validation_rank_weight,
                    "better_residual": np.ones_like(vby, dtype=np.float32),
                    "worse_residual": np.ones_like(vwy, dtype=np.float32),
                },
            ),
            epochs=pairwise_epochs,
            batch_size=1024,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=4,
                    min_delta=1e-5,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
                ),
            ],
        )
        metrics, predictions = _evaluate_adaptive(
            scorer,
            scaler,
            states,
            costs,
            validation_states,
            metadata,
            rates,
            fold_number=fold_number,
        )
        predictions.to_csv(fold_dir / "predictions.csv", index=False)
        pd.DataFrame(regression_history.history).to_csv(
            fold_dir / "regression_history.csv", index=False
        )
        pd.DataFrame(pair_history.history).to_csv(
            fold_dir / "pairwise_history.csv", index=False
        )
        scorer.save(fold_dir / "candidate_ranker.keras")
        (fold_dir / "feature_scaler.json").write_text(
            json.dumps(scaler, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record: dict[str, object] = {
            "fold": fold_number,
            "fit_base_groups": len(set(groups[fit_states])),
            "validation_base_groups": len(set(groups[validation_states])),
            "regression_epochs": len(regression_history.history["loss"]),
            "pairwise_epochs": len(pair_history.history["loss"]),
            "elapsed_seconds": time.perf_counter() - fold_started,
            **metrics,
        }
        fold_records.append(record)
        prediction_frames.append(predictions)
        print(json.dumps(record, sort_keys=True), flush=True)
        del pair_model, scorer, pair_data, validation_pair_data
        del bx, wx, blb, wlb, by, wy, gap, vbx, vwx, vblb, vwlb, vby, vwy, vgap
        tf.keras.backend.clear_session()
        gc.collect()

    fold_frame = pd.DataFrame(fold_records)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    overall = _metrics(all_predictions)
    by_k = (
        all_predictions.groupby("charging_pad_count")
        .apply(lambda frame: pd.Series(_metrics(frame)), include_groups=False)
        .reset_index()
    )
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    all_predictions.to_csv(output_dir / "all_out_of_fold_predictions.csv", index=False)
    by_k.to_csv(output_dir / "out_of_fold_metrics_by_k.csv", index=False)
    report: dict[str, object] = {
        "status": "pass",
        "requested_folds": folds,
        "completed_folds": len(splits),
        "group_leakage_count": 0,
        "adaptive_top_n_by_k": DEFAULT_TOP_N_BY_K,
        "overall_out_of_fold": overall,
        "fold_mean_adaptive_global_optimal_rate": float(
            fold_frame["adaptive_global_optimal_rate"].mean()
        ),
        "fold_std_adaptive_global_optimal_rate": float(
            fold_frame["adaptive_global_optimal_rate"].std(ddof=1)
        ) if len(fold_frame) > 1 else 0.0,
        "minimum_fold_adaptive_global_optimal_rate": float(
            fold_frame["adaptive_global_optimal_rate"].min()
        ),
        "total_seconds": time.perf_counter() - started,
    }
    (output_dir / "cross_validation_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
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
        "--output-dir",
        type=Path,
        default=DEFAULT_DIR / "adaptive_topn_cross_validation",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--regression-epochs", type=int, default=60)
    parser.add_argument("--pairwise-epochs", type=int, default=15)
    parser.add_argument("--random-seed", type=int, default=20260820)
    args = parser.parse_args()
    report = run_cross_validation(
        states_path=args.states,
        costs_path=args.costs,
        class_table_path=args.class_table,
        rate_table_path=args.rate_table,
        output_dir=args.output_dir,
        folds=args.folds,
        max_folds=args.max_folds,
        regression_epochs=args.regression_epochs,
        pairwise_epochs=args.pairwise_epochs,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
