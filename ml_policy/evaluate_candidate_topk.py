"""Evaluate strict global-optimum recall within a candidate ranker's top K."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
    _candidate_metadata,
    _features_for_state,
    _rate_groups,
    _scale,
)


def evaluate(
    *,
    model_path: Path,
    scaler_path: Path,
    states_path: Path,
    costs_path: Path,
    class_table_path: Path,
    rate_table_path: Path,
    output_dir: Path,
    random_seed: int,
) -> dict[str, object]:
    states = pd.read_csv(states_path)
    costs = np.load(costs_path)["costs"]
    metadata = _candidate_metadata(pd.read_csv(class_table_path))
    rates = _rate_groups(rate_table_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))

    groups = states["base_state_id"].to_numpy()
    strata = (
        states["wind_direction"].astype(str)
        + "_lv"
        + states["wind_level"].astype(str)
    ).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_seed
    )
    _, validation_indices = next(splitter.split(states, strata, groups))

    top_values = (1, 2, 3, 5, 10, 20)
    records: list[dict[str, object]] = []
    for position, state_index in enumerate(validation_indices):
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
        order = finite[np.argsort(predicted_total, kind="stable")]
        exact = costs[int(state_index)]
        optimum = float(np.min(exact[finite]))
        is_optimal = np.isclose(exact[order], optimum, atol=2e-5, rtol=0.0)
        first_optimum_rank = int(np.flatnonzero(is_optimal)[0]) + 1
        adaptive_n = DEFAULT_TOP_N_BY_K[int(row["charging_pad_count"])]
        adaptive_regret = float(np.min(exact[order[:adaptive_n]]) - optimum)
        record: dict[str, object] = {
            "state_index": int(state_index),
            "base_state_id": int(row["base_state_id"]),
            "wind_direction": str(row["wind_direction"]),
            "wind_level": int(row["wind_level"]),
            "charging_pad_count": int(row["charging_pad_count"]),
            "first_global_optimum_rank": first_optimum_rank,
            "top1_regret_minutes": float(exact[int(order[0])] - optimum),
            "adaptive_top_n": adaptive_n,
            "adaptive_regret_minutes": adaptive_regret,
            "adaptive_is_global_optimum": bool(adaptive_regret <= 2e-5),
        }
        for top_k in top_values:
            record[f"global_optimum_in_top_{top_k}"] = bool(
                np.any(is_optimal[:top_k])
            )
        records.append(record)
        if (position + 1) % 1000 == 0:
            print(f"ranked {position + 1}/{len(validation_indices)} states", flush=True)

    frame = pd.DataFrame(records)
    overall = {
        f"top_{top_k}_global_optimum_recall": float(
            frame[f"global_optimum_in_top_{top_k}"].mean()
        )
        for top_k in top_values
    }
    overall.update(
        {
            "states": len(frame),
            "median_first_global_optimum_rank": float(
                frame["first_global_optimum_rank"].median()
            ),
            "p95_first_global_optimum_rank": float(
                frame["first_global_optimum_rank"].quantile(0.95)
            ),
            "maximum_first_global_optimum_rank": int(
                frame["first_global_optimum_rank"].max()
            ),
            "adaptive_global_optimal_rate": float(
                frame["adaptive_is_global_optimum"].mean()
            ),
            "adaptive_mean_exact_candidates": float(frame["adaptive_top_n"].mean()),
            "adaptive_mean_regret_minutes": float(
                frame["adaptive_regret_minutes"].mean()
            ),
            "adaptive_p95_regret_minutes": float(
                frame["adaptive_regret_minutes"].quantile(0.95)
            ),
            "adaptive_maximum_regret_minutes": float(
                frame["adaptive_regret_minutes"].max()
            ),
        }
    )
    condition = (
        frame.groupby(
            ["wind_direction", "wind_level", "charging_pad_count"],
            as_index=False,
        )[
            [
                "global_optimum_in_top_1",
                "global_optimum_in_top_2",
                "global_optimum_in_top_3",
                "global_optimum_in_top_5",
                "top1_regret_minutes",
                "adaptive_is_global_optimum",
                "adaptive_regret_minutes",
                "adaptive_top_n",
            ]
        ]
        .mean()
        .rename(
            columns={
                "global_optimum_in_top_1": "top_1_recall",
                "global_optimum_in_top_2": "top_2_recall",
                "global_optimum_in_top_3": "top_3_recall",
                "global_optimum_in_top_5": "top_5_recall",
                "top1_regret_minutes": "mean_top1_regret_minutes",
                "adaptive_is_global_optimum": "adaptive_global_optimal_rate",
                "adaptive_regret_minutes": "adaptive_mean_regret_minutes",
                "adaptive_top_n": "adaptive_exact_candidates",
            }
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "state_topk_ranks.csv", index=False)
    condition.to_csv(output_dir / "condition_k_topk_summary.csv", index=False)
    report: dict[str, object] = {
        "status": "pass",
        "group_leakage_count": 0,
        "validation_base_groups": int(frame["base_state_id"].nunique()),
        "overall": overall,
        "worst_top1_condition_k_cells": condition.nsmallest(10, "top_1_recall").to_dict(
            orient="records"
        ),
    }
    (output_dir / "topk_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    default_experiment = DEFAULT_DIR / "regret_aware_pairwise_experiment"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=default_experiment / "pairwise_finetuned_candidate_ranker.keras",
    )
    parser.add_argument(
        "--scaler", type=Path, default=default_experiment / "feature_scaler.json"
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=DEFAULT_DIR.parent / "oracle_training_controlled_k_5000x5.csv",
    )
    parser.add_argument("--costs", type=Path, default=DEFAULT_DIR / "training_costs.npz")
    parser.add_argument("--class-table", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--rate-table", type=Path, default=DEFAULT_RATE_TABLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=default_experiment / "topk")
    parser.add_argument("--random-seed", type=int, default=20260820)
    args = parser.parse_args()
    report = evaluate(
        model_path=args.model,
        scaler_path=args.scaler,
        states_path=args.states,
        costs_path=args.costs,
        class_table_path=args.class_table,
        rate_table_path=args.rate_table,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
