"""Find robust adaptive shortlist sizes from saved out-of-fold neural rankings."""

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

from ml_policy.oracle_optimizer import DEFAULT_RATE_TABLE_PATH
from ml_policy.train_controlled_k_candidate_cost_ranker import (
    DEFAULT_CLASSES,
    DEFAULT_DIR,
    _candidate_metadata,
    _features_for_state,
    _rate_groups,
    _scale,
)


def analyze(
    *,
    states_path: Path,
    costs_path: Path,
    class_table_path: Path,
    rate_table_path: Path,
    cv_dir: Path,
    target_cell_recall: float,
) -> dict[str, object]:
    states = pd.read_csv(states_path)
    costs = np.load(costs_path)["costs"]
    metadata = _candidate_metadata(pd.read_csv(class_table_path))
    rates = _rate_groups(rate_table_path)
    assignments = pd.read_csv(cv_dir / "all_out_of_fold_predictions.csv")
    records: list[dict[str, object]] = []

    for fold_number, fold_rows in assignments.groupby("fold", sort=True):
        fold_dir = cv_dir / f"fold_{int(fold_number)}"
        model = tf.keras.models.load_model(
            fold_dir / "candidate_ranker.keras", compile=False
        )
        scaler = json.loads(
            (fold_dir / "feature_scaler.json").read_text(encoding="utf-8")
        )
        for position, state_index in enumerate(
            fold_rows["state_index"].to_numpy(dtype=int), start=1
        ):
            row = states.iloc[state_index]
            finite = np.flatnonzero(np.isfinite(costs[state_index])).astype(np.int32)
            features, lower_bound = _features_for_state(
                row,
                finite,
                metadata,
                rates[(str(row["wind_direction"]), int(row["wind_level"]))],
            )
            predicted_log = model(_scale(features, scaler), training=False).numpy().reshape(-1)
            predicted_total = lower_bound + np.expm1(np.maximum(predicted_log, 0.0))
            ordered = finite[np.argsort(predicted_total, kind="stable")]
            exact = costs[state_index]
            optimum = float(np.min(exact[finite]))
            optimal = np.isclose(exact[ordered], optimum, atol=2e-5, rtol=0.0)
            first_rank = int(np.flatnonzero(optimal)[0]) + 1
            records.append(
                {
                    "fold": int(fold_number),
                    "state_index": state_index,
                    "base_state_id": int(row["base_state_id"]),
                    "wind_direction": str(row["wind_direction"]),
                    "wind_level": int(row["wind_level"]),
                    "charging_pad_count": int(row["charging_pad_count"]),
                    "first_global_optimum_rank": first_rank,
                }
            )
            if position % 1000 == 0:
                print(f"fold {fold_number}: ranked {position}/{len(fold_rows)}", flush=True)
        del model
        tf.keras.backend.clear_session()

    frame = pd.DataFrame(records)
    cells: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["wind_direction", "wind_level", "charging_pad_count"], sort=True
    ):
        ranks = group["first_global_optimum_rank"].to_numpy(dtype=int)
        required_n = int(np.quantile(ranks, target_cell_recall, method="higher"))
        cells.append(
            {
                "wind_direction": keys[0],
                "wind_level": int(keys[1]),
                "charging_pad_count": int(keys[2]),
                "states": len(group),
                "required_top_n": required_n,
                "achieved_recall": float(np.mean(ranks <= required_n)),
            }
        )
    cell_frame = pd.DataFrame(cells)
    robust_by_k = {
        int(k): int(group["required_top_n"].max())
        for k, group in cell_frame.groupby("charging_pad_count")
    }
    frame["recommended_top_n"] = frame["charging_pad_count"].map(robust_by_k)
    frame["recommended_policy_is_global_optimum"] = (
        frame["first_global_optimum_rank"] <= frame["recommended_top_n"]
    )
    overall_recall = float(frame["recommended_policy_is_global_optimum"].mean())
    by_cell_recall = (
        frame.groupby(
            ["wind_direction", "wind_level", "charging_pad_count"], as_index=False
        )["recommended_policy_is_global_optimum"]
        .mean()
        .rename(columns={"recommended_policy_is_global_optimum": "recall"})
    )
    output_dir = cv_dir / "shortlist_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "out_of_fold_optimum_ranks.csv", index=False)
    cell_frame.to_csv(output_dir / "required_top_n_by_condition_k.csv", index=False)
    by_cell_recall.to_csv(output_dir / "recommended_policy_recall_by_condition_k.csv", index=False)
    report: dict[str, object] = {
        "status": "pass",
        "target_minimum_condition_k_recall": target_cell_recall,
        "recommended_top_n_by_k": robust_by_k,
        "mean_exact_candidates": float(frame["recommended_top_n"].mean()),
        "overall_out_of_fold_global_optimal_rate": overall_recall,
        "minimum_condition_k_global_optimal_rate": float(by_cell_recall["recall"].min()),
        "maximum_required_top_n": int(max(robust_by_k.values())),
    }
    (output_dir / "shortlist_recommendation.json").write_text(
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
        "--cv-dir", type=Path, default=DEFAULT_DIR / "adaptive_topn_cross_validation"
    )
    parser.add_argument("--target-cell-recall", type=float, default=0.90)
    args = parser.parse_args()
    report = analyze(
        states_path=args.states,
        costs_path=args.costs,
        class_table_path=args.class_table,
        rate_table_path=args.rate_table,
        cv_dir=args.cv_dir,
        target_cell_recall=args.target_cell_recall,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
