"""Train a scalable pair-scoring model for Hungarian position decoding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_policy.pairwise_position_policy import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    expand_candidate_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_TRAINING = DEFAULT_DIR / "position_aware_training_candidates.csv"


def _decoded_exact_rate(model: Pipeline, candidate_frame: pd.DataFrame) -> float:
    pair_frame = expand_candidate_rows(candidate_frame, include_target=True)
    probability = model.predict_proba(pair_frame[FEATURES])[:, 1]
    pair_frame["assignment_probability"] = probability
    exact: list[bool] = []
    for candidate_index, group in pair_frame.groupby("candidate_index", sort=False):
        scores = group["assignment_probability"].to_numpy().reshape(5, 5)
        drones, slots = linear_sum_assignment(-scores)
        predicted = np.empty(5, dtype=int)
        predicted[drones] = slots + 1
        row = candidate_frame.loc[candidate_index]
        truth = np.asarray(
            [int(row[f"assigned_slot_index_d{index}"]) for index in range(1, 6)]
        )
        exact.append(bool(np.array_equal(predicted, truth)))
    return float(np.mean(exact))


def train_pairwise_position_policy(
    training_csv: Path,
    output_dir: Path,
    *,
    random_seed: int,
) -> dict[str, object]:
    candidates = pd.read_csv(training_csv)
    candidates = candidates[candidates["eligible_for_selection"].eq(1)].copy()
    scenario_ids = np.asarray(sorted(candidates["scenario_id"].unique()))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(scenario_ids)
    validation_count = max(1, int(round(0.20 * len(scenario_ids))))
    validation_ids = set(int(value) for value in scenario_ids[:validation_count])
    train_candidates = candidates[~candidates["scenario_id"].isin(validation_ids)].copy()
    validation_candidates = candidates[candidates["scenario_id"].isin(validation_ids)].copy()

    started_expand = time.perf_counter()
    train_pairs = expand_candidate_rows(train_candidates, include_target=True)
    validation_pairs = expand_candidate_rows(validation_candidates, include_target=True)
    expansion_seconds = time.perf_counter() - started_expand

    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=0.2,
        random_state=random_seed,
    )
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    target = train_pairs["is_assigned"].to_numpy(dtype=int)
    sample_weight = np.where(target == 1, 4.0, 1.0)
    started_train = time.perf_counter()
    model.fit(
        train_pairs[FEATURES],
        target,
        classifier__sample_weight=sample_weight,
    )
    training_seconds = time.perf_counter() - started_train

    validation_probability = model.predict_proba(validation_pairs[FEATURES])[:, 1]
    validation_prediction = (validation_probability >= 0.5).astype(int)
    pair_accuracy = float(
        np.mean(validation_prediction == validation_pairs["is_assigned"].to_numpy())
    )
    decoded_exact = _decoded_exact_rate(model, validation_candidates)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pairwise_position_policy.joblib"
    joblib.dump(model, model_path)
    metrics: dict[str, object] = {
        "model_type": "HistGradientBoostingClassifier_pair_scores_with_Hungarian_decoder",
        "training_csv": str(training_csv.resolve()),
        "training_scenarios": len(scenario_ids) - validation_count,
        "internal_validation_scenarios": validation_count,
        "training_candidate_rows": len(train_candidates),
        "internal_validation_candidate_rows": len(validation_candidates),
        "training_pair_rows": len(train_pairs),
        "internal_validation_pair_rows": len(validation_pairs),
        "pair_expansion_seconds": expansion_seconds,
        "training_seconds": training_seconds,
        "pair_classification_accuracy": pair_accuracy,
        "decoded_exact_position_rate": decoded_exact,
        "features": FEATURES,
        "random_seed": random_seed,
        "scope": {
            "decoder_complexity": "O(N^3)",
            "position_enumeration": False,
            "drone_count_in_current_training": 5,
        },
    }
    (output_dir / "pairwise_position_policy_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--random-seed", type=int, default=20260819)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metrics = train_pairwise_position_policy(
        args.training_csv,
        args.output_dir,
        random_seed=args.random_seed,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
