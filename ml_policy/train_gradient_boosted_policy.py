"""Train and evaluate the first fast formation/spacing policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_TRAINING_CSV = DEFAULT_OUTPUT_DIR / "oracle_training_states_0p25_25m.csv"

CATEGORICAL_FEATURES = ["wind_direction"]
NUMERIC_FEATURES = [
    "wind_level",
    "charging_pad_count",
    "remaining_distance_m",
    "soc_lowest",
    "soc_second_lowest",
    "soc_middle",
    "soc_second_highest",
    "soc_highest",
    "soc_range",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "oracle_structure"


def _split_frames(
    frame: pd.DataFrame,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Keep very rare Oracle choices in training and stratify the rest."""
    counts = frame[TARGET].value_counts()
    rare_labels = counts[counts < 5].index.tolist()
    rare_frame = frame[frame[TARGET].isin(rare_labels)]
    common_frame = frame[~frame[TARGET].isin(rare_labels)]
    train_common, holdout = train_test_split(
        common_frame,
        test_size=0.40,
        random_state=random_seed,
        stratify=common_frame[TARGET],
    )
    validation_frame, test_frame = train_test_split(
        holdout,
        test_size=0.50,
        random_state=random_seed + 1,
        stratify=holdout[TARGET],
    )
    train_frame = pd.concat([train_common, rare_frame], ignore_index=True)
    return train_frame, validation_frame, test_frame, rare_labels


def _masked_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    masked = probabilities.copy()
    safe_matrix = np.zeros_like(masked, dtype=bool)
    backup_matrix = np.zeros_like(masked, dtype=bool)
    for class_index, label in enumerate(classes):
        time_column = f"time__{label}"
        tier_column = f"tier__{label}"
        if time_column not in frame or tier_column not in frame:
            masked[:, class_index] = 0.0
            continue
        available = frame[time_column].notna().to_numpy()
        safe = frame[tier_column].eq("safe").to_numpy()
        backup = frame[tier_column].eq("backup_only").to_numpy()
        safe_matrix[:, class_index] = available & safe
        backup_matrix[:, class_index] = available & backup
    has_safe = safe_matrix.any(axis=1)
    eligible_matrix = np.where(has_safe[:, None], safe_matrix, backup_matrix)
    masked[~eligible_matrix] = 0.0
    row_totals = masked.sum(axis=1, keepdims=True)
    zero_rows = row_totals[:, 0] <= 0
    if np.any(zero_rows):
        fallback = eligible_matrix[zero_rows].astype(float)
        fallback_totals = fallback.sum(axis=1, keepdims=True)
        if np.any(fallback_totals <= 0):
            raise RuntimeError("At least one evaluation row has no safe predicted class")
        masked[zero_rows] = fallback / fallback_totals
        row_totals = masked.sum(axis=1, keepdims=True)
    return masked / row_totals


def _decision_metrics(
    model: Pipeline,
    frame: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    started = time.perf_counter()
    probabilities = model.predict_proba(frame[FEATURES])
    inference_ms = (time.perf_counter() - started) * 1000.0
    classes = model.named_steps["classifier"].classes_
    masked_probabilities = _masked_probabilities(probabilities, classes, frame)
    rankings = np.argsort(-masked_probabilities, axis=1)
    top1 = classes[rankings[:, 0]]
    top2 = [
        [
            classes[class_index]
            for class_index in ranking
            if masked_probabilities[row_index, class_index] > 0
        ][:2]
        for row_index, ranking in enumerate(rankings)
    ]
    truth = frame[TARGET].to_numpy()
    top2_hit = np.array(
        [target in candidates for target, candidates in zip(truth, top2)],
        dtype=bool,
    )

    predicted_times = np.array(
        [
            frame.iloc[index][f"time__{label}"]
            for index, label in enumerate(top1)
        ],
        dtype=float,
    )
    oracle_times = frame["oracle_total_minutes"].to_numpy(dtype=float)
    regret = predicted_times - oracle_times

    predictions = frame[
        [
            "scenario_id",
            "wind_direction",
            "wind_level",
            "charging_pad_count",
            "remaining_distance_m",
            TARGET,
            "oracle_total_minutes",
        ]
    ].copy()
    predictions["predicted_top1"] = top1
    predictions["predicted_top2"] = ["|".join(values) for values in top2]
    predictions["top1_correct"] = top1 == truth
    predictions["top2_contains_oracle"] = top2_hit
    predictions["predicted_total_minutes"] = predicted_times
    predictions["regret_minutes"] = regret
    predictions["confidence"] = masked_probabilities.max(axis=1)

    metrics = {
        "row_count": float(len(frame)),
        "top1_accuracy": float(accuracy_score(truth, top1)),
        "top2_coverage": float(top2_hit.mean()),
        "mean_regret_minutes": float(regret.mean()),
        "median_regret_minutes": float(np.median(regret)),
        "p95_regret_minutes": float(np.quantile(regret, 0.95)),
        "maximum_regret_minutes": float(regret.max()),
        "total_inference_ms": inference_ms,
        "mean_inference_ms_per_state": inference_ms / len(frame),
        "mean_oracle_runtime_ms": float(frame["oracle_runtime_ms"].mean()),
    }
    metrics["oracle_to_policy_speedup"] = (
        metrics["mean_oracle_runtime_ms"]
        / metrics["mean_inference_ms_per_state"]
    )
    return metrics, predictions


def _majority_baseline_metrics(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    group_columns: list[str],
) -> dict[str, float]:
    """Evaluate a lookup baseline learned only from coarse condition groups."""
    group_rankings = {
        key: list(values.value_counts().index)
        for key, values in train_frame.groupby(group_columns, dropna=False)[TARGET]
    }
    global_ranking = list(train_frame[TARGET].value_counts().index)
    predicted_labels = []
    predicted_times = []
    for _, row in evaluation_frame.iterrows():
        key_values = tuple(row[column] for column in group_columns)
        key = key_values[0] if len(key_values) == 1 else key_values
        ranking = list(group_rankings.get(key, []))
        ranking.extend(label for label in global_ranking if label not in ranking)
        safe_candidates = [
            label
            for label in ranking
            if pd.notna(row.get(f"time__{label}"))
            and row.get(f"tier__{label}") == "safe"
        ]
        backup_candidates = [
            label
            for label in ranking
            if pd.notna(row.get(f"time__{label}"))
            and row.get(f"tier__{label}") == "backup_only"
        ]
        candidates = safe_candidates or backup_candidates
        if not candidates:
            raise RuntimeError("Baseline found no feasible safe or backup candidate")
        label = candidates[0]
        predicted_labels.append(label)
        predicted_times.append(float(row[f"time__{label}"]))

    truth = evaluation_frame[TARGET].to_numpy()
    regret = np.asarray(predicted_times) - evaluation_frame[
        "oracle_total_minutes"
    ].to_numpy(dtype=float)
    return {
        "top1_accuracy": float(accuracy_score(truth, predicted_labels)),
        "mean_regret_minutes": float(regret.mean()),
        "median_regret_minutes": float(np.median(regret)),
        "p95_regret_minutes": float(np.quantile(regret, 0.95)),
        "maximum_regret_minutes": float(regret.max()),
    }


def train_policy(
    *,
    training_csv: Path,
    output_dir: Path,
    random_seed: int,
) -> None:
    frame = pd.read_csv(training_csv)
    training_manifest_path = training_csv.with_suffix(".manifest.json")
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    missing = set(FEATURES + [TARGET]).difference(frame.columns)
    if missing:
        raise ValueError("Training data is missing columns: " + ", ".join(sorted(missing)))

    train_frame, validation_frame, test_frame, train_only_rare_labels = _split_frames(
        frame,
        random_seed,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "wind",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    classifier = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=random_seed,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    class_counts = train_frame[TARGET].value_counts()
    sample_weight = train_frame[TARGET].map(
        {label: len(train_frame) / (len(class_counts) * count) for label, count in class_counts.items()}
    ).to_numpy()
    model.fit(
        train_frame[FEATURES],
        train_frame[TARGET],
        classifier__sample_weight=sample_weight,
    )

    validation_metrics, validation_predictions = _decision_metrics(
        model,
        validation_frame,
    )
    test_metrics, test_predictions = _decision_metrics(model, test_frame)
    condition_baseline_metrics = _majority_baseline_metrics(
        train_frame,
        test_frame,
        ["wind_direction", "wind_level"],
    )
    condition_k_baseline_metrics = _majority_baseline_metrics(
        train_frame,
        test_frame,
        ["wind_direction", "wind_level", "charging_pad_count"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gradient_boosted_policy.joblib"
    joblib.dump(model, model_path)
    validation_predictions.to_csv(
        output_dir / "gradient_boosted_validation_predictions.csv",
        index=False,
    )
    test_predictions.to_csv(
        output_dir / "gradient_boosted_test_predictions.csv",
        index=False,
    )

    classes = list(model.named_steps["classifier"].classes_)
    matrix = confusion_matrix(
        test_predictions[TARGET],
        test_predictions["predicted_top1"],
        labels=classes,
    )
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        output_dir / "gradient_boosted_confusion_matrix.csv"
    )
    class_counts.sort_index().rename("train_count").to_csv(
        output_dir / "gradient_boosted_train_class_counts.csv"
    )

    metadata = {
        "model_type": "GradientBoostingClassifier",
        "model_parameters": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
        },
        "features": FEATURES,
        "target": TARGET,
        "classes": classes,
        "random_seed": random_seed,
        "training_csv": str(training_csv),
        "training_manifest": str(training_manifest_path),
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "test_rows": len(test_frame),
        "train_only_rare_classes": train_only_rare_labels,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "condition_only_baseline_test_metrics": condition_baseline_metrics,
        "condition_and_k_baseline_test_metrics": condition_k_baseline_metrics,
        "scope": {
            "drone_count": 5,
            "charging_pad_count": "1-5",
            "charging_model": training_manifest["charging_model"],
            "fully_charged_soc": training_manifest["fully_charged_soc"],
            "zero_to_fully_charged_minutes": training_manifest[
                "zero_to_fully_charged_minutes"
            ],
            "charging_time_constant_minutes": training_manifest[
                "charging_time_constant_minutes"
            ],
            "decision_interval_seconds": training_manifest[
                "decision_interval_seconds"
            ],
            "remaining_distance_m": {
                "minimum": float(frame["remaining_distance_m"].min()),
                "maximum": float(frame["remaining_distance_m"].max()),
                "note": "Simulated range; experimental rates were measured over 2.5 m.",
            },
        },
    }
    (output_dir / "gradient_boosted_policy_metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata["test_metrics"], indent=2, sort_keys=True))
    print(f"Saved model to {model_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=20260817)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    train_policy(
        training_csv=args.training_csv,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
