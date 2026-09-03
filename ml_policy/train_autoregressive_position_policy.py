"""Train an autoregressive learned permutation policy with beam decoding."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_policy.train_permutation_position_ablation import (
    CATEGORICAL,
    add_engineered_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT / "analysis_outputs" / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_TRAINING = DEFAULT_DIR / "position_aware_training_candidates.csv"
PREVIOUS_FEATURES = [f"previous_slot_d{index}" for index in range(1, 5)]


def _prepare_previous_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for index in range(1, 5):
        result[f"previous_slot_d{index}"] = 0
    return result


def decode_position(
    models: list[Pipeline],
    base_row: pd.DataFrame,
    feature_columns: list[str],
    *,
    beam_width: int = 10,
) -> tuple[int, ...]:
    """Decode a one-based slot permutation without enumerating all N! outputs."""

    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for drone_offset, model in enumerate(models):
        expanded: list[tuple[tuple[int, ...], float]] = []
        for assigned, log_probability in beams:
            row = base_row.copy()
            for previous_offset, slot in enumerate(assigned, start=1):
                row[f"previous_slot_d{previous_offset}"] = slot
            probabilities = model.predict_proba(row[feature_columns])[0]
            probability_by_class = dict(zip(model.named_steps["classifier"].classes_, probabilities))
            for slot in range(1, 6):
                if slot in assigned:
                    continue
                probability = max(float(probability_by_class.get(slot, 0.0)), 1e-12)
                expanded.append((assigned + (slot,), log_probability + math.log(probability)))
        beams = sorted(expanded, key=lambda item: (-item[1], item[0]))[:beam_width]
    best = beams[0][0]
    remaining = next(slot for slot in range(1, 6) if slot not in best)
    return best + (remaining,)


def train_autoregressive_position_policy(
    training_csv: Path,
    output_dir: Path,
    *,
    random_seed: int,
    beam_width: int,
) -> dict[str, object]:
    raw = pd.read_csv(training_csv)
    raw = raw[raw["eligible_for_selection"].eq(1)].copy()
    scenario_ids = np.asarray(sorted(raw["scenario_id"].unique()))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(scenario_ids)
    validation_count = int(round(0.20 * len(scenario_ids)))
    validation_ids = set(int(value) for value in scenario_ids[:validation_count])
    train_raw = raw[~raw["scenario_id"].isin(validation_ids)].copy()
    validation_raw = raw[raw["scenario_id"].isin(validation_ids)].copy()
    train, numeric = add_engineered_features(train_raw)
    validation, _ = add_engineered_features(validation_raw)
    train = _prepare_previous_features(train)
    validation = _prepare_previous_features(validation)
    feature_columns = CATEGORICAL + numeric + PREVIOUS_FEATURES

    models: list[Pipeline] = []
    for drone_index in range(1, 5):
        step_train = train.copy()
        for previous in range(1, drone_index):
            step_train[f"previous_slot_d{previous}"] = step_train[
                f"assigned_slot_index_d{previous}"
            ]
        preprocessor = ColumnTransformer(
            [
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL,
                ),
                ("numeric", "passthrough", numeric + PREVIOUS_FEATURES),
            ]
        )
        classifier = ExtraTreesClassifier(
            n_estimators=350,
            max_features=0.8,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_seed + drone_index,
        )
        model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        model.fit(
            step_train[feature_columns],
            step_train[f"assigned_slot_index_d{drone_index}"].to_numpy(dtype=int),
        )
        models.append(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "models": models,
        "feature_columns": feature_columns,
        "beam_width": beam_width,
    }
    joblib.dump(artifact, output_dir / "autoregressive_position_policy.joblib")

    validation_sample = validation.sample(
        n=min(1000, len(validation)), random_state=random_seed
    )
    predicted: list[tuple[int, ...]] = []
    truth: list[tuple[int, ...]] = []
    for _, row in validation_sample.iterrows():
        base = pd.DataFrame([row])
        predicted.append(
            decode_position(models, base, feature_columns, beam_width=beam_width)
        )
        truth.append(
            tuple(int(row[f"assigned_slot_index_d{index}"]) for index in range(1, 6))
        )
    exact_rate = float(np.mean([left == right for left, right in zip(predicted, truth)]))

    metrics: dict[str, object] = {
        "model_type": "four_step_ExtraTrees_autoregressive_position_policy",
        "decoder": f"beam search width {beam_width}; final slot is the unused slot",
        "training_scenarios": len(scenario_ids) - validation_count,
        "validation_scenarios": validation_count,
        "training_candidate_rows": len(train),
        "validation_candidate_rows": len(validation),
        "internal_position_evaluation_rows": len(validation_sample),
        "internal_exact_position_rate": exact_rate,
        "output_complexity": "O(B*N^2) for fixed beam width B",
        "position_enumeration": False,
        "training_csv": str(training_csv.resolve()),
        "random_seed": random_seed,
    }
    (output_dir / "autoregressive_position_policy_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--random-seed", type=int, default=20260819)
    parser.add_argument("--beam-width", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metrics = train_autoregressive_position_policy(
        args.training_csv,
        args.output_dir,
        random_seed=args.random_seed,
        beam_width=args.beam_width,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
