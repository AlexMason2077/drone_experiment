"""Train a fixed-five-drone permutation classifier as a position ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_policy.charging_model import exponential_charging_minutes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    PROJECT_ROOT / "analysis_outputs" / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_TRAINING = DEFAULT_DIR / "position_aware_training_candidates.csv"
CATEGORICAL = ["wind_direction", "formation", "structure", "charging_pad_count"]


def add_engineered_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    numeric = [
        "wind_level", "remaining_distance_m", "inter_drone_spacing_cm",
        *[f"soc_d{index}" for index in range(1, 6)],
        *[f"slot_{index}_rate_pp_per_min" for index in range(1, 6)],
    ]
    forward_minutes = result["remaining_distance_m"].to_numpy(dtype=float) / 6.0
    for drone in range(1, 6):
        soc = result[f"soc_d{drone}"].to_numpy(dtype=float)
        for slot in range(1, 6):
            rate = result[f"slot_{slot}_rate_pp_per_min"].to_numpy(dtype=float)
            arrival = soc - rate * forward_minutes
            arrival_name = f"pair_arrival_d{drone}_s{slot}"
            charge_name = f"pair_charge_d{drone}_s{slot}"
            result[arrival_name] = arrival
            result[charge_name] = [
                exponential_charging_minutes(max(0.0, value))
                for value in arrival
            ]
            numeric.extend([arrival_name, charge_name])
    return result, numeric


def permutation_label(frame: pd.DataFrame) -> pd.Series:
    return frame[[f"assigned_slot_index_d{index}" for index in range(1, 6)]].astype(
        str
    ).agg("|".join, axis=1)


def train_permutation_ablation(
    training_csv: Path,
    output_dir: Path,
    *,
    random_seed: int,
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
    target = permutation_label(train)
    validation_target = permutation_label(validation)

    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("numeric", "passthrough", numeric),
        ]
    )
    classifier = ExtraTreesClassifier(
        n_estimators=500,
        max_features=0.8,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_seed,
    )
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    model.fit(train[CATEGORICAL + numeric], target)
    prediction = model.predict(validation[CATEGORICAL + numeric])
    exact = float(np.mean(prediction == validation_target.to_numpy()))

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "permutation_position_ablation.joblib")
    metadata: dict[str, object] = {
        "model_type": "ExtraTreesClassifier_120_permutation_ablation",
        "training_csv": str(training_csv.resolve()),
        "training_scenarios": len(scenario_ids) - validation_count,
        "validation_scenarios": validation_count,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "permutation_class_count": int(target.nunique()),
        "internal_exact_position_rate": exact,
        "categorical_features": CATEGORICAL,
        "numeric_features": numeric,
        "scalability_caveat": (
            "This classifier has N! output classes and is retained only as a fixed-N "
            "ablation, not as the scalable final position decoder."
        ),
    }
    (output_dir / "permutation_position_ablation_metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--random-seed", type=int, default=20260819)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = train_permutation_ablation(
        args.training_csv, args.output_dir, random_seed=args.random_seed
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
