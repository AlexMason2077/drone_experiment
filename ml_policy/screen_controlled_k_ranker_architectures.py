"""Screen neural capacity on one fixed grouped validation fold only."""

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

from ml_policy.train_controlled_k_cost_ranker import (
    DEFAULT_CLASSES,
    DEFAULT_DIR,
    _add_derived_features,
    _fit_preprocessor,
    _metrics,
    _select,
    _soft_cost_targets,
    _transform,
)


SPECS = (
    {
        "name": "baseline_extended",
        "hidden": (512, 512, 256),
        "dropout": 0.20,
        "learning_rate": 7e-4,
        "weight_decay": 1e-4,
    },
    {
        "name": "wider",
        "hidden": (768, 768, 512, 256),
        "dropout": 0.15,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
    },
    {
        "name": "deeper",
        "hidden": (512, 512, 512, 512, 256),
        "dropout": 0.15,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
    },
)


def _build_model(
    input_width: int,
    class_count: int,
    spec: dict[str, object],
    random_seed: int,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(random_seed)
    inputs = tf.keras.Input(shape=(input_width,), name="observed_state")
    x = inputs
    for index, width in enumerate(spec["hidden"], start=1):
        x = tf.keras.layers.Dense(
            int(width), activation=tf.nn.gelu, name=f"dense_{index}"
        )(x)
        x = tf.keras.layers.LayerNormalization(name=f"norm_{index}")(x)
        x = tf.keras.layers.Dropout(float(spec["dropout"]), name=f"dropout_{index}")(x)
    logits = tf.keras.layers.Dense(class_count, name="configuration_logits")(x)
    model = tf.keras.Model(inputs, logits, name=str(spec["name"]))
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=float(spec["learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )
    return model


def screen(
    states_path: Path,
    costs_path: Path,
    class_table_path: Path,
    output_dir: Path,
    *,
    max_epochs: int,
    patience: int,
    temperature_minutes: float,
    random_seed: int,
) -> dict[str, object]:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    states = _add_derived_features(pd.read_csv(states_path))
    costs = np.load(costs_path)["costs"]
    classes = pd.read_csv(class_table_path).sort_values("class_index")
    groups = states["base_state_id"].to_numpy()
    strata = (
        states["wind_direction"].astype(str)
        + "_lv"
        + states["wind_level"].astype(str)
    ).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_seed
    )
    fit_indices, validation_indices = next(splitter.split(states, strata, groups))
    if set(groups[fit_indices]).intersection(set(groups[validation_indices])):
        raise RuntimeError("Group leakage in architecture screen")
    preprocessor = _fit_preprocessor(states.iloc[fit_indices])
    x_fit = _transform(states.iloc[fit_indices], preprocessor)
    x_validation = _transform(states.iloc[validation_indices], preprocessor)
    y_fit = _soft_cost_targets(costs[fit_indices], temperature_minutes)
    y_validation = _soft_cost_targets(costs[validation_indices], temperature_minutes)
    validation_costs = costs[validation_indices]
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for trial_number, raw_spec in enumerate(SPECS, start=1):
        spec = dict(raw_spec)
        print(f"=== architecture {trial_number}/{len(SPECS)}: {spec['name']} ===", flush=True)
        model = _build_model(
            x_fit.shape[1], len(classes), spec, random_seed + trial_number
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
            batch_size=256,
            verbose=0,
            callbacks=callbacks,
        )
        logits = model.predict(x_validation, batch_size=512, verbose=0)
        selected = _select(logits, validation_costs)
        record: dict[str, object] = {
            **spec,
            "hidden": list(spec["hidden"]),
            "parameter_count": int(model.count_params()),
            "epochs_completed": len(history.history["loss"]),
            "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
            "best_val_loss": float(min(history.history["val_loss"])),
            "elapsed_seconds": time.perf_counter() - started,
            **_metrics(selected, validation_costs),
        }
        records.append(record)
        pd.DataFrame(history.history).to_csv(
            output_dir / f"{spec['name']}_history.csv", index=False
        )
        print(json.dumps(record, sort_keys=True), flush=True)
        tf.keras.backend.clear_session()

    report = {
        "status": "pass",
        "selection_data": "one fixed grouped validation fold; independent test not used",
        "fit_groups": int(len(set(groups[fit_indices]))),
        "validation_groups": int(len(set(groups[validation_indices]))),
        "temperature_minutes": temperature_minutes,
        "architectures": records,
    }
    pd.DataFrame(records).to_csv(output_dir / "architecture_comparison.csv", index=False)
    (output_dir / "architecture_comparison.json").write_text(
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
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DIR / "architecture_screen"
    )
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--temperature-minutes", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=20260820)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = screen(
        args.states,
        args.costs,
        args.class_table,
        args.output_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        temperature_minutes=args.temperature_minutes,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
