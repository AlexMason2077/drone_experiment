"""Feature encoding and decoding for the full-configuration neural policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


WIND_DIRECTIONS = ("head", "side", "tail")
FORMATIONS = ("column", "diamond", "echalon", "front", "vee")
NUMERIC_FEATURES = (
    "wind_level",
    "charging_pad_count",
    "remaining_distance_m",
    "soc_d1",
    "soc_d2",
    "soc_d3",
    "soc_d4",
    "soc_d5",
    "inter_drone_spacing_cm",
    "slot_1_rate_pp_per_min",
    "slot_2_rate_pp_per_min",
    "slot_3_rate_pp_per_min",
    "slot_4_rate_pp_per_min",
    "slot_5_rate_pp_per_min",
)


def fit_preprocessor(frame: pd.DataFrame) -> dict[str, object]:
    numeric = frame.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float64)
    mean = numeric.mean(axis=0)
    scale = numeric.std(axis=0)
    scale[scale < 1e-12] = 1.0
    time = frame["total_completion_minutes"].to_numpy(dtype=np.float64)
    time_scale = float(time.std())
    if time_scale < 1e-12:
        time_scale = 1.0
    return {
        "numeric_features": list(NUMERIC_FEATURES),
        "numeric_mean": mean.tolist(),
        "numeric_scale": scale.tolist(),
        "wind_directions": list(WIND_DIRECTIONS),
        "formations": list(FORMATIONS),
        "time_mean": float(time.mean()),
        "time_scale": time_scale,
    }


def transform_features(
    frame: pd.DataFrame,
    preprocessor: dict[str, object],
) -> np.ndarray:
    numeric_features = list(preprocessor["numeric_features"])
    numeric = frame.loc[:, numeric_features].to_numpy(dtype=np.float32)
    mean = np.asarray(preprocessor["numeric_mean"], dtype=np.float32)
    scale = np.asarray(preprocessor["numeric_scale"], dtype=np.float32)
    numeric = (numeric - mean) / scale

    wind_values = frame["wind_direction"].astype(str).to_numpy()
    wind = np.column_stack(
        [wind_values == value for value in preprocessor["wind_directions"]]
    ).astype(np.float32)
    formation_values = frame["formation"].astype(str).to_numpy()
    formation = np.column_stack(
        [formation_values == value for value in preprocessor["formations"]]
    ).astype(np.float32)
    return np.concatenate([numeric, wind, formation], axis=1)


def transform_time(
    values: Sequence[float],
    preprocessor: dict[str, object],
) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float32) - float(preprocessor["time_mean"])
    ) / float(preprocessor["time_scale"])


def inverse_time(values: np.ndarray, preprocessor: dict[str, object]) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float64) * float(preprocessor["time_scale"])
        + float(preprocessor["time_mean"])
    )


def assignment_targets(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame[f"assigned_slot_index_d{index}"].to_numpy(dtype=np.int32) - 1
            for index in range(1, 6)
        ]
    )


def decode_assignment(logits: np.ndarray) -> tuple[int, ...]:
    """Return zero-based slot index by drone using one-to-one matching."""

    scores = np.asarray(logits, dtype=np.float64)
    if scores.shape != (5, 5):
        raise ValueError(f"Expected a 5x5 assignment score matrix, received {scores.shape}")
    drone_indices, slot_indices = linear_sum_assignment(-scores)
    slot_by_drone = np.empty(5, dtype=np.int32)
    slot_by_drone[drone_indices] = slot_indices
    return tuple(int(value) for value in slot_by_drone)


def save_preprocessor(preprocessor: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(preprocessor, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_preprocessor(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
