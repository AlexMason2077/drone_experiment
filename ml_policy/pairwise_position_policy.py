"""Pairwise features for a scalable learned drone-to-slot assignment policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_policy.charging_model import exponential_charging_minutes


CATEGORICAL_FEATURES = ["wind_direction", "formation", "drone_id", "slot_index"]
NUMERIC_FEATURES = [
    "wind_level",
    "charging_pad_count",
    "remaining_distance_m",
    "inter_drone_spacing_cm",
    "drone_soc",
    "slot_rate_pp_per_min",
    "projected_arrival_soc",
    "pair_charging_minutes",
    "drone_soc_rank",
    "slot_rate_rank",
    "soc_lowest",
    "soc_middle",
    "soc_highest",
    "soc_range",
    "rate_lowest",
    "rate_middle",
    "rate_highest",
    "rate_range",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def expand_candidate_rows(frame: pd.DataFrame, *, include_target: bool) -> pd.DataFrame:
    """Expand each state/structure into 25 drone-slot pair rows."""

    records: list[dict[str, object]] = []
    for candidate_index, row in frame.iterrows():
        soc_values = np.asarray(
            [float(row[f"soc_d{index}"]) for index in range(1, 6)]
        )
        rate_values = np.asarray(
            [float(row[f"slot_{index}_rate_pp_per_min"]) for index in range(1, 6)]
        )
        soc_order = np.argsort(np.argsort(soc_values, kind="stable"), kind="stable")
        rate_order = np.argsort(np.argsort(rate_values, kind="stable"), kind="stable")
        forward_minutes = float(row["remaining_distance_m"]) / 0.10 / 60.0
        soc_sorted = np.sort(soc_values)
        rate_sorted = np.sort(rate_values)

        for drone_offset in range(5):
            for slot_offset in range(5):
                projected_soc = (
                    soc_values[drone_offset]
                    - rate_values[slot_offset] * forward_minutes
                )
                pair_charge = (
                    exponential_charging_minutes(projected_soc)
                    if projected_soc >= 0.0
                    else 90.0
                )
                record: dict[str, object] = {
                    "candidate_index": int(candidate_index),
                    "scenario_id": int(row["scenario_id"]),
                    "structure": row["structure"],
                    "wind_direction": row["wind_direction"],
                    "formation": row["formation"],
                    "drone_id": f"D{drone_offset + 1}",
                    "slot_index": str(slot_offset + 1),
                    "wind_level": int(row["wind_level"]),
                    "charging_pad_count": int(row["charging_pad_count"]),
                    "remaining_distance_m": float(row["remaining_distance_m"]),
                    "inter_drone_spacing_cm": float(
                        row["inter_drone_spacing_cm"]
                    ),
                    "drone_soc": soc_values[drone_offset],
                    "slot_rate_pp_per_min": rate_values[slot_offset],
                    "projected_arrival_soc": projected_soc,
                    "pair_charging_minutes": pair_charge,
                    "drone_soc_rank": float(soc_order[drone_offset]),
                    "slot_rate_rank": float(rate_order[slot_offset]),
                    "soc_lowest": soc_sorted[0],
                    "soc_middle": soc_sorted[2],
                    "soc_highest": soc_sorted[4],
                    "soc_range": soc_sorted[4] - soc_sorted[0],
                    "rate_lowest": rate_sorted[0],
                    "rate_middle": rate_sorted[2],
                    "rate_highest": rate_sorted[4],
                    "rate_range": rate_sorted[4] - rate_sorted[0],
                }
                if include_target:
                    record["is_assigned"] = int(
                        int(row[f"assigned_slot_index_d{drone_offset + 1}"])
                        == slot_offset + 1
                    )
                records.append(record)
    return pd.DataFrame.from_records(records)
