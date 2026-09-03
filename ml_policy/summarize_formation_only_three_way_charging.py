"""Compare fixed, formation-only, and full-global charging makespans."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "controlled_k_sweep"
    / "ablation"
    / "formation_only_current_configuration"
)


def _metrics(group: pd.DataFrame) -> dict[str, float]:
    fixed = group["fixed_charging_minutes"].to_numpy(dtype=float)
    formation = group["formation_only_exact_charging_minutes"].to_numpy(dtype=float)
    model = group["formation_only_guarded_model_charging_minutes"].to_numpy(dtype=float)
    full = group["full_global_charging_minutes"].to_numpy(dtype=float)
    formation_saving = fixed - formation
    full_saving = fixed - full
    return {
        "states": int(len(group)),
        "mean_fixed_charging_minutes": float(fixed.mean()),
        "mean_formation_only_exact_charging_minutes": float(formation.mean()),
        "mean_formation_only_guarded_model_charging_minutes": float(model.mean()),
        "mean_full_global_charging_minutes": float(full.mean()),
        "mean_formation_only_saving_vs_fixed_minutes": float(formation_saving.mean()),
        "mean_full_saving_vs_fixed_minutes": float(full_saving.mean()),
        "mean_formation_only_gap_to_full_minutes": float((formation - full).mean()),
        "aggregate_formation_only_improvement_vs_fixed_percent": float(
            100.0 * formation_saving.sum() / fixed.sum()
        ),
        "aggregate_full_improvement_vs_fixed_percent": float(
            100.0 * full_saving.sum() / fixed.sum()
        ),
        "formation_only_fraction_of_full_saving_captured_percent": float(
            100.0 * formation_saving.sum() / full_saving.sum()
        ),
        "aggregate_guarded_model_improvement_vs_fixed_percent": float(
            100.0 * (fixed.sum() - model.sum()) / fixed.sum()
        ),
    }


def main() -> int:
    states = pd.read_csv(ROOT / "independent_states.csv")
    audit = pd.read_csv(ROOT / "model" / "independent_ablation_audit_rows.csv")
    if len(states) != len(audit):
        raise ValueError("State and audit row counts differ")
    flight = states["remaining_distance_m"].to_numpy(dtype=float) / 0.10 / 60.0
    frame = states[
        ["base_state_id", "wind_direction", "wind_level", "charging_pad_count"]
    ].copy()
    frame["fixed_charging_minutes"] = audit["fixed_minutes"].to_numpy(dtype=float) - flight
    frame["formation_only_exact_charging_minutes"] = (
        audit["restricted_optimum_minutes"].to_numpy(dtype=float) - flight
    )
    frame["formation_only_guarded_model_charging_minutes"] = (
        audit["guarded_minutes"].to_numpy(dtype=float) - flight
    )
    frame["full_global_charging_minutes"] = (
        audit["full_optimum_minutes"].to_numpy(dtype=float) - flight
    )
    if not np.all(
        frame["full_global_charging_minutes"]
        <= frame["formation_only_exact_charging_minutes"] + 2e-5
    ):
        raise RuntimeError("Full optimum exceeds a formation-only optimum")
    if not np.all(
        frame["formation_only_exact_charging_minutes"]
        <= frame["fixed_charging_minutes"] + 2e-5
    ):
        raise RuntimeError("Formation-only optimum exceeds fixed configuration")

    overall = {"scope": "all K and all wind conditions", **_metrics(frame)}
    by_k = pd.DataFrame(
        [
            {"charging_pad_count": int(k), **_metrics(group)}
            for k, group in frame.groupby("charging_pad_count", sort=True)
        ]
    )
    by_condition_k = pd.DataFrame(
        [
            {
                "wind_direction": direction,
                "wind_level": int(level),
                "charging_pad_count": int(k),
                **_metrics(group),
            }
            for (direction, level, k), group in frame.groupby(
                ["wind_direction", "wind_level", "charging_pad_count"], sort=True
            )
        ]
    )
    by_k_path = ROOT / "formation_only_charging_three_way_by_k.csv"
    by_condition_k_path = ROOT / "formation_only_charging_three_way_by_condition_k.csv"
    row_path = ROOT / "formation_only_charging_three_way_rows.csv"
    by_k.to_csv(by_k_path, index=False)
    by_condition_k.to_csv(by_condition_k_path, index=False)
    frame.to_csv(row_path, index=False)
    report = {
        "status": "pass",
        "comparison": [
            "fixed current configuration",
            "exact formation-only optimum with current spacing and position fixed",
            "unrestricted full global optimum over formation, spacing, and position",
        ],
        "charging_time_definition": "minimum K-pad makespan until all five batteries are fully charged",
        "paired_independent_states": len(frame),
        "paired_base_state_groups": int(frame["base_state_id"].nunique()),
        "overall": overall,
        "by_k_csv": str(by_k_path.resolve()),
        "by_condition_k_csv": str(by_condition_k_path.resolve()),
        "row_level_csv": str(row_path.resolve()),
    }
    output = ROOT / "formation_only_charging_three_way_comparison.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(by_k.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
