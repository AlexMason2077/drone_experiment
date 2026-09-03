"""Audit the position-only model against fixed and unrestricted references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "controlled_k_sweep"
    / "ablation"
    / "position_only_current_configuration"
)


def _metrics(cost: np.ndarray, fixed: np.ndarray, restricted: np.ndarray, full: np.ndarray) -> dict[str, float]:
    regret = cost - restricted
    saving = fixed - cost
    return {
        "strict_restricted_optimal_rate": float(np.mean(np.abs(regret) <= 2e-5)),
        "within_0p1_minute_of_restricted_optimum_rate": float(np.mean(regret <= 0.1 + 2e-5)),
        "within_0p5_minute_of_restricted_optimum_rate": float(np.mean(regret <= 0.5 + 2e-5)),
        "mean_restricted_regret_minutes": float(np.mean(regret)),
        "mean_saving_vs_fixed_minutes": float(np.mean(saving)),
        "aggregate_time_improvement_vs_fixed_percent": float(100.0 * np.sum(saving) / np.sum(fixed)),
        "mean_per_state_improvement_vs_fixed_percent": float(np.mean(100.0 * saving / fixed)),
        "fraction_strictly_better_than_fixed": float(np.mean(saving > 2e-5)),
        "fraction_equal_to_fixed": float(np.mean(np.abs(saving) <= 2e-5)),
        "fraction_worse_than_fixed": float(np.mean(saving < -2e-5)),
        "mean_gap_to_unrestricted_full_optimum_minutes": float(np.mean(cost - full)),
        "fraction_equal_to_unrestricted_full_optimum": float(np.mean(np.abs(cost - full) <= 2e-5)),
    }


def audit(states_path: Path, costs_path: Path, predictions_path: Path, output_path: Path) -> dict[str, object]:
    states = pd.read_csv(states_path)
    costs = np.load(costs_path)["costs"]
    predictions = pd.read_csv(predictions_path)
    if not (len(states) == len(costs) == len(predictions)):
        raise ValueError("State, cost, and prediction row counts differ")
    selected_class = predictions["selected_class_index"].to_numpy(dtype=np.int64)
    selected = costs[np.arange(len(costs)), selected_class]
    if not np.isfinite(selected).all():
        raise RuntimeError("Model selected an infeasible position")
    restricted = np.min(costs, axis=1)
    fixed = states["fixed_current_total_minutes"].to_numpy(dtype=float)
    full = states["full_oracle_total_minutes_reference"].to_numpy(dtype=float)
    guarded = np.minimum(selected, fixed)
    report: dict[str, object] = {
        "status": "pass",
        "scope": "independent single-decision states; not yet a dynamic end-to-end rollout",
        "states": len(states),
        "base_state_groups": int(states["base_state_id"].nunique()),
        "model_selection": _metrics(selected, fixed, restricted, full),
        "model_with_keep_current_guard": _metrics(guarded, fixed, restricted, full),
        "exact_position_only_reference": _metrics(restricted, fixed, restricted, full),
        "guard_definition": "Compare the model proposal with the current configuration and keep the lower exact predicted mission cost.",
    }
    rows = states[["wind_direction", "wind_level", "charging_pad_count", "base_state_id"]].copy()
    rows["fixed_minutes"] = fixed
    rows["model_minutes"] = selected
    rows["guarded_minutes"] = guarded
    rows["restricted_optimum_minutes"] = restricted
    rows["full_optimum_minutes"] = full
    rows["model_restricted_regret_minutes"] = selected - restricted
    rows["model_saving_vs_fixed_minutes"] = fixed - selected
    row_output = output_path.with_name("independent_ablation_audit_rows.csv")
    rows.to_csv(row_output, index=False)
    subgroups: list[dict[str, object]] = []
    for (direction, level, k), group in rows.groupby(
        ["wind_direction", "wind_level", "charging_pad_count"], sort=True
    ):
        idx = group.index.to_numpy(dtype=np.int64)
        subgroups.append({
            "wind_direction": direction,
            "wind_level": int(level),
            "charging_pad_count": int(k),
            "states": len(group),
            **_metrics(guarded[idx], fixed[idx], restricted[idx], full[idx]),
        })
    subgroup_output = output_path.with_name("independent_ablation_audit_by_condition_k.csv")
    pd.DataFrame(subgroups).to_csv(subgroup_output, index=False)
    report["row_audit_csv"] = str(row_output.resolve())
    report["condition_k_audit_csv"] = str(subgroup_output.resolve())
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=DEFAULT_ROOT / "independent_states.csv")
    parser.add_argument("--costs", type=Path, default=DEFAULT_ROOT / "independent_costs.npz")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_ROOT / "model" / "independent_predictions.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "model" / "position_only_ablation_audit.json")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = audit(args.states, args.costs, args.predictions, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
