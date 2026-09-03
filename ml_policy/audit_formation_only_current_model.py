"""Audit the formal formation-only model against fixed and full references."""

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
    / "formation_only_current_configuration"
)


def _comparison_metrics(
    method_cost: np.ndarray,
    fixed_cost: np.ndarray,
    restricted_optimum: np.ndarray,
    full_optimum: np.ndarray,
) -> dict[str, float]:
    restricted_regret = method_cost - restricted_optimum
    savings = fixed_cost - method_cost
    return {
        "strict_restricted_optimal_rate": float(
            np.mean(np.abs(restricted_regret) <= 2e-5)
        ),
        "within_0p1_minute_of_restricted_optimum_rate": float(
            np.mean(restricted_regret <= 0.1 + 2e-5)
        ),
        "within_0p5_minute_of_restricted_optimum_rate": float(
            np.mean(restricted_regret <= 0.5 + 2e-5)
        ),
        "mean_restricted_regret_minutes": float(np.mean(restricted_regret)),
        "mean_saving_vs_fixed_minutes": float(np.mean(savings)),
        "aggregate_time_improvement_vs_fixed_percent": float(
            100.0 * np.sum(savings) / np.sum(fixed_cost)
        ),
        "mean_per_state_improvement_vs_fixed_percent": float(
            np.mean(100.0 * savings / fixed_cost)
        ),
        "fraction_strictly_better_than_fixed": float(np.mean(savings > 2e-5)),
        "fraction_equal_to_fixed": float(np.mean(np.abs(savings) <= 2e-5)),
        "fraction_worse_than_fixed": float(np.mean(savings < -2e-5)),
        "mean_gap_to_unrestricted_full_optimum_minutes": float(
            np.mean(method_cost - full_optimum)
        ),
        "fraction_equal_to_unrestricted_full_optimum": float(
            np.mean(np.abs(method_cost - full_optimum) <= 2e-5)
        ),
    }


def audit(
    *,
    states_path: Path,
    costs_path: Path,
    predictions_path: Path,
    output_path: Path,
) -> dict[str, object]:
    states = pd.read_csv(states_path)
    costs = np.load(costs_path)["costs"]
    predictions = pd.read_csv(predictions_path)
    if not (len(states) == len(costs) == len(predictions)):
        raise ValueError("State, cost, and prediction row counts differ")
    selected_class = predictions["selected_class_index"].to_numpy(dtype=np.int64)
    selected_cost = costs[np.arange(len(costs)), selected_class]
    restricted_optimum = np.min(costs, axis=1)
    fixed_cost = states["fixed_current_total_minutes"].to_numpy(dtype=float)
    full_optimum = states["full_oracle_total_minutes_reference"].to_numpy(dtype=float)
    if not np.isfinite(selected_cost).all():
        raise RuntimeError("Model selected an infeasible formation")

    # A one-candidate exact guard is not a new search.  It simply keeps the
    # current configuration when the model's proposed formation is predicted
    # to have a higher exact mission cost than the current one.
    guarded_cost = np.minimum(selected_cost, fixed_cost)
    report: dict[str, object] = {
        "status": "pass",
        "scope": "independent single-decision states; not yet a dynamic end-to-end rollout",
        "states": len(states),
        "base_state_groups": int(states["base_state_id"].nunique()),
        "model_selection": _comparison_metrics(
            selected_cost, fixed_cost, restricted_optimum, full_optimum
        ),
        "model_with_keep_current_guard": _comparison_metrics(
            guarded_cost, fixed_cost, restricted_optimum, full_optimum
        ),
        "exact_formation_only_reference": _comparison_metrics(
            restricted_optimum, fixed_cost, restricted_optimum, full_optimum
        ),
        "guard_definition": (
            "Evaluate the model proposal and current configuration, then keep "
            "the lower-cost one; this adds one comparison and prevents predicted regression."
        ),
    }

    detail = states[
        ["wind_direction", "wind_level", "charging_pad_count", "base_state_id"]
    ].copy()
    detail["fixed_minutes"] = fixed_cost
    detail["model_minutes"] = selected_cost
    detail["guarded_minutes"] = guarded_cost
    detail["restricted_optimum_minutes"] = restricted_optimum
    detail["full_optimum_minutes"] = full_optimum
    detail["model_restricted_regret_minutes"] = selected_cost - restricted_optimum
    detail["model_saving_vs_fixed_minutes"] = fixed_cost - selected_cost
    detail["guarded_saving_vs_fixed_minutes"] = fixed_cost - guarded_cost
    detail_output = output_path.with_name("independent_ablation_audit_rows.csv")
    detail.to_csv(detail_output, index=False)

    subgroup_records: list[dict[str, object]] = []
    for (direction, level, k), group in detail.groupby(
        ["wind_direction", "wind_level", "charging_pad_count"], sort=True
    ):
        idx = group.index.to_numpy(dtype=np.int64)
        metrics = _comparison_metrics(
            guarded_cost[idx],
            fixed_cost[idx],
            restricted_optimum[idx],
            full_optimum[idx],
        )
        subgroup_records.append(
            {
                "wind_direction": direction,
                "wind_level": int(level),
                "charging_pad_count": int(k),
                "states": len(group),
                **metrics,
            }
        )
    subgroup_output = output_path.with_name("independent_ablation_audit_by_condition_k.csv")
    pd.DataFrame(subgroup_records).to_csv(subgroup_output, index=False)
    report["row_audit_csv"] = str(detail_output.resolve())
    report["condition_k_audit_csv"] = str(subgroup_output.resolve())
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=DEFAULT_ROOT / "independent_states.csv")
    parser.add_argument("--costs", type=Path, default=DEFAULT_ROOT / "independent_costs.npz")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_ROOT / "model" / "independent_predictions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "model" / "formation_only_ablation_audit.json",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = audit(
        states_path=args.states,
        costs_path=args.costs,
        predictions_path=args.predictions,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
