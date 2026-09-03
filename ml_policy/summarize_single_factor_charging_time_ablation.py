"""Summarize charging-makespan differences for paired single-factor ablations."""

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
)


def _paths(factor: str) -> tuple[Path, Path]:
    base = ROOT / f"{factor}_only_current_configuration"
    return base / "independent_states.csv", base / "model" / "independent_ablation_audit_rows.csv"


def _summarize(factor: str, states: pd.DataFrame, rows: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]]]:
    flight = states["remaining_distance_m"].to_numpy(dtype=float) / 0.10 / 60.0
    fixed = rows["fixed_minutes"].to_numpy(dtype=float) - flight
    restricted = rows["restricted_optimum_minutes"].to_numpy(dtype=float) - flight
    guarded = rows["guarded_minutes"].to_numpy(dtype=float) - flight
    full = rows["full_optimum_minutes"].to_numpy(dtype=float) - flight

    def metrics(index: np.ndarray) -> dict[str, float]:
        f = fixed[index]
        r = restricted[index]
        g = guarded[index]
        o = full[index]
        possible = f - o
        exact_saving = f - r
        model_saving = f - g
        return {
            "states": int(len(index)),
            "mean_fixed_current_charging_minutes": float(f.mean()),
            "mean_exact_factor_only_charging_minutes": float(r.mean()),
            "mean_guarded_model_charging_minutes": float(g.mean()),
            "mean_full_global_optimum_charging_minutes": float(o.mean()),
            "mean_exact_saving_vs_fixed_minutes": float(exact_saving.mean()),
            "mean_guarded_model_saving_vs_fixed_minutes": float(model_saving.mean()),
            "aggregate_exact_charge_improvement_vs_fixed_percent": float(
                100.0 * exact_saving.sum() / f.sum()
            ),
            "aggregate_guarded_model_charge_improvement_vs_fixed_percent": float(
                100.0 * model_saving.sum() / f.sum()
            ),
            "mean_exact_factor_only_gap_to_full_minutes": float((r - o).mean()),
            "exact_factor_only_excess_over_full_percent": float(
                100.0 * (r.sum() - o.sum()) / o.sum()
            ),
            "fraction_of_full_possible_charging_saving_captured_percent": float(
                100.0 * exact_saving.sum() / possible.sum()
            ),
        }

    overall = {"factor": factor, "charging_time_definition": "optimal K-pad makespan until all five batteries are fully charged", **metrics(np.arange(len(states)))}
    by_k: list[dict[str, object]] = []
    k_values = states["charging_pad_count"].to_numpy(dtype=int)
    for k in range(1, 6):
        index = np.flatnonzero(k_values == k)
        by_k.append({"factor": factor, "charging_pad_count": k, **metrics(index)})
    return overall, by_k


def main() -> int:
    overall_rows: list[dict[str, object]] = []
    k_rows: list[dict[str, object]] = []
    shared_columns = [
        "base_state_id",
        "charging_pad_count",
        "current_class_index",
        "current_structure",
        "current_position_slot_indices_json",
        "fixed_current_total_minutes",
        "full_oracle_total_minutes_reference",
    ]
    shared_reference: pd.DataFrame | None = None
    for factor in ("formation", "spacing", "position"):
        states_path, rows_path = _paths(factor)
        states = pd.read_csv(states_path)
        rows = pd.read_csv(rows_path)
        if shared_reference is None:
            shared_reference = states[shared_columns].copy()
        elif not shared_reference.equals(states[shared_columns]):
            raise RuntimeError(f"{factor}: current configurations are not paired")
        overall, by_k = _summarize(factor, states, rows)
        overall_rows.append(overall)
        k_rows.extend(by_k)

    overall_frame = pd.DataFrame(overall_rows).sort_values(
        "mean_exact_factor_only_charging_minutes"
    )
    by_k_frame = pd.DataFrame(k_rows).sort_values(["charging_pad_count", "factor"])
    overall_path = ROOT / "single_factor_charging_time_comparison.csv"
    by_k_path = ROOT / "single_factor_charging_time_comparison_by_k.csv"
    overall_frame.to_csv(overall_path, index=False)
    by_k_frame.to_csv(by_k_path, index=False)
    fixed = float(overall_frame["mean_fixed_current_charging_minutes"].iloc[0])
    full = float(overall_frame["mean_full_global_optimum_charging_minutes"].iloc[0])
    report = {
        "status": "pass",
        "comparison_is_paired": True,
        "charging_time_definition": "minimum K-pad makespan: time from arrival until all five batteries are fully charged",
        "shared_independent_states": int(overall_frame["states"].iloc[0]),
        "mean_fixed_current_charging_minutes": fixed,
        "mean_full_global_optimum_charging_minutes": full,
        "mean_full_possible_saving_minutes": fixed - full,
        "aggregate_full_charge_improvement_vs_fixed_percent": 100.0 * (fixed - full) / fixed,
        "overall_csv": str(overall_path.resolve()),
        "by_k_csv": str(by_k_path.resolve()),
        "overall": overall_frame.to_dict(orient="records"),
    }
    output = ROOT / "single_factor_charging_time_comparison.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
