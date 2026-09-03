"""Build a spacing-only ablation using shared current configurations.

The current complete configuration is inherited row-for-row from the formal
formation-only dataset. This makes factor comparisons paired: every factor
sees the same wind, K, distance, SOC vector, and C_current.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = PROJECT_ROOT / "analysis_outputs" / "ml_policy"
CONTROLLED_ROOT = ML_ROOT / "controlled_k_sweep"
FULL_COST_ROOT = CONTROLLED_ROOT / "cost_aware_ranker"
SHARED_CURRENT_ROOT = CONTROLLED_ROOT / "ablation" / "formation_only_current_configuration"
OUTPUT_ROOT = CONTROLLED_ROOT / "ablation" / "spacing_only_current_configuration"
CLASS_PATH = (
    ML_ROOT
    / "expanded_25m_exponential_90min_interval30s"
    / "joint_full_configuration_ranker"
    / "complete_configuration_classes.csv"
)
SPACINGS = (50, 75)


def _lookup(classes: pd.DataFrame) -> dict[tuple[str, int, tuple[int, ...]], int]:
    result: dict[tuple[str, int, tuple[int, ...]], int] = {}
    for _, row in classes.iterrows():
        formation, spacing = str(row["structure"]).rsplit("_", 1)
        position = tuple(int(row[f"slot_index_d{i}"]) for i in range(1, 6))
        result[(formation, int(spacing), position)] = int(row["class_index"])
    if len(result) != 1200:
        raise ValueError(f"Expected 1200 complete classes, found {len(result)}")
    return result


def _parse_position(value: str) -> tuple[int, ...]:
    parsed = json.loads(value)
    return tuple(int(parsed[f"D{i}"]) for i in range(1, 6))


def _reference_metrics(
    fixed: np.ndarray, restricted: np.ndarray, full: np.ndarray
) -> dict[str, float]:
    saving = fixed - restricted
    return {
        "mean_exact_improvement_vs_fixed_percent": float(
            np.mean(100.0 * saving / fixed)
        ),
        "aggregate_exact_improvement_vs_fixed_percent": float(
            100.0 * np.sum(saving) / np.sum(fixed)
        ),
        "mean_exact_saving_vs_fixed_minutes": float(np.mean(saving)),
        "fraction_with_strict_improvement_vs_fixed": float(np.mean(saving > 2e-5)),
        "mean_exact_gap_to_full_minutes": float(np.mean(restricted - full)),
        "fraction_equal_to_full_optimum": float(
            np.mean(np.abs(restricted - full) <= 2e-5)
        ),
    }


def _build_split(
    *,
    name: str,
    shared_states_path: Path,
    full_costs_path: Path,
    classes: pd.DataFrame,
    lookup: dict[tuple[str, int, tuple[int, ...]], int],
    output_dir: Path,
) -> dict[str, object]:
    states = pd.read_csv(shared_states_path)
    full_costs = np.load(full_costs_path)["costs"]
    if full_costs.shape != (len(states), len(classes)):
        raise ValueError(f"{name}: state and cost dimensions differ")
    required = {
        "current_class_index",
        "current_structure",
        "current_formation",
        "current_inter_drone_spacing_cm",
        "current_position_slot_indices_json",
        "fixed_current_total_minutes",
    }
    missing = required.difference(states.columns)
    if missing:
        raise ValueError(f"{name}: shared current-state table lacks {sorted(missing)}")

    masked = np.full_like(full_costs, np.inf, dtype=np.float32)
    restricted_structure = np.empty(len(states), dtype=object)
    restricted_cost = np.zeros(len(states), dtype=np.float32)
    candidate_count = np.zeros(len(states), dtype=np.int8)
    fixed = states["fixed_current_total_minutes"].to_numpy(dtype=np.float32)
    full = np.min(full_costs, axis=1).astype(np.float32)

    for _, raw_indices in states.groupby("base_state_id", sort=True).indices.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        group = states.iloc[indices]
        if set(group["charging_pad_count"].astype(int)) != set(range(1, 6)):
            raise ValueError(f"{name}: a base-state group lacks K=1,...,5")
        for column in (
            "current_class_index",
            "current_structure",
            "current_formation",
            "current_inter_drone_spacing_cm",
            "current_position_slot_indices_json",
        ):
            if group[column].nunique() != 1:
                raise ValueError(f"{name}: current configuration changes across K in {column}")
        formation = str(group["current_formation"].iloc[0])
        position = _parse_position(str(group["current_position_slot_indices_json"].iloc[0]))
        candidates = [lookup[(formation, spacing, position)] for spacing in SPACINGS]
        current_class = int(group["current_class_index"].iloc[0])
        if current_class not in candidates:
            raise ValueError(f"{name}: inherited current configuration is not a spacing candidate")
        group_costs = full_costs[np.ix_(indices, candidates)]
        if not np.isfinite(group_costs[:, candidates.index(current_class)]).all():
            raise RuntimeError(f"{name}: inherited current configuration is infeasible")
        for row_index, row_costs in zip(indices, group_costs):
            finite = np.isfinite(row_costs)
            finite_classes = np.asarray(candidates)[finite]
            masked[row_index, finite_classes] = row_costs[finite]
            local_best = int(np.argmin(row_costs))
            restricted_structure[row_index] = str(
                classes.iloc[candidates[local_best]]["structure"]
            )
            restricted_cost[row_index] = float(row_costs[local_best])
            candidate_count[row_index] = int(finite.sum())

    if not np.isfinite(masked).any(axis=1).all():
        raise RuntimeError(f"{name}: a row has no spacing-only candidate")
    if not np.all(restricted_cost <= fixed + 2e-5):
        raise RuntimeError(f"{name}: spacing-only optimum exceeds fixed current cost")

    output = states.copy()
    output["spacing_only_oracle_structure"] = restricted_structure
    output["spacing_only_oracle_total_minutes"] = restricted_cost
    output["full_oracle_total_minutes_reference"] = full
    output["spacing_only_feasible_candidate_count"] = candidate_count
    output["spacing_only_improvement_vs_fixed_percent"] = np.where(
        fixed > 0, 100.0 * (fixed - restricted_cost) / fixed, 0.0
    )
    output["spacing_only_gap_to_full_minutes"] = restricted_cost - full
    states_output = output_dir / f"{name}_states.csv"
    costs_output = output_dir / f"{name}_costs.npz"
    output.to_csv(states_output, index=False)
    np.savez_compressed(
        costs_output,
        costs=masked,
        base_state_ids=output["base_state_id"].to_numpy(dtype=np.int64),
        charging_pad_counts=output["charging_pad_count"].to_numpy(dtype=np.int64),
    )

    base = output.drop_duplicates("base_state_id")
    coverage = (
        base.groupby(
            ["current_formation", "current_position_slot_indices_json"], sort=True
        )
        .size()
        .rename("base_state_groups")
        .reset_index()
    )
    coverage_output = output_dir / f"{name}_current_context_coverage.csv"
    coverage.to_csv(coverage_output, index=False)
    return {
        "input_states": len(states),
        "base_state_groups": int(states["base_state_id"].nunique()),
        "same_current_configuration_as_formation_only_dataset": True,
        "all_rows_have_a_feasible_current_configuration": True,
        "all_rows_have_a_spacing_only_label": True,
        "formation_position_contexts_used": len(coverage),
        "possible_formation_position_contexts": 600,
        "mean_feasible_spacings_per_state": float(candidate_count.mean()),
        **_reference_metrics(fixed, restricted_cost, full),
        "source_shared_current_states": str(shared_states_path.resolve()),
        "states_output": str(states_output.resolve()),
        "costs_output": str(costs_output.resolve()),
        "context_coverage_output": str(coverage_output.resolve()),
    }


def build(
    *,
    training_shared_states: Path,
    independent_shared_states: Path,
    training_costs: Path,
    independent_costs: Path,
    classes_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = pd.read_csv(classes_path).sort_values("class_index").reset_index(drop=True)
    if not np.array_equal(classes["class_index"].to_numpy(), np.arange(len(classes))):
        raise ValueError("Class indices must be contiguous and zero-based")
    lookup = _lookup(classes)
    class_output = output_dir / "complete_configuration_classes.csv"
    classes.to_csv(class_output, index=False)
    report = {
        "status": "pass",
        "ablation": "inter_drone_spacing_only_with_shared_current_configuration",
        "paired_factor_comparison": True,
        "variable_factor": "inter_drone_spacing",
        "fixed_factors": ["formation", "position"],
        "candidate_rule": "retain f_current and p_current; vary spacing over {50,75} cm",
        "switching_time_assumption": "zero",
        "class_table": str(class_output.resolve()),
        "training": _build_split(
            name="training",
            shared_states_path=training_shared_states,
            full_costs_path=training_costs,
            classes=classes,
            lookup=lookup,
            output_dir=output_dir,
        ),
        "independent": _build_split(
            name="independent",
            shared_states_path=independent_shared_states,
            full_costs_path=independent_costs,
            classes=classes,
            lookup=lookup,
            output_dir=output_dir,
        ),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-shared-states",
        type=Path,
        default=SHARED_CURRENT_ROOT / "training_states.csv",
    )
    parser.add_argument(
        "--independent-shared-states",
        type=Path,
        default=SHARED_CURRENT_ROOT / "independent_states.csv",
    )
    parser.add_argument(
        "--training-costs", type=Path, default=FULL_COST_ROOT / "training_costs.npz"
    )
    parser.add_argument(
        "--independent-costs", type=Path, default=FULL_COST_ROOT / "independent_costs.npz"
    )
    parser.add_argument("--classes", type=Path, default=CLASS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build(
        training_shared_states=args.training_shared_states,
        independent_shared_states=args.independent_shared_states,
        training_costs=args.training_costs,
        independent_costs=args.independent_costs,
        classes_path=args.classes,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
