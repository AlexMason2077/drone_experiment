"""Build the formation-only ablation dataset from the verified full cost matrix.

The state distribution and exact mission-cost calculations are unchanged.  The
candidate set is restricted to configurations with a fixed 75 cm spacing and
the identity drone-to-slot assignment (D1->slot 1, ..., D5->slot 5); only the
formation is allowed to vary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "analysis_outputs" / "ml_policy"
DEFAULT_CONTROLLED = DEFAULT_ROOT / "controlled_k_sweep"
DEFAULT_FULL_COSTS = DEFAULT_CONTROLLED / "cost_aware_ranker"
DEFAULT_CLASSES = (
    DEFAULT_ROOT
    / "expanded_25m_exponential_90min_interval30s"
    / "joint_full_configuration_ranker"
    / "complete_configuration_classes.csv"
)
DEFAULT_OUTPUT = DEFAULT_CONTROLLED / "ablation" / "formation_only_d75_identity"


def _formation_only_classes(
    full_classes: pd.DataFrame, spacing_cm: int
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = full_classes["structure"].astype(str).str.endswith(f"_{spacing_cm}")
    for drone in range(1, 6):
        mask &= full_classes[f"slot_index_d{drone}"].eq(drone)
    selected = full_classes.loc[mask].copy().sort_values("structure")
    expected_formations = {"column", "diamond", "echelon", "front", "vee"}
    actual_formations = {
        str(value).rsplit("_", 1)[0] for value in selected["structure"]
    }
    if actual_formations != expected_formations:
        raise ValueError(
            f"Expected {sorted(expected_formations)}, found {sorted(actual_formations)}"
        )
    source_indices = selected["class_index"].to_numpy(dtype=np.int64)
    selected.insert(0, "source_full_class_index", source_indices)
    selected["class_index"] = np.arange(len(selected), dtype=np.int64)
    return selected.reset_index(drop=True), source_indices


def _build_split(
    *,
    split_name: str,
    states_path: Path,
    full_costs_path: Path,
    output_dir: Path,
    restricted_classes: pd.DataFrame,
    source_indices: np.ndarray,
) -> dict[str, object]:
    states = pd.read_csv(states_path)
    full_archive = np.load(full_costs_path)
    full_costs = full_archive["costs"]
    if len(states) != len(full_costs):
        raise ValueError(
            f"{split_name}: {len(states)} states but {len(full_costs)} cost rows"
        )

    restricted_costs_all = full_costs[:, source_indices].astype(np.float32)
    feasible_counts_all = np.isfinite(restricted_costs_all).sum(axis=1)
    keep = feasible_counts_all > 0
    kept_states = states.loc[keep].copy().reset_index(drop=True)
    kept_costs = restricted_costs_all[keep]
    full_optimum = np.min(full_costs[keep], axis=1)
    restricted_optimum = np.min(kept_costs, axis=1)
    best_local_index = np.argmin(kept_costs, axis=1)

    kept_states["formation_only_reference_spacing_cm"] = 75
    kept_states["formation_only_reference_position"] = "D1:1|D2:2|D3:3|D4:4|D5:5"
    kept_states["formation_only_oracle_structure"] = restricted_classes.iloc[
        best_local_index
    ]["structure"].to_numpy()
    kept_states["formation_only_oracle_total_minutes"] = restricted_optimum
    kept_states["full_oracle_total_minutes_reference"] = full_optimum
    kept_states["formation_only_restriction_gap_minutes"] = (
        restricted_optimum - full_optimum
    )
    kept_states["formation_only_feasible_candidate_count"] = np.isfinite(
        kept_costs
    ).sum(axis=1)

    states_output = output_dir / f"{split_name}_states.csv"
    costs_output = output_dir / f"{split_name}_costs.npz"
    excluded_output = output_dir / f"{split_name}_infeasible_states.csv"
    kept_states.to_csv(states_output, index=False)
    states.loc[~keep].assign(
        formation_only_reference_spacing_cm=75,
        formation_only_reference_position="D1:1|D2:2|D3:3|D4:4|D5:5",
        formation_only_feasible_candidate_count=0,
    ).to_csv(excluded_output, index=False)
    np.savez_compressed(
        costs_output,
        costs=kept_costs,
        base_state_ids=kept_states["base_state_id"].to_numpy(dtype=np.int64),
        charging_pad_counts=kept_states["charging_pad_count"].to_numpy(dtype=np.int64),
    )

    condition_counts = (
        pd.DataFrame(
            {
                "wind_direction": states["wind_direction"],
                "wind_level": states["wind_level"],
                "eligible": keep.astype(int),
            }
        )
        .groupby(["wind_direction", "wind_level"], sort=True)["eligible"]
        .agg(["count", "sum"])
        .reset_index()
    )
    condition_counts["excluded"] = condition_counts["count"] - condition_counts["sum"]
    condition_counts["feasible_rate"] = condition_counts["sum"] / condition_counts["count"]
    condition_counts.rename(columns={"count": "states", "sum": "feasible"}).to_csv(
        output_dir / f"{split_name}_feasibility_by_condition.csv", index=False
    )

    return {
        "source_states": str(states_path.resolve()),
        "source_full_costs": str(full_costs_path.resolve()),
        "input_states": int(len(states)),
        "feasible_states": int(keep.sum()),
        "infeasible_states": int((~keep).sum()),
        "feasible_rate": float(keep.mean()),
        "mean_restricted_candidate_count_on_feasible_states": float(
            np.isfinite(kept_costs).sum(axis=1).mean()
        ),
        "mean_restriction_gap_minutes": float(
            np.mean(restricted_optimum - full_optimum)
        ),
        "median_restriction_gap_minutes": float(
            np.median(restricted_optimum - full_optimum)
        ),
        "states_output": str(states_output.resolve()),
        "costs_output": str(costs_output.resolve()),
        "infeasible_states_output": str(excluded_output.resolve()),
    }


def build_dataset(
    *,
    training_states: Path,
    independent_states: Path,
    training_costs: Path,
    independent_costs: Path,
    full_classes_path: Path,
    output_dir: Path,
    spacing_cm: int = 75,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_classes = pd.read_csv(full_classes_path)
    restricted_classes, source_indices = _formation_only_classes(
        full_classes, spacing_cm
    )
    class_output = output_dir / "formation_only_classes.csv"
    restricted_classes.to_csv(class_output, index=False)
    report = {
        "status": "pass",
        "ablation": "formation_only",
        "unchanged_state_distribution": True,
        "variable_factor": "formation",
        "fixed_inter_drone_spacing_cm": spacing_cm,
        "fixed_position_rule": "identity slot-index assignment: Di -> slot i",
        "candidate_formations": sorted(
            value.rsplit("_", 1)[0]
            for value in restricted_classes["structure"].astype(str)
        ),
        "candidate_count_before_condition_safety_filter": len(restricted_classes),
        "class_table": str(class_output.resolve()),
        "training": _build_split(
            split_name="training",
            states_path=training_states,
            full_costs_path=training_costs,
            output_dir=output_dir,
            restricted_classes=restricted_classes,
            source_indices=source_indices,
        ),
        "independent": _build_split(
            split_name="independent",
            states_path=independent_states,
            full_costs_path=independent_costs,
            output_dir=output_dir,
            restricted_classes=restricted_classes,
            source_indices=source_indices,
        ),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-states",
        type=Path,
        default=DEFAULT_CONTROLLED / "oracle_training_controlled_k_5000x5.csv",
    )
    parser.add_argument(
        "--independent-states",
        type=Path,
        default=DEFAULT_CONTROLLED / "oracle_independent_controlled_k_1000x5.csv",
    )
    parser.add_argument(
        "--training-costs",
        type=Path,
        default=DEFAULT_FULL_COSTS / "training_costs.npz",
    )
    parser.add_argument(
        "--independent-costs",
        type=Path,
        default=DEFAULT_FULL_COSTS / "independent_costs.npz",
    )
    parser.add_argument("--full-classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spacing-cm", type=int, default=75)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_dataset(
        training_states=args.training_states,
        independent_states=args.independent_states,
        training_costs=args.training_costs,
        independent_costs=args.independent_costs,
        full_classes_path=args.full_classes,
        output_dir=args.output_dir,
        spacing_cm=args.spacing_cm,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
