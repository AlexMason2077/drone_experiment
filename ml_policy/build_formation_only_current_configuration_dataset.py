"""Build a formal formation-only ablation dataset with a current configuration.

For each controlled-K base-state group, this builder selects one feasible
current complete configuration C_current=(f_current,d_current,p_current).  The
same C_current is used for all K=1,...,5 rows in that group.  Exact labels are
then recomputed over only the formations that retain d_current and p_current.
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = PROJECT_ROOT / "analysis_outputs" / "ml_policy"
CONTROLLED_ROOT = ML_ROOT / "controlled_k_sweep"
FULL_COST_ROOT = CONTROLLED_ROOT / "cost_aware_ranker"
CLASS_PATH = (
    ML_ROOT
    / "expanded_25m_exponential_90min_interval30s"
    / "joint_full_configuration_ranker"
    / "complete_configuration_classes.csv"
)
OUTPUT_ROOT = (
    CONTROLLED_ROOT / "ablation" / "formation_only_current_configuration"
)
FORMATIONS = ("column", "diamond", "echelon", "front", "vee")
SPACINGS = (50, 75)


def _class_lookup(classes: pd.DataFrame) -> dict[tuple[str, int, tuple[int, ...]], int]:
    lookup: dict[tuple[str, int, tuple[int, ...]], int] = {}
    for _, row in classes.iterrows():
        structure = str(row["structure"])
        formation, spacing_text = structure.rsplit("_", 1)
        permutation = tuple(
            int(row[f"slot_index_d{drone}"]) for drone in range(1, 6)
        )
        lookup[(formation, int(spacing_text), permutation)] = int(row["class_index"])
    expected = len(FORMATIONS) * len(SPACINGS) * 120
    if len(lookup) != expected:
        raise ValueError(f"Expected {expected} complete classes, found {len(lookup)}")
    return lookup


def _position_json(permutation: tuple[int, ...]) -> str:
    return json.dumps(
        {f"D{drone}": int(slot) for drone, slot in enumerate(permutation, start=1)},
        separators=(",", ":"),
    )


def _build_split(
    *,
    name: str,
    states_path: Path,
    full_costs_path: Path,
    classes: pd.DataFrame,
    lookup: dict[tuple[str, int, tuple[int, ...]], int],
    output_dir: Path,
    random_seed: int,
) -> dict[str, object]:
    states = pd.read_csv(states_path)
    full_archive = np.load(full_costs_path)
    full_costs = full_archive["costs"]
    if full_costs.shape != (len(states), len(classes)):
        raise ValueError(
            f"{name}: costs shape {full_costs.shape} does not match "
            f"({len(states)}, {len(classes)})"
        )

    rng = np.random.default_rng(random_seed)
    contexts = [
        (spacing, tuple(int(value) for value in permutation))
        for spacing in SPACINGS
        for permutation in permutations(range(1, 6))
    ]
    rng.shuffle(contexts)
    masked_costs = np.full_like(full_costs, np.inf, dtype=np.float32)
    output_states = states.copy()
    current_class = np.full(len(states), -1, dtype=np.int32)
    current_structure = np.empty(len(states), dtype=object)
    current_formation = np.empty(len(states), dtype=object)
    current_spacing = np.zeros(len(states), dtype=np.int16)
    current_position = np.empty(len(states), dtype=object)
    restricted_structure = np.empty(len(states), dtype=object)
    restricted_cost = np.zeros(len(states), dtype=np.float32)
    fixed_cost = np.zeros(len(states), dtype=np.float32)
    full_optimum = np.min(full_costs, axis=1).astype(np.float32)
    candidate_count = np.zeros(len(states), dtype=np.int8)
    attempts_per_group: list[int] = []
    context_usage: dict[tuple[int, tuple[int, ...]], int] = {}

    groups = list(states.groupby("base_state_id", sort=True).indices.items())
    for group_number, (_, raw_indices) in enumerate(groups):
        indices = np.asarray(raw_indices, dtype=np.int64)
        if set(states.iloc[indices]["charging_pad_count"].astype(int)) != set(range(1, 6)):
            raise ValueError(f"{name}: a base-state group does not contain K=1,...,5")

        selected_context: tuple[int, tuple[int, ...]] | None = None
        selected_candidates: list[int] | None = None
        start = group_number % len(contexts)
        attempts = 0
        for offset in range(len(contexts)):
            spacing, permutation = contexts[(start + offset) % len(contexts)]
            candidates = [
                lookup[(formation, spacing, permutation)] for formation in FORMATIONS
            ]
            # A current configuration should be feasible for every controlled-K
            # copy of the same physical state.  Feasibility is independent of K,
            # but checking all rows also protects against corrupted cost matrices.
            finite_for_all_k = np.isfinite(full_costs[np.ix_(indices, candidates)]).all(axis=0)
            attempts += 1
            if finite_for_all_k.any():
                selected_context = (spacing, permutation)
                selected_candidates = candidates
                break
        if selected_context is None or selected_candidates is None:
            raise RuntimeError(f"{name}: no feasible formation-only context for a group")

        attempts_per_group.append(attempts)
        context_usage[selected_context] = context_usage.get(selected_context, 0) + 1
        spacing, permutation = selected_context
        group_candidate_costs = full_costs[np.ix_(indices, selected_candidates)]
        common_feasible = np.isfinite(group_candidate_costs).all(axis=0)
        feasible_current_candidates = np.asarray(selected_candidates)[common_feasible]
        chosen_current = int(rng.choice(feasible_current_candidates))
        current_row = classes.iloc[chosen_current]

        for row_index, row_costs in zip(indices, group_candidate_costs):
            finite = np.isfinite(row_costs)
            masked_costs[row_index, np.asarray(selected_candidates)[finite]] = row_costs[finite]
            local_best = int(np.argmin(row_costs))
            best_class = selected_candidates[local_best]
            current_class[row_index] = chosen_current
            current_structure[row_index] = str(current_row["structure"])
            current_formation[row_index] = str(current_row["structure"]).rsplit("_", 1)[0]
            current_spacing[row_index] = spacing
            current_position[row_index] = _position_json(permutation)
            restricted_structure[row_index] = str(classes.iloc[best_class]["structure"])
            restricted_cost[row_index] = float(row_costs[local_best])
            fixed_cost[row_index] = float(full_costs[row_index, chosen_current])
            candidate_count[row_index] = int(finite.sum())

    if (current_class < 0).any() or not np.isfinite(masked_costs).any(axis=1).all():
        raise RuntimeError(f"{name}: incomplete formation-only labels")
    if not np.isfinite(fixed_cost).all():
        raise RuntimeError(f"{name}: a sampled current configuration is infeasible")
    if not np.all(restricted_cost <= fixed_cost + 2e-5):
        raise RuntimeError(f"{name}: restricted optimum is worse than fixed current config")

    output_states["current_class_index"] = current_class
    output_states["current_structure"] = current_structure
    output_states["current_formation"] = current_formation
    output_states["current_inter_drone_spacing_cm"] = current_spacing
    output_states["current_position_slot_indices_json"] = current_position
    output_states["fixed_current_total_minutes"] = fixed_cost
    output_states["formation_only_oracle_structure"] = restricted_structure
    output_states["formation_only_oracle_total_minutes"] = restricted_cost
    output_states["full_oracle_total_minutes_reference"] = full_optimum
    output_states["formation_only_feasible_candidate_count"] = candidate_count
    output_states["formation_only_improvement_vs_fixed_percent"] = np.where(
        fixed_cost > 0,
        100.0 * (fixed_cost - restricted_cost) / fixed_cost,
        0.0,
    )
    output_states["formation_only_gap_to_full_minutes"] = restricted_cost - full_optimum

    states_output = output_dir / f"{name}_states.csv"
    costs_output = output_dir / f"{name}_costs.npz"
    output_states.to_csv(states_output, index=False)
    np.savez_compressed(
        costs_output,
        costs=masked_costs,
        base_state_ids=output_states["base_state_id"].to_numpy(dtype=np.int64),
        charging_pad_counts=output_states["charging_pad_count"].to_numpy(dtype=np.int64),
    )

    context_frame = pd.DataFrame(
        [
            {
                "spacing_cm": spacing,
                "position_slot_indices_json": _position_json(permutation),
                "base_state_groups": count,
            }
            for (spacing, permutation), count in sorted(
                context_usage.items(), key=lambda item: (item[0][0], item[0][1])
            )
        ]
    )
    context_output = output_dir / f"{name}_current_context_coverage.csv"
    context_frame.to_csv(context_output, index=False)

    return {
        "input_states": int(len(states)),
        "base_state_groups": int(states["base_state_id"].nunique()),
        "all_rows_have_a_feasible_current_configuration": True,
        "all_rows_have_a_formation_only_label": True,
        "current_contexts_used": int(len(context_usage)),
        "possible_spacing_position_contexts": int(len(contexts)),
        "spacing_group_counts": {
            str(int(key)): int(value)
            for key, value in context_frame.groupby("spacing_cm")["base_state_groups"].sum().items()
        },
        "mean_context_search_attempts": float(np.mean(attempts_per_group)),
        "mean_feasible_formations_per_state": float(candidate_count.mean()),
        "mean_exact_improvement_vs_fixed_percent": float(
            output_states["formation_only_improvement_vs_fixed_percent"].mean()
        ),
        "median_exact_improvement_vs_fixed_percent": float(
            output_states["formation_only_improvement_vs_fixed_percent"].median()
        ),
        "fraction_with_strict_improvement_vs_fixed": float(
            np.mean(restricted_cost < fixed_cost - 2e-5)
        ),
        "mean_exact_gap_to_full_minutes": float(np.mean(restricted_cost - full_optimum)),
        "fraction_equal_to_full_optimum": float(
            np.mean(np.abs(restricted_cost - full_optimum) <= 2e-5)
        ),
        "states_output": str(states_output.resolve()),
        "costs_output": str(costs_output.resolve()),
        "context_coverage_output": str(context_output.resolve()),
    }


def build(
    *,
    training_states: Path,
    independent_states: Path,
    training_costs: Path,
    independent_costs: Path,
    classes_path: Path,
    output_dir: Path,
    random_seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = pd.read_csv(classes_path).sort_values("class_index").reset_index(drop=True)
    if not np.array_equal(classes["class_index"].to_numpy(), np.arange(len(classes))):
        raise ValueError("Class table must have contiguous zero-based class_index values")
    lookup = _class_lookup(classes)
    class_output = output_dir / "complete_configuration_classes.csv"
    classes.to_csv(class_output, index=False)
    report = {
        "status": "pass",
        "ablation": "formation_only_with_current_configuration",
        "state_input_includes_current_configuration": True,
        "variable_factor": "formation",
        "fixed_factors": ["inter_drone_spacing", "position"],
        "candidate_rule": "retain d_current and p_current; vary formation only",
        "switching_time_assumption": "zero; current formation does not alter the new optimum label",
        "controlled_k_rule": "one identical current configuration for all K=1,...,5 rows of each base_state_id",
        "random_seed": random_seed,
        "class_table": str(class_output.resolve()),
        "training": _build_split(
            name="training",
            states_path=training_states,
            full_costs_path=training_costs,
            classes=classes,
            lookup=lookup,
            output_dir=output_dir,
            random_seed=random_seed,
        ),
        "independent": _build_split(
            name="independent",
            states_path=independent_states,
            full_costs_path=independent_costs,
            classes=classes,
            lookup=lookup,
            output_dir=output_dir,
            random_seed=random_seed + 10_000,
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
        default=CONTROLLED_ROOT / "oracle_training_controlled_k_5000x5.csv",
    )
    parser.add_argument(
        "--independent-states",
        type=Path,
        default=CONTROLLED_ROOT / "oracle_independent_controlled_k_1000x5.csv",
    )
    parser.add_argument(
        "--training-costs", type=Path, default=FULL_COST_ROOT / "training_costs.npz"
    )
    parser.add_argument(
        "--independent-costs", type=Path, default=FULL_COST_ROOT / "independent_costs.npz"
    )
    parser.add_argument("--classes", type=Path, default=CLASS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--random-seed", type=int, default=20260824)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build(
        training_states=args.training_states,
        independent_states=args.independent_states,
        training_costs=args.training_costs,
        independent_costs=args.independent_costs,
        classes_path=args.classes,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
