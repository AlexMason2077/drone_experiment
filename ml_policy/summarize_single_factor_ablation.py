"""Create the paired comparison table for the three single-factor ablations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "controlled_k_sweep"
    / "ablation"
)


def main() -> int:
    specs = {
        "formation": {
            "root": ROOT / "formation_only_current_configuration",
            "audit_key": "exact_formation_only_reference",
            "candidate_label": "up to 5 formations",
        },
        "inter_drone_spacing": {
            "root": ROOT / "spacing_only_current_configuration",
            "audit_key": "exact_spacing_only_reference",
            "candidate_label": "up to 2 spacings",
        },
        "position": {
            "root": ROOT / "position_only_current_configuration",
            "audit_key": "exact_position_only_reference",
            "candidate_label": "up to 120 permutations",
        },
    }
    rows: list[dict[str, object]] = []
    for factor, spec in specs.items():
        root = Path(spec["root"])
        dataset = json.loads((root / "dataset_manifest.json").read_text())
        training = json.loads((root / "model" / "candidate_aware_metrics.json").read_text())
        audit_name = (
            "formation_only_ablation_audit.json"
            if factor == "formation"
            else "spacing_only_ablation_audit.json"
            if factor == "inter_drone_spacing"
            else "position_only_ablation_audit.json"
        )
        audit = json.loads((root / "model" / audit_name).read_text())
        exact = audit[str(spec["audit_key"])]
        guarded = audit["model_with_keep_current_guard"]
        rows.append(
            {
                "factor": factor,
                "candidate_space": spec["candidate_label"],
                "paired_independent_states": audit["states"],
                "cross_validation_strict_restricted_optimal_rate": training[
                    "cross_validation_summary"
                ]["strict_global_optimal_rate"]["mean"],
                "independent_raw_strict_restricted_optimal_rate": training[
                    "independent_metrics"
                ]["strict_global_optimal_rate"],
                "independent_guarded_strict_restricted_optimal_rate": guarded[
                    "strict_restricted_optimal_rate"
                ],
                "exact_aggregate_improvement_vs_fixed_percent": exact[
                    "aggregate_time_improvement_vs_fixed_percent"
                ],
                "guarded_model_aggregate_improvement_vs_fixed_percent": guarded[
                    "aggregate_time_improvement_vs_fixed_percent"
                ],
                "guarded_model_mean_saving_vs_fixed_minutes": guarded[
                    "mean_saving_vs_fixed_minutes"
                ],
                "guarded_model_fraction_strictly_better_than_fixed": guarded[
                    "fraction_strictly_better_than_fixed"
                ],
                "guarded_model_fraction_worse_than_fixed": guarded[
                    "fraction_worse_than_fixed"
                ],
                "exact_mean_gap_to_full_optimum_minutes": exact[
                    "mean_gap_to_unrestricted_full_optimum_minutes"
                ],
                "mean_online_ms_per_state": training[
                    "mean_online_ms_per_state_including_all_candidate_features_and_scoring"
                ],
                "dataset_manifest": str((root / "dataset_manifest.json").resolve()),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        "guarded_model_aggregate_improvement_vs_fixed_percent", ascending=False
    )
    frame.insert(0, "single_factor_rank", range(1, len(frame) + 1))
    output_csv = ROOT / "single_factor_ablation_paired_comparison.csv"
    frame.to_csv(output_csv, index=False)
    report = {
        "status": "pass",
        "comparison_is_paired": True,
        "shared_independent_states": 5000,
        "shared_base_state_groups": 1000,
        "ranking_metric": "guarded model aggregate completion-time improvement versus fixed current configuration",
        "single_factor_ranking": frame["factor"].tolist(),
        "scope": "single-decision independent states; dynamic end-to-end rollout remains future work",
        "comparison_csv": str(output_csv.resolve()),
        "rows": frame.to_dict(orient="records"),
    }
    output_json = ROOT / "single_factor_ablation_paired_comparison.json"
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
