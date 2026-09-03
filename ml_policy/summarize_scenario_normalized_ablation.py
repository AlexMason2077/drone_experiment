"""Compute scenario-normalized paired ablation improvements."""

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
FACTORS = ("formation", "spacing", "position")


def _load(factor: str) -> pd.DataFrame:
    base = ROOT / f"{factor}_only_current_configuration"
    states = pd.read_csv(base / "independent_states.csv")
    audit = pd.read_csv(base / "model" / "independent_ablation_audit_rows.csv")
    flight = states["remaining_distance_m"].to_numpy(dtype=float) / 0.10 / 60.0
    frame = states[
        [
            "base_state_id",
            "wind_direction",
            "wind_level",
            "charging_pad_count",
            "remaining_distance_m",
            "current_class_index",
        ]
    ].copy()
    frame["fixed_charging_minutes"] = audit["fixed_minutes"].to_numpy(dtype=float) - flight
    frame["factor_only_charging_minutes"] = (
        audit["restricted_optimum_minutes"].to_numpy(dtype=float) - flight
    )
    frame["full_global_charging_minutes"] = (
        audit["full_optimum_minutes"].to_numpy(dtype=float) - flight
    )
    frame["factor"] = factor
    frame["scenario_improvement_percent"] = 100.0 * (
        frame["fixed_charging_minutes"] - frame["factor_only_charging_minutes"]
    ) / frame["fixed_charging_minutes"]
    frame["full_scenario_improvement_percent"] = 100.0 * (
        frame["fixed_charging_minutes"] - frame["full_global_charging_minutes"]
    ) / frame["fixed_charging_minutes"]
    return frame


def _bootstrap_grouped_mean(
    frame: pd.DataFrame, column: str, *, seed: int, draws: int = 3000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    grouped = {
        int(base_id): group[column].to_numpy(dtype=float)
        for base_id, group in frame.groupby("base_state_id", sort=False)
    }
    keys = np.asarray(list(grouped), dtype=np.int64)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        means[draw] = np.mean(np.concatenate([grouped[int(key)] for key in sampled]))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _summary(frame: pd.DataFrame, column: str, *, seed: int) -> dict[str, float]:
    value = frame[column].to_numpy(dtype=float)
    fixed = frame["fixed_charging_minutes"].to_numpy(dtype=float)
    if column == "full_scenario_improvement_percent":
        method = frame["full_global_charging_minutes"].to_numpy(dtype=float)
    else:
        method = frame["factor_only_charging_minutes"].to_numpy(dtype=float)
    stratum_means = frame.groupby(
        ["wind_direction", "wind_level", "charging_pad_count"], sort=True
    )[column].mean()
    low, high = _bootstrap_grouped_mean(frame, column, seed=seed)
    return {
        "states": int(len(frame)),
        "base_state_groups": int(frame["base_state_id"].nunique()),
        "mean_scenario_improvement_percent": float(value.mean()),
        "median_scenario_improvement_percent": float(np.median(value)),
        "standard_deviation_percent": float(value.std(ddof=1)),
        "q25_percent": float(np.quantile(value, 0.25)),
        "q75_percent": float(np.quantile(value, 0.75)),
        "grouped_bootstrap_95ci_low_percent": low,
        "grouped_bootstrap_95ci_high_percent": high,
        "equal_weight_mean_over_30_condition_k_strata_percent": float(
            stratum_means.mean()
        ),
        "aggregate_time_weighted_improvement_percent": float(
            100.0 * (fixed.sum() - method.sum()) / fixed.sum()
        ),
    }


def main() -> int:
    frames = {factor: _load(factor) for factor in FACTORS}
    reference = frames["formation"]
    key_columns = [
        "base_state_id",
        "wind_direction",
        "wind_level",
        "charging_pad_count",
        "remaining_distance_m",
        "current_class_index",
        "fixed_charging_minutes",
        "full_global_charging_minutes",
    ]
    for factor, frame in frames.items():
        if not reference[key_columns].equals(frame[key_columns]):
            raise RuntimeError(f"{factor}: paired states differ")

    records: list[dict[str, object]] = []
    for index, factor in enumerate(FACTORS):
        records.append(
            {
                "method": f"{factor}_only_exact",
                **_summary(
                    frames[factor], "scenario_improvement_percent", seed=20260825 + index
                ),
            }
        )
    records.append(
        {
            "method": "full_global_exact",
            **_summary(
                reference, "full_scenario_improvement_percent", seed=20260830
            ),
        }
    )
    summary = pd.DataFrame(records).sort_values(
        "mean_scenario_improvement_percent", ascending=False
    )
    summary_path = ROOT / "scenario_normalized_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    by_k_records: list[dict[str, object]] = []
    for factor, frame in frames.items():
        for k, group in frame.groupby("charging_pad_count", sort=True):
            by_k_records.append(
                {
                    "method": f"{factor}_only_exact",
                    "charging_pad_count": int(k),
                    "states": len(group),
                    "mean_scenario_improvement_percent": float(
                        group["scenario_improvement_percent"].mean()
                    ),
                    "median_scenario_improvement_percent": float(
                        group["scenario_improvement_percent"].median()
                    ),
                }
            )
    for k, group in reference.groupby("charging_pad_count", sort=True):
        by_k_records.append(
            {
                "method": "full_global_exact",
                "charging_pad_count": int(k),
                "states": len(group),
                "mean_scenario_improvement_percent": float(
                    group["full_scenario_improvement_percent"].mean()
                ),
                "median_scenario_improvement_percent": float(
                    group["full_scenario_improvement_percent"].median()
                ),
            }
        )
    by_k = pd.DataFrame(by_k_records).sort_values(
        ["charging_pad_count", "mean_scenario_improvement_percent"],
        ascending=[True, False],
    )
    by_k_path = ROOT / "scenario_normalized_ablation_by_k.csv"
    by_k.to_csv(by_k_path, index=False)
    report = {
        "status": "pass",
        "primary_metric": "mean of per-scenario relative charging-makespan improvements",
        "per_scenario_formula": "100 * (T_fixed_i - T_method_i) / T_fixed_i",
        "reason": "prevents longer-distance or longer-charging states from receiving larger implicit weight",
        "secondary_metric": "aggregate time-weighted improvement = 100 * sum(T_fixed_i - T_method_i) / sum(T_fixed_i)",
        "paired_states": len(reference),
        "paired_base_state_groups": int(reference["base_state_id"].nunique()),
        "summary_csv": str(summary_path.resolve()),
        "by_k_csv": str(by_k_path.resolve()),
        "summary": summary.to_dict(orient="records"),
    }
    output = ROOT / "scenario_normalized_ablation_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
