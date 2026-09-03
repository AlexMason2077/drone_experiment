#!/usr/bin/env python3
"""Audit the association between initial SOC and processed forward discharge rate.

The analysis uses the paper's current run-level output: forward-movement-only,
Bideal-normalized discharge rates.  The primary observation is one physical run,
summarised across the five drones, so the five nested drone rows are not treated
as five independent experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis_outputs" / "initial_soc_effect_study"
RATE_FILE = (
    ROOT
    / "analysis_outputs"
    / "configuration_condition_rate_bar_charts"
    / "run_level_configuration_drone_Bideal_forward_rates.csv"
)
SELECTION_FILE = (
    ROOT
    / "analysis_outputs"
    / "forward_discharge_rate_modeling"
    / "selected_runs_by_database_cell.csv"
)

CELL_COLUMNS = [
    "formation",
    "inter_drone_spacing_cm",
    "wind_direction",
    "wind_level",
]
RUN_KEY = CELL_COLUMNS + ["experiment_directory", "run_id"]


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["wind_level"] = pd.to_numeric(frame["wind_level"], errors="coerce").astype("Int64")
    frame["inter_drone_spacing_cm"] = pd.to_numeric(
        frame["inter_drone_spacing_cm"], errors="coerce"
    ).astype("Int64")
    frame["run_id"] = frame["run_id"].astype(str)
    return frame


def load_run_level_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rates = _normalise_keys(pd.read_csv(RATE_FILE))
    selections = _normalise_keys(pd.read_csv(SELECTION_FILE))

    selected = selections.loc[
        selections["selection_status"].eq("selected"),
        RUN_KEY
        + [
            "run_start_soc_min_pct",
            "run_start_soc_median_pct",
            "run_start_soc_mean_pct",
            "run_start_soc_max_pct",
            "soc_selection_stratum",
            "primary_analysis_status",
            "flat_full_window_drone_count",
        ],
    ].copy()

    run_rates = (
        rates.groupby(RUN_KEY, as_index=False, dropna=False)
        .agg(
            run_rate_mean_pp_per_min=("run_Bideal_forward_rate_pp_per_min", "mean"),
            run_rate_median_pp_per_min=("run_Bideal_forward_rate_pp_per_min", "median"),
            run_rate_sd_pp_per_min=("run_Bideal_forward_rate_pp_per_min", "std"),
            drone_count=("drone_name", "nunique"),
        )
        .merge(selected, on=RUN_KEY, how="left", validate="one_to_one")
    )
    run_rates["condition"] = run_rates[CELL_COLUMNS].astype(str).agg(" | ".join, axis=1)
    run_rates["run_timestamp"] = pd.to_datetime(
        run_rates["run_id"], format="%Y%m%d_%H%M%S", errors="coerce"
    )
    run_rates["chronological_rank_in_condition"] = run_rates.groupby("condition")[
        "run_timestamp"
    ].rank(method="first")
    run_rates["within_swarm_soc_spread_pp"] = (
        run_rates["run_start_soc_max_pct"] - run_rates["run_start_soc_min_pct"]
    )

    if len(rates) != 780 or len(run_rates) != 156:
        raise ValueError(
            f"Unexpected selected data size: {len(rates)} drone rows, {len(run_rates)} runs"
        )
    if run_rates["run_start_soc_mean_pct"].isna().any():
        raise ValueError("Selected run rates did not join completely to starting-SOC metadata")
    if not run_rates["drone_count"].eq(5).all():
        raise ValueError("A selected run does not contain exactly five drone-rate rows")

    return rates, run_rates


def build_condition_summary(run_rates: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for condition, group in run_rates.groupby("condition", sort=True):
        group = group.sort_values("run_start_soc_mean_pct")
        first = group.iloc[0]
        soc = group["run_start_soc_mean_pct"].to_numpy(dtype=float)
        rate_mean = group["run_rate_mean_pp_per_min"].to_numpy(dtype=float)
        rate_median = group["run_rate_median_pp_per_min"].to_numpy(dtype=float)
        comparable = len(group) >= 2 and np.unique(soc).size >= 2
        middle_index = len(group) // 2
        slope_mean = float(np.polyfit(soc, rate_mean, 1)[0]) if comparable else np.nan
        slope_median = float(np.polyfit(soc, rate_median, 1)[0]) if comparable else np.nan
        records.append(
            {
                **{column: first[column] for column in CELL_COLUMNS},
                "condition": condition,
                "selected_run_count": int(len(group)),
                "comparable_soc": bool(comparable),
                "soc_low_pct": float(soc[0]),
                "soc_middle_pct": float(soc[middle_index]),
                "soc_high_pct": float(soc[-1]),
                "soc_range_pp": float(soc[-1] - soc[0]),
                "rate_at_low_soc_pp_per_min": float(rate_mean[0]),
                "rate_at_middle_soc_pp_per_min": float(rate_mean[middle_index]),
                "rate_at_high_soc_pp_per_min": float(rate_mean[-1]),
                "low_minus_high_rate_pp_per_min": float(rate_mean[0] - rate_mean[-1]),
                "effect_of_10pp_lower_soc_pp_per_min": -10.0 * slope_mean,
                "median_drone_effect_of_10pp_lower_soc_pp_per_min": -10.0
                * slope_median,
                "strict_low_gt_middle_gt_high": bool(
                    len(group) == 3 and rate_mean[0] > rate_mean[1] > rate_mean[2]
                ),
                "low_soc_run_later_than_high_soc_run": bool(
                    comparable
                    and group.iloc[0]["run_timestamp"] > group.iloc[-1]["run_timestamp"]
                ),
                "low_and_high_runs_same_day": bool(
                    comparable
                    and group.iloc[0]["run_timestamp"].date()
                    == group.iloc[-1]["run_timestamp"].date()
                ),
            }
        )
    result = pd.DataFrame.from_records(records)
    result["direction"] = np.select(
        [
            ~result["comparable_soc"],
            result["low_minus_high_rate_pp_per_min"] > 0,
            result["low_minus_high_rate_pp_per_min"] < 0,
        ],
        ["not_comparable", "lower_soc_higher_rate", "opposite_direction"],
        default="tie",
    )
    return result


def fixed_effect_bootstrap(
    run_rates: pd.DataFrame,
    outcome: str,
    *,
    seed: int = 20260827,
    draws: int = 100_000,
) -> dict[str, float]:
    pieces = []
    for _, group in run_rates.groupby("condition"):
        x = group["run_start_soc_mean_pct"].to_numpy(dtype=float)
        y = group[outcome].to_numpy(dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        pieces.append(((x * y).sum(), (x * x).sum()))
    components = np.asarray(pieces, dtype=float)
    slope = components[:, 0].sum() / components[:, 1].sum()
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, len(components), size=(draws, len(components)), endpoint=False
    )
    bootstrap_slopes = (
        components[sampled_indices, 0].sum(axis=1)
        / components[sampled_indices, 1].sum(axis=1)
    )
    effect = -10.0 * slope
    effect_draws = -10.0 * bootstrap_slopes
    return {
        "effect_of_10pp_lower_soc_pp_per_min": float(effect),
        "bootstrap_ci_low": float(np.quantile(effect_draws, 0.025)),
        "bootstrap_ci_high": float(np.quantile(effect_draws, 0.975)),
        "bootstrap_draws": int(draws),
    }


def build_explicit_strata_summary(run_rates: pd.DataFrame) -> pd.DataFrame:
    explicit = run_rates.loc[
        run_rates["soc_selection_stratum"].isin(["low", "middle", "high"])
    ].copy()
    records = []
    for condition, group in explicit.groupby("condition"):
        if set(group["soc_selection_stratum"]) != {"low", "middle", "high"}:
            continue
        indexed = group.set_index("soc_selection_stratum")
        records.append(
            {
                **{column: group.iloc[0][column] for column in CELL_COLUMNS},
                "condition": condition,
                "high_soc_pct": float(indexed.loc["high", "run_start_soc_mean_pct"]),
                "middle_soc_pct": float(indexed.loc["middle", "run_start_soc_mean_pct"]),
                "low_soc_pct": float(indexed.loc["low", "run_start_soc_mean_pct"]),
                "high_rate_pp_per_min": float(
                    indexed.loc["high", "run_rate_mean_pp_per_min"]
                ),
                "middle_rate_pp_per_min": float(
                    indexed.loc["middle", "run_rate_mean_pp_per_min"]
                ),
                "low_rate_pp_per_min": float(
                    indexed.loc["low", "run_rate_mean_pp_per_min"]
                ),
                "low_minus_high_rate_pp_per_min": float(
                    indexed.loc["low", "run_rate_mean_pp_per_min"]
                    - indexed.loc["high", "run_rate_mean_pp_per_min"]
                ),
                "strict_low_gt_middle_gt_high": bool(
                    indexed.loc["low", "run_rate_mean_pp_per_min"]
                    > indexed.loc["middle", "run_rate_mean_pp_per_min"]
                    > indexed.loc["high", "run_rate_mean_pp_per_min"]
                ),
                "low_swarm_soc_spread_pp": float(
                    indexed.loc["low", "within_swarm_soc_spread_pp"]
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def primary_only_sensitivity(run_rates: pd.DataFrame) -> dict[str, float | int]:
    primary = run_rates.loc[
        run_rates["primary_analysis_status"].eq("eligible_primary_75_to_40")
    ]
    slopes = []
    deltas = []
    for _, group in primary.groupby("condition"):
        if len(group) < 2 or group["run_start_soc_mean_pct"].nunique() < 2:
            continue
        ordered = group.sort_values("run_start_soc_mean_pct")
        slopes.append(
            float(
                np.polyfit(
                    group["run_start_soc_mean_pct"],
                    group["run_rate_mean_pp_per_min"],
                    1,
                )[0]
            )
        )
        deltas.append(
            float(
                ordered.iloc[0]["run_rate_mean_pp_per_min"]
                - ordered.iloc[-1]["run_rate_mean_pp_per_min"]
            )
        )
    return {
        "selected_primary_runs": int(len(primary)),
        "comparable_conditions": int(len(slopes)),
        "lower_soc_higher_rate_conditions": int(np.sum(np.asarray(slopes) < 0)),
        "opposite_direction_conditions": int(np.sum(np.asarray(slopes) > 0)),
        "median_low_minus_high_rate_pp_per_min": float(np.median(deltas)),
    }


def drone_level_direction_sensitivity(
    drone_rates: pd.DataFrame, run_rates: pd.DataFrame
) -> dict[str, int | float]:
    metadata = run_rates[RUN_KEY + ["run_start_soc_mean_pct"]]
    nested = drone_rates.merge(metadata, on=RUN_KEY, how="inner", validate="many_to_one")
    nested["condition"] = nested[CELL_COLUMNS].astype(str).agg(" | ".join, axis=1)
    slopes = []
    for _, group in nested.groupby(["condition", "drone_name"]):
        if len(group) < 2 or group["run_start_soc_mean_pct"].nunique() < 2:
            continue
        slopes.append(
            float(
                np.polyfit(
                    group["run_start_soc_mean_pct"],
                    group["run_Bideal_forward_rate_pp_per_min"],
                    1,
                )[0]
            )
        )
    slopes_array = np.asarray(slopes)
    return {
        "comparable_condition_drone_series": int(len(slopes_array)),
        "lower_soc_higher_rate_series": int(np.sum(slopes_array < 0)),
        "opposite_direction_series": int(np.sum(slopes_array > 0)),
        "share_lower_soc_higher_rate": float(np.mean(slopes_array < 0)),
    }


def plot_condition_effects(condition_summary: pd.DataFrame) -> None:
    plotted = condition_summary.loc[condition_summary["comparable_soc"]].copy()
    formation_order = sorted(plotted["formation"].unique())
    palette = {
        formation: color
        for formation, color in zip(
            formation_order,
            ["#315C8C", "#C58A1A", "#D46A3A", "#758C3A", "#B85C8A"],
            strict=False,
        )
    }
    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    for formation in formation_order:
        subset = plotted.loc[plotted["formation"].eq(formation)]
        axis.scatter(
            subset["soc_range_pp"],
            subset["low_minus_high_rate_pp_per_min"],
            s=54,
            alpha=0.85,
            label=formation,
            color=palette[formation],
            edgecolor="#24303B",
            linewidth=0.45,
        )
    axis.axhline(0, color="#2F3640", linewidth=1.1)
    axis.set_title("Initial-SOC contrast across experimental conditions", loc="left")
    axis.set_xlabel("Difference between highest and lowest starting SOC (percentage points)")
    axis.set_ylabel("Low-SOC rate minus high-SOC rate (pp/min)")
    axis.grid(axis="both", color="#D7DCE1", linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(title="Formation", frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    axis.text(
        0.99,
        0.99,
        "✦",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=16,
        color="#315C8C",
    )
    fig.text(
        0.01,
        0.01,
        "Each point is one formation × spacing × wind direction × level condition; rates are forward-only and Bideal-normalized.",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUTPUT / "condition_soc_effect_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_analysis() -> dict[str, object]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    drone_rates, run_rates = load_run_level_data()
    condition_summary = build_condition_summary(run_rates)
    explicit = build_explicit_strata_summary(run_rates)

    comparable = condition_summary.loc[condition_summary["comparable_soc"]]
    sufficient_range = comparable.loc[comparable["soc_range_pp"].ge(5)]
    explicit_low_minus_high = explicit["low_minus_high_rate_pp_per_min"]

    within_soc = run_rates["run_start_soc_mean_pct"] - run_rates.groupby("condition")[
        "run_start_soc_mean_pct"
    ].transform("mean")
    within_order = run_rates["chronological_rank_in_condition"] - run_rates.groupby(
        "condition"
    )["chronological_rank_in_condition"].transform("mean")
    order_association = stats.spearmanr(within_soc, within_order)

    summary: dict[str, object] = {
        "source_drone_rows": int(len(drone_rates)),
        "selected_physical_runs": int(len(run_rates)),
        "represented_conditions": int(condition_summary["condition"].nunique()),
        "comparable_conditions": int(len(comparable)),
        "lower_soc_higher_rate_conditions": int(
            comparable["low_minus_high_rate_pp_per_min"].gt(0).sum()
        ),
        "opposite_direction_conditions": int(
            comparable["low_minus_high_rate_pp_per_min"].lt(0).sum()
        ),
        "conditions_with_soc_range_at_least_5pp": int(len(sufficient_range)),
        "range_5pp_lower_soc_higher_rate_conditions": int(
            sufficient_range["low_minus_high_rate_pp_per_min"].gt(0).sum()
        ),
        "median_low_minus_high_rate_pp_per_min": float(
            comparable["low_minus_high_rate_pp_per_min"].median()
        ),
        "strict_monotonic_three_run_conditions": int(
            condition_summary["strict_low_gt_middle_gt_high"].sum()
        ),
        "three_selected_run_conditions": int(
            condition_summary["selected_run_count"].eq(3).sum()
        ),
        "explicit_three_strata_conditions": int(len(explicit)),
        "explicit_low_higher_than_high_conditions": int(
            explicit_low_minus_high.gt(0).sum()
        ),
        "explicit_strict_low_middle_high_conditions": int(
            explicit["strict_low_gt_middle_gt_high"].sum()
        ),
        "explicit_strata_mean_soc_pct": {
            stratum: float(
                run_rates.loc[
                    run_rates["soc_selection_stratum"].eq(stratum),
                    "run_start_soc_mean_pct",
                ].mean()
            )
            for stratum in ["high", "middle", "low"]
        },
        "explicit_strata_mean_rate_pp_per_min": {
            stratum: float(
                run_rates.loc[
                    run_rates["soc_selection_stratum"].eq(stratum),
                    "run_rate_mean_pp_per_min",
                ].mean()
            )
            for stratum in ["high", "middle", "low"]
        },
        "run_mean_fixed_effect_model": fixed_effect_bootstrap(
            run_rates, "run_rate_mean_pp_per_min"
        ),
        "run_median_fixed_effect_sensitivity": fixed_effect_bootstrap(
            run_rates, "run_rate_median_pp_per_min", seed=20260828
        ),
        "primary_only_sensitivity": primary_only_sensitivity(run_rates),
        "drone_level_direction_sensitivity": drone_level_direction_sensitivity(
            drone_rates, run_rates
        ),
        "low_soc_run_later_than_high_soc_run_conditions": int(
            comparable["low_soc_run_later_than_high_soc_run"].sum()
        ),
        "low_and_high_runs_same_day_conditions": int(
            comparable["low_and_high_runs_same_day"].sum()
        ),
        "within_condition_soc_order_spearman": float(order_association.statistic),
        "within_condition_soc_order_p_value": float(order_association.pvalue),
        "selected_runs_with_swarm_soc_spread_le_5pp": int(
            run_rates["within_swarm_soc_spread_pp"].le(5).sum()
        ),
        "selected_runs_total_for_swarm_spread": int(len(run_rates)),
        "explicit_low_runs_with_swarm_soc_spread_le_5pp": int(
            run_rates.loc[
                run_rates["soc_selection_stratum"].eq("low"),
                "within_swarm_soc_spread_pp",
            ].le(5).sum()
        ),
        "explicit_low_runs_total": int(
            run_rates["soc_selection_stratum"].eq("low").sum()
        ),
        "causal_interpretation": "not_supported_due_to_trial_order_and_within_swarm_SOC_confounding",
    }

    run_rates.to_csv(OUTPUT / "run_level_soc_rate_data.csv", index=False)
    condition_summary.to_csv(OUTPUT / "condition_soc_effect_summary.csv", index=False)
    explicit.to_csv(OUTPUT / "explicit_low_middle_high_summary.csv", index=False)
    with (OUTPUT / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    plot_condition_effects(condition_summary)
    return summary


if __name__ == "__main__":
    result = run_analysis()
    print(json.dumps(result, indent=2, ensure_ascii=False))
