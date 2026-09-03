"""Build the canonical portable-report artifact for the position-energy method study."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "position_energy_method_study"

METHOD_SQL = """SELECT
  method,
  available_run_drone_rows,
  coverage_fraction_of_780,
  zero_rate_fraction_among_available,
  median_estimated_rate_pp_per_min,
  median_effective_duration_sec,
  median_pairwise_spearman_slot_rank_across_runs,
  cells_with_replicate_rank_comparison
FROM method_comparison_summary
ORDER BY method"""

SIMULATION_SQL = """SELECT
  method,
  AVG(bias_pp_per_min) AS bias_pp_per_min,
  AVG(rmse_pp_per_min) AS rmse_pp_per_min,
  AVG(coverage_fraction) AS coverage_fraction,
  MAX(runs_pooled_per_condition) AS runs_pooled
FROM integer_soc_quantization_simulation
GROUP BY method
ORDER BY rmse_pp_per_min"""

RECOMMENDATION_SQL = """SELECT
  method,
  what_it_measures,
  strength,
  main_failure,
  recommended_role
FROM method_recommendations
ORDER BY method"""


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_specs() -> list[dict]:
    return [
        {
            "id": "src_method_study",
            "label": "Cleaned DJI Tello experiment method study",
            "query": {
                "engine": "SQLite over reviewed Python-analysis snapshot",
                "language": "sql",
                "description": (
                    "Phase-aware reconstruction of the first trajectory-defined 250 cm, "
                    "battery-specific Bideal normalization, and six alternative discharge-rate estimators."
                ),
                "sql": METHOD_SQL,
                "tables_used": [
                    "cleaning_master_run_index.csv",
                    "trajectory_drone_segments.csv",
                    "selected_runs_by_database_cell.csv",
                    "battery_ideal_normalization.csv",
                    "raw all_coordination.csv files",
                ],
                "filters": [
                    "formal runs marked eligible after trajectory cleaning",
                    "five drones complete the trajectory-defined first 250 cm",
                    "no-wind, prepare/pre, marked-outlier, and incomplete runs excluded",
                    "up to three starting-SOC-representative runs per database condition cell",
                ],
                "metric_definitions": [
                    "coverage = finite run-by-drone rate estimates / 780 selected run-by-drone rows",
                    "zero-rate share = available rows whose estimated rate is numerically zero",
                    "rank stability = median pairwise Spearman correlation of five slot rates across replicate runs within a cell",
                    "Bideal rate transforms physical-battery percentage-point drops using battery-specific hover-baseline slopes",
                ],
            },
        },
        {
            "id": "src_quantization_simulation",
            "label": "Integer-SOC quantization simulation",
            "query": {
                "engine": "SQLite over reviewed Python-simulation snapshot",
                "language": "sql",
                "description": (
                    "Three repeated 25 s curves per condition, five true slot rates from 8 to 12 pp/min, "
                    "0.1 s sampling, uniformly random fractional starting SOC, and integer floor reporting."
                ),
                "sql": SIMULATION_SQL,
                "tables_used": ["integer_soc_quantization_simulation.csv"],
                "filters": ["3,000 Monte Carlo repetitions", "three pooled runs per condition"],
                "metric_definitions": [
                    "bias = mean estimated rate minus true rate",
                    "RMSE = square root of mean squared estimation error",
                ],
            },
        },
        {
            "id": "src_method_recommendations",
            "label": "Estimator recommendation table",
            "query": {
                "engine": "SQLite over reviewed method-study snapshot",
                "language": "sql",
                "description": "Decision table assembled from the empirical coverage, stability, and simulation checks.",
                "sql": RECOMMENDATION_SQL,
                "tables_used": ["method_recommendations"],
                "metric_definitions": ["recommended role is the evidence-based intended use in the final pipeline"],
            },
        },
    ]


def main() -> None:
    summary = pd.read_csv(OUT / "method_comparison_summary.csv")
    simulation = pd.read_csv(OUT / "integer_soc_quantization_simulation.csv")
    run_coverage = pd.read_csv(OUT / "method_run_coverage.csv")

    short_names = {
        "M0 total selected-window rate": "M0 Total window",
        "M1 forward-event endpoint rate": "M1 Forward endpoint",
        "M2 after all-five observed-drop onset": "M2 Post-first-drop",
        "M3 strict all-five simultaneous-forward rate": "M3 Strict overlap",
        "M5 within-forward-island fixed-effects rate": "M5 Island FE",
        "M6 two-clock fixed-effects rate": "M6 Two-clock FE",
    }
    connection = sqlite3.connect(":memory:")
    summary.to_sql("method_comparison_summary", connection, index=False)
    simulation.to_sql("integer_soc_quantization_simulation", connection, index=False)
    queried_summary = pd.read_sql_query(METHOD_SQL, connection)
    method_rows = []
    for row in queried_summary.to_dict("records"):
        row["short_method"] = short_names[row["method"]]
        method_rows.append(row)

    sim_summary = pd.read_sql_query(SIMULATION_SQL, connection)
    sim_labels = {
        "endpoint": "Pooled endpoint",
        "free_intercept_curve": "Free-intercept curve",
        "post_all_five_first_drop": "Post-all-five first drop",
        "through_origin_curve": "Through-origin curve",
    }
    sim_rows = []
    for row in sim_summary.to_dict("records"):
        row["method_label"] = sim_labels[row["method"]]
        sim_rows.append(row)

    common_overlap_median = float(run_coverage["strict_all_five_forward_overlap_sec"].median())
    all_five_drop_runs = int(run_coverage["all_five_have_forward_drop"].sum())
    total_runs = len(run_coverage)
    m2 = summary[summary["method"].str.startswith("M2")].iloc[0]
    m3 = summary[summary["method"].str.startswith("M3")].iloc[0]
    m6 = summary[summary["method"].str.startswith("M6")].iloc[0]
    free_rmse = float(
        sim_summary.loc[sim_summary["method"].eq("free_intercept_curve"), "rmse_pp_per_min"].iloc[0]
    )

    headline = [
        {
            "post_first_drop_coverage": float(m2["coverage_fraction_of_780"]),
            "strict_overlap_median_sec": common_overlap_median,
            "free_intercept_simulation_rmse": free_rmse,
            "replicate_rank_stability": float(m6["median_pairwise_spearman_slot_rank_across_runs"]),
        }
    ]

    recommendation_rows = [
        {
            "method": "M0 Total selected-window drop/rate",
            "what_it_measures": "Flight plus programmed waiting/correction/hover",
            "strength": "Complete coverage; simple audit benchmark",
            "main_failure": "Confounds formation position with the collector's waiting schedule",
            "recommended_role": "Benchmark only",
        },
        {
            "method": "M1 Forward-event endpoint/curve",
            "what_it_measures": "SOC decrements timestamped during detected forward intervals",
            "strength": "Complete coverage; directly implements forward-only intent",
            "main_failure": "Integer SOC updates can lag across movement-state boundaries",
            "recommended_role": "Primary descriptive sensitivity",
        },
        {
            "method": "M2 Start after all five show a SOC drop",
            "what_it_measures": "Remaining forward curve after outcome-defined common onset",
            "strength": "Visually removes initial flat plateaus",
            "main_failure": "Selects on the measured outcome, loses runs/exposure, and increases zero-rate curves",
            "recommended_role": "Diagnostic overlay only",
        },
        {
            "method": "M3 Strict all-five simultaneous-forward overlap",
            "what_it_measures": "Only wall-clock intervals where all five masks are forward",
            "strength": "Closest observed window to simultaneous formation translation",
            "main_failure": "Median exposure is about 9 s; integer SOC supplies too few transitions",
            "recommended_role": "Robustness check only",
        },
        {
            "method": "M4 Forward-distance curve",
            "what_it_measures": "Bideal drop per detected metre / normalized 250 cm progress",
            "strength": "Matches equal-distance missions and removes total-duration comparisons",
            "main_failure": "Does not solve integer quantization or delayed telemetry updates",
            "recommended_role": "Secondary algorithm-facing unit",
        },
        {
            "method": "M5 Within-forward-island fixed effects",
            "what_it_measures": "Within-island curve slopes with a separate intercept per movement island",
            "strength": "Uses all runs, retains plateaus, and excludes between-island hover shifts",
            "main_failure": "A delayed SOC update at the next island boundary may be absorbed by the intercept",
            "recommended_role": "Strong secondary estimator",
        },
        {
            "method": "M6 Quantization-aware hierarchical two-clock model",
            "what_it_measures": "Latent slot forward rate while non-forward exposure is modeled as a nuisance process",
            "strength": "Uses repeated curves jointly, retains all samples, and separates forward/non-forward clocks",
            "main_failure": "The current constrained least-squares prototype is not yet a full interval-censored model",
            "recommended_role": "Recommended final modeling direction",
        },
    ]
    recommendation_frame = pd.DataFrame(recommendation_rows)
    recommendation_frame.to_sql("method_recommendations", connection, index=False)
    recommendation_rows = pd.read_sql_query(RECOMMENDATION_SQL, connection).to_dict("records")
    connection.close()

    generated = datetime.now(timezone.utc).isoformat()
    sources = source_specs()
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Estimating Position-Specific Forward Energy Use from Integer DJI Tello SOC",
        "description": "Method comparison for configuration-level position energy estimation.",
        "generatedAt": generated,
        "sources": sources,
        "charts": [
            {
                "id": "chart_coverage",
                "title": "Usable-data coverage by estimator",
                "subtitle": "Selected formal data: 156 runs and 780 run-by-drone curves",
                "showDescription": True,
                "intent": "comparison",
                "question": "How much of the selected dataset does each estimator retain?",
                "rationale": "A bar chart makes outcome-defined data loss directly comparable across estimators.",
                "type": "bar",
                "dataset": "method_summary",
                "sourceId": "src_method_recommendations",
                "encodings": {
                    "x": {"field": "short_method", "type": "nominal", "label": "Estimator"},
                    "y": {"field": "coverage_fraction_of_780", "type": "quantitative", "format": "percent", "label": "Coverage"},
                },
                "layout": "full",
            },
            {
                "id": "chart_zero_rate",
                "title": "Zero estimated-rate share by estimator",
                "subtitle": "High shares indicate sensitivity to integer-SOC flat curves or short windows",
                "showDescription": True,
                "intent": "comparison",
                "question": "Which estimators turn quantized flat curves into zero energy rates?",
                "rationale": "A common-scale bar chart exposes measurement fragility rather than only central estimates.",
                "type": "bar",
                "dataset": "method_summary",
                "sourceId": "src_method_study",
                "encodings": {
                    "x": {"field": "short_method", "type": "nominal", "label": "Estimator"},
                    "y": {"field": "zero_rate_fraction_among_available", "type": "quantitative", "format": "percent", "label": "Zero-rate share"},
                },
                "layout": "full",
            },
            {
                "id": "chart_simulation_rmse",
                "title": "Integer-SOC simulation error",
                "subtitle": "Three repeated 25 s curves pooled per condition; lower RMSE is better",
                "showDescription": True,
                "intent": "comparison",
                "question": "Which curve estimator best recovers a known latent discharge rate after integer quantization?",
                "rationale": "RMSE combines estimator bias and variance on a common percentage-points-per-minute scale.",
                "type": "bar",
                "dataset": "simulation_summary",
                "sourceId": "src_quantization_simulation",
                "encodings": {
                    "x": {"field": "method_label", "type": "nominal", "label": "Estimator"},
                    "y": {"field": "rmse_pp_per_min", "type": "quantitative", "format": "number", "label": "RMSE (pp/min)"},
                },
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "table_methods",
                "title": "Estimator decision table",
                "subtitle": "Purpose, failure mode, and recommended role in the final analysis pipeline",
                "showDescription": True,
                "dataset": "recommendations",
                "sourceId": "src_method_study",
                "defaultSort": {"field": "method", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "method", "label": "Method", "type": "text"},
                    {"field": "what_it_measures", "label": "What it measures", "type": "text"},
                    {"field": "strength", "label": "Strength", "type": "text"},
                    {"field": "main_failure", "label": "Main failure", "type": "text"},
                    {"field": "recommended_role", "label": "Recommended role", "type": "text"},
                ],
            }
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# Estimating Position-Specific Forward Energy Use from Integer DJI Tello SOC", "layout": "full"},
            {
                "id": "summary",
                "type": "markdown",
                "body": (
                    "## Technical summary\n\n"
                    "The initial flat portion of a Tello discharge curve should **not** be deleted: it is primarily a consequence of integer SOC quantization and asynchronous threshold crossings, not evidence of zero physical energy use. Starting after all five drones have shown a reported drop is outcome-based selection; it removes data, shortens exposure, and increases zero-rate estimates. Strict five-drone simultaneous-forward windows are physically attractive but too short for one-percentage-point telemetry.\n\n"
                    "The recommended target is a latent, position-specific forward discharge rate estimated jointly from repeated runs. The final model should retain all curve samples, use battery normalization and starting-SOC adjustment, include a separate intercept for each run-by-drone curve, and represent forward and non-forward exposure with separate clocks. The current two-clock constrained least-squares model is a useful prototype, but an interval-censored hierarchical version is needed before producing algorithm-facing slot rankings."
                ),
                "layout": "full",
            },
            {
                "id": "finding_plateau",
                "type": "markdown",
                "sourceId": "src_method_study",
                "body": (
                    "## Deleting the initial plateau loses information and selects on the outcome\n\n"
                    f"Only {all_five_drop_runs} of {total_runs} selected runs had at least one forward-period SOC decrement for every drone. The post-first-drop method retained {m2['coverage_fraction_of_780']:.1%} of run-by-drone rows, reduced the median usable forward exposure to {m2['median_effective_duration_sec']:.2f} s, and produced zero rates in {m2['zero_rate_fraction_among_available']:.1%} of the remaining rows. Its higher median rate is therefore partly mechanical: early exposure is removed conditional on having observed the outcome."
                ),
                "layout": "full",
            },
            {"id": "coverage_chart", "type": "chart", "chartId": "chart_coverage", "layout": "full"},
            {"id": "zero_chart", "type": "chart", "chartId": "chart_zero_rate", "layout": "full"},
            {
                "id": "finding_overlap",
                "type": "markdown",
                "sourceId": "src_method_study",
                "body": (
                    "## Strict simultaneous-forward analysis is too sparse to be the primary estimator\n\n"
                    f"The strict all-five overlap lasted a median {common_overlap_median:.2f} s per run. On that window, {m3['zero_rate_fraction_among_available']:.1%} of available run-by-drone estimates were zero. This method is valuable as a formation-validity sensitivity check, but not as the sole source of slot energy rates. A one-percentage-point sensor commonly supplies only one or two transitions in such a short window."
                ),
                "layout": "full",
            },
            {
                "id": "finding_simulation",
                "type": "markdown",
                "sourceId": "src_quantization_simulation",
                "body": (
                    "## Repeated curves with free intercepts recover latent rates more accurately\n\n"
                    f"In a controlled integer-SOC simulation with three pooled 25 s runs, the free-intercept curve estimator achieved an average RMSE of {free_rmse:.3f} pp/min. The pooled endpoint estimator reached {float(sim_summary.loc[sim_summary['method'].eq('endpoint'), 'rmse_pp_per_min'].iloc[0]):.3f}, the through-origin curve {float(sim_summary.loc[sim_summary['method'].eq('through_origin_curve'), 'rmse_pp_per_min'].iloc[0]):.3f}, and the post-all-five-first-drop estimator {float(sim_summary.loc[sim_summary['method'].eq('post_all_five_first_drop'), 'rmse_pp_per_min'].iloc[0]):.3f} pp/min. The simulation isolates quantization only; it does not reproduce telemetry lag or aerodynamic variability, so it supports estimator choice rather than validating real-data truth."
                ),
                "layout": "full",
            },
            {"id": "simulation_chart", "type": "chart", "chartId": "chart_simulation_rmse", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "body": (
                    "## Target estimand and analysis scope\n\n"
                    "The algorithm needs the expected battery cost assigned to each formation slot under a wind condition and spacing, not the wall-clock duration of the data-collection procedure. The primary estimand should therefore be the **latent Bideal-normalized forward discharge rate for condition × configuration × slot**, with uncertainty. For mission integration, retain both pp/min and pp/m: pp/min is needed when segment duration is predicted, while pp/m maps directly to equal-distance travel.\n\n"
                    "The selected population contains formal five-drone runs completing the trajectory-defined first 250 cm. Programmed waiting is not part of the target, but its elapsed exposure must be modeled as a nuisance process because delayed integer SOC changes cannot always be assigned to one instantaneous state."
                ),
                "layout": "full",
            },
            {"id": "methods_heading", "type": "markdown", "body": "## The methods serve different roles; no single cropped curve is sufficient", "layout": "full"},
            {"id": "methods_table", "type": "table", "tableId": "table_methods", "layout": "full"},
            {
                "id": "model",
                "type": "markdown",
                "body": (
                    "## Recommended final model: quantization-aware hierarchical two-clock regression\n\n"
                    "For drone *d* in run *r*, model latent standardized SOC as a run-specific intercept minus a slot-specific forward rate multiplied by cumulative forward time, minus a nuisance non-forward rate multiplied by cumulative non-forward time. The observed integer SOC is treated as an interval containing the latent value rather than as an exact continuous measurement.\n\n"
                    "The model should include battery-specific Bideal scaling, starting-SOC effects, run-level random effects, and partial pooling across wind conditions. It should output a posterior or bootstrap distribution for every slot rate, not a hard winner. The online configuration algorithm can then optimize expected total mission-plus-charging time while penalizing uncertain choices."
                ),
                "layout": "full",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "src_method_study",
                "body": (
                    "## Current data supports relative reported-SOC rates, not absolute electrical energy\n\n"
                    f"Replicate slot rankings remain weak: the median pairwise Spearman correlation across repeated runs is about {m6['median_pairwise_spearman_slot_rank_across_runs']:.2f}. Integer SOC, battery-specific behavior, only one physical drone/battery per slot, and a small number of repeated runs make drone identity inseparable from position without position swaps. Absolute energy in Wh cannot be recovered without voltage/current telemetry. The present dataset can support provisional relative battery-cost estimates, but algorithm-facing rankings must carry uncertainty and should be validated with held-out runs."
                ),
                "layout": "full",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. Preserve the original forward-aligned repeated-run curves as the descriptive evidence; do not delete initial flat plateaus.\n"
                    "2. Implement the interval-censored hierarchical two-clock model and compare it with M1 and M5 using leave-one-run-out prediction.\n"
                    "3. Define a formation-validity indicator from relative geometry and use strict all-five overlap only as a robustness subset.\n"
                    "4. Produce condition × configuration × slot estimates with uncertainty, then aggregate them into swarm total drop, maximum per-drone drop, and end-of-mission charging time.\n"
                    "5. For future experiments, rotate drones/batteries across slots or add voltage/current logging; either change is more valuable than collecting many additional runs with the same fixed assignment."
                ),
                "layout": "full",
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "- What geometric tolerance defines an intact formation under each spacing?\n"
                    "- Should the algorithm optimize expected swarm energy, the maximum individual battery drop, charging completion time, or a weighted combination?\n"
                    "- Can any future validation runs swap drone/battery assignments across slots to identify position effects separately from hardware effects?"
                ),
                "layout": "full",
            },
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "method_summary": method_rows,
                "simulation_summary": sim_rows,
                "recommendations": recommendation_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    (OUT / "artifact.json").write_text(
        json.dumps(json_safe(artifact), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(OUT / "artifact.json")


if __name__ == "__main__":
    main()
