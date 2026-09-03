"""Build the canonical portable-report artifact for the configuration analysis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "configuration_energy_analysis"
ARTIFACT = OUT / "artifact.json"


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy().astype(object)
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def markdown(block_id: str, body: str, source_id: str | None = None) -> dict:
    block = {"id": block_id, "type": "markdown", "body": body, "layout": "full"}
    if source_id:
        block["sourceId"] = source_id
    return block


def main() -> None:
    ranking = pd.read_csv(OUT / "configuration_ranking_by_condition.csv")
    leaders = pd.read_csv(OUT / "condition_configuration_leaders.csv")
    coverage = pd.read_csv(OUT / "configuration_condition_coverage.csv")
    calibration = pd.read_csv(OUT / "hover_battery_calibration_75_40.csv")
    validation = json.loads((OUT / "validation_checks.json").read_text(encoding="utf-8"))

    generated_at = datetime.now(ZoneInfo("Australia/Sydney")).isoformat(timespec="seconds")
    source_analysis = {
        "id": "source_analysis",
        "label": "Cleaned condition-by-configuration analysis",
        "path": "analysis_outputs/configuration_energy_analysis/configuration_ranking_by_condition.csv",
        "query": {
            "engine": "Python/pandas",
            "language": "python",
            "description": "Reproducible five-drone run aggregation and condition-specific configuration ranking.",
            "query": "output_py/analyze_configuration_energy.py",
            "sql": "SELECT * FROM read_csv_auto('analysis_outputs/configuration_energy_analysis/configuration_ranking_by_condition.csv');",
            "executed_at": generated_at,
            "tables_used": [
                "db_copy_for_cleaning/_cleaning_admin/trajectory_qc/trajectory_drone_segments.csv",
                "db_copy_for_cleaning/_cleaning_admin/cleaning_master_run_index.csv",
                "db_copy_for_cleaning/baselines",
            ],
            "filters": [
                "primary_analysis_status = eligible_primary_75_to_40",
                "wind_direction in {head, side, tail}",
                "exactly five completed drone segments per run",
                "first trajectory-observed 250 cm only",
            ],
            "metric_definitions": [
                "Per-drone hover-equivalent seconds = 60 × reported SOC percentage-point drop ÷ physical-battery 75%–40% hover discharge rate.",
                "Adjusted hover-equivalent seconds = hover-equivalent seconds − trajectory-confirmed stationary-wait seconds, clipped at zero.",
                "Run score = mean adjusted hover-equivalent seconds across the five drones; configuration score = mean across eligible runs within one wind condition.",
            ],
        },
    }
    source_validation = {
        "id": "source_validation",
        "label": "Independent validation checks",
        "path": "analysis_outputs/configuration_energy_analysis/validation_checks.json",
        "query": {
            "engine": "Python/pandas",
            "language": "python",
            "description": "Independent grain, eligibility, join, normalization, and leader-reproduction checks.",
            "query": "output_py/validate_configuration_energy_analysis.py",
            "sql": "SELECT * FROM read_json_auto('analysis_outputs/configuration_energy_analysis/validation_checks.json');",
            "executed_at": generated_at,
            "tables_used": [
                "analysis_outputs/configuration_energy_analysis/primary_drone_metrics.csv",
                "analysis_outputs/configuration_energy_analysis/primary_run_metrics.csv",
                "analysis_outputs/configuration_energy_analysis/configuration_ranking_by_condition.csv",
            ],
        },
    }
    source_hover = {
        "id": "source_hover",
        "label": "Battery-specific 75%–40% hover calibration",
        "path": "analysis_outputs/configuration_energy_analysis/hover_battery_calibration_75_40.csv",
        "query": {
            "engine": "Python/numpy",
            "language": "python",
            "description": "Linear fits to clean hover traces in the selected battery range.",
            "query": "hover_calibration() in output_py/analyze_configuration_energy.py",
            "sql": "SELECT * FROM read_csv_auto('analysis_outputs/configuration_energy_analysis/hover_battery_calibration_75_40.csv');",
            "executed_at": generated_at,
            "tables_used": ["db_copy_for_cleaning/baselines"],
            "filters": ["reported battery level from 75% to 40%"],
        },
    }
    sources = [source_analysis, source_validation, source_hover]

    headline = pd.DataFrame(
        [{
            "primary_runs": 168,
            "observed_cells": 55,
            "possible_cells": 60,
            "stable_leaders": 1,
            "assessment": validation["assessment"],
        }]
    )
    leader_table = leaders.copy()
    leader_table["leader_adjusted_mean_sec"] = leader_table["leader_adjusted_mean_sec"].round(1)
    leader_table["runner_up_adjusted_mean_sec"] = leader_table["runner_up_adjusted_mean_sec"].round(1)
    leader_table["gap_to_runner_up_pct"] = leader_table["gap_to_runner_up_pct"].round(1)
    leader_table["bootstrap_winner_probability"] = leader_table["bootstrap_winner_probability"].round(3)

    condition_specs = [
        ("head", 1, "Head wind · Level 1", "Front · 75 cm is the only comparatively stable leader: it ranks first under all five metric variants and is 36.3% below the runner-up score. This is the strongest current result, although it still has only three runs."),
        ("head", 2, "Head wind · Level 2", "Vee · 75 cm leads the primary ranking, but the gap to Echelon · 75 cm is modest and excluding drone 5 changes the leader to Column · 75 cm. Treat this as provisional."),
        ("side", 1, "Side wind · Level 1", "Front · 50 cm and Front · 75 cm differ by only 0.6 adjusted hover-equivalent seconds (1.3%). Median and no-drone-5 analyses favor Front · 75 cm, so there is no clear winner."),
        ("side", 2, "Side wind · Level 2", "Front · 75 cm ranks first in all five metric variants, but it has only two runs and is just 4.7% below Echelon · 50 cm. The direction is consistent but still provisional."),
        ("tail", 1, "Tail wind · Level 1", "Front · 50 cm leads narrowly. Diamond · 75 cm and Echelon · 50 cm are close, and the median selects Echelon · 50 cm; this is not a decisive separation."),
        ("tail", 2, "Tail wind · Level 2", "No defensible winner can be declared. Column · 50 cm leads only after conservative wait correction and has one run; raw and uncorrected metrics instead favor Echelon · 50 cm."),
    ]

    datasets: dict[str, list[dict]] = {
        "headline": records(headline),
        "leaders": records(leader_table),
        "coverage": records(coverage),
        "calibration": records(calibration.round(4)),
    }
    charts = []
    blocks = [
        markdown("title", "# Configuration Energy Comparison by Wind Condition"),
        markdown(
            "technical_summary",
            """## Technical Summary

The analysis identifies the lowest observed energy score for each wind condition while keeping condition fixed and comparing **configuration = formation + inter-drone spacing**. **Front · 75 cm under Head wind · Level 1 is the only comparatively stable leader.** The remaining condition-level results are provisional, tied, or under-replicated. In particular, Tail wind · Level 2 is unresolved rather than a valid single winner.

The five-drone run is the analytical unit. Scores are based on the first trajectory-observed 250 cm, normalized by each physical battery's 75%–40% hover discharge rate, and reduced only by trajectory-confirmed stationary waiting. Lower is better. These are descriptive rankings—not statistical-significance or causal-optimality claims.""",
            "source_analysis",
        ),
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["card_runs", "card_coverage", "card_stable"], "layout": "full"},
        markdown(
            "metric_definition",
            """## Comparison Basis

**Adjusted hover-equivalent seconds per drone per 250 cm** converts each drone's integer SOC drop into the hover time that would produce the same drop for that physical battery, then subtracts only confirmed stationary waiting. Five drone values are averaged to one run; eligible runs are averaged within one condition and configuration.

The unit is a normalized SOC proxy, not literal elapsed time, motor-on time, or electrical energy in joules. It is used because flight duration and fixed battery identity differ across the records.""",
            "source_analysis",
        ),
        markdown(
            "leader_intro",
            """## Condition-Level Leaders

The table reports the primary leader and nearest observed alternative in every condition. The evidence label combines replication, separation from the runner-up, and agreement across five ranking variants. “Most stable” does not mean statistically proven; it means the current descriptive evidence is materially less sensitive than the other conditions.""",
            "source_analysis",
        ),
        {"id": "leader_table_block", "type": "table", "tableId": "leader_table", "layout": "full"},
        markdown(
            "ranking_intro",
            """## Full Rankings Within Each Condition

Each chart includes every observed configuration in that condition. Bars begin at zero and are sorted from lowest to highest adjusted score. Read comparisons only within a panel; do not compare bar positions across different wind conditions as a causal wind effect.""",
            "source_analysis",
        ),
    ]

    for direction, level, condition, narrative in condition_specs:
        dataset_id = f"rank_{direction}_{level}"
        chart_id = f"chart_{direction}_{level}"
        subset = ranking[
            ranking["wind_direction"].eq(direction) & ranking["wind_level"].eq(level)
        ].sort_values("adjusted_mean_sec")
        dataset = subset[[
            "configuration", "adjusted_mean_sec", "run_count", "adjusted_ci95_low_sec",
            "adjusted_ci95_high_sec", "raw_mean_drop_pp", "uncorrected_mean_hover_equivalent_sec",
            "mean_confirmed_wait_sec", "bootstrap_winner_probability", "adjusted_mean_rank",
        ]].copy()
        for column in dataset.select_dtypes(include="number").columns:
            dataset[column] = dataset[column].round(3)
        datasets[dataset_id] = records(dataset)
        charts.append({
            "id": chart_id,
            "title": condition,
            "subtitle": "Adjusted hover-equivalent seconds per drone per 250 cm; lower is better",
            "intent": "comparison",
            "question": f"Which observed configuration has the lowest normalized energy score under {condition}?",
            "rationale": "A sorted horizontal bar chart supports exact within-condition ranking while retaining readable configuration labels.",
            "comparisonContext": {
                "grain": "five-drone run averaged within configuration",
                "normalization": "physical-battery 75%–40% hover rate and confirmed stationary-wait correction",
                "unit": "adjusted hover-equivalent seconds per drone per 250 cm",
            },
            "type": "horizontalBar",
            "dataset": dataset_id,
            "sourceId": "source_analysis",
            "encodings": {
                "x": {"field": "configuration", "type": "nominal", "label": "Configuration"},
                "y": {"field": "adjusted_mean_sec", "type": "quantitative", "label": "Adjusted hover-equivalent seconds", "unit": "s"},
                "tooltip": [
                    {"field": "configuration", "type": "nominal", "label": "Configuration"},
                    {"field": "adjusted_mean_sec", "type": "quantitative", "label": "Adjusted score", "unit": "s"},
                    {"field": "run_count", "type": "quantitative", "label": "Runs"},
                    {"field": "raw_mean_drop_pp", "type": "quantitative", "label": "Raw mean SOC drop", "unit": "percentage points"},
                    {"field": "mean_confirmed_wait_sec", "type": "quantitative", "label": "Mean confirmed wait", "unit": "s"},
                ],
            },
            "valueFormat": "number",
            "unit": "s",
            "layout": "full",
            "labels": {"values": "all"},
            "settings": {"orientation": "horizontal", "sort": "ascending", "showValues": True},
            "surface": {"surface": "card", "viewMode": "visualization", "showControls": False},
            "maxRows": 10,
        })
        blocks.append(markdown(f"note_{direction}_{level}", f"### {condition}\n\n{narrative}", "source_analysis"))
        blocks.append({"id": f"block_{chart_id}", "type": "chart", "chartId": chart_id, "layout": "full"})

    blocks.extend([
        markdown(
            "robustness",
            """## Robustness and Battery-Replacement Sensitivity

Five rankings were compared: adjusted mean (primary), adjusted median, raw reported SOC drop, normalized but uncorrected mean, and adjusted mean excluding drone 5. Excluding drone 5 is the direct sensitivity check for the fixed drone/battery assignment and the later B15-to-B12 replacement.

- Head wind · Level 1 and Side wind · Level 2 retain the same leader in all five variants.
- Head wind · Level 2 changes to Column · 75 cm when drone 5 is excluded.
- Side wind · Level 1 changes to Front · 75 cm under the median and no-drone-5 variants.
- Tail wind · Level 1 changes to Echelon · 50 cm under the median.
- Tail wind · Level 2 changes to Echelon · 50 cm under both raw and uncorrected rankings.

The battery-specific normalization reduces systematic B12/B15 discharge-rate differences, but it cannot fully separate battery, drone position, and run-order effects because they were not randomized.""",
            "source_analysis",
        ),
        markdown(
            "coverage_intro",
            """## Design Coverage

The current primary dataset contains 55 of 60 possible condition-configuration cells. Replication is unbalanced (1–6 runs per observed cell), so a low mean from one or two runs should not be treated as equivalent evidence to a replicated result. Zeros in the table are missing design cells, not zero energy.""",
            "source_analysis",
        ),
        {"id": "coverage_table_block", "type": "table", "tableId": "coverage_table", "layout": "full"},
        markdown(
            "limitations",
            """## Limitations and Validation Status

**Assessment: Share with caveats.** Independent checks confirmed the primary run set, five-drone grain, 75%–40% eligibility, battery calibration coverage, absence of marked outliers/no-wind runs, non-negative adjusted values, and exact reproduction of all six leaders.

Material limitations remain:

1. Tello reports integer battery percentage; 36 of 840 drone segments show zero reported percentage-point drop.
2. Five design cells are missing and three observed cells have only one run.
3. Confirmed stationary-wait removal is intentionally conservative and changes Tail wind · Level 2.
4. Fixed drone positions and batteries prevent clean separation of configuration, position, airframe, and battery effects.
5. Bootstrap winner probabilities are descriptive and can be degenerate for a one-run cell.
6. The design is not a balanced randomized block experiment, so the rankings are not causal estimates or significance tests.""",
            "source_validation",
        ),
        markdown(
            "next_steps",
            """## Recommended Next Experiments

1. Replicate **Tail wind · Level 2, Column · 50 cm** before interpreting its corrected lead; also repeat Echelon · 50 cm as the raw/uncorrected comparator.
2. Run matched **Front · 50 cm versus Front · 75 cm** trials for Side wind · Level 1 because their present difference is practically negligible.
3. Add matched runs for Vee · 75 cm, Echelon · 75 cm, and Column · 75 cm under Head wind · Level 2.
4. Complete the five missing design cells and target the same number of runs per cell.
5. Randomize configuration order within battery-state blocks where practical, and record a higher-resolution electrical measure if available.
6. Keep position analysis separate and within formation, as planned; do not reinterpret fixed drone ID as a transferable position effect.""",
            "source_analysis",
        ),
        markdown(
            "reproducibility",
            """## Reproducibility

The analysis is reproducible from the cleaned trajectory segments and cleaning master index. The executed notebook, full condition rankings, run-level metrics, calibration table, validation report, and static six-panel figure are supplied next to this report.""",
            "source_validation",
        ),
    ])

    cards = [
        {
            "id": "card_runs", "dataset": "headline", "sourceId": "source_analysis",
            "description": "Eligible five-drone runs in the primary 75%–40% analysis.",
            "metrics": [{"label": "Primary runs", "field": "primary_runs", "format": "number"}],
        },
        {
            "id": "card_coverage", "dataset": "headline", "sourceId": "source_analysis",
            "description": "Observed cells across six conditions and ten possible configurations.",
            "metrics": [
                {"label": "Observed cells", "field": "observed_cells", "format": "number"},
                {"label": "Possible", "field": "possible_cells", "format": "number"},
            ],
        },
        {
            "id": "card_stable", "dataset": "headline", "sourceId": "source_validation",
            "description": "Condition leaders stable across all sensitivity variants with material separation and replication.",
            "metrics": [{"label": "Comparatively stable leader", "field": "stable_leaders", "format": "number"}],
        },
    ]
    tables = [
        {
            "id": "leader_table",
            "title": "Primary leader and runner-up by condition",
            "subtitle": "Primary adjusted mean ranking with descriptive robustness evidence",
            "dataset": "leaders",
            "sourceId": "source_analysis",
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "condition", "label": "Condition", "type": "text"},
                {"field": "leading_configuration", "label": "Leader", "type": "text"},
                {"field": "leader_adjusted_mean_sec", "label": "Score (s)", "format": "number"},
                {"field": "leader_run_count", "label": "Runs", "format": "number"},
                {"field": "runner_up_configuration", "label": "Runner-up", "type": "text"},
                {"field": "gap_to_runner_up_pct", "label": "Gap (%)", "format": "number"},
                {"field": "variant_agreement_count_of_5", "label": "Variants agreeing", "format": "number"},
                {"field": "evidence_label", "label": "Evidence", "type": "text"},
            ],
        },
        {
            "id": "coverage_table",
            "title": "Run count by configuration and condition",
            "subtitle": "Zero indicates an unobserved design cell",
            "dataset": "coverage",
            "sourceId": "source_analysis",
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": column, "label": column, "type": "text" if column == "configuration" else "number", **({} if column == "configuration" else {"format": "number"})}
                for column in coverage.columns
            ],
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Configuration Energy Comparison by Wind Condition",
            "description": "Technical comparison of formation and inter-drone spacing under six wind conditions.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": sources,
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print(ARTIFACT)


if __name__ == "__main__":
    main()
