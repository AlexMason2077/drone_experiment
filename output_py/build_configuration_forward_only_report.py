"""Build the English portable report for the forward-only energy analysis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "configuration_energy_analysis"
BASE_ARTIFACT = OUT / "artifact.json"
ARTIFACT = OUT / "forward_only_artifact.json"


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy().astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def main() -> None:
    artifact = json.loads(BASE_ARTIFACT.read_text(encoding="utf-8"))
    ranking = pd.read_csv(OUT / "forward_only_configuration_ranking_by_condition.csv")
    leaders = pd.read_csv(OUT / "forward_only_condition_configuration_leaders.csv")
    changes = pd.read_csv(OUT / "forward_only_leader_change_comparison.csv")
    calibration = pd.read_csv(OUT / "forward_only_hover_battery_calibration_75_40.csv")
    validation = json.loads((OUT / "forward_only_validation_checks.json").read_text())
    coverage = pd.read_csv(OUT / "configuration_condition_coverage.csv")
    diagnostic = pd.read_csv(
        OUT / "echalon50_forward_only_examples" / "echalon50_side_movement_time_diagnostic_runs.csv"
    )
    generated_at = datetime.now(ZoneInfo("Australia/Sydney")).isoformat(timespec="seconds")
    validation_counts = validation["counts"]
    threshold_span = validation["movement_timing"]["threshold_median_span_sec"]
    leader_map = leaders.set_index("condition").to_dict(orient="index")
    lv1_diagnostic = diagnostic[diagnostic["wind_level"].eq(1)]
    lv2_diagnostic = diagnostic[diagnostic["wind_level"].eq(2)]
    lv1_move_mean = float(lv1_diagnostic["mean_forward_movement_sec"].mean())
    lv2_move_mean = float(lv2_diagnostic["mean_forward_movement_sec"].mean())
    lv1_speed_mean = float(lv1_diagnostic["mean_forward_speed_cm_s"].mean())
    lv2_speed_mean = float(lv2_diagnostic["mean_forward_speed_cm_s"].mean())
    lv1_lateral_mean = float(lv1_diagnostic["mean_lateral_per_forward"].mean())
    lv2_lateral_mean = float(lv2_diagnostic["mean_lateral_per_forward"].mean())
    lv1_distance_mean = float(lv1_diagnostic["mean_forward_distance_cm"].mean())
    lv2_distance_mean = float(lv2_diagnostic["mean_forward_distance_cm"].mean())
    slow_lv2 = lv2_diagnostic.sort_values("mean_forward_movement_sec", ascending=False).iloc[0]
    other_lv2 = lv2_diagnostic.sort_values("mean_forward_movement_sec", ascending=False).iloc[-1]

    source_analysis = {
        "id": "source_analysis",
        "label": "Forward-movement-only condition-by-configuration analysis",
        "path": "analysis_outputs/configuration_energy_analysis/forward_only_configuration_ranking_by_condition.csv",
        "query": {
            "engine": "Python/pandas",
            "language": "python",
            "description": "Trajectory-state segmentation, battery-specific hover normalization, and five-drone run ranking.",
            "query": "output_py/build_forward_motion_segments.py; output_py/analyze_configuration_forward_only.py",
            "sql": "SELECT * FROM read_csv_auto('analysis_outputs/configuration_energy_analysis/forward_only_configuration_ranking_by_condition.csv');",
            "executed_at": generated_at,
            "tables_used": [
                "db_copy_for_cleaning/_cleaning_admin/trajectory_qc/trajectory_drone_segments.csv",
                "db_copy_for_cleaning/_cleaning_admin/trajectory_qc/forward_motion_drone_segments.csv",
                "db_copy_for_cleaning/_cleaning_admin/cleaning_master_run_index.csv",
                "db_copy_for_cleaning/baselines",
            ],
            "filters": [
                "primary_analysis_status = eligible_primary_75_to_40",
                "first trajectory-observed 250 cm",
                "exactly five completed drones per run",
                "exclude every in-flight interval without detected forward progress",
            ],
            "metric_definitions": [
                "Forward movement: within each logged data_collector node-segment phase, smoothed forward-progress velocity >= 2 cm/s; gaps <= 1.0 s are bridged only inside the same phase and islands < 0.5 s are removed.",
                "Per-drone total hover-equivalent seconds = 60 × selected-window SOC percentage-point drop ÷ physical-battery 75%–40% hover rate.",
                "Forward-only score = total hover-equivalent seconds − detected in-flight non-forward seconds, clipped at zero.",
                "Run score = mean of five drones; configuration score = mean of eligible runs within one wind condition.",
            ],
        },
    }
    source_validation = {
        "id": "source_validation",
        "label": "Independent forward-only validation",
        "path": "analysis_outputs/configuration_energy_analysis/forward_only_validation_checks.json",
        "query": {
            "engine": "Python/pandas",
            "language": "python",
            "description": "Independent grain, eligibility, formula, threshold-robustness, and leader checks.",
            "query": "output_py/validate_configuration_forward_only.py",
            "sql": "SELECT * FROM read_json_auto('analysis_outputs/configuration_energy_analysis/forward_only_validation_checks.json');",
            "executed_at": generated_at,
            "tables_used": [
                "analysis_outputs/configuration_energy_analysis/forward_only_primary_drone_metrics.csv",
                "analysis_outputs/configuration_energy_analysis/forward_only_primary_run_metrics.csv",
            ],
        },
    }
    source_hover = {
        "id": "source_hover",
        "label": "Physical-battery 75%–40% hover calibration",
        "path": "analysis_outputs/configuration_energy_analysis/forward_only_hover_battery_calibration_75_40.csv",
        "query": {
            "engine": "Python/numpy",
            "language": "python",
            "description": "Linear hover-discharge fits used to normalize fixed physical batteries and model excluded hover energy.",
            "query": "hover_calibration() in output_py/analyze_configuration_energy.py",
            "sql": "SELECT * FROM read_csv_auto('analysis_outputs/configuration_energy_analysis/forward_only_hover_battery_calibration_75_40.csv');",
            "executed_at": generated_at,
            "tables_used": ["db_copy_for_cleaning/baselines"],
            "filters": ["reported battery level from 75% to 40%"],
        },
    }
    source_diagnostic = {
        "id": "source_diagnostic",
        "label": "Echelon 50 cm side-wind movement-time diagnostic",
        "path": "analysis_outputs/configuration_energy_analysis/echalon50_forward_only_examples/echalon50_side_movement_time_diagnostic_runs.csv",
        "query": {
            "engine": "Python/pandas",
            "language": "python",
            "description": "Run-level decomposition of moving time, detected forward speed, lateral correction, spacing error, and roll.",
            "query": "output_py/diagnose_echalon50_side_movement_time.py",
            "sql": "SELECT * FROM read_csv_auto('analysis_outputs/configuration_energy_analysis/echalon50_forward_only_examples/echalon50_side_movement_time_diagnostic_runs.csv');",
            "executed_at": generated_at,
            "tables_used": [
                "analysis_outputs/configuration_energy_analysis/forward_only_primary_drone_metrics.csv",
                "db_copy_for_cleaning/*_all_coordination.csv",
            ],
        },
    }
    sources = [source_analysis, source_validation, source_hover, source_diagnostic]

    title = "Forward-Movement-Only Configuration Energy Comparison"
    manifest = artifact["manifest"]
    manifest.update(
        {
            "title": title,
            "description": "Technical comparison after removing all detected in-flight hovering, waiting, and non-forward time.",
            "generatedAt": generated_at,
            "sources": sources,
        }
    )
    artifact["sources"] = sources
    artifact["snapshot"]["generatedAt"] = generated_at

    leader_table = leaders.copy()
    leader_table["leader_forward_only_mean_sec"] = leader_table[
        "leader_forward_only_mean_sec"
    ].round(1)
    leader_table["runner_up_forward_only_mean_sec"] = leader_table[
        "runner_up_forward_only_mean_sec"
    ].round(1)
    leader_table["gap_to_runner_up_pct"] = leader_table["gap_to_runner_up_pct"].round(1)
    artifact["snapshot"]["datasets"] = {
        "headline": [{
            "primary_runs": validation["counts"]["primary_runs"],
            "median_moving_sec": round(validation["movement_timing"]["median_forward_movement_sec"], 2),
            "median_removed_sec": round(validation["movement_timing"]["median_removed_nonforward_sec"], 2),
            "changed_condition_leaders": int(changes["leader_changed_after_full_hover_removal"].sum()),
        }],
        "leaders": records(leader_table),
        "coverage": records(coverage),
        "calibration": records(calibration.round(4)),
        "echalon50_side_movement_diagnostic": records(diagnostic.round(3)),
    }
    for direction, level in [
        ("head", 1), ("head", 2), ("side", 1), ("side", 2), ("tail", 1), ("tail", 2)
    ]:
        subset = ranking[
            ranking["wind_direction"].eq(direction) & ranking["wind_level"].eq(level)
        ].sort_values("forward_only_mean_sec")
        dataset = subset[[
            "configuration",
            "forward_only_mean_sec",
            "run_count",
            "forward_only_ci95_low_sec",
            "forward_only_ci95_high_sec",
            "raw_mean_drop_pp",
            "total_mean_hover_equivalent_sec",
            "mean_in_flight_nonforward_sec",
            "mean_forward_movement_sec",
            "bootstrap_winner_probability",
            "forward_only_mean_rank",
        ]].copy()
        for column in dataset.select_dtypes(include="number").columns:
            dataset[column] = dataset[column].round(3)
        artifact["snapshot"]["datasets"][f"rank_{direction}_{level}"] = records(dataset)

    cards = {card["id"]: card for card in manifest["cards"]}
    cards["card_runs"].update(
        {
            "description": "Eligible five-drone runs after the primary cleaning rules.",
            "metrics": [{"label": "Primary runs", "field": "primary_runs", "format": "number"}],
        }
    )
    cards["card_coverage"].update(
        {
            "description": "Typical forward-moving time compared with the 25 s commanded benchmark.",
            "metrics": [{"label": "Median moving time (s)", "field": "median_moving_sec", "format": "number"}],
        }
    )
    cards["card_stable"].update(
        {
            "description": "Typical time removed as in-flight hovering, waiting, or lateral-only correction.",
            "metrics": [{"label": "Median removed time (s)", "field": "median_removed_sec", "format": "number"}],
        }
    )

    for chart in manifest["charts"]:
        chart["subtitle"] = "Estimated forward-only hover-equivalent seconds per drone per 250 cm; lower is better"
        chart["question"] = chart["question"].replace("normalized energy score", "estimated forward-only energy score")
        chart["comparisonContext"]["normalization"] = (
            "physical-battery 75%–40% hover rate; all detected in-flight non-forward time removed"
        )
        chart["comparisonContext"]["unit"] = "forward-only hover-equivalent seconds per drone per 250 cm"
        chart["encodings"]["y"].update(
            {
                "field": "forward_only_mean_sec",
                "label": "Forward-only hover-equivalent seconds",
            }
        )
        tooltips = chart["encodings"].get("tooltip", [])
        for tooltip in tooltips:
            if tooltip.get("field") == "adjusted_mean_sec":
                tooltip.update(
                    {"field": "forward_only_mean_sec", "label": "Forward-only score"}
                )
            elif tooltip.get("field") == "mean_confirmed_wait_sec":
                tooltip.update(
                    {"field": "mean_in_flight_nonforward_sec", "label": "Mean removed non-forward time"}
                )

    manifest["charts"].append(
        {
            "id": "chart_echalon50_side_movement_time",
            "title": "Echelon · 50 cm · Side wind: moving time by run",
            "subtitle": "Mean across five drones; the commanded distance is the same 250 cm",
            "intent": "comparison",
            "question": "Why is detected forward-moving time longer at Level 2?",
            "rationale": "Run-level bars show whether the Level 2 mean is broad or concentrated in one run.",
            "comparisonContext": {
                "grain": "five-drone run",
                "denominator": "first trajectory-observed 250 cm",
                "unit": "seconds",
            },
            "type": "bar",
            "dataset": "echalon50_side_movement_diagnostic",
            "sourceId": "source_diagnostic",
            "encodings": {
                "x": {"field": "run_label", "type": "nominal", "label": "Run"},
                "y": {
                    "field": "mean_forward_movement_sec",
                    "type": "quantitative",
                    "label": "Mean forward-moving time",
                    "unit": "s",
                },
                "tooltip": [
                    {"field": "run_label", "type": "nominal", "label": "Run"},
                    {"field": "mean_forward_movement_sec", "type": "quantitative", "label": "Moving time", "unit": "s"},
                    {"field": "mean_forward_speed_cm_s", "type": "quantitative", "label": "Detected forward speed", "unit": "cm/s"},
                    {"field": "mean_lateral_per_forward", "type": "quantitative", "label": "Lateral/forward path ratio"},
                    {"field": "mean_spacing_error_cm", "type": "quantitative", "label": "Mean spacing error", "unit": "cm"},
                ],
            },
            "valueFormat": "number",
            "unit": "s",
            "layout": "full",
            "labels": {"values": "all"},
            "settings": {"sort": "none", "showValues": True},
            "surface": {"surface": "card", "viewMode": "visualization", "showControls": False},
            "maxRows": 5,
        }
    )

    tables = {table["id"]: table for table in manifest["tables"]}
    tables["leader_table"].update(
        {
            "subtitle": "Estimated forward-only mean with movement-threshold and battery-replacement sensitivity",
            "columns": [
                {"field": "condition", "label": "Condition", "type": "text"},
                {"field": "leading_configuration", "label": "Leader", "type": "text"},
                {"field": "leader_forward_only_mean_sec", "label": "Score (s)", "format": "number"},
                {"field": "leader_run_count", "label": "Runs", "format": "number"},
                {"field": "runner_up_configuration", "label": "Runner-up", "type": "text"},
                {"field": "gap_to_runner_up_pct", "label": "Gap (%)", "format": "number"},
                {"field": "variant_agreement_count_of_5", "label": "Variants agreeing", "format": "number"},
                {"field": "evidence_label", "label": "Evidence", "type": "text"},
            ],
        }
    )

    block_updates = {
        "title": f"# {title}",
        "technical_summary": """## Technical Summary

This revision removes every detected in-flight interval without forward progression, including formation-induced hovering and waiting after the common timer starts. **Front · 75 cm under Head wind · Level 1 remains the only comparatively stable leader with at least three runs.**

Full non-forward removal changes two side-wind rankings: Side wind · Level 1 changes from Front · 50 cm to Front · 75 cm, and Side wind · Level 2 changes from Front · 75 cm to Echelon · 50 cm. These changed results remain provisional. Head-wind and tail-wind leaders do not change.

The score is an estimated forward-only SOC energy proxy. The dataset has no synchronized current or voltage trace, so hovering cannot be removed as directly measured joules.""",
        "metric_definition": """## Forward-Only Comparison Basis

Forward movement is defined from the reconstructed mission-pad trajectory as smoothed forward-progress velocity of at least 2 cm/s. Classification is performed separately inside each logged `node_segment_i_of_n` phase from `data_collector.py`. This prevents release waits and end-of-segment synchronization holds from being bridged across segment boundaries. Gaps up to 1.0 s are bridged only within the same phase to tolerate coordinate quantization, and movement islands shorter than 0.5 s are removed.

The collector logic explains the main non-forward intervals: staggered formations wait for group pad lock and programmed release delays; Column can additionally wait behind a spacing safety gate; all formations can pause for marker/position correction and after an early arrival while the group synchronizes.

For each drone, the selected-window SOC drop is converted to physical-battery 75%–40% hover-equivalent seconds. The hover-baseline energy corresponding to every detected non-forward second is subtracted, with negative values clipped to zero. Five drone scores are averaged to one run, then eligible runs are averaged within condition and configuration.""",
        "leader_intro": """## Revised Condition-Level Leaders

The table reports the lowest estimated forward-only score in each condition. Evidence labels account for run count, separation, bootstrap stability, movement-threshold sensitivity, median-versus-mean ranking, and exclusion of drone 5 as a B15-to-B12 sensitivity check.""",
        "ranking_intro": """## Full Forward-Only Rankings

Each chart compares only configurations observed within the same wind condition. Bars start at zero and use a common direction: lower estimated forward-only energy is better. Run counts are shown because replication remains unbalanced.""",
        "robustness": f"""## Robustness and Interpretation

The primary leader was compared across the mean, median, 1 cm/s and 4 cm/s movement thresholds, and an analysis excluding drone 5. The detected median forward-moving time changes by only {threshold_span:.2f} s across thresholds from 1 to 4 cm/s. The primary median is {validation['movement_timing']['median_forward_movement_sec']:.2f} s, close to the 25 s ideal for 250 cm at 10 cm/s.

- Head wind · Level 1 retains Front · 75 cm across all checks.
- Head wind · Level 2 retains Vee · 75 cm except when drone 5 is excluded, when Column · 75 cm leads.
- Side wind · Level 1 retains Front · 75 cm across the five primary forward-only variants, but raw and less-strict wait correction identify other configurations.
- Side wind · Level 2 retains Echelon · 50 cm across the five primary forward-only variants, but it has two runs and raw/less-strict measures favor Front · 75 cm.
- Tail wind · Level 1 changes to Echelon · 50 cm under the median and no-drone-5 analyses.
- Tail wind · Level 2 remains a one-run Column · 50 cm result and is unresolved.""",
        "coverage_intro": """## Design Coverage

The primary dataset still contains 55 of 60 possible condition-configuration cells, with 1–6 runs per observed cell. Removing hover changes the metric, not the experiment coverage. Zeros in the coverage table indicate missing design cells.""",
        "limitations": f"""## Validation Status and Limitations

**Assessment: Share with caveats.** Independent checks reproduced the formula and all six revised leaders, confirmed five drones per run, verified 75%–40% eligibility, and showed that the movement threshold has little effect on median moving time.

Material limitations remain:

1. Hover removal is model-based: excluded non-forward time is charged at the battery-specific hover-baseline rate because current and voltage were not recorded.
2. Tello SOC is integer-valued; {validation_counts['forward_only_values_clipped_to_zero']} of {validation_counts['primary_drone_rows']} drone scores clip to zero after subtraction.
3. {validation_counts['forward_segmentation_flagged_drone_rows']} trajectory rows have duration, detected-distance, or phase-coverage review flags and remain included with explicit markers.
4. Side wind · Level 2 changes leader only under the new method and has two runs for Echelon · 50 cm.
5. Tail wind · Level 2 has a one-run leader and remains unresolved.
6. Sub-second pauses inside one segment cannot be separated reliably from mission-pad coordinate quantization; the 1.0 s bridge is never allowed to cross a collector phase boundary.
7. The design is unbalanced and not randomized as a blocked experiment, so rankings are descriptive rather than causal.""",
        "next_steps": f"""## Recommended Next Experiments

1. Replicate **Echelon · 50 cm versus Front · 75 cm under Side wind · Level 2** to test the method-dependent change.
2. Run matched **Front · 50 cm versus Front · 75 cm under Side wind · Level 1**.
3. Replicate **Column · 50 cm and Echelon · 50 cm under Tail wind · Level 2** before declaring a winner.
4. Review the {validation_counts['forward_segmentation_flagged_drone_rows']} flagged trajectory rows; retain or exclude them using a documented rule before formal inference.
5. For future experiments, record synchronized battery voltage and current if feasible. That would permit direct integration of movement-only electrical energy instead of hover-baseline subtraction.""",
        "reproducibility": """## Reproducibility

The forward-motion sidecars, rerunnable companion notebook, drone- and run-level metrics, condition rankings, flagged-row list, validation report, and static six-panel chart are supplied next to this report. The previous wait-corrected analysis remains preserved for audit but is superseded for the current scientific question.""",
    }
    condition_notes = {}
    condition_block_ids = {
        "Head wind · Level 1": "note_head_1",
        "Head wind · Level 2": "note_head_2",
        "Side wind · Level 1": "note_side_1",
        "Side wind · Level 2": "note_side_2",
        "Tail wind · Level 1": "note_tail_1",
        "Tail wind · Level 2": "note_tail_2",
    }
    for condition, block_id in condition_block_ids.items():
        item = leader_map[condition]
        qualification = {
            "Head wind · Level 1": "This remains the strongest result and agrees across all five primary forward-only variants.",
            "Head wind · Level 2": "Excluding drone 5 changes the winner to Column · 75 cm, so this result remains provisional.",
            "Side wind · Level 1": "The numerical separation is small, so the ranking change is not decisive.",
            "Side wind · Level 2": "The result is threshold-robust but provisional because the leading cell has only two runs and less-strict measures favor Front · 75 cm.",
            "Tail wind · Level 1": "Median and no-drone-5 checks favor Echelon · 50 cm, so there is no clear separation.",
            "Tail wind · Level 2": "The leading cell contains one run, so the condition remains unresolved.",
        }[condition]
        condition_notes[block_id] = (
            f"### {condition}\n\n{item['leading_configuration']} has the lowest estimated score "
            f"({item['leader_forward_only_mean_sec']:.1f} s; {int(item['leader_run_count'])} run"
            f"{'s' if int(item['leader_run_count']) != 1 else ''}), {item['gap_to_runner_up_pct']:.1f}% below "
            f"{item['runner_up_configuration']}. {qualification}"
        )
    block_updates.update(condition_notes)
    for block in manifest["blocks"]:
        if block["id"] in block_updates:
            block["body"] = block_updates[block["id"]]

    diagnostic_blocks = [
        {
            "id": "echalon50_side_time_diagnostic",
            "type": "markdown",
            "layout": "full",
            "sourceId": "source_diagnostic",
            "body": f"""## Why Echelon · 50 cm Takes Longer at Side Wind Level 2

The Level 2 mean moving time is {lv2_move_mean:.2f} s versus {lv1_move_mean:.2f} s at Level 1, but the difference is concentrated rather than broad. Level 2 run {str(slow_lv2['run_id'])[-6:]} averages {slow_lv2['mean_forward_movement_sec']:.2f} s, while run {str(other_lv2['run_id'])[-6:]} is {other_lv2['mean_forward_movement_sec']:.2f} s.

The assigned distance is essentially unchanged ({lv2_distance_mean:.1f} cm versus {lv1_distance_mean:.1f} cm). Level 2 instead has lower detected forward speed ({lv2_speed_mean:.2f} cm/s versus {lv1_speed_mean:.2f} cm/s) and more lateral correction relative to forward travel ({lv2_lateral_mean:.2f} versus {lv1_lateral_mean:.2f}). This pattern is consistent with stronger crosswind requiring more correction and reducing the forward component, but two Level 2 runs are insufficient for a causal wind-level claim. The slow Level 2 run also contains a segmentation review flag.""",
        },
        {
            "id": "block_chart_echalon50_side_movement_time",
            "type": "chart",
            "chartId": "chart_echalon50_side_movement_time",
            "layout": "full",
        },
    ]
    limitation_index = next(
        index for index, block in enumerate(manifest["blocks"]) if block["id"] == "limitations"
    )
    manifest["blocks"][limitation_index:limitation_index] = diagnostic_blocks

    ARTIFACT.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print(ARTIFACT)


if __name__ == "__main__":
    main()
