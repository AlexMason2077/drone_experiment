"""Build the canonical Data Analytics report artifact for swarm energy findings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "swarm_analysis"


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    factors = pd.read_csv(OUT / "factor_effect_summary.csv")
    cells = pd.read_csv(OUT / "scenario_cell_summary.csv")
    runs = pd.read_csv(OUT / "clean_swarm_runs.csv")

    formation = factors[factors.factor == "formation"].copy()
    order = {"front": 0, "vee": 1, "echalon": 2, "diamond": 3, "column": 4}
    formation["order"] = formation.level.map(order)
    formation = formation.sort_values("order")
    formation_long = formation.melt(
        id_vars=["level", "order"],
        value_vars=["mean_battery_drop", "mean_max_battery_drop"],
        var_name="metric", value_name="battery_drop",
    )
    formation_long["metric"] = formation_long.metric.map({
        "mean_battery_drop": "Average drone", "mean_max_battery_drop": "Most depleted drone",
    })

    front = formation.set_index("level").loc["front"]
    decomposition = []
    for row in formation.itertuples(index=False):
        for metric, value in [
            ("Total battery drop", row.mean_battery_drop / front.mean_battery_drop * 100),
            ("Traversal duration", row.mean_duration_sec / front.mean_duration_sec * 100),
            ("Drop rate", row.mean_drop_rate / front.mean_drop_rate * 100),
        ]:
            decomposition.append({"formation": row.level, "order": row.order, "metric": metric, "index": round(value, 2)})

    cells["condition"] = cells.apply(
        lambda r: f"{int(r.distance)}cm · {r.wind_direction_short} · lv{int(r.wind_level)}", axis=1
    )
    cells["scenario"] = cells.apply(
        lambda r: f"{r.formation} · {int(r.distance)}cm · {r.wind_direction_short} · lv{int(r.wind_level)}", axis=1
    )
    ranked = pd.concat([cells.nsmallest(10, "mean_battery_drop"), cells.nlargest(10, "mean_battery_drop")])
    ranked = ranked.drop_duplicates("scenario").sort_values("mean_battery_drop", ascending=False)
    ranked = ranked[["scenario", "run_count", "mean_battery_drop", "mean_max_battery_drop", "mean_duration_sec"]].round(2)

    sources = [
        {
            "id": "swarm-clean-runs", "label": "Clean swarm run dataset",
            "path": "swarm_analysis/clean_swarm_runs.csv",
            "query": {"sql": "SELECT * FROM read_csv_auto('swarm_analysis/clean_swarm_runs.csv')"},
        },
        {
            "id": "scenario-cells", "label": "Scenario cell summary",
            "path": "swarm_analysis/scenario_cell_summary.csv",
            "query": {"sql": "SELECT * FROM read_csv_auto('swarm_analysis/scenario_cell_summary.csv')"},
        },
        {
            "id": "factor-effects", "label": "Cell-balanced factor effect summary",
            "path": "swarm_analysis/factor_effect_summary.csv",
            "query": {"sql": "SELECT * FROM read_csv_auto('swarm_analysis/factor_effect_summary.csv')"},
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# Swarm flight energy patterns"},
        {"id": "executive-summary", "type": "markdown", "sourceId": "factor-effects", "body": """## Executive Summary

- **Front is the lowest-consumption formation on common conditions.** Its mean drop is 7.33 percentage points per node, versus 8.22 for echalon, 8.56 for vee, 9.73 for diamond, and 10.88 for column.
- **Formation differences are primarily duration-driven.** Column uses about 48% more battery than front and takes about 48% longer, while its mean drop rate is only about 2% higher.
- **Wind and spacing act through interactions.** Side wind is highest on average, but formation-specific reversals mean the optimization model should not use one global wind or spacing penalty.
- **Charging decisions should model the most depleted drone.** Depending on formation, the maximum drone-level drop is 16%–32% above the swarm mean and can determine charging completion time."""},
        {"id": "metric-definition", "type": "markdown", "body": """## What the energy metric represents

The primary metric is the **battery percentage-point drop required for the five-drone swarm to complete one 250 cm node traversal**. The report shows both the average drone and the most depleted drone. The latter is the relevant bottleneck when the next mission cannot start until every required drone has sufficient charge."""},
        {"id": "formation-finding", "type": "markdown", "sourceId": "factor-effects", "body": """## Front minimizes energy; column creates the largest charging burden

The formation ranking remains visible when the comparison is restricted to the ten distance–direction–wind-level settings shared by all five formations. The orange series matters for the future scheduler: it approximates the largest recharge requirement created by one node traversal."""},
        {"id": "formation-chart", "type": "chart", "chartId": "formation-burden"},
        {"id": "duration-finding", "type": "markdown", "sourceId": "factor-effects", "body": """## Longer traversal time explains most of the formation energy gap

Unit-time consumption is comparatively similar across formations, while traversal duration changes sharply. This supports a two-part cost model: first predict time to reach the node, then predict conditional discharge intensity. Multiplying the two produces the expected battery loss used by the charging optimizer."""},
        {"id": "duration-chart", "type": "chart", "chartId": "duration-decomposition"},
        {"id": "wind-finding", "type": "markdown", "sourceId": "scenario-cells", "body": """## Wind direction and spacing cannot be reduced to global penalties

Side wind is highest on average, but the detailed matrix contains large formation-specific differences and several reversals. Similarly, the global difference between 50 cm and 75 cm is small, yet the sign changes by formation and wind level. The future algorithm therefore needs condition-specific or interaction-aware lookup values."""},
        {"id": "wind-chart", "type": "chart", "chartId": "condition-heatmap"},
        {"id": "ranking-finding", "type": "markdown", "sourceId": "scenario-cells", "body": """## Observed scenario costs span a wide range

The best observed cells require roughly 6 battery percentage points per drone, while the highest observed cells exceed 13–14 points. Some extreme cells have only one or two retained runs, so these values should seed a provisional lookup table rather than be treated as final deterministic constants."""},
        {"id": "ranking-table", "type": "table", "tableId": "scenario-ranking"},
        {"id": "recommendations", "type": "markdown", "body": """## Recommended next steps

1. Build the optimization state around per-drone state of charge, current formation, forecast wind direction/level, spacing, next-node distance, and charging-pad availability.
2. Predict traversal duration and discharge intensity separately, then derive expected per-drone battery loss.
3. Optimize against the maximum charging completion time when all drones must depart together; use total charging work as a secondary objective when pads can operate in parallel.
4. Add uncertainty penalties for sparse scenario cells so the algorithm avoids apparently efficient but weakly supported choices.
5. Collect charging curves and pad service rates; battery percentage loss alone cannot yet be converted into exact charging minutes."""},
        {"id": "further-questions", "type": "markdown", "body": """## Further questions

- Are charging rates linear across the relevant state-of-charge range, or does the scheduler need a nonlinear charging model?
- Can formation be changed between nodes, and what energy/time cost does reconfiguration add?
- Does the next mission require all five drones, or can a partially charged subset depart?
- How accurately will wind direction and speed be forecast at decision time?"""},
        {"id": "caveats", "type": "markdown", "body": """## Caveats and assumptions

- The two column, 50 cm, level-2 head/side conditions were deliberately discontinued because repeated collisions created a safety risk; they are not imputed.
- Diamond, 50 cm, side wind, level 2 has no retained non-outlier run.
- Scenario replication is unbalanced, ranging from one to eight retained runs per observed cell.
- Battery telemetry is integer-valued, so individual-drone short-run discharge rates are noisy.
- Drone position and battery identity are not fully separable in the existing design; position-specific effects should not be interpreted as causal battery-independent effects."""},
    ]

    charts = [
        {
            "id": "formation-burden", "title": "Formation-level battery burden",
            "subtitle": "Common-condition mean percentage-point drop per node",
            "type": "bar", "intent": "comparison", "dataset": "formation_burden", "sourceId": "factor-effects",
            "encodings": {
                "x": {"field": "level", "type": "ordinal", "label": "Formation"},
                "y": {"field": "battery_drop", "type": "quantitative", "label": "Battery drop", "unit": "% points"},
                "color": {"field": "metric", "type": "nominal", "label": "Drone summary"},
                "tooltip": [{"field": "metric"}, {"field": "battery_drop", "format": "number"}],
            },
            "settings": {"groupMode": "grouped", "sort": "custom"}, "legend": {"position": "bottom"}, "layout": "full",
        },
        {
            "id": "duration-decomposition", "title": "Formation energy, duration, and intensity indices",
            "subtitle": "Front = 100; formation comparison uses common experimental settings",
            "type": "line", "intent": "decomposition", "dataset": "duration_decomposition", "sourceId": "factor-effects",
            "encodings": {
                "x": {"field": "formation", "type": "ordinal", "label": "Formation"},
                "y": {"field": "index", "type": "quantitative", "label": "Index"},
                "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                "tooltip": [{"field": "metric"}, {"field": "index", "format": "number"}],
            },
            "referenceLines": [{"axis": "y", "value": 100, "label": "Front baseline", "lineStyle": "dashed", "color": "neutral"}],
            "legend": {"position": "bottom"}, "layout": "full",
        },
        {
            "id": "condition-heatmap", "title": "Formation and wind-condition energy comparison",
            "subtitle": "Observed mean battery drop per node; unavailable cells are omitted",
            "type": "bar", "intent": "comparison", "dataset": "scenario_cells", "sourceId": "scenario-cells",
            "encodings": {
                "x": {"field": "condition", "type": "nominal", "label": "Spacing · direction · wind level"},
                "y": {"field": "mean_battery_drop", "type": "quantitative", "label": "Mean battery drop", "unit": "% points"},
                "color": {"field": "formation", "type": "nominal", "label": "Formation"},
                "tooltip": [{"field": "run_count"}, {"field": "mean_battery_drop", "format": "number"}, {"field": "mean_max_battery_drop", "format": "number"}],
            },
            "settings": {"groupMode": "grouped", "categoryLabelPolicy": "rotate"}, "legend": {"position": "bottom"}, "layout": "full",
        },
    ]

    tables = [{
        "id": "scenario-ranking", "title": "Lowest- and highest-consumption observed scenarios",
        "subtitle": "Exact observed cell means with retained run counts",
        "dataset": "scenario_ranking", "sourceId": "scenario-cells", "layout": "full", "density": "spacious",
        "defaultSort": {"field": "mean_battery_drop", "direction": "desc"},
        "columns": [
            {"field": "scenario", "label": "Scenario", "type": "text"},
            {"field": "run_count", "label": "Runs", "format": "number"},
            {"field": "mean_battery_drop", "label": "Mean drop", "format": "number"},
            {"field": "mean_max_battery_drop", "label": "Most depleted", "format": "number"},
            {"field": "mean_duration_sec", "label": "Duration (s)", "format": "number"},
        ],
    }]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": "Swarm flight energy patterns",
            "description": "Energy patterns across formation, wind, and spacing for charging-aware swarm optimization.",
            "generatedAt": generated, "blocks": blocks, "charts": charts, "tables": tables, "sources": sources,
        },
        "snapshot": {
            "version": 1, "generatedAt": generated, "status": "ready",
            "datasets": {
                "formation_burden": records(formation_long[["level", "order", "metric", "battery_drop"]].round(3)),
                "duration_decomposition": decomposition,
                "scenario_cells": records(cells[["formation", "condition", "run_count", "mean_battery_drop", "mean_max_battery_drop", "mean_duration_sec"]].round(3)),
                "scenario_ranking": records(ranked),
            },
        },
        "sources": sources,
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT / "artifact.json")


if __name__ == "__main__":
    main()
