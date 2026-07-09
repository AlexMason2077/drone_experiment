"""
Generate interactive hover battery comparison charts from baseline data.

Default input:
    db_copy_for_cleaning/baselines

Default outputs:
    output_graph/hover_battery_comparison.html
    output_graph/hover_battery_runs_summary.csv

The HTML is standalone: it uses embedded SVG/JavaScript and does not require
Plotly, internet access, or any Python package beyond pandas.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "db_copy_for_cleaning" / "baselines"
DEFAULT_OUTPUT = BASE_DIR / "output_graph"
SELECTED_MEAN_BATTERIES = ["B10", "B11", "B12", "B13", "B14", "B15"]

NUMERIC_COLS = [
    "elapsed_time",
    "hover_elapsed_time",
    "node_elapsed_time",
    "battery",
    "battery_start",
    "battery_drop_from_start",
]


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def clean_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def first_nonempty(df: pd.DataFrame, columns: list[str], fallback: str = "") -> str:
    for col in columns:
        if col in df.columns:
            values = df[col].dropna().map(clean_label)
            values = values[values != ""]
            if not values.empty:
                return values.iloc[0]
    return fallback


def parse_battery_id(path: Path) -> str:
    match = re.search(r"(?:^|[_/-])(B\d{1,3})(?:[_/-]|$)", path.as_posix(), re.IGNORECASE)
    return match.group(1).upper() if match else ""


def parse_drone_name(path: Path) -> str:
    match = re.search(r"(drone[_-]\d+)", path.as_posix(), re.IGNORECASE)
    return match.group(1).replace("-", "_").lower() if match else ""


def choose_time_column(df: pd.DataFrame) -> str | None:
    candidates = ["node_elapsed_time", "hover_elapsed_time", "elapsed_time"]
    best = None
    best_max = -math.inf
    for col in candidates:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        current_max = series.max(skipna=True)
        if pd.notna(current_max) and current_max > best_max:
            best = col
            best_max = float(current_max)
    return best


def downsample_points(points: list[dict], max_points: int) -> list[dict]:
    if len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def align_discharge_points(points: list[dict]) -> tuple[list[dict], float]:
    if not points:
        return [], 0.0
    start_battery = float(points[0]["battery"])
    align_index = 0
    if start_battery >= 99.5:
        first_drop_index = None
        for idx, point in enumerate(points):
            if float(point["battery"]) < 99.5:
                first_drop_index = idx
                break
        if first_drop_index is not None and first_drop_index > 0:
            align_index = first_drop_index - 1
    align_time = float(points[align_index]["t"])
    aligned = [
        {
            "t": round(float(point["t"]) - align_time, 3),
            "battery": float(point["battery"]),
        }
        for point in points[align_index:]
        if float(point["t"]) >= align_time
    ]
    return aligned, align_time


def interpolate_battery(points: list[dict], target_t: float) -> float | None:
    if not points:
        return None
    if target_t < points[0]["t"] or target_t > points[-1]["t"]:
        return None
    for idx in range(1, len(points)):
        left = points[idx - 1]
        right = points[idx]
        if target_t <= right["t"]:
            if right["t"] == left["t"]:
                return float(right["battery"])
            ratio = (target_t - left["t"]) / (right["t"] - left["t"])
            return float(left["battery"]) + ratio * (float(right["battery"]) - float(left["battery"]))
    return float(points[-1]["battery"])


def mean_trace_for_battery(battery_id: str, traces: list[dict], max_points: int) -> dict | None:
    if not traces:
        return None
    aligned_sets = []
    for trace in traces:
        aligned = trace.get("alignedPoints") or []
        if len(aligned) >= 2:
            aligned_sets.append(aligned)
    if not aligned_sets:
        return None

    max_duration = max(points[-1]["t"] for points in aligned_sets)
    grid = list(range(0, int(math.floor(max_duration)) + 1))
    mean_points = []
    sample_counts = []
    for t_value in grid:
        values = []
        for points in aligned_sets:
            value = interpolate_battery(points, float(t_value))
            if value is not None:
                values.append(value)
        if values:
            mean_points.append({
                "t": round(float(t_value), 3),
                "battery": round(sum(values) / len(values), 3),
                "n": len(values),
            })
            sample_counts.append(len(values))

    if len(mean_points) < 2:
        return None

    battery_start = mean_points[0]["battery"]
    battery_end = mean_points[-1]["battery"]
    duration_sec = mean_points[-1]["t"]
    return {
        "traceId": f"{battery_id}|aligned_mean|{len(aligned_sets)}",
        "batteryId": battery_id,
        "droneName": "aligned mean",
        "runId": f"{len(aligned_sets)} runs",
        "baselineId": f"{battery_id} aligned mean",
        "sourceFile": f"Aligned mean of {len(aligned_sets)} hover runs",
        "durationSec": round(duration_sec, 3),
        "batteryStart": round(battery_start, 3),
        "batteryEnd": round(battery_end, 3),
        "batteryDrop": round(battery_start - battery_end, 3),
        "dropPerMin": round((battery_start - battery_end) / (duration_sec / 60.0), 4) if duration_sec > 0 else None,
        "points": downsample_points(mean_points, max_points),
        "rawPointCount": len(mean_points),
        "sampleCountMin": min(sample_counts),
        "sampleCountMax": max(sample_counts),
        "kind": "mean",
    }


def threshold_aligned_mean_trace(mean_trace: dict, threshold: float = 75.0, max_points: int = 1200) -> dict | None:
    points = mean_trace.get("points") or []
    if len(points) < 2:
        return None

    crossing_time = None
    crossing_point = None
    first = points[0]
    if float(first["battery"]) <= threshold:
        crossing_time = float(first["t"])
        crossing_point = {"t": crossing_time, "battery": float(first["battery"]), "n": first.get("n")}
    else:
        for idx in range(1, len(points)):
            left = points[idx - 1]
            right = points[idx]
            left_battery = float(left["battery"])
            right_battery = float(right["battery"])
            if left_battery >= threshold >= right_battery:
                if right_battery == left_battery:
                    crossing_time = float(right["t"])
                else:
                    ratio = (threshold - left_battery) / (right_battery - left_battery)
                    crossing_time = float(left["t"]) + ratio * (float(right["t"]) - float(left["t"]))
                crossing_point = {
                    "t": crossing_time,
                    "battery": threshold,
                    "n": min(
                        int(left.get("n", 1) or 1),
                        int(right.get("n", 1) or 1),
                    ),
                }
                break

    if crossing_time is None or crossing_point is None:
        return None

    enriched = points + [crossing_point]
    enriched = sorted(enriched, key=lambda point: float(point["t"]))
    shifted = []
    seen = set()
    for point in enriched:
        shifted_t = round(float(point["t"]) - crossing_time, 3)
        key = (shifted_t, round(float(point["battery"]), 3))
        if key in seen:
            continue
        seen.add(key)
        shifted.append({
            "t": shifted_t,
            "battery": round(float(point["battery"]), 3),
            "n": point.get("n"),
        })

    out = dict(mean_trace)
    out["traceId"] = f"{mean_trace['batteryId']}|mean_aligned_at_{int(threshold)}"
    out["baselineId"] = f"{mean_trace['batteryId']} mean aligned at {threshold:g}%"
    out["sourceFile"] = f"{mean_trace['sourceFile']}; time zero aligned at {threshold:g}%"
    out["points"] = downsample_points(shifted, max_points)
    out["durationSec"] = round(max(abs(point["t"]) for point in shifted), 3)
    out["thresholdPercent"] = threshold
    out["thresholdOffsetSec"] = round(crossing_time, 3)
    out["kind"] = "mean75"
    return out


def aligned_display_trace(trace: dict) -> dict:
    aligned_points = trace.get("alignedPoints") or trace.get("points") or []
    out = dict(trace)
    out["traceId"] = f"{trace['traceId']}|aligned"
    out["points"] = aligned_points
    out["durationSec"] = round(aligned_points[-1]["t"], 3) if aligned_points else 0
    out["kind"] = "aligned_run"
    return out


def load_hover_timeseries(path: Path, max_points: int) -> tuple[dict | None, dict | None]:
    source_rel = display_path(path)
    try:
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    except Exception as exc:
        return None, {"source_file": source_rel, "status": "read_error", "notes": str(exc)}

    df.columns = df.columns.str.strip()
    if df.empty:
        return None, {"source_file": source_rel, "status": "empty_rows", "notes": ""}
    if "battery" not in df.columns:
        return None, {"source_file": source_rel, "status": "missing_battery", "notes": ""}

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    time_col = choose_time_column(df)
    if time_col is None:
        return None, {"source_file": source_rel, "status": "missing_time", "notes": ""}

    df = df[[col for col in df.columns if col in set(NUMERIC_COLS + [
        "run_id",
        "baseline_id",
        "experiment_id",
        "drone_name",
        "drone_number",
        "drone_ip",
        "battery_id",
        "mode",
        "wind_direction",
        "phase",
        "timestamp",
    ])]].copy()
    df["time_sec"] = pd.to_numeric(df[time_col], errors="coerce")
    df["battery"] = pd.to_numeric(df["battery"], errors="coerce")
    df = df.dropna(subset=["time_sec", "battery"])
    df = df[(df["time_sec"] >= 0) & (df["battery"].between(0, 100))]
    df = df.sort_values("time_sec").drop_duplicates(subset=["time_sec", "battery"])
    if len(df) < 2:
        return None, {"source_file": source_rel, "status": "too_few_rows", "notes": ""}

    start_time = float(df["time_sec"].min())
    df["time_sec"] = df["time_sec"] - start_time

    battery_id = first_nonempty(df, ["battery_id"], parse_battery_id(path))
    drone_name = first_nonempty(df, ["drone_name"], parse_drone_name(path))
    run_id = first_nonempty(df, ["run_id"], "")
    baseline_id = first_nonempty(df, ["baseline_id", "experiment_id"], path.stem)

    battery_start = float(df["battery"].iloc[0])
    battery_end = float(df["battery"].iloc[-1])
    duration_sec = float(df["time_sec"].max())
    battery_drop = battery_start - battery_end
    drop_per_min = battery_drop / (duration_sec / 60.0) if duration_sec > 0 else None

    points = [
        {
            "t": round(float(row.time_sec), 3),
            "battery": round(float(row.battery), 3),
        }
        for row in df[["time_sec", "battery"]].itertuples(index=False)
    ]
    aligned_points, align_offset_sec = align_discharge_points(points)

    trace = {
        "traceId": f"{battery_id}|{drone_name}|{run_id or baseline_id}|{path.name}",
        "batteryId": battery_id or "unknown",
        "droneName": drone_name or "unknown_drone",
        "runId": run_id,
        "baselineId": baseline_id,
        "sourceFile": source_rel,
        "durationSec": round(duration_sec, 3),
        "batteryStart": round(battery_start, 3),
        "batteryEnd": round(battery_end, 3),
        "batteryDrop": round(battery_drop, 3),
        "dropPerMin": round(drop_per_min, 4) if drop_per_min is not None else None,
        "points": downsample_points(points, max_points),
        "alignedPoints": downsample_points(aligned_points, max_points),
        "alignOffsetSec": round(align_offset_sec, 3),
        "rawPointCount": len(points),
        "kind": "run",
    }
    summary = {
        "battery_id": trace["batteryId"],
        "drone_name": trace["droneName"],
        "run_id": run_id,
        "baseline_id": baseline_id,
        "duration_sec": trace["durationSec"],
        "battery_start": trace["batteryStart"],
        "battery_end": trace["batteryEnd"],
        "battery_drop": trace["batteryDrop"],
        "battery_drop_per_min": trace["dropPerMin"],
        "align_offset_sec": trace["alignOffsetSec"],
        "raw_point_count": trace["rawPointCount"],
        "plotted_point_count": len(trace["points"]),
        "source_file": source_rel,
        "status": "included",
        "notes": "",
    }
    return trace, summary


def find_hover_timeseries(input_dir: Path) -> list[Path]:
    paths = []
    for path in input_dir.rglob("*timeseries.csv"):
        lower = path.as_posix().lower()
        if "hover" not in lower:
            continue
        if "source_raw" in lower:
            continue
        paths.append(path)
    return sorted(paths)


def build_html(traces: list[dict]) -> str:
    batteries = sorted({trace["batteryId"] for trace in traces})
    battery_traces = {
        battery_id: [trace for trace in traces if trace["batteryId"] == battery_id]
        for battery_id in batteries
    }
    aligned_battery_traces = {
        battery_id: [aligned_display_trace(trace) for trace in battery_traces[battery_id]]
        for battery_id in batteries
    }
    mean_traces = {
        battery_id: mean_trace_for_battery(battery_id, battery_traces[battery_id], max_points=1200)
        for battery_id in batteries
    }
    mean_traces = {battery_id: trace for battery_id, trace in mean_traces.items() if trace is not None}
    selected_mean_traces = [
        mean_traces[battery_id]
        for battery_id in SELECTED_MEAN_BATTERIES
        if battery_id in mean_traces
    ]
    selected_threshold_mean_traces = [
        trace
        for trace in (
            threshold_aligned_mean_trace(mean_traces[battery_id], threshold=75.0, max_points=1200)
            for battery_id in SELECTED_MEAN_BATTERIES
            if battery_id in mean_traces
        )
        if trace is not None
    ]
    payload = {
        "traces": traces,
        "batteries": batteries,
        "batteryTraces": battery_traces,
        "alignedBatteryTraces": aligned_battery_traces,
        "meanTraces": mean_traces,
        "selectedMeanTraces": selected_mean_traces,
        "selectedThresholdMeanTraces": selected_threshold_mean_traces,
        "selectedMeanBatteries": SELECTED_MEAN_BATTERIES,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hover Battery Comparison</title>
<style>
  :root {{
    --text: #17212b;
    --muted: #647080;
    --line: #d9e0e7;
    --panel: #ffffff;
    --bg: #f5f7fa;
  }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
  }}
  main {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 24px;
  }}
  h1 {{ margin: 0 0 6px; font-size: 28px; }}
  h2 {{ margin: 28px 0 10px; font-size: 20px; }}
  .note {{ color: var(--muted); margin-bottom: 18px; }}
  .chart {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 18px;
  }}
  .chart-title {{
    font-weight: 650;
    margin: 0 0 10px;
  }}
  svg {{
    width: 100%;
    height: 520px;
    display: block;
    background: #fff;
  }}
  .small svg {{ height: 360px; }}
  .axis text {{ fill: var(--muted); font-size: 12px; }}
  .axis line, .axis path, .grid line {{ stroke: var(--line); }}
  .grid line {{ stroke-dasharray: 3 4; }}
  .trace {{ fill: none; stroke-width: 2; opacity: .9; }}
  .trace.mean {{ stroke-width: 2; opacity: 1; stroke-dasharray: 9 5; }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    color: var(--muted);
    font-size: 13px;
    margin: 8px 0 0;
  }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
  .tooltip {{
    position: fixed;
    z-index: 10;
    pointer-events: none;
    background: rgba(20, 29, 38, .94);
    color: #fff;
    padding: 9px 10px;
    border-radius: 7px;
    font-size: 13px;
    line-height: 1.45;
    min-width: 220px;
    box-shadow: 0 8px 24px rgba(0,0,0,.18);
    display: none;
  }}
  .empty {{
    color: var(--muted);
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 18px;
  }}
</style>
</head>
<body>
<main>
  <h1>Hover Battery Comparison</h1>
  <div class="note">
    Hover over a line to see battery ID, discharge time, current battery,
    drone, run ID, and source file. Curves show remaining battery percentage over time.
  </div>

  <section class="chart">
    <div class="chart-title">Different Batteries: Remaining Battery vs Hover Time</div>
    <div id="all-chart"></div>
    <div id="all-legend" class="legend"></div>
  </section>

  <section class="chart">
    <div class="chart-title">Selected Batteries Mean Curves: B10-B15</div>
    <div id="selected-mean-chart"></div>
    <div id="selected-mean-legend" class="legend"></div>
  </section>

  <section class="chart">
    <div class="chart-title">Selected Batteries Mean Curves: Aligned At 75% Battery</div>
    <div id="selected-mean-75-chart"></div>
    <div id="selected-mean-75-legend" class="legend"></div>
  </section>

  <h2>Battery Consumption By Battery ID</h2>
  <div id="battery-charts"></div>
</main>
<div id="tooltip" class="tooltip"></div>
<script>
const DATA = {data_json};
const COLORS = [
  "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
  "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
  "#005f73", "#9b2226", "#0a9396", "#ca6702", "#6a4c93"
];
const batteryColor = new Map(DATA.batteries.map((id, i) => [id, COLORS[i % COLORS.length]]));

function fmtTime(sec) {{
  if (!Number.isFinite(sec)) return "";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${{m}}m ${{String(s).padStart(2, "0")}}s`;
}}

function makeLegend(containerId, traces) {{
  const container = document.getElementById(containerId);
  if (!container) return;
  const ids = Array.from(new Set(traces.map(t => t.batteryId))).sort();
  container.innerHTML = ids.map(id => `
    <span class="legend-item"><span class="swatch" style="background:${{batteryColor.get(id)}}"></span>${{id}}</span>
  `).join("");
}}

function extent(values, pad = 0) {{
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {{ min -= 1; max += 1; }}
  const span = max - min;
  return [min - span * pad, max + span * pad];
}}

function pathFor(points, xScale, yScale) {{
  return points.map((p, i) => `${{i ? "L" : "M"}} ${{xScale(p.t).toFixed(2)}} ${{yScale(p.battery).toFixed(2)}}`).join(" ");
}}

function splitSegments(points, predicate) {{
  const segments = [];
  let current = [];
  for (const point of points) {{
    if (predicate(point)) {{
      current.push(point);
    }} else {{
      if (current.length > 1) segments.push(current);
      current = [];
    }}
  }}
  if (current.length > 1) segments.push(current);
  return segments;
}}

function traceStroke(trace, options) {{
  if ((trace.kind === "mean" || trace.kind === "mean75") && !options.colorMeanByBattery) {{
    return "#111827";
  }}
  return batteryColor.get(trace.batteryId) || "#333";
}}

function renderTrace(trace, xScale, yScale, options) {{
  const color = traceStroke(trace, options);
  const meanClass = trace.kind === "mean" || trace.kind === "mean75" ? "mean" : "";
  if (options.weakAbove75) {{
    const weakSegments = splitSegments(trace.points, point => point.battery >= 75);
    const strongSegments = splitSegments(trace.points, point => point.battery <= 75);
    return [
      ...weakSegments.map(segment => `
        <path class="trace ${{meanClass}}" data-trace="${{trace.traceId}}"
          d="${{pathFor(segment, xScale, yScale)}}"
          stroke="${{color}}" opacity="0.25" stroke-dasharray="4 7"></path>
      `),
      ...strongSegments.map(segment => `
        <path class="trace ${{meanClass}}" data-trace="${{trace.traceId}}"
          d="${{pathFor(segment, xScale, yScale)}}"
          stroke="${{color}}"></path>
      `),
    ].join("");
  }}
  return `
    <path class="trace ${{meanClass}}" data-trace="${{trace.traceId}}"
      d="${{pathFor(trace.points, xScale, yScale)}}"
      stroke="${{color}}"></path>
  `;
}}

function createChart(containerId, traces, options = {{}}) {{
  const container = document.getElementById(containerId);
  if (!container || traces.length === 0) return;
  const width = 1080;
  const height = options.small ? 360 : 520;
  const margin = {{left: 64, right: 24, top: 20, bottom: 54}};
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const allPoints = traces.flatMap(t => t.points);
  const [xMin, xMax] = extent(allPoints.map(p => p.t), .02);
  const [yMinRaw, yMaxRaw] = extent(allPoints.map(p => p.battery), .05);
  const yMin = Math.max(0, Math.floor(yMinRaw / 5) * 5);
  const yMax = Math.min(100, Math.ceil(yMaxRaw / 5) * 5);
  const xScale = x => margin.left + ((x - xMin) / (xMax - xMin)) * innerW;
  const yScale = y => margin.top + (1 - ((y - yMin) / (yMax - yMin))) * innerH;

  const xTicks = Array.from({{length: 7}}, (_, i) => xMin + (xMax - xMin) * i / 6);
  const yTicks = Array.from({{length: 6}}, (_, i) => yMin + (yMax - yMin) * i / 5);
  const tracePaths = traces.map(trace => renderTrace(trace, xScale, yScale, options)).join("");

  container.innerHTML = `
    <svg viewBox="0 0 ${{width}} ${{height}}" role="img">
      <g class="grid">
        ${{xTicks.map(t => `<line x1="${{xScale(t)}}" x2="${{xScale(t)}}" y1="${{margin.top}}" y2="${{height - margin.bottom}}"></line>`).join("")}}
        ${{yTicks.map(t => `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{yScale(t)}}" y2="${{yScale(t)}}"></line>`).join("")}}
      </g>
      <g class="axis">
        <line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{height - margin.bottom}}" y2="${{height - margin.bottom}}"></line>
        <line x1="${{margin.left}}" x2="${{margin.left}}" y1="${{margin.top}}" y2="${{height - margin.bottom}}"></line>
        ${{xTicks.map(t => `<text x="${{xScale(t)}}" y="${{height - 18}}" text-anchor="middle">${{fmtTime(t)}}</text>`).join("")}}
        ${{yTicks.map(t => `<text x="${{margin.left - 10}}" y="${{yScale(t) + 4}}" text-anchor="end">${{t.toFixed(0)}}%</text>`).join("")}}
        <text x="${{width / 2}}" y="${{height - 4}}" text-anchor="middle">Hover discharge time</text>
        <text x="16" y="${{height / 2}}" text-anchor="middle" transform="rotate(-90 16 ${{height / 2}})">Current battery (%)</text>
      </g>
      <g>${{tracePaths}}</g>
      <circle id="${{containerId}}-hover-dot" r="5" fill="#111" stroke="#fff" stroke-width="2" style="display:none"></circle>
    </svg>
  `;

  const svg = container.querySelector("svg");
  const hoverDot = container.querySelector(`#${{containerId}}-hover-dot`);
  const tooltip = document.getElementById("tooltip");
  const screenPoints = [];
  for (const trace of traces) {{
    for (const point of trace.points) {{
      screenPoints.push({{
        x: xScale(point.t),
        y: yScale(point.battery),
        point,
        trace,
      }});
    }}
  }}

  svg.addEventListener("mousemove", event => {{
    const rect = svg.getBoundingClientRect();
    const sx = (event.clientX - rect.left) * (width / rect.width);
    const sy = (event.clientY - rect.top) * (height / rect.height);
    let nearest = null;
    let best = Infinity;
    for (const item of screenPoints) {{
      const dx = item.x - sx;
      const dy = item.y - sy;
      const dist = dx * dx + dy * dy;
      if (dist < best) {{
        best = dist;
        nearest = item;
      }}
    }}
    if (!nearest || best > 900) {{
      tooltip.style.display = "none";
      hoverDot.style.display = "none";
      return;
    }}
    hoverDot.setAttribute("cx", nearest.x);
    hoverDot.setAttribute("cy", nearest.y);
    hoverDot.setAttribute("fill", batteryColor.get(nearest.trace.batteryId) || "#111");
    hoverDot.style.display = "block";
    tooltip.innerHTML = `
      <strong>Battery ${{nearest.trace.batteryId}}</strong><br>
      Curve: ${{nearest.trace.kind === "mean75" ? "mean aligned at 75%" : nearest.trace.kind === "mean" ? "aligned mean" : nearest.trace.kind === "aligned_run" ? "aligned run" : "single run"}}<br>
      Discharge time: ${{fmtTime(nearest.point.t)}} (${{nearest.point.t.toFixed(1)}} s)<br>
      Current battery: ${{nearest.point.battery.toFixed(1)}}%<br>
      Drone: ${{nearest.trace.droneName}}<br>
      Run: ${{nearest.trace.runId || nearest.trace.baselineId}}<br>
      Total duration: ${{fmtTime(nearest.trace.durationSec)}}<br>
      Total drop: ${{nearest.trace.batteryDrop.toFixed(1)}}%<br>
      ${{nearest.trace.kind === "mean" ? `Samples at this time: ${{nearest.point.n || ""}}<br>` : ""}}
      <span style="color:#cbd5df">${{nearest.trace.sourceFile}}</span>
    `;
    tooltip.style.left = `${{event.clientX + 14}}px`;
    tooltip.style.top = `${{event.clientY + 14}}px`;
    tooltip.style.display = "block";
  }});
  svg.addEventListener("mouseleave", () => {{
    tooltip.style.display = "none";
    hoverDot.style.display = "none";
  }});
}}

createChart("all-chart", DATA.traces);
makeLegend("all-legend", DATA.traces);
createChart("selected-mean-chart", DATA.selectedMeanTraces, {{colorMeanByBattery: true}});
makeLegend("selected-mean-legend", DATA.selectedMeanTraces);
createChart("selected-mean-75-chart", DATA.selectedThresholdMeanTraces, {{colorMeanByBattery: true, weakAbove75: true}});
makeLegend("selected-mean-75-legend", DATA.selectedThresholdMeanTraces);

const batteryRoot = document.getElementById("battery-charts");
const batteryIds = DATA.batteries;
batteryRoot.innerHTML = batteryIds.map(id => `
    <section class="chart small">
      <div class="chart-title">Battery ${{id}}: hover discharge runs + aligned mean</div>
      <div id="battery-${{id}}"></div>
    </section>
  `).join("");
for (const id of batteryIds) {{
  const traces = [...(DATA.alignedBatteryTraces[id] || [])];
  if (DATA.meanTraces[id] && traces.length > 1) {{
    traces.push(DATA.meanTraces[id]);
  }}
  createChart(`battery-${{id}}`, traces, {{small: true}});
}}
</script>
</body>
</html>
"""


def write_summary(rows: list[dict], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["status", "battery_id", "drone_name", "baseline_id"], na_position="last")
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interactive hover battery baseline charts.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-points-per-run", type=int, default=1200)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    summary_rows = []
    for path in find_hover_timeseries(input_dir):
        trace, summary = load_hover_timeseries(path, max_points=args.max_points_per_run)
        if trace is not None:
            traces.append(trace)
        if summary is not None:
            summary_rows.append(summary)

    traces = sorted(traces, key=lambda item: (item["batteryId"], item["droneName"], item["baselineId"]))
    summary_path = output_dir / "hover_battery_runs_summary.csv"
    html_path = output_dir / "hover_battery_comparison.html"
    write_summary(summary_rows, summary_path)

    if not traces:
        raise SystemExit(f"No usable hover timeseries found in {input_dir}")

    html_path.write_text(build_html(traces), encoding="utf-8")
    print(f"Included hover runs: {len(traces)}")
    print(f"Summary CSV: {summary_path}")
    print(f"Interactive HTML: {html_path}")


if __name__ == "__main__":
    main()
