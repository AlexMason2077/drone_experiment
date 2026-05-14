from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "discharge_logs"
OUT_DIR = BASE_DIR / "analysis_outputs"

MIN_LINEARIZATION_SOC_SPAN_PCT = 30.0
MIN_END_BATTERY_PCT = 40.0
PALETTE = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
]


def discover_battery_logs() -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for path in sorted(LOG_DIR.glob("battery_*.csv")):
        match = re.search(r"battery_(\d+)_", path.name)
        if not match:
            continue
        battery_id = match.group(1).zfill(2)
        # If repeated runs exist for the same battery, keep the newest filename.
        discovered[battery_id] = path
    return dict(sorted(discovered.items(), key=lambda item: int(item[0])))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def constrained_cubic_fit(
    x: np.ndarray,
    y: np.ndarray,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    z = (x - x_end) / (x_start - x_end)
    baseline = y_end + (y_start - y_end) * z
    basis = np.column_stack([z * (1.0 - z), z * (1.0 - z) * (2.0 * z - 1.0)])
    params, *_ = np.linalg.lstsq(basis, y - baseline, rcond=None)
    p0 = float(params[0])
    p1 = float(params[1])
    y_fit = baseline + basis @ params
    z_poly = np.poly1d([1.0 / (x_start - x_end), -x_end / (x_start - x_end)])
    poly = (
        (y_start - y_end) * z_poly
        + y_end
        + z_poly * (1 - z_poly) * p0
        + z_poly * (1 - z_poly) * (2 * z_poly - 1) * p1
    )
    meta = {
        "x_start": float(x_start),
        "x_end": float(x_end),
        "y_start": float(y_start),
        "y_end": float(y_end),
        "basis_params": params.tolist(),
        "standard_poly_coeff": poly.c.tolist(),
    }
    return y_fit, poly.c, meta


def load_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["elapsed_s", "battery_pct"]].dropna().copy()
    df["elapsed_min"] = df["elapsed_s"] / 60.0
    return df


def grouped_nodes(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("battery_pct", as_index=False)
        .agg(
            first_elapsed_s=("elapsed_s", "first"),
            last_elapsed_s=("elapsed_s", "last"),
            mean_elapsed_s=("elapsed_s", "mean"),
            samples=("elapsed_s", "size"),
        )
        .sort_values("battery_pct", ascending=False)
        .reset_index(drop=True)
    )
    grouped["mid_elapsed_s"] = (
        grouped["first_elapsed_s"] + grouped["last_elapsed_s"]
    ) / 2.0
    return grouped


def analyze_one(battery_id: str, path: Path) -> tuple[pd.DataFrame, dict]:
    df = load_curve(path)
    nodes = grouped_nodes(df)

    start_s = float(df["elapsed_s"].iloc[0])
    end_s = float(df["elapsed_s"].iloc[-1])
    span_s = end_s - start_s
    if span_s <= 0:
        raise ValueError(f"battery {battery_id} has non-positive duration")

    start_pct = float(df["battery_pct"].iloc[0])
    end_pct = float(df["battery_pct"].iloc[-1])
    soc_span = start_pct - end_pct
    if soc_span < MIN_LINEARIZATION_SOC_SPAN_PCT or end_pct > MIN_END_BATTERY_PCT:
        raise ValueError(
            f"battery {battery_id} skipped: SoC span {soc_span:.1f}% "
            f"({start_pct:.0f}->{end_pct:.0f}%) is too short for full-curve "
            f"linearization; need span >= {MIN_LINEARIZATION_SOC_SPAN_PCT:.0f}% "
            f"and end <= {MIN_END_BATTERY_PCT:.0f}%"
        )

    nodes["u"] = (nodes["mid_elapsed_s"] - start_s) / span_s
    nodes["linear_remaining_pct"] = 100.0 * (1.0 - nodes["u"])

    raw_x = nodes["battery_pct"].to_numpy(dtype=float)
    linear_y = nodes["linear_remaining_pct"].to_numpy(dtype=float)
    u = nodes["u"].to_numpy(dtype=float)
    raw_y = nodes["battery_pct"].to_numpy(dtype=float)

    # Cubic conversion from reported BMS percentage to time-linear remaining percentage.
    # The endpoints are constrained so the tested start maps to 100 and cutoff maps to 0.
    lin_fit, lin_coeff, lin_meta = constrained_cubic_fit(
        raw_x,
        linear_y,
        x_start=start_pct,
        x_end=end_pct,
        y_start=100.0,
        y_end=0.0,
    )
    nodes["linearized_fit_pct"] = lin_fit
    nodes["linearization_residual_pct"] = (
        nodes["linear_remaining_pct"] - nodes["linearized_fit_pct"]
    )

    # Cubic curve fit for the original discharge curve, also endpoint-constrained.
    raw_fit, raw_coeff, raw_meta = constrained_cubic_fit(
        u,
        raw_y,
        x_start=0.0,
        x_end=1.0,
        y_start=start_pct,
        y_end=end_pct,
    )
    nodes["raw_curve_fit_pct"] = raw_fit
    nodes["raw_curve_residual_pct"] = nodes["battery_pct"] - nodes["raw_curve_fit_pct"]

    # Apply the conversion to every raw sample for plotting/export.
    df["u"] = (df["elapsed_s"] - start_s) / span_s
    df["linear_remaining_pct"] = 100.0 * (1.0 - df["u"])
    df["linearized_fit_pct"] = np.polyval(lin_coeff, df["battery_pct"].to_numpy(dtype=float))
    df["battery_id"] = battery_id

    summary = {
        "battery_id": battery_id,
        "source_file": str(path.relative_to(BASE_DIR)),
        "samples": int(len(df)),
        "duration_s": end_s,
        "duration_min": end_s / 60.0,
        "start_battery_pct": int(df["battery_pct"].iloc[0]),
        "end_battery_pct": int(df["battery_pct"].iloc[-1]),
        "unique_battery_levels": int(nodes.shape[0]),
        "mean_drop_pct_per_min": float(
            (df["battery_pct"].iloc[0] - df["battery_pct"].iloc[-1])
            / (end_s / 60.0)
        ),
        "raw_curve_constrained_poly3_battery_pct_vs_u": raw_meta,
        "raw_curve_poly3_r2": r2_score(raw_y, nodes["raw_curve_fit_pct"].to_numpy()),
        "raw_curve_poly3_rmse_pct": rmse(
            raw_y, nodes["raw_curve_fit_pct"].to_numpy()
        ),
        "linearization_constrained_poly3_linear_remaining_vs_raw_pct": lin_meta,
        "linearization_poly3_r2": r2_score(
            linear_y, nodes["linearized_fit_pct"].to_numpy()
        ),
        "linearization_poly3_rmse_pct": rmse(
            linear_y, nodes["linearized_fit_pct"].to_numpy()
        ),
    }
    return df, nodes, summary


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def draw_line_plot(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[dict],
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    hline: float | None = None,
) -> None:
    left, top, right, bottom = box
    title_font = load_font(28, bold=True)
    label_font = load_font(20)
    tick_font = load_font(17)
    legend_font = load_font(18)

    plot_left = left + 88
    plot_top = top + 58
    plot_right = right - 28
    plot_bottom = bottom - 72
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    xs = np.concatenate([np.asarray(s["x"], dtype=float) for s in series])
    ys = np.concatenate([np.asarray(s["y"], dtype=float) for s in series])
    xmin, xmax = xlim if xlim else (float(np.nanmin(xs)), float(np.nanmax(xs)))
    ymin, ymax = ylim if ylim else (float(np.nanmin(ys)), float(np.nanmax(ys)))
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    ypad = (ymax - ymin) * 0.06
    if not ylim:
        ymin -= ypad
        ymax += ypad

    def sx(x: float) -> float:
        return plot_left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return plot_bottom - (y - ymin) / (ymax - ymin) * plot_h

    draw.rectangle([left, top, right, bottom], fill=(250, 250, 247), outline=(220, 220, 215))
    draw.text((left + 18, top + 16), title, fill=(30, 30, 30), font=title_font)

    for i in range(6):
        x = plot_left + i * plot_w / 5
        y = plot_top + i * plot_h / 5
        draw.line([(x, plot_top), (x, plot_bottom)], fill=(225, 225, 220), width=1)
        draw.line([(plot_left, y), (plot_right, y)], fill=(225, 225, 220), width=1)

        xv = xmin + i * (xmax - xmin) / 5
        yv = ymax - i * (ymax - ymin) / 5
        draw.text((x - 18, plot_bottom + 8), f"{xv:.1f}", fill=(70, 70, 70), font=tick_font)
        draw.text((plot_left - 72, y - 9), f"{yv:.0f}", fill=(70, 70, 70), font=tick_font)

    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=(70, 70, 70), width=2)
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=(70, 70, 70), width=2)

    if hline is not None and ymin <= hline <= ymax:
        y = sy(hline)
        draw.line([(plot_left, y), (plot_right, y)], fill=(70, 70, 70), width=2)

    for s in series:
        color = hex_to_rgb(s["color"])
        points = [
            (sx(float(x)), sy(float(y)))
            for x, y in zip(np.asarray(s["x"], dtype=float), np.asarray(s["y"], dtype=float))
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(points) >= 2 and s.get("width", 3) > 0:
            draw.line(points, fill=color, width=s.get("width", 3), joint="curve")
        if s.get("scatter"):
            radius = s.get("radius", 3)
            for x, y in points:
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    draw.text(((plot_left + plot_right) / 2 - 80, bottom - 38), xlabel, fill=(40, 40, 40), font=label_font)
    draw.text((left + 14, (plot_top + plot_bottom) / 2 - 12), ylabel, fill=(40, 40, 40), font=label_font)

    legend_x = plot_right - 265
    legend_y = plot_top + 12
    for i, s in enumerate(series):
        y = legend_y + i * 25
        color = hex_to_rgb(s["color"])
        draw.line([(legend_x, y + 9), (legend_x + 34, y + 9)], fill=color, width=4)
        draw.text((legend_x + 44, y), s["label"], fill=(35, 35, 35), font=legend_font)


def make_plots(results: dict[str, dict]) -> None:
    colors = {
        battery_id: PALETTE[i % len(PALETTE)]
        for i, battery_id in enumerate(results.keys())
    }
    image = Image.new("RGB", (2400, 1600), (245, 245, 240))
    draw = ImageDraw.Draw(image)
    draw.text(
        (55, 28),
        "Tello Battery Discharge and Time-Linearization",
        fill=(20, 20, 20),
        font=load_font(42, bold=True),
    )

    raw_series = []
    for battery_id, result in results.items():
        df = result["df"].iloc[::8]
        nodes = result["nodes"]
        raw_series.append(
            {
                "x": df["elapsed_min"],
                "y": df["battery_pct"],
                "color": colors[battery_id],
                "label": f"B{battery_id} raw",
                "width": 2,
            }
        )
        raw_series.append(
            {
                "x": nodes["mid_elapsed_s"] / 60.0,
                "y": nodes["raw_curve_fit_pct"],
                "color": colors[battery_id],
                "label": f"B{battery_id} cubic fit",
                "width": 5,
            }
        )
    draw_line_plot(
        draw,
        (55, 105, 1170, 775),
        raw_series,
        "Original Reported Discharge Curve",
        "Elapsed time (min)",
        "Reported battery (%)",
        ylim=(0, 100),
    )

    mapping_series = []
    for battery_id, result in results.items():
        nodes = result["nodes"]
        sort_nodes = nodes.sort_values("battery_pct")
        mapping_series.append(
            {
                "x": nodes["battery_pct"],
                "y": nodes["linear_remaining_pct"],
                "color": colors[battery_id],
                "label": f"B{battery_id} nodes",
                "width": 0,
                "scatter": True,
                "radius": 4,
            }
        )
        mapping_series.append(
            {
                "x": sort_nodes["battery_pct"],
                "y": sort_nodes["linearized_fit_pct"],
                "color": colors[battery_id],
                "label": f"B{battery_id} mapping",
                "width": 5,
            }
        )
    draw_line_plot(
        draw,
        (1230, 105, 2345, 775),
        mapping_series,
        "Raw % to Time-Linear Remaining %",
        "Reported battery (%)",
        "Linearized remaining (%)",
        ylim=(0, 105),
    )

    converted_series = []
    for battery_id, result in results.items():
        df = result["df"].iloc[::8]
        converted_series.append(
            {
                "x": df["elapsed_min"],
                "y": df["linearized_fit_pct"],
                "color": colors[battery_id],
                "label": f"B{battery_id} converted",
                "width": 3,
            }
        )
        converted_series.append(
            {
                "x": df["elapsed_min"],
                "y": df["linear_remaining_pct"],
                "color": "#555555",
                "label": f"B{battery_id} ideal",
                "width": 2,
            }
        )
    draw_line_plot(
        draw,
        (55, 845, 1170, 1515),
        converted_series,
        "Discharge After Linearization",
        "Elapsed time (min)",
        "Linearized remaining (%)",
        ylim=(0, 105),
    )

    residual_series = []
    for battery_id, result in results.items():
        nodes = result["nodes"].sort_values("battery_pct")
        residual_series.append(
            {
                "x": nodes["battery_pct"],
                "y": nodes["linearization_residual_pct"],
                "color": colors[battery_id],
                "label": f"B{battery_id}",
                "width": 3,
                "scatter": True,
                "radius": 3,
            }
        )
    draw_line_plot(
        draw,
        (1230, 845, 2345, 1515),
        residual_series,
        "Linearization Residual",
        "Reported battery (%)",
        "Residual (pct pts)",
        hline=0,
    )
    image.save(OUT_DIR / "battery_all_discharge_linearization.png")

    raw = Image.new("RGB", (1600, 950), (245, 245, 240))
    raw_draw = ImageDraw.Draw(raw)
    raw_only_series = []
    for battery_id, result in results.items():
        df = result["df"].iloc[::5]
        raw_only_series.append(
            {
                "x": df["elapsed_min"],
                "y": df["battery_pct"],
                "color": colors[battery_id],
                "label": f"Battery {battery_id}",
                "width": 3,
            }
        )
    draw_line_plot(
        raw_draw,
        (50, 50, 1550, 900),
        raw_only_series,
        "Battery Reported Discharge Comparison",
        "Elapsed time (min)",
        "Reported battery (%)",
        ylim=(0, 100),
    )
    raw.save(OUT_DIR / "battery_all_raw_discharge.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    summaries = []
    skipped = []
    converted_frames = []

    batteries = discover_battery_logs()
    if not batteries:
        raise FileNotFoundError(f"No battery CSV files found in {LOG_DIR}")

    for battery_id, path in batteries.items():
        try:
            df, nodes, summary = analyze_one(battery_id, path)
        except ValueError as exc:
            short_df = load_curve(path)
            skipped.append(
                {
                    "battery_id": battery_id,
                    "source_file": str(path.relative_to(BASE_DIR)),
                    "samples": int(len(short_df)),
                    "duration_s": float(short_df["elapsed_s"].iloc[-1]),
                    "duration_min": float(short_df["elapsed_s"].iloc[-1] / 60.0),
                    "start_battery_pct": int(short_df["battery_pct"].iloc[0]),
                    "end_battery_pct": int(short_df["battery_pct"].iloc[-1]),
                    "reason": str(exc),
                }
            )
            continue
        results[battery_id] = {"df": df, "nodes": nodes, "summary": summary}
        summaries.append(summary)
        converted_frames.append(df)
        nodes.to_csv(OUT_DIR / f"battery_{battery_id}_linearization_nodes.csv", index=False)

    if not results:
        raise ValueError("No logs were long enough for full-curve linearization")

    converted = pd.concat(converted_frames, ignore_index=True)
    converted.to_csv(OUT_DIR / "battery_all_linearized_samples.csv", index=False)
    make_plots(results)

    with (OUT_DIR / "fit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "included": summaries,
                "skipped": skipped,
                "settings": {
                    "min_linearization_soc_span_pct": MIN_LINEARIZATION_SOC_SPAN_PCT,
                    "min_end_battery_pct": MIN_END_BATTERY_PCT,
                },
            },
            f,
            indent=2,
        )

    print(json.dumps({"included": summaries, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
