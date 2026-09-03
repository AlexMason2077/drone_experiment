"""Plot title-free DJI Tello hover battery curves in the 75%--40% range."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_hover_battery_charts import (  # noqa: E402
    SELECTED_MEAN_BATTERIES,
    cleaning_reason,
    find_hover_timeseries,
    load_hover_timeseries,
    mean_trace_for_battery,
)


INPUT_DIR = ROOT / "database_csv" / "baselines"
OUTPUT_PATH = ROOT / "output_graph" / "hover_baseline_battery_discharge_75_40.png"
COMPARISON_OUTPUT_PATH = ROOT / "output_graph" / "hover_baseline_B10_B15_raw_and_linear_75_40.png"
FIT_ONLY_OUTPUT_PATH = ROOT / "output_graph" / "hover_baseline_B10_B15_linear_fit_75_40.png"


def clipped_segment(points, upper=75.0, lower=40.0):
    """Interpolate a monotone discharge trace at its upper/lower crossings."""
    xs = np.asarray([float(point["t"]) for point in points], dtype=float)
    ys = np.asarray([float(point["battery"]) for point in points], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    def crossing(level):
        candidates = np.where((ys[:-1] >= level) & (ys[1:] <= level))[0]
        if not len(candidates):
            return None
        i = int(candidates[0])
        if ys[i + 1] == ys[i]:
            return xs[i + 1]
        return xs[i] + (level - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])

    t_upper, t_lower = crossing(upper), crossing(lower)
    if t_upper is None or t_lower is None or t_lower <= t_upper:
        return None
    mask = (xs > t_upper) & (xs < t_lower)
    x = np.concatenate(([t_upper], xs[mask], [t_lower])) - t_upper
    y = np.concatenate(([upper], ys[mask], [lower]))
    return x / 60.0, y


def main():
    traces = []
    for path in find_hover_timeseries(INPUT_DIR):
        trace, _ = load_hover_timeseries(path, max_points=2000)
        if trace is not None and not cleaning_reason(trace):
            traces.append(trace)

    colors = {
        "B10": "#0072B2", "B11": "#E69F00", "B12": "#D55E00",
        "B13": "#6B7F2A", "B14": "#CC79A7", "B15": "#6F63A8",
    }
    mean_curves = {}
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    plotted = 0
    for battery_id in SELECTED_MEAN_BATTERIES:
        group = [trace for trace in traces if trace["batteryId"] == battery_id]
        mean_trace = mean_trace_for_battery(battery_id, group, max_points=2000)
        if mean_trace is None:
            continue
        mean_curves[battery_id] = mean_trace
        segment = clipped_segment(mean_trace["points"])
        if segment is None:
            continue
        x, y = segment
        coeff = np.polyfit(x, y, 1)
        fitted = np.polyval(coeff, x)
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
        ax.plot(x, y, color=colors[battery_id], linewidth=1.7,
                label=f"{battery_id} ($R^2$={r_squared:.3f})")
        plotted += 1

    if not plotted:
        raise RuntimeError("No complete hover baseline curves covered the 75%--40% range")

    ax.set_xlabel("Hover time from 75% battery (min)")
    ax.set_ylabel("Remaining battery (%)")
    ax.set_ylim(39, 76)
    ax.set_xlim(left=0)
    ax.set_yticks(np.arange(40, 76, 5))
    ax.grid(True, color="#D9DEE3", linewidth=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.2, ncol=2, loc="upper right")
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Reference-style two-panel figure: complete curves on the left and
    # full curves plus linear fits within the selected window on the right.
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=300, sharey=True)
    fit_metrics = []
    for battery_id in SELECTED_MEAN_BATTERIES:
        mean_trace = mean_curves.get(battery_id)
        if mean_trace is None:
            continue
        points = mean_trace["points"]
        full_x = np.asarray([float(point["t"]) / 60.0 for point in points])
        full_y = np.asarray([float(point["battery"]) for point in points])
        segment = clipped_segment(points)
        if segment is None:
            continue
        x, y = segment
        coeff = np.polyfit(x, y, 1)
        fitted = np.polyval(coeff, x)
        t_at_75 = float(np.interp(75.0, full_y[::-1], full_x[::-1]))
        fit_x_absolute = x + t_at_75
        residuals = y - fitted
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
        fit_metrics.append((battery_id, r_squared, rmse, abs(float(coeff[0]))))

        axes[0].plot(full_x, full_y, color=colors[battery_id], linewidth=1.45,
                     label=battery_id)
        axes[1].plot(full_x, full_y, color=colors[battery_id], linewidth=1.0,
                     alpha=0.35, label=battery_id)
        axes[1].plot(fit_x_absolute, fitted, color=colors[battery_id], linewidth=1.8,
                     linestyle="--")

    for axis in axes:
        axis.axhspan(40, 75, color="#DDEEDD", alpha=0.65, zorder=0)
        axis.set_xlabel("Elapsed hover time (min)")
        axis.set_xlim(left=0)
        axis.set_ylim(5, 101)
        axis.set_yticks(np.arange(10, 101, 10))
        axis.grid(True, color="#D9DEE3", linestyle="--", linewidth=0.55, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=True, framealpha=0.82, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Remaining battery (%)")

    if fit_metrics:
        r2_values = np.asarray([item[1] for item in fit_metrics])
        rmse_values = np.asarray([item[2] for item in fit_metrics])
        rates = np.asarray([item[3] for item in fit_metrics])
        rate_cv = float(np.std(rates, ddof=1) / np.mean(rates)) if len(rates) > 1 else 0.0
        summary = (
            f"75–40% window: min $R^2$={r2_values.min():.3f}, "
            f"max RMSE={rmse_values.max():.2f} pp, rate CV={rate_cv:.3f}"
        )
        axes[1].text(
            0.02, 0.025, summary, transform=axes[1].transAxes,
            fontsize=8.4, ha="left", va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#BFC7CE", "alpha": 0.9, "pad": 4},
        )

    fig.tight_layout(w_pad=2.4)
    fig.savefig(COMPARISON_OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Publication version: the fitted panel only, with statistics left to the caption/text.
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=300)
    for battery_id in SELECTED_MEAN_BATTERIES:
        mean_trace = mean_curves.get(battery_id)
        if mean_trace is None:
            continue
        points = mean_trace["points"]
        full_x = np.asarray([float(point["t"]) / 60.0 for point in points])
        full_y = np.asarray([float(point["battery"]) for point in points])
        segment = clipped_segment(points)
        if segment is None:
            continue
        x, y = segment
        fitted = np.polyval(np.polyfit(x, y, 1), x)
        t_at_75 = float(np.interp(75.0, full_y[::-1], full_x[::-1]))
        fit_x_absolute = x + t_at_75

        ax.plot(
            full_x, full_y, color=colors[battery_id], linewidth=0.95,
            alpha=0.44, label=battery_id,
        )
        ax.plot(
            fit_x_absolute, fitted, color=colors[battery_id], linewidth=2.35,
            linestyle="--",
        )

    ax.axhspan(40, 75, color="#DDEEDD", alpha=0.68, zorder=0)
    ax.set_xlabel("Elapsed hover time (min)")
    ax.set_ylabel("Reported battery level (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(5, 101)
    ax.set_yticks(np.arange(10, 101, 10))
    ax.grid(True, color="#D9DEE3", linestyle="--", linewidth=0.55, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, framealpha=0.82, fontsize=8.5, loc="upper right")
    ax.text(
        0.025, 0.035, "Linear fitting range: 75%–40%",
        transform=ax.transAxes, fontsize=9.2, ha="left", va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#BFC7CE", "alpha": 0.92, "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(FIT_ONLY_OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(FIT_ONLY_OUTPUT_PATH)


if __name__ == "__main__":
    main()
