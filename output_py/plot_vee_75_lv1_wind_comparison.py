"""Compare battery use by wind direction for vee, 75 cm, wind level 1."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEVEL = 1
OUT = ROOT / "swarm_analysis" / "wind_direction" / f"vee_75_lv{LEVEL}"
OUTPUT_PNG = OUT / f"vee_75cm_lv{LEVEL}_wind_battery_comparison.png"
OUTPUT_CSV = OUT / f"vee_75cm_lv{LEVEL}_wind_run_summary.csv"
BAR_OUTPUT_PNG = OUT / f"vee_75cm_lv{LEVEL}_wind_battery_by_drone_bar.png"
BAR_OUTPUT_CSV = OUT / f"vee_75cm_lv{LEVEL}_wind_battery_by_drone_summary.csv"
WINDS = ["head", "tail", "side"]
LABELS = {"head": "Head wind", "tail": "Tail wind", "side": "Side wind"}
COLORS = {"head": "#2878B5", "tail": "#5B8C3A", "side": "#D9911B"}


def load_run_summary() -> pd.DataFrame:
    source = ROOT / "swarm_analysis" / "multidimensional" / "swarm_rows_with_baseline_adjustment.csv"
    rows = pd.read_csv(source, low_memory=False)
    rows = rows[
        (rows["formation"] == "vee")
        & (rows["distance"] == 75)
        & (rows["wind_level"] == LEVEL)
        & (rows["wind_direction_short"].isin(WINDS))
    ].copy()
    summary = (
        rows.groupby(["wind_direction_short", "experiment_id", "csv_run_id"], as_index=False)
        .agg(
            drone_count=("csv_drone_name", "nunique"),
            duration_sec=("csv_node_duration_sec", "first"),
            mean_start_battery=("csv_battery_hover_start", "mean"),
            total_reported_drop=("csv_battery_drop", "sum"),
            mean_reported_drop_per_drone=("csv_battery_drop", "mean"),
            total_expected_baseline_drop=("baseline_expected_drop", "sum"),
            total_excess_vs_baseline=("excess_vs_baseline", "sum"),
            mean_excess_vs_baseline_per_drone=("excess_vs_baseline", "mean"),
        )
    )
    if not (summary["drone_count"] == 5).all():
        raise RuntimeError("At least one selected run does not contain five drones")
    summary["wind_order"] = summary["wind_direction_short"].map({wind: i for i, wind in enumerate(WINDS)})
    return summary.sort_values(["wind_order", "csv_run_id"]).drop(columns="wind_order")


def load_drone_rows() -> pd.DataFrame:
    source = ROOT / "swarm_analysis" / "multidimensional" / "swarm_rows_with_baseline_adjustment.csv"
    rows = pd.read_csv(source, low_memory=False)
    rows = rows[
        (rows["formation"] == "vee")
        & (rows["distance"] == 75)
        & (rows["wind_level"] == LEVEL)
        & (rows["wind_direction_short"].isin(WINDS))
    ].copy()
    rows["position"] = rows["position"].astype(int)
    return rows


def draw_group(ax, values, index, color):
    values = np.asarray(values, dtype=float)
    offsets = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.array([0.0])
    ax.scatter(
        index + offsets, values, s=52, color=color, alpha=0.75,
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    ax.vlines(index, q1, q3, color=color, linewidth=7, alpha=0.23, zorder=2)
    ax.hlines(median, index - 0.20, index + 0.20, color=color, linewidth=3.2, zorder=4)


def plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), dpi=260)
    metrics = [
        ("mean_reported_drop_per_drone", "Reported battery consumption\n(% points per drone)"),
        ("mean_excess_vs_baseline_per_drone", "SOC-adjusted excess battery consumption\n(% points per drone)"),
    ]
    tick_labels = []
    for wind in WINDS:
        n_runs = summary.loc[summary["wind_direction_short"] == wind, "csv_run_id"].nunique()
        tick_labels.append(f"{LABELS[wind]}\n(n={n_runs} runs)")

    for ax, (metric, ylabel) in zip(axes, metrics):
        for index, wind in enumerate(WINDS):
            values = summary.loc[summary["wind_direction_short"] == wind, metric]
            draw_group(ax, values, index, COLORS[wind])
        ax.set_xticks(range(len(WINDS)), tick_labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDE2E6", linewidth=0.75)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].axhline(0, color="#4A4A4A", linestyle="--", linewidth=1.1)
    fig.suptitle(
        f"Vee formation · 75 cm spacing · wind level {LEVEL}",
        x=0.075, y=0.985, ha="left", fontsize=14, weight="bold",
    )
    fig.text(
        0.075, 0.925,
        "Each point is one five-drone run; horizontal bars show medians and shaded bars show IQR",
        color="#59636E", fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89], w_pad=2.8)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def drone_bar_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (position, wind), group in rows.groupby(["position", "wind_direction_short"]):
        q1, median, q3 = np.percentile(group["csv_battery_drop"].to_numpy(float), [25, 50, 75])
        records.append({
            "position": int(position),
            "battery_id": group["csv_battery_id"].iloc[0],
            "wind_direction": wind,
            "n_runs": int(group["csv_run_id"].nunique()),
            "median_reported_battery_consumption": median,
            "q1_reported_battery_consumption": q1,
            "q3_reported_battery_consumption": q3,
            "mean_reported_battery_consumption": float(group["csv_battery_drop"].mean()),
        })
    summary = pd.DataFrame(records)
    summary["wind_order"] = summary["wind_direction"].map({wind: i for i, wind in enumerate(WINDS)})
    return summary.sort_values(["position", "wind_order"]).drop(columns="wind_order")


def plot_drone_bars(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=260)
    positions = np.arange(1, 6, dtype=float)
    width = 0.24
    offsets = {"head": -width, "tail": 0.0, "side": width}
    for wind in WINDS:
        subset = summary[summary["wind_direction"] == wind].set_index("position").reindex(range(1, 6))
        medians = subset["median_reported_battery_consumption"].to_numpy(float)
        lower = medians - subset["q1_reported_battery_consumption"].to_numpy(float)
        upper = subset["q3_reported_battery_consumption"].to_numpy(float) - medians
        bars = ax.bar(
            positions + offsets[wind], medians, width=width * 0.92,
            color=COLORS[wind], alpha=0.88, label=LABELS[wind],
            yerr=np.vstack([lower, upper]), capsize=3.5,
            error_kw={"elinewidth": 1.25, "capthick": 1.25, "ecolor": COLORS[wind]},
        )
        for bar, value in zip(bars, medians):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.22,
                f"{value:.1f}", ha="center", va="bottom", fontsize=8.3,
                color="#333333",
            )
    battery_by_position = (
        summary.groupby("position")["battery_id"].first().reindex(range(1, 6))
    )
    ax.set_xticks(
        positions,
        [f"Drone {position}\n({battery_by_position.loc[position]})" for position in range(1, 6)],
    )
    ax.set_xlabel("Position in vee formation")
    ax.set_ylabel("Reported battery consumption (% points)")
    ax.set_ylim(0, max(summary["q3_reported_battery_consumption"].max() + 2.0, 14))
    ax.grid(axis="y", color="#DDE2E6", linewidth=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left", title="Wind direction")
    ax.set_title(f"Vee formation · 75 cm spacing · wind level {LEVEL}", loc="left", weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(BAR_OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    global LEVEL, OUT, OUTPUT_PNG, OUTPUT_CSV, BAR_OUTPUT_PNG, BAR_OUTPUT_CSV
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    LEVEL = args.level
    OUT = ROOT / "swarm_analysis" / "wind_direction" / f"vee_75_lv{LEVEL}"
    OUTPUT_PNG = OUT / f"vee_75cm_lv{LEVEL}_wind_battery_comparison.png"
    OUTPUT_CSV = OUT / f"vee_75cm_lv{LEVEL}_wind_run_summary.csv"
    BAR_OUTPUT_PNG = OUT / f"vee_75cm_lv{LEVEL}_wind_battery_by_drone_bar.png"
    BAR_OUTPUT_CSV = OUT / f"vee_75cm_lv{LEVEL}_wind_battery_by_drone_summary.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_run_summary()
    drone_rows = load_drone_rows()
    bar_summary = drone_bar_summary(drone_rows)
    summary.to_csv(OUTPUT_CSV, index=False)
    bar_summary.to_csv(BAR_OUTPUT_CSV, index=False)
    plot(summary)
    plot_drone_bars(bar_summary)
    medians = summary.groupby("wind_direction_short")[[
        "mean_reported_drop_per_drone", "mean_excess_vs_baseline_per_drone"
    ]].median().reindex(WINDS)
    print(summary.groupby("wind_direction_short")["csv_run_id"].nunique().reindex(WINDS).to_string())
    print(medians.round(3).to_string())
    print(BAR_OUTPUT_PNG)


if __name__ == "__main__":
    main()
