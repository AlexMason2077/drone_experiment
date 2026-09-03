"""Plot formation wind comparisons after removing mission-pad wait energy.

The source table estimates active movement segment-by-segment from the 10%--90%
trajectory crossing. Battery use during the remaining stationary time is estimated
from the matching battery/SOC hover rate and removed from the reported node drop.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "swarm_analysis" / "pure_forward" / "pure_forward_drone_rows.csv"
WINDS = ["head", "tail", "side"]
LABELS = {"head": "Head wind", "tail": "Tail wind", "side": "Side wind"}
COLORS = {"head": "#2878B5", "tail": "#5B8C3A", "side": "#D9911B"}


def load_rows(level: int, formation: str) -> pd.DataFrame:
    rows = pd.read_csv(SOURCE, low_memory=False)
    rows = rows[
        (rows["formation"] == formation)
        & (rows["distance"] == 75)
        & (rows["wind_level"] == level)
        & (rows["wind_direction"].isin(WINDS))
    ].copy()
    rows["position"] = rows["position"].astype(int)
    if rows.empty:
        raise RuntimeError(f"No {formation}/75 cm/wind level {level} rows found")
    run_sizes = rows.groupby(["experiment_id", "run_id"])["drone_name"].nunique()
    if not (run_sizes == 5).all():
        raise RuntimeError("At least one selected run does not contain all five drones")
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (position, wind), group in rows.groupby(["position", "wind_direction"]):
        energy = group["pure_forward_drop_250cm"].to_numpy(float)
        q1, median, q3 = np.percentile(energy, [25, 50, 75])
        records.append(
            {
                "position": int(position),
                "battery_id": "/".join(sorted(group["battery_id"].dropna().astype(str).unique())),
                "wind_direction": wind,
                "n_runs": int(group["run_id"].nunique()),
                "median_motion_only_battery_consumption_pct_points": median,
                "q1_motion_only_battery_consumption_pct_points": q1,
                "q3_motion_only_battery_consumption_pct_points": q3,
                "mean_motion_only_battery_consumption_pct_points": float(np.mean(energy)),
                "median_stationary_wait_removed_sec": float(group["stationary_wait_sec"].median()),
                "median_estimated_wait_consumption_removed_pct_points": float(
                    group["estimated_wait_hover_drop"].median()
                ),
            }
        )
    summary = pd.DataFrame(records)
    summary["wind_order"] = summary["wind_direction"].map(
        {wind: index for index, wind in enumerate(WINDS)}
    )
    return summary.sort_values(["position", "wind_order"]).drop(columns="wind_order")


def plot(summary: pd.DataFrame, level: int, formation: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.7), dpi=260)
    positions = np.arange(1, 6, dtype=float)
    width = 0.24
    offsets = {"head": -width, "tail": 0.0, "side": width}

    for wind in WINDS:
        subset = (
            summary[summary["wind_direction"] == wind]
            .set_index("position")
            .reindex(range(1, 6))
        )
        medians = subset["median_motion_only_battery_consumption_pct_points"].to_numpy(float)
        q1 = subset["q1_motion_only_battery_consumption_pct_points"].to_numpy(float)
        q3 = subset["q3_motion_only_battery_consumption_pct_points"].to_numpy(float)
        bars = ax.bar(
            positions + offsets[wind],
            medians,
            width=width * 0.92,
            color=COLORS[wind],
            alpha=0.9,
            label=LABELS[wind],
            yerr=np.vstack([medians - q1, q3 - medians]),
            capsize=3.5,
            error_kw={"elinewidth": 1.25, "capthick": 1.25, "ecolor": COLORS[wind]},
        )
        for bar, value in zip(bars, medians):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.22,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
                color="#333333",
            )

    batteries = (
        summary.groupby("position")["battery_id"]
        .agg(lambda values: "/".join(sorted({bid for value in values for bid in str(value).split("/")})))
        .reindex(range(1, 6))
    )
    ax.set_xticks(
        positions,
        [f"Drone {position}\n({batteries.loc[position]})" for position in range(1, 6)],
    )
    ax.set_xlabel(f"Position in {formation} formation")
    ax.set_ylabel("Battery consumption excluding pad waiting\n(% points per 250 cm)")
    ymin = min(-1.5, summary["q1_motion_only_battery_consumption_pct_points"].min() - 0.8)
    ymax = max(13.5, summary["q3_motion_only_battery_consumption_pct_points"].max() + 1.8)
    ax.set_ylim(ymin, ymax)
    ax.axhline(0, color="#60666B", linewidth=0.9)
    ax.grid(axis="y", color="#DDE2E6", linewidth=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left", title="Wind direction")
    ax.set_title(
        f"{formation.capitalize()} formation · 75 cm spacing · wind level {level}",
        loc="left",
        weight="bold",
        pad=12,
    )
    fig.text(
        0.125,
        0.015,
        "Bars show medians; error bars show IQR. Mission-pad waiting/calibration hover use is removed.",
        fontsize=8.8,
        color="#59636E",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_summary(rows: pd.DataFrame) -> pd.DataFrame:
    per_run = (
        rows.groupby(["experiment_id", "run_id", "wind_direction"], as_index=False)
        .agg(
            drone_count=("drone_name", "nunique"),
            mean_motion_only_battery_consumption_pct_points=("pure_forward_drop_250cm", "mean"),
            median_motion_only_battery_consumption_pct_points=("pure_forward_drop_250cm", "median"),
            median_stationary_wait_removed_sec=("stationary_wait_sec", "median"),
            median_estimated_wait_consumption_removed_pct_points=("estimated_wait_hover_drop", "median"),
        )
    )
    per_run["wind_order"] = per_run["wind_direction"].map(
        {wind: index for index, wind in enumerate(WINDS)}
    )
    return per_run.sort_values(["wind_order", "experiment_id"]).drop(columns="wind_order")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1, 2], required=True)
    parser.add_argument(
        "--formation",
        choices=["front", "vee", "diamond", "echalon", "column"],
        default="vee",
    )
    args = parser.parse_args()

    output_dir = ROOT / "swarm_analysis" / "wind_direction" / f"{args.formation}_75_lv{args.level}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / f"{args.formation}_75cm_lv{args.level}_wind_motion_only_by_drone_bar.png"
    output_csv = output_dir / f"{args.formation}_75cm_lv{args.level}_wind_motion_only_by_drone_summary.csv"
    output_run_csv = output_dir / f"{args.formation}_75cm_lv{args.level}_wind_motion_only_run_summary.csv"

    rows = load_rows(args.level, args.formation)
    summary = summarize(rows)
    runs = run_summary(rows)
    summary.to_csv(output_csv, index=False)
    runs.to_csv(output_run_csv, index=False)
    plot(summary, args.level, args.formation, output_png)

    wind_summary = runs.groupby("wind_direction").agg(
        n_runs=("run_id", "nunique"),
        median_run_mean=("mean_motion_only_battery_consumption_pct_points", "median"),
        median_wait_removed_sec=("median_stationary_wait_removed_sec", "median"),
        median_wait_drop_removed=("median_estimated_wait_consumption_removed_pct_points", "median"),
    ).reindex(WINDS)
    print(wind_summary.round(3).to_string())
    print(output_png)


if __name__ == "__main__":
    main()
