"""Compare front-formation head-wind energy and time at 50 cm and 75 cm."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "swarm_analysis" / "algorithm_energy_knowledge_base"
OUT = KB / "charts" / "front_head_50_75_comparison"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {50: "#2878B5", 75: "#D9911B"}
LINESTYLES = {50: "-", 75: "--"}
MARKERS = {50: "o", 75: "s"}
POSITION_LABELS = {
    1: "Drone 1\nB11",
    2: "Drone 2\nB10",
    3: "Drone 3\nB13",
    4: "Drone 4\nB14",
    5: "Drone 5\nB15/B12",
}


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#DCE2E8", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def add_figure_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.07, y=0.985, ha="left", fontsize=19, fontweight="bold", color="#15191E")
    fig.text(0.07, 0.935, subtitle, ha="left", fontsize=10.5, color="#5D6772")


def grouped_bar(
    summary: pd.DataFrame,
    value: str,
    q25: str,
    q75: str,
    ylabel: str,
    title: str,
    subtitle: str,
    filename: str,
    baseline_by_position: dict[int, float] | None = None,
    baseline_label: str = "Single-drone baseline",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.7), sharey=True)
    x = np.arange(1, 6)
    width = 0.34
    for ax, level in zip(axes, [1, 2]):
        sub = summary[summary.wind_level == level]
        for idx, distance in enumerate([50, 75]):
            d = sub[sub.distance == distance].set_index("position").reindex(x)
            vals = d[value].to_numpy(float)
            lo = vals - d[q25].to_numpy(float)
            hi = d[q75].to_numpy(float) - vals
            offset = (-0.5 if idx == 0 else 0.5) * width
            ax.bar(
                x + offset, vals, width=width, color=COLORS[distance], alpha=0.88,
                edgecolor="#27313A", linewidth=0.8, label=f"{distance} cm",
                yerr=np.vstack([lo, hi]), capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#66717C"},
            )
        if baseline_by_position:
            base = np.array([baseline_by_position[i] for i in x])
            ax.plot(x, base, color="#22272C", marker="D", markersize=5, linewidth=1.6,
                    linestyle=":", label=baseline_label)
        ax.set_title(f"Head wind · level {level}", loc="left", fontsize=13, fontweight="bold")
        ax.set_xticks(x, [POSITION_LABELS[i] for i in x])
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.set_ylim(bottom=0)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.94, 0.955), frameon=False, ncol=3)
    add_figure_header(fig, title, subtitle)
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.88])
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def position_line(
    summary: pd.DataFrame,
    value: str,
    q25: str,
    q75: str,
    ylabel: str,
    title: str,
    subtitle: str,
    filename: str,
    baseline_by_position: dict[int, float] | None = None,
    baseline_label: str = "Single-drone baseline",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.7), sharey=True)
    x = np.arange(1, 6)
    for ax, level in zip(axes, [1, 2]):
        sub = summary[summary.wind_level == level]
        for distance in [50, 75]:
            d = sub[sub.distance == distance].set_index("position").reindex(x)
            y = d[value].to_numpy(float)
            low = d[q25].to_numpy(float)
            high = d[q75].to_numpy(float)
            ax.plot(
                x, y, color=COLORS[distance], linestyle=LINESTYLES[distance],
                marker=MARKERS[distance], markersize=7, linewidth=2.4,
                markerfacecolor=(COLORS[distance] if distance == 50 else "white"),
                markeredgewidth=1.6, label=f"{distance} cm",
            )
            ax.fill_between(x, low, high, color=COLORS[distance], alpha=0.12)
        if baseline_by_position:
            base = np.array([baseline_by_position[i] for i in x])
            ax.plot(x, base, color="#22272C", marker="D", markersize=4.5,
                    linewidth=1.5, linestyle=":", label=baseline_label)
        ax.set_title(f"Head wind · level {level}", loc="left", fontsize=13, fontweight="bold")
        ax.set_xticks(x, [POSITION_LABELS[i] for i in x])
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.set_ylim(bottom=0)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.94, 0.955), frameon=False, ncol=3)
    add_figure_header(fig, title, subtitle)
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.88])
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def total_bar(total: pd.DataFrame, total_baseline: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3))
    x = np.arange(2)
    width = 0.34
    for idx, distance in enumerate([50, 75]):
        d = total[total.distance == distance].set_index("wind_level").reindex([1, 2])
        offset = (-0.5 if idx == 0 else 0.5) * width
        energy = d.total_energy_median.to_numpy(float)
        axes[0].bar(
            x + offset, energy, width=width, color=COLORS[distance], alpha=0.88,
            edgecolor="#27313A", linewidth=0.8, label=f"{distance} cm",
            yerr=np.vstack([energy - d.total_energy_q25, d.total_energy_q75 - energy]), capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#66717C"},
        )
        time = d.completion_time_median.to_numpy(float)
        axes[1].bar(
            x + offset, time, width=width, color=COLORS[distance], alpha=0.88,
            edgecolor="#27313A", linewidth=0.8, label=f"{distance} cm",
            yerr=np.vstack([time - d.completion_time_q25, d.completion_time_q75 - time]), capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#66717C"},
        )
        for ax, values, counts in zip(axes, [energy, time], [d.run_count, d.time_run_count]):
            for xx, yy, n in zip(x + offset, values, counts):
                ax.text(xx, yy, f"n={int(n)}", ha="center", va="bottom", fontsize=8, color="#4D5862")
    axes[0].axhline(total_baseline, color="#22272C", linestyle=":", linewidth=1.7)
    axes[1].axhline(25, color="#22272C", linestyle=":", linewidth=1.7)
    axes[0].text(0.99, total_baseline, "5-drone baseline", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8.5, color="#30363C")
    axes[1].text(0.99, 25, "25 s speed reference", transform=axes[1].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8.5, color="#30363C")
    axes[0].set_title("Five-drone total energy", loc="left", fontsize=13, fontweight="bold")
    axes[1].set_title("Swarm forward-completion time", loc="left", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Equivalent hover seconds / 250 cm")
    axes[1].set_ylabel("Seconds / 250 cm")
    for ax in axes:
        ax.set_xticks(x, ["Wind level 1", "Wind level 2"])
        ax.set_ylim(bottom=0)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.95, 0.95), frameon=False, ncol=3)
    add_figure_header(
        fig, "Front formation · head wind: total energy and time",
        "Medians with run-level IQR; 75 cm missions normalized to the first 250 cm",
    )
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.86])
    fig.savefig(OUT / "01_total_energy_time_bar.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def total_slope(total: pd.DataFrame, total_baseline: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3))
    for level, color, marker in [(1, "#2878B5", "o"), (2, "#D9911B", "s")]:
        d = total[total.wind_level == level].sort_values("distance")
        axes[0].plot(d.distance, d.total_energy_median, color=color, marker=marker,
                     linewidth=2.5, markersize=8, label=f"Wind level {level}")
        axes[1].plot(d.distance, d.completion_time_median, color=color, marker=marker,
                     linewidth=2.5, markersize=8, label=f"Wind level {level}")
        for ax, field in [(axes[0], "total_energy_median"), (axes[1], "completion_time_median")]:
            for row in d.itertuples(index=False):
                ax.annotate(f"{getattr(row, field):.1f}", (row.distance, getattr(row, field)),
                            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    axes[0].axhline(total_baseline, color="#22272C", linestyle=":", linewidth=1.7)
    axes[1].axhline(25, color="#22272C", linestyle=":", linewidth=1.7)
    axes[0].text(0.99, total_baseline, "5-drone baseline", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8.5, color="#30363C")
    axes[1].text(0.99, 25, "25 s speed reference", transform=axes[1].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8.5, color="#30363C")
    axes[0].set_title("Five-drone total energy", loc="left", fontsize=13, fontweight="bold")
    axes[1].set_title("Swarm forward-completion time", loc="left", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Equivalent hover seconds / 250 cm")
    axes[1].set_ylabel("Seconds / 250 cm")
    for ax in axes:
        ax.set_xticks([50, 75], ["50 cm", "75 cm"])
        ax.set_xlim(45, 80)
        ax.set_ylim(bottom=0)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.95, 0.95), frameon=False, ncol=3)
    add_figure_header(
        fig, "Front formation · head wind: 50–75 cm slope comparison",
        "The line is a two-point spacing comparison, not a continuous-distance model",
    )
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.86])
    fig.savefig(OUT / "02_total_energy_time_slope.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = pd.read_csv(KB / "swarm_drone_energy_rows.csv")
    configs = pd.read_csv(KB / "configuration_energy_knowledge_base.csv")
    baselines = pd.read_csv(KB / "single_drone_baseline_energy_models.csv")
    runs = pd.read_csv(KB / "swarm_run_energy_profiles.csv")

    filt = (
        (rows.formation == "front")
        & (rows.wind_direction_short == "head")
        & rows.distance.isin([50, 75])
        & rows.wind_level.isin([1, 2])
    )
    front = rows.loc[filt].copy()
    valid_map = runs[["experiment_id", "csv_run_id", "physically_valid_total_energy"]].drop_duplicates()
    front = front.merge(valid_map, on=["experiment_id", "csv_run_id"], how="left", validate="many_to_one")
    front["distance_scale"] = 250 / front.commanded_distance_cm
    front["forward_time_250cm"] = front.active_forward_sec_adjusted * front.distance_scale

    def q25(s: pd.Series) -> float:
        return float(s.quantile(0.25))

    def q75(s: pd.Series) -> float:
        return float(s.quantile(0.75))

    energy_drone = (
        front[front.physically_valid_total_energy]
        .groupby(["distance", "wind_level", "position", "effective_battery_id"], as_index=False)
        .agg(
            energy_median=("pure_forward_energy_250cm", "median"),
            energy_q25=("pure_forward_energy_250cm", q25),
            energy_q75=("pure_forward_energy_250cm", q75),
            energy_observation_count=("csv_run_id", "size"),
            energy_run_count=("csv_run_id", "nunique"),
        )
    )
    time_drone = (
        front.groupby(["distance", "wind_level", "position", "effective_battery_id"], as_index=False)
        .agg(
            forward_time_median=("forward_time_250cm", "median"),
            forward_time_q25=("forward_time_250cm", q25),
            forward_time_q75=("forward_time_250cm", q75),
            start_soc_median=("csv_battery_hover_start", "median"),
            time_observation_count=("csv_run_id", "size"),
            time_run_count=("csv_run_id", "nunique"),
            motion_floor_share=("motion_floor_applied", "mean"),
        )
    )
    drone = energy_drone.merge(
        time_drone, on=["distance", "wind_level", "position", "effective_battery_id"],
        how="outer", validate="one_to_one",
    )

    per_run_time = (
        front.groupby(["experiment_id", "csv_run_id", "distance", "wind_level"], as_index=False)
        .agg(completion_time_250cm=("forward_time_250cm", "max"))
    )
    time_summary = (
        per_run_time.groupby(["distance", "wind_level"], as_index=False)
        .agg(
            completion_time_median=("completion_time_250cm", "median"),
            completion_time_q25=("completion_time_250cm", q25),
            completion_time_q75=("completion_time_250cm", q75),
            time_run_count=("csv_run_id", "size"),
        )
    )
    total = configs[
        (configs.formation == "front")
        & (configs.wind_direction_short == "head")
        & configs.distance.isin([50, 75])
        & configs.wind_level.isin([1, 2])
    ][[
        "distance", "wind_level", "total_energy_median", "total_energy_q25", "total_energy_q75",
        "mean_drone_energy_median", "relative_vs_baseline_median_pct", "run_count", "evidence_strength",
    ]].merge(time_summary, on=["distance", "wind_level"], how="left")

    baseline_by_battery = baselines.set_index("battery_id").baseline_energy_median.to_dict()
    battery_by_position = {1: "B11", 2: "B10", 3: "B13", 4: "B14", 5: "B15"}
    baseline_by_position = {p: float(baseline_by_battery[b]) for p, b in battery_by_position.items()}
    total_baseline = sum(baseline_by_position.values())

    total.to_csv(OUT / "front_head_total_summary.csv", index=False)
    drone.to_csv(OUT / "front_head_per_drone_summary.csv", index=False)
    per_run_time.to_csv(OUT / "front_head_run_completion_times.csv", index=False)

    total_bar(total, total_baseline)
    total_slope(total, total_baseline)
    grouped_bar(
        drone, "energy_median", "energy_q25", "energy_q75",
        "Equivalent hover seconds / 250 cm",
        "Front formation · head wind: per-drone energy (bar)",
        "Median and IQR by assigned position/battery; dotted diamonds show matched single-drone baselines",
        "03_per_drone_energy_bar.png", baseline_by_position,
    )
    position_line(
        drone, "energy_median", "energy_q25", "energy_q75",
        "Equivalent hover seconds / 250 cm",
        "Front formation · head wind: per-drone energy profile (line)",
        "Position is an ordered spatial profile; shaded bands show run-level IQR",
        "04_per_drone_energy_line.png", baseline_by_position,
    )
    grouped_bar(
        drone, "forward_time_median", "forward_time_q25", "forward_time_q75",
        "Active forward seconds / 250 cm",
        "Front formation · head wind: per-drone forward time (bar)",
        "75 cm missions normalized to 250 cm; 25 s is the commanded-speed physical reference",
        "05_per_drone_time_bar.png", {i: 25.0 for i in range(1, 6)}, "25 s speed reference",
    )
    position_line(
        drone, "forward_time_median", "forward_time_q25", "forward_time_q75",
        "Active forward seconds / 250 cm",
        "Front formation · head wind: per-drone forward-time profile (line)",
        "Position is an ordered spatial profile; shaded bands show run-level IQR",
        "06_per_drone_time_line.png", {i: 25.0 for i in range(1, 6)}, "25 s speed reference",
    )

    print(f"rows={len(front)} runs={front.csv_run_id.nunique()} charts=6")
    print(total.sort_values(["wind_level", "distance"]).to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
