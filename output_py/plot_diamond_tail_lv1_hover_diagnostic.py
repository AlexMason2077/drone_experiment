"""Plot one-run battery and motion-state diagnostics for user review."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY = (
    ROOT
    / "db_copy_for_cleaning"
    / "_cleaning_admin"
    / "trajectory_qc"
    / "trajectory_drone_segments.csv"
)
FORWARD = (
    ROOT
    / "db_copy_for_cleaning"
    / "_cleaning_admin"
    / "trajectory_qc"
    / "forward_motion_drone_segments.csv"
)
OUTPUT_DIR = ROOT / "analysis_outputs" / "run_diagnostics"
EXPERIMENT = "diamond_75_tail_lv1_new_001"
RUN_ID = "20260621_152905"


def main() -> None:
    trajectory = pd.read_csv(TRAJECTORY, dtype={"run_id": "string"}, low_memory=False)
    forward = pd.read_csv(FORWARD, dtype={"run_id": "string"}, low_memory=False)
    keys = [
        "experiment_directory",
        "run_id",
        "drone_name",
        "formation",
        "inter_drone_spacing_cm",
        "wind_direction",
        "wind_level",
        "battery_id",
    ]
    data = trajectory.merge(forward, on=keys, how="inner", suffixes=("", "_forward"))
    data = data[
        data["experiment_directory"].eq(EXPERIMENT)
        & data["run_id"].eq(RUN_ID)
    ].copy()
    data["drone_number"] = pd.to_numeric(
        data["drone_name"].str.extract(r"(\d+)")[0], errors="raise"
    )
    data = data.sort_values("drone_number").reset_index(drop=True)
    if len(data) != 5:
        raise RuntimeError(f"Expected five drone rows, found {len(data)}")

    export_columns = [
        "experiment_directory",
        "run_id",
        "drone_name",
        "battery_id",
        "battery_at_motion_start_pct",
        "battery_at_250cm_end_pct",
        "reported_battery_drop_pct_points",
        "reported_drop_during_forward_events_pp",
        "reported_drop_during_nonforward_events_pp",
        "selected_window_sec",
        "forward_movement_sec",
        "in_flight_nonforward_sec",
        "detected_forward_distance_cm",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{EXPERIMENT}_{RUN_ID}_hover_comparison.csv"
    data[export_columns].to_csv(csv_path, index=False)

    labels = [f"Drone {int(number)}" for number in data["drone_number"]]
    x = np.arange(len(labels))
    width = 0.34
    total_drop = data["reported_battery_drop_pct_points"].to_numpy(float)
    forward_event_drop = data["reported_drop_during_forward_events_pp"].to_numpy(float)
    forward_sec = data["forward_movement_sec"].to_numpy(float)
    nonforward_sec = data["in_flight_nonforward_sec"].to_numpy(float)

    blue = "#2F6F9F"
    orange = "#D98B3A"
    grey = "#AAB4BE"
    ink = "#243447"
    grid = "#DDE3E8"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.6),
        dpi=220,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.42},
    )
    fig.suptitle(
        "Battery-drop diagnostic: full window vs forward-motion events",
        fontsize=13.5,
        fontweight="semibold",
        color=ink,
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        "Diamond · 75 cm · Tail wind Level 1 · Run 20260621_152905",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#5A6978",
    )

    ax = axes[0]
    bars_total = ax.bar(
        x - width / 2,
        total_drop,
        width,
        color=blue,
        edgecolor="#214F72",
        linewidth=0.7,
        label="Recorded drop over full selected window",
    )
    bars_forward = ax.bar(
        x + width / 2,
        forward_event_drop,
        width,
        color=orange,
        edgecolor="#9A5E23",
        linewidth=0.7,
        label="Drop events timestamped during forward motion",
    )
    ax.set_title("Reported battery-level decrease", loc="left", fontsize=11, color=ink)
    ax.set_ylabel("Reported battery drop (percentage points)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(7.0, float(total_drop.max()) + 2.0))
    ax.grid(axis="y", color=grid, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for bars in (bars_total, bars_forward):
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8.5, color=ink)
    for index, row in data.iterrows():
        ax.text(
            index,
            -0.17,
            f"{int(row['battery_at_motion_start_pct'])}%→{int(row['battery_at_250cm_end_pct'])}%",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.2,
            color="#5A6978",
        )

    ax = axes[1]
    ax.bar(
        x,
        forward_sec,
        width=0.58,
        color=blue,
        edgecolor="#214F72",
        linewidth=0.7,
        label="Forward movement retained",
    )
    ax.bar(
        x,
        nonforward_sec,
        width=0.58,
        bottom=forward_sec,
        color=grey,
        edgecolor="#77838E",
        linewidth=0.7,
        label="Non-forward time removed",
    )
    ax.set_title("Selected-window time composition", loc="left", fontsize=11, color=ink)
    ax.set_ylabel("Time (s)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, float((forward_sec + nonforward_sec).max()) + 6)
    ax.grid(axis="y", color=grid, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for index, (moving, waiting) in enumerate(zip(forward_sec, nonforward_sec)):
        ax.text(index, moving / 2, f"{moving:.1f}s", ha="center", va="center", fontsize=8, color="white")
        if waiting >= 3:
            ax.text(
                index,
                moving + waiting / 2,
                f"{waiting:.1f}s",
                ha="center",
                va="center",
                fontsize=8,
                color=ink,
            )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#8D99A5")
        ax.tick_params(colors=ink)

    fig.text(
        0.075,
        0.018,
        "Note: the forward-event measure is diagnostic only. Tello reports integer SOC and updates may lag; "
        "Drone 3 and Drone 4 stayed at 72% and 73% throughout the full selected window.",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#5A6978",
        wrap=True,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.89, bottom=0.12)
    image_path = OUTPUT_DIR / f"{EXPERIMENT}_{RUN_ID}_hover_comparison.png"
    fig.savefig(image_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(image_path)
    print(csv_path)


if __name__ == "__main__":
    main()
