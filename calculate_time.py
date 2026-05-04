"""
Build battery-consumption summary plots from hover column experiments.

Behavior
1. Ignore all trials up to and including the reset marker.
2. Renumber included trials as Trial 1, Trial 2, Trial 3, ...
3. Plot per-trial battery drop for left / middle / right drones.
4. Plot per-trial percentage share plus overall mean percentage share.
5. Click a legend item to isolate that dataset; click again to restore all.
"""

import csv
import os
import sys
from datetime import datetime
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUMMARY = os.path.join(OUTPUT_DIR, "hover_front_summary.csv")
RESET_MARKER_FILE = os.path.join(OUTPUT_DIR, "calculate_time_front_reset_marker.txt")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "calculate_time_front_history.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "calculate_time_front_overview.png")

DRONE_ORDER = ["drone_left", "drone_middle", "drone_right"]
DRONE_LABELS = {
    "drone_left": "Left",
    "drone_middle": "Middle",
    "drone_right": "Right",
}
DRONE_COLORS = {
    "drone_left": "#3a86ff",
    "drone_middle": "#fb8500",
    "drone_right": "#2a9d8f",
}
CHARGING_BAR_COLORS = ["#7f5539", "#6c8f4e", "#3d5a80"]


def safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_summary_rows(summary_path):
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No data rows found in: {summary_path}")

    return rows


def read_reset_marker():
    if not os.path.exists(RESET_MARKER_FILE):
        return 0
    with open(RESET_MARKER_FILE, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid reset marker file: {RESET_MARKER_FILE}") from exc


def filter_rows_after_reset(summary_rows, reset_trial_id):
    filtered = []
    for row in summary_rows:
        try:
            actual_trial_id = int(row["trial_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if actual_trial_id > reset_trial_id:
            row["actual_trial_id"] = actual_trial_id
            filtered.append(row)
    return filtered


def group_rows_by_trial(rows):
    grouped = {}
    for row in rows:
        session_key = (
            row.get("hover_start_timestamp", "").strip(),
            row["actual_trial_id"],
        )
        grouped.setdefault(session_key, []).append(row)
    return grouped


def parse_trial_data(rows, display_trial_id):
    drone_rows = []
    for row in rows:
        drone_name = row.get("drone_name", "")
        if drone_name not in DRONE_ORDER:
            continue
        battery_drop = safe_float(row.get("battery_drop"), 0.0)
        drone_rows.append({
            "actual_trial_id": row["actual_trial_id"],
            "display_trial_id": display_trial_id,
            "trial_label": f"Trial {display_trial_id}",
            "drone_name": drone_name,
            "drone_ip": row.get("drone_ip", ""),
            "drone_role": row.get("drone_role", ""),
            "battery_drop": battery_drop,
        })

    if not drone_rows:
        raise ValueError(f"No usable drone rows found for display trial {display_trial_id}.")

    drone_rows.sort(key=lambda item: DRONE_ORDER.index(item["drone_name"]))
    total_drop = sum(item["battery_drop"] for item in drone_rows)
    for item in drone_rows:
        item["share_pct"] = (item["battery_drop"] / total_drop * 100.0) if total_drop > 0 else 0.0
        item["total_drop"] = total_drop
    return drone_rows


def build_charging_totals(drone_rows):
    totals = {}
    for pad_count in (1, 2, 3):
        totals[pad_count] = build_schedule(drone_rows, pad_count)
    return totals


def build_schedule(drone_rows, pad_count):
    pad_available_times = [0.0] * pad_count
    charge_needs = sorted(
        [row["battery_drop"] for row in drone_rows],
        reverse=True,
    )

    for charge_minutes in charge_needs:
        pad_index = min(range(pad_count), key=lambda idx: pad_available_times[idx])
        pad_available_times[pad_index] += charge_minutes

    return max(pad_available_times) if pad_available_times else 0.0


def build_trial_records(grouped_rows):
    records = []
    session_keys = sorted(grouped_rows, key=lambda item: (item[0], item[1]))
    for display_trial_id, session_key in enumerate(session_keys, start=1):
        actual_trial_id = session_key[1]
        drone_rows = parse_trial_data(grouped_rows[session_key], display_trial_id)
        records.append({
            "dataset_id": f"trial_{display_trial_id}",
            "dataset_label": f"Trial {display_trial_id}",
            "actual_trial_id": actual_trial_id,
            "display_trial_id": display_trial_id,
            "hover_start_timestamp": session_key[0],
            "drone_rows": drone_rows,
            "charging_totals": build_charging_totals(drone_rows),
            "is_mean": False,
        })
    return records


def build_mean_record(records):
    per_drone = {drone_name: [] for drone_name in DRONE_ORDER}
    for record in records:
        for row in record["drone_rows"]:
            per_drone[row["drone_name"]].append(row)

    mean_drone_rows = []
    total_mean_drop = 0.0
    for drone_name in DRONE_ORDER:
        rows = per_drone[drone_name]
        if not rows:
            continue
        mean_drop = mean(row["battery_drop"] for row in rows)
        mean_drone_rows.append({
            "actual_trial_id": "",
            "display_trial_id": "mean",
            "trial_label": "Mean",
            "drone_name": drone_name,
            "drone_ip": rows[0]["drone_ip"],
            "drone_role": rows[0]["drone_role"],
            "battery_drop": mean_drop,
        })
        total_mean_drop += mean_drop

    for row in mean_drone_rows:
        row["share_pct"] = (row["battery_drop"] / total_mean_drop * 100.0) if total_mean_drop > 0 else 0.0
        row["total_drop"] = total_mean_drop

    return {
        "dataset_id": "mean",
        "dataset_label": f"Mean ({len(records)} trials)",
        "actual_trial_id": "",
        "display_trial_id": "mean",
        "drone_rows": mean_drone_rows,
        "charging_totals": build_charging_totals(mean_drone_rows),
        "is_mean": True,
    }


def save_history_csv(records, mean_record, reset_trial_id):
    generated_at = datetime.now().isoformat(timespec="seconds")
    header = [
        "generated_at",
        "reset_trial_id",
        "dataset_id",
        "dataset_label",
        "is_mean",
        "actual_trial_id",
        "display_trial_id",
        "drone_name",
        "drone_ip",
        "drone_role",
        "battery_drop",
        "share_pct",
        "total_drop",
        "charging_time_1_pad",
        "charging_time_2_pads",
        "charging_time_3_pads",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for record in records + [mean_record]:
            for row in record["drone_rows"]:
                writer.writerow([
                    generated_at,
                    reset_trial_id,
                    record["dataset_id"],
                    record["dataset_label"],
                    record["is_mean"],
                    row["actual_trial_id"],
                    row["display_trial_id"],
                    row["drone_name"],
                    row["drone_ip"],
                    row["drone_role"],
                    row["battery_drop"],
                    row["share_pct"],
                    row["total_drop"],
                    record["charging_totals"][1],
                    record["charging_totals"][2],
                    record["charging_totals"][3],
                ])


def set_dataset_visibility(dataset_artists, active_dataset_id=None):
    for dataset_id, artists in dataset_artists.items():
        visible = active_dataset_id is None or dataset_id == active_dataset_id
        for artist in artists:
            artist.set_visible(visible)


def annotate_bar(ax, bar, text, y_offset):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + y_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=8,
    )


def build_interactive_plot(records, mean_record):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18))
    dataset_artists = {}
    legend_map = {}

    all_records = records + [mean_record]
    x_positions = list(range(len(all_records)))
    group_width = 0.72
    bar_width = group_width / 3.0
    offsets = [-bar_width, 0.0, bar_width]

    max_drop = max(
        (row["battery_drop"] for record in all_records for row in record["drone_rows"]),
        default=1.0,
    )
    max_charge = max(
        (record["charging_totals"][pad_count] for record in all_records for pad_count in (1, 2, 3)),
        default=1.0,
    )

    for idx, record in enumerate(all_records):
        dataset_id = record["dataset_id"]
        dataset_artists[dataset_id] = []
        alpha = 0.95 if record["is_mean"] else 0.75
        edgecolor = "black" if record["is_mean"] else "none"
        linewidth = 1.2 if record["is_mean"] else 0.0

        for drone_idx, drone_name in enumerate(DRONE_ORDER):
            row = next((r for r in record["drone_rows"] if r["drone_name"] == drone_name), None)
            if row is None:
                continue

            bar = ax1.bar(
                x_positions[idx] + offsets[drone_idx],
                row["battery_drop"],
                width=bar_width * 0.9,
                color=DRONE_COLORS[drone_name],
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )[0]
            dataset_artists[dataset_id].append(bar)
            annotate_bar(ax1, bar, f"{row['share_pct']:.1f}%", max_drop * 0.03)

        bottom = 0.0
        for drone_name in DRONE_ORDER:
            row = next((r for r in record["drone_rows"] if r["drone_name"] == drone_name), None)
            if row is None:
                continue
            stack = ax2.bar(
                x_positions[idx],
                row["share_pct"],
                width=0.62,
                bottom=bottom,
                color=DRONE_COLORS[drone_name],
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )[0]
            dataset_artists[dataset_id].append(stack)
            if row["share_pct"] > 6:
                ax2.text(
                    x_positions[idx],
                    bottom + row["share_pct"] / 2,
                    f"{DRONE_LABELS[drone_name]}\n{row['share_pct']:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if drone_name != "drone_middle" else "black",
                )
            bottom += row["share_pct"]

        charging_offsets = [-bar_width, 0.0, bar_width]
        for pad_idx, pad_count in enumerate((1, 2, 3)):
            charge_bar = ax3.bar(
                x_positions[idx] + charging_offsets[pad_idx],
                record["charging_totals"][pad_count],
                width=bar_width * 0.9,
                color=CHARGING_BAR_COLORS[pad_idx],
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )[0]
            dataset_artists[dataset_id].append(charge_bar)
            annotate_bar(ax3, charge_bar, f"{record['charging_totals'][pad_count]:.1f}", max_charge * 0.03)

    ax1.set_title("Battery Drop Per Trial And Mean")
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Battery Drop (%)")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([record["dataset_label"] for record in all_records], rotation=25, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.set_title("Battery Share Per Trial And Overall Mean")
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Share Of Total Drop (%)")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([record["dataset_label"] for record in all_records], rotation=25, ha="right")
    ax2.grid(True, axis="y", alpha=0.3)

    ax3.set_title("Charging Time Per Trial And Mean")
    ax3.set_xlabel("Dataset")
    ax3.set_ylabel("Total Charging Time (minutes)")
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels([record["dataset_label"] for record in all_records], rotation=25, ha="right")
    ax3.grid(True, axis="y", alpha=0.3)

    drone_handles = [
        Line2D([0], [0], color=DRONE_COLORS[drone_name], lw=8, label=DRONE_LABELS[drone_name])
        for drone_name in DRONE_ORDER
    ]
    ax2.legend(handles=drone_handles, loc="upper right", title="Drone")

    charging_handles = [
        Line2D([0], [0], color=CHARGING_BAR_COLORS[0], lw=8, label="1 pad"),
        Line2D([0], [0], color=CHARGING_BAR_COLORS[1], lw=8, label="2 pads"),
        Line2D([0], [0], color=CHARGING_BAR_COLORS[2], lw=8, label="3 pads"),
    ]
    ax3.legend(handles=charging_handles, loc="upper right", title="Charging Setup")

    dataset_handles = []
    for record in all_records:
        handle = Line2D(
            [0], [0],
            color="black" if record["is_mean"] else "dimgray",
            lw=3 if record["is_mean"] else 2,
            linestyle="-" if record["is_mean"] else "--",
            marker="o",
            label=record["dataset_label"],
            picker=5,
        )
        dataset_handles.append(handle)

    legend = ax1.legend(handles=dataset_handles, loc="upper left", title="Click To Isolate / Restore")
    for legend_artist, record in zip(legend.legend_handles, all_records):
        legend_artist.set_picker(5)
        legend_map[legend_artist] = record["dataset_id"]

    active_dataset_id = {"value": None}

    def on_pick(event):
        dataset_id = legend_map.get(event.artist)
        if dataset_id is None:
            return
        if active_dataset_id["value"] == dataset_id:
            active_dataset_id["value"] = None
        else:
            active_dataset_id["value"] = dataset_id
        set_dataset_visibility(dataset_artists, active_dataset_id["value"])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    return fig


def main():
    summary_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SUMMARY
    summary_rows = load_summary_rows(summary_path)
    reset_trial_id = read_reset_marker()
    filtered_rows = filter_rows_after_reset(summary_rows, reset_trial_id)

    if not filtered_rows:
        print("No new trials found after reset marker.")
        print(f"Reset marker trial_id: {reset_trial_id}")
        print(f"Summary source: {summary_path}")
        return

    grouped_rows = group_rows_by_trial(filtered_rows)
    records = build_trial_records(grouped_rows)
    mean_record = build_mean_record(records)
    save_history_csv(records, mean_record, reset_trial_id)

    print(f"Loaded {len(records)} trial(s) after reset marker {reset_trial_id} from {summary_path}")
    for record in records:
        total_drop = record["drone_rows"][0]["total_drop"] if record["drone_rows"] else 0.0
        print(f"\n{record['dataset_label']} (actual trial {record['actual_trial_id']}): total_drop={total_drop:.1f}%")
        for row in record["drone_rows"]:
            print(
                f"  {DRONE_LABELS[row['drone_name']]}: "
                f"drop={row['battery_drop']:.1f}% share={row['share_pct']:.1f}%"
            )

    print("\nOverall mean:")
    total_mean_drop = mean_record["drone_rows"][0]["total_drop"] if mean_record["drone_rows"] else 0.0
    print(f"  Mean total_drop={total_mean_drop:.2f}%")
    for row in mean_record["drone_rows"]:
        print(
            f"  {DRONE_LABELS[row['drone_name']]}: "
            f"mean_drop={row['battery_drop']:.2f}% mean_share={row['share_pct']:.2f}%"
        )

    build_interactive_plot(records, mean_record)
    print(f"\nSaved history CSV: {OUTPUT_CSV}")
    print(f"Saved overview plot: {OUTPUT_PNG}")
    print(f"Reset marker file: {RESET_MARKER_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
