"""Real 2.5 m processing window with approximately 25 s of forward motion."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_trajectory_cleaning_segments import prepare_run_groups, centered_rolling_median
from build_forward_motion_segments import build_forward_mask, durations_and_distance
from plot_ieee_battery_consumption_phases import brace

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "column_75_head_lv1_new_003"
RUN = "20260620_175512"
DRONE = "drone_2"
SOURCE = ROOT / "db_copy_for_cleaning" / EXPERIMENT / f"{EXPERIMENT}_{RUN}_all_coordination.csv"
QC = ROOT / "db_copy_for_cleaning/_cleaning_admin/trajectory_qc"
OUT = ROOT / "analysis_outputs/methodology_figures"
STEM = "fig_remove_hover_battery_consumption_25s_first2pp"


def selected_row(filename):
    table = pd.read_csv(QC / filename, dtype={"run_id": str})
    return table.loc[table.run_id.eq(RUN) & table.drone_name.eq(DRONE)].iloc[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    item = prepare_run_groups(pd.read_csv(SOURCE, low_memory=False))[DRONE]
    segment = selected_row("trajectory_drone_segments.csv")
    summary = selected_row("forward_motion_drone_segments.csv")
    times = item["times"]
    progress = centered_rolling_median(item["relative"] @ item["run_direction"], 11)
    progress = progress * float(segment.trajectory_distance_calibration_factor)
    group = item["group"]
    moving, inside, _ = build_forward_mask(
        times, progress, group.phase.fillna("").astype(str).to_numpy(),
        float(segment.motion_onset_sec), float(segment.selected_250cm_end_sec),
        threshold_cm_s=2.0,
    )
    forward_s, nonforward_s, _ = durations_and_distance(times, np.maximum.accumulate(progress), moving, inside)
    assert abs(forward_s - summary.forward_movement_sec) < 1e-6
    assert abs(nonforward_s - summary.in_flight_nonforward_sec) < 1e-6
    data = group.loc[inside].copy()
    t = times[inside]
    mask = moving[inside]
    assert np.all(np.diff(np.flatnonzero(inside)) == 1)
    total_s = t[-1] - t[0]
    assert abs(total_s - forward_s - nonforward_s) < 1e-8
    assert 24 <= forward_s <= 26 and total_s > 25
    assert not data.phase.str.contains("takeoff|landing", case=False).any()
    consumption = float(data.battery.iloc[0]) - data.battery.to_numpy(float)
    assert np.all(np.diff(consumption) >= 0)
    data["source_time_s"] = t
    data["time_s"] = t - t[0]
    data["battery_consumption_pp"] = consumption
    # The existing pipeline assigns each time interval to its left sample.
    data["forward_interval_to_next_sample"] = np.r_[mask[:-1], False]
    data["interval_duration_s"] = np.r_[np.diff(t), 0.0]
    data.to_csv(OUT / f"{STEM}_data.csv", index=False)

    cuts = np.r_[0, np.flatnonzero(mask[1:-1] != mask[:-2]) + 1, len(t)-1]
    intervals = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        intervals.append({"phase": "Forward flight" if mask[a] else "Non-forward",
                          "retained": bool(mask[a]), "start_s": float(t[a]-t[0]),
                          "end_s": float(t[b]-t[0]), "duration_s": float(t[b]-t[a])})
    assert abs(sum(p["duration_s"] for p in intervals if p["retained"]) - forward_s) < 1e-8
    first_forward = data[data.time_s.lt(intervals[0]["end_s"])]
    assert first_forward.battery.iloc[0] - first_forward.battery.iloc[-1] == 2
    pd.DataFrame(intervals).to_csv(OUT / f"{STEM}_intervals.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 11, "axes.labelsize": 13, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11, 4.9))
    fig.subplots_adjust(left=.09, right=.98, bottom=.15, top=.77)
    for p in intervals:
        if not p["retained"]:
            ax.axvspan(p["start_s"], p["end_s"], color="#F0F2F4", zorder=0)
        brace(ax, p["start_s"], p["end_s"], "Forward\nflight" if p["retained"] else ("Non-\nforward" if p["duration_s"] < 3 else "Non-forward"))
        if p["start_s"] > 0:
            ax.axvline(p["start_s"], color="#A3A8AE", ls=(0, (4, 4)), lw=.8, zorder=1)
    ax.step(data.time_s, consumption, where="post", lw=2, color="#246B9A", zorder=3)
    change = np.r_[True, np.diff(consumption) != 0]
    ax.scatter(data.time_s.to_numpy()[change], consumption[change], s=20, color="#246B9A", clip_on=False, zorder=4)
    ax.scatter([total_s], [consumption[-1]], s=20, color="#246B9A", clip_on=False, zorder=4)
    ax.set(xlim=(0, total_s), ylim=(0, 10), xlabel="Time (s)", ylabel="Battery consumption (%)")
    ax.set_xticks(np.arange(0, total_s, 5))
    ax.set_yticks(np.arange(0, 11, 2))
    ax.text(.02, .965, f"Total: {total_s:.2f} s     Forward: {forward_s:.2f} s     Non-forward: {nonforward_s:.2f} s",
            transform=ax.transAxes, ha="left", va="top", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.1)
    ax.tick_params(direction="out", width=1, length=4)
    fig.savefig(OUT / f"{STEM}.pdf")
    fig.savefig(OUT / f"{STEM}.png", dpi=300)
    plt.close(fig)

    caption = ("Battery consumption over a recorded flight interval. Shaded non-forward intervals are excluded, "
               f"reducing the analysed time from {total_s:.2f} s to {forward_s:.2f} s of forward flight. "
               "Time is measured from the first sample of the displayed analysis window.")
    metadata = {"experiment": EXPERIMENT, "run_id": RUN, "drone": DRONE, "battery": str(data.battery_id.iloc[0]),
                "source": str(SOURCE), "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "source_window_s": [float(t[0]), float(t[-1])], "recorded_interval_s": total_s,
                "retained_forward_s": forward_s, "excluded_nonforward_s": nonforward_s,
                "first_forward_reported_soc_drop_pp": 2, "sample_count": len(data), "soc_start_end": [float(data.battery.iloc[0]), float(data.battery.iloc[-1])],
                "selected_distance_cm": float(segment.selected_distance_cm),
                "method": "Existing trajectory cleaning and forward-motion classifier, threshold 2 cm/s; left-sample time-interval assignment, unchanged from pipeline.",
                "selection": "Illustrative real record selected for a 24-26 s forward duration, longer unfiltered window and a 2-point reported SOC decrease during the first forward interval.",
                "limitations": ["Only the recorded analysis window is shown, not the entire trial.",
                                "Take-off and landing are excluded.",
                                "SOC is integer telemetry with possible update lag; flat portions do not imply zero physical consumption.",
                                "Non-forward denotes the existing motion classifier, not a confirmed coordinate-detection cause.",
                                "Consumption is measured reported-SOC decrease in percentage points; no normalization, interpolation or smoothing is applied."],
                "caption": caption}
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUT / f"{STEM}_caption.txt").write_text(caption + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
