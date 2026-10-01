"""Plot a real flight excerpt showing retained and excluded flight intervals.

No smoothing, interpolation of SOC, or synthetic take-off samples are used.
The existing cleaning sidecar supplies forward intervals. Take-off and landing
are omitted. Adjacent sample midpoints delimit forward transitions.
"""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "front_75_head_lv2_new_002"
RUN = "20260605_200752"
DRONE = "drone_5"
SOURCE = ROOT / "db_copy_for_cleaning" / EXPERIMENT / f"{EXPERIMENT}_{RUN}_all_coordination.csv"
INTERVAL_SOURCE = ROOT / "db_copy_for_cleaning/_cleaning_admin/trajectory_qc/forward_motion_intervals.csv"
OUT = ROOT / "analysis_outputs/methodology_figures"
STEM = "fig_remove_hover_battery_consumption"
BLUE = "#246B9A"


def brace(ax, left, right, label):
    """Upward brace in data-x / axes-y coordinates."""
    inset = min(0.12, (right - left) * 0.045)
    a, b = left + inset, right - inset
    middle = (a + b) / 2
    r = min((b - a) * .12, .5)
    y, dy = 1.025, .028
    vertices = [(a, y), (a, y + dy), (a + r, y + dy),
                (middle - r, y + dy), (middle, y + dy), (middle, y + 2*dy),
                (middle, y + dy), (middle + r, y + dy),
                (b - r, y + dy), (b, y + dy), (b, y)]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3]
    ax.add_patch(PathPatch(MplPath(vertices, codes), transform=ax.get_xaxis_transform(),
                          fill=False, lw=1.1, color="#202428", clip_on=False))
    ax.text(middle, 1.103, label, transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=10, linespacing=1.1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, low_memory=False)
    full = raw.loc[raw.drone_name.eq(DRONE)].sort_values("node_elapsed_time").copy()
    intervals = pd.read_csv(INTERVAL_SOURCE, dtype={"run_id": str})
    intervals = intervals.loc[intervals.run_id.eq(RUN) & intervals.drone_name.eq(DRONE)].copy()
    forwards = intervals.loc[intervals.interval_type.eq("forward_movement")].tail(2)
    assert len(forwards) == 2
    begin = float(forwards.iloc[0].start_time_sec)
    landing = float(full.loc[full.phase.eq("landing"), "node_elapsed_time"].min())
    end = float(full.loc[full.node_elapsed_time.lt(landing), "node_elapsed_time"].max())
    first_end = float(forwards.iloc[0].end_time_sec)
    second_start = float(forwards.iloc[1].start_time_sec)
    second_end = float(forwards.iloc[1].end_time_sec)
    ts = full.node_elapsed_time.to_numpy()

    def after_midpoint(t):
        return (t + ts[ts > t][0]) / 2

    def before_midpoint(t):
        return (ts[ts < t][-1] + t) / 2

    boundaries = [begin, after_midpoint(first_end), before_midpoint(second_start),
                  after_midpoint(second_end), end]
    labels = ["Forward flight", "Non-forward", "Forward flight", "Non-forward"]
    kept = [True, False, True, False]
    data = full.loc[full.node_elapsed_time.between(begin, end)].copy()
    data["time_s"] = data.node_elapsed_time - begin
    data["battery_consumption_pp"] = float(data.battery.iloc[0]) - data.battery
    # The measured SOC is monotone in this particular illustrative excerpt.
    assert (data.battery.diff().dropna() <= 0).all()
    assert data.battery.iloc[0] == 60 and data.battery.iloc[-1] == 54
    data["display_interval"] = pd.cut(data.node_elapsed_time, boundaries,
                                      labels=False, right=False, include_lowest=True).fillna(3).astype(int)
    data["display_phase"] = data.display_interval.map(dict(enumerate(labels)))
    data["retained_forward"] = data.display_interval.map(dict(enumerate(kept)))
    data.to_csv(OUT / f"{STEM}_data.csv", index=False)
    phase_table = pd.DataFrame({"phase": labels, "retained": kept,
                               "source_start_s": boundaries[:-1], "source_end_s": boundaries[1:]})
    phase_table["plot_start_s"] = phase_table.source_start_s - begin
    phase_table["plot_end_s"] = phase_table.source_end_s - begin
    phase_table.to_csv(OUT / f"{STEM}_intervals.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 11, "axes.labelsize": 13, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    fig.subplots_adjust(left=.115, right=.975, bottom=.16, top=.78)
    for _, p in phase_table.iterrows():
        if not p.retained:
            ax.axvspan(p.plot_start_s, p.plot_end_s, facecolor="#F0F2F4", zorder=0)
        brace(ax, p.plot_start_s, p.plot_end_s, p.phase.replace(" ", "\n") if p.phase == "Forward flight" else p.phase)
    for b in boundaries[1:]:
        ax.axvline(b - begin, color="#A3A8AE", lw=.85, ls=(0, (4, 4)), zorder=1)
    ax.step(data.time_s, data.battery_consumption_pp, where="post", color=BLUE, lw=2, zorder=3)
    change = data.battery.diff().ne(0)
    ax.scatter(data.loc[change, "time_s"], data.loc[change, "battery_consumption_pp"],
               s=21, color=BLUE, zorder=4, clip_on=False)
    ax.scatter([data.time_s.iloc[-1]], [data.battery_consumption_pp.iloc[-1]], s=21, color=BLUE, zorder=4, clip_on=False)
    ax.set(xlim=(0, end-begin), ylim=(0, 7), xlabel="Time (s)", ylabel="Battery consumption (%)")
    ax.set_xticks(np.arange(0, end-begin, 5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.15)
    ax.tick_params(direction="out", width=1, length=4)
    fig.savefig(OUT / f"{STEM}.pdf", metadata={"Title": "Measured battery consumption during forward and non-forward flight intervals"})
    fig.savefig(OUT / f"{STEM}.png", dpi=300)
    plt.close(fig)

    # A full-record check image retains the original recording-time origin.
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    before_landing = full.loc[full.node_elapsed_time.le(end)]
    ax.step(before_landing.node_elapsed_time, before_landing.battery.iloc[0]-before_landing.battery, where="post", color=BLUE, lw=1.6)
    ax.axvspan(begin, end, color="#246B9A", alpha=.1, label="Displayed excerpt")
    ax.set(xlabel="Time from recording start (s)", ylabel="Battery consumption (%)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / f"{STEM}_full_record.png", dpi=200)
    plt.close(fig)

    metadata = {
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "forward_interval_source": str(INTERVAL_SOURCE.relative_to(ROOT)),
        "forward_interval_source_sha256": hashlib.sha256(INTERVAL_SOURCE.read_bytes()).hexdigest(),
        "run_id": RUN, "experiment": EXPERIMENT, "drone": DRONE,
        "battery_id": str(data.battery_id.iloc[0]), "source_time_window_s": [begin, end],
        "plot_time_origin": "Start of the displayed excerpt; not take-off or whole-flight start",
        "sample_count": len(data), "initial_reported_soc_percent": int(data.battery.iloc[0]),
        "final_reported_soc_percent": int(data.battery.iloc[-1]),
        "consumption": "Initial reported SOC minus observed SOC, in percentage points; no normalization or smoothing",
        "boundary_rule": "Midpoints between adjacent forward/non-forward samples; display ends at the final recorded sample before landing",
        "selection_purpose": "Illustrative excerpt with two forward bouts and an intervening visible pause; not a performance comparison",
        "limitations": ["Recording begins after take-off; no take-off consumption is reconstructed.",
                       "Non-forward labels do not establish whether a pause was specifically for coordinate detection.",
                       "Integer SOC telemetry can lag physical consumption; individual 1-point drops must not be attributed precisely to a flight mode.",
                       "The final non-forward interval continues beyond the sidecar endpoint to the final sample before landing; reported planar velocities are zero in this continuation.",
                       "Take-off and landing are omitted from both images, as requested."],
        "caption": "Battery consumption during an excerpt of a recorded flight. Unshaded intervals correspond to forward flight. Shaded intervals are excluded from forward-flight analysis. The curve shows the measured SOC decrease from the start of the displayed excerpt."
    }
    extension = full.loc[full.node_elapsed_time.gt(float(intervals.end_time_sec.max())) & full.node_elapsed_time.lt(landing)]
    assert (extension[["vgx", "vgy"]].abs().max() == 0).all()
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUT / f"{STEM}_caption.txt").write_text(metadata["caption"] + "\n")
    print(json.dumps({"png": str(OUT / f"{STEM}.png"), "pdf": str(OUT / f"{STEM}.pdf"),
                      "source_time_s": [begin, end], "samples": len(data), "soc": [60, 54]}, indent=2))


if __name__ == "__main__":
    main()
