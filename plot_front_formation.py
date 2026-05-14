import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CSV_CANDIDATES = [
    "front_experiment_trajectory_log.csv",
]

LANE_X_TARGETS = {"drone_left": 0, "drone_middle": 50, "drone_right": 100}
LANE_SPACING_CM = 50
TOLERANCE_CM = 8

DRONE_STYLES = {
    "drone_left":   {"color": "#2980b9", "marker": "o",  "label": "Left"},
    "drone_middle": {"color": "#e67e22", "marker": "s",  "label": "Middle"},
    "drone_right":  {"color": "#27ae60", "marker": "^",  "label": "Right"},
}

TRIAL_LINESTYLES = ["-", "--", "-.", ":"]

PHASE_COLORS = {
    "coordinate_climb":      "#d5e8d4",
    "forward_to_y_250":      "#dae8fc",
    "coordinate_correction": "#fff2cc",
    "final_descent_prepare": "#ffe6cc",
    "final_global_check":    "#f8cecc",
    "landing":               "#e1d5e7",
}

SPACING_PAIR_STYLES = {
    "y_distance_diff_12_cm": {"color": "#8e44ad", "label": "Left–Middle"},
    "y_distance_diff_23_cm": {"color": "#c0392b", "label": "Middle–Right"},
    "y_distance_diff_13_cm": {"color": "#16a085", "label": "Left–Right",  "linestyle": "--"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_csv_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for p in DEFAULT_CSV_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No front formation CSV found. Pass the CSV path as an argument."
    )


def load_csv(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
    df.columns = df.columns.str.strip()
    numeric = [
        "trial_id", "elapsed_time",
        "X_global", "Y_global", "Z_global",
        "x", "y", "z",
        "y_distance_diff_mean",
        "y_distance_diff_12_cm", "y_distance_diff_23_cm", "y_distance_diff_13_cm",
        "battery",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trial_id"] = df["trial_id"].astype(int)
    return df.sort_values(["trial_id", "drone_name", "elapsed_time"]).copy()


def phase_spans(trial_df, drone_name):
    """Return list of (t_start, t_end, phase) for a drone in one trial."""
    ddf = trial_df[trial_df["drone_name"] == drone_name].dropna(
        subset=["elapsed_time", "phase"]
    ).sort_values("elapsed_time")
    if ddf.empty:
        return []
    spans = []
    for _, grp in ddf.groupby(
        (ddf["phase"] != ddf["phase"].shift()).cumsum()
    ):
        spans.append((
            grp["elapsed_time"].iloc[0],
            grp["elapsed_time"].iloc[-1],
            grp["phase"].iloc[0],
        ))
    return spans


def add_phase_background(ax, trial_df):
    """Shade Y-axis backgrounds by phase using the left drone as reference."""
    for t_start, t_end, phase in phase_spans(trial_df, "drone_left"):
        color = PHASE_COLORS.get(phase, "#f5f5f5")
        ax.axvspan(t_start, t_end, color=color, alpha=0.25, zorder=0)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def build_plot(df, csv_path):
    trial_ids = sorted(df["trial_id"].unique())
    n_trials = len(trial_ids)

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
    ax_xy   = fig.add_subplot(gs[0, 0])   # top-view
    ax_y    = fig.add_subplot(gs[0, 1])   # Y vs time
    ax_x    = fig.add_subplot(gs[1, 0])   # X vs time (lane adherence)
    ax_gap  = fig.add_subplot(gs[1, 1])   # Y-spacing between drones

    # ---- reference lines -----------------------------------------------
    for name, xval in LANE_X_TARGETS.items():
        style = DRONE_STYLES[name]
        ax_xy.axvline(xval, color=style["color"], linestyle=":", linewidth=1, alpha=0.5)
        ax_x.axhline(xval, color=style["color"], linestyle=":", linewidth=1, alpha=0.5,
                     label=f"ideal {style['label']} (X={xval})")

    ax_gap.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
                   label="ideal (0 cm gap)")
    ax_gap.axhspan(-TOLERANCE_CM, TOLERANCE_CM, color="gray", alpha=0.08,
                   label=f"±{TOLERANCE_CM} cm tolerance")

    # ---- data per trial --------------------------------------------------
    trial_legend_handles = []
    pick_map = {}
    hover_map = {}
    trial_artists = {tid: [] for tid in trial_ids}
    trial_meta = {}

    for t_idx, tid in enumerate(trial_ids):
        trial_df = df[df["trial_id"] == tid].copy()
        ls = TRIAL_LINESTYLES[t_idx % len(TRIAL_LINESTYLES)]
        trial_meta[tid] = {}

        # phase background on time-series axes (use first trial for clarity)
        if t_idx == 0:
            for ax_ts in (ax_y, ax_x, ax_gap):
                add_phase_background(ax_ts, trial_df)

        for drone_name, style in DRONE_STYLES.items():
            ddf = trial_df[trial_df["drone_name"] == drone_name].dropna(
                subset=["X_global", "Y_global", "elapsed_time"]
            ).sort_values("elapsed_time")
            if ddf.empty:
                continue

            color = style["color"]
            marker = style["marker"]
            alpha = 0.85 if t_idx == 0 else 0.55

            kw = dict(linewidth=1.5, color=color, linestyle=ls,
                      marker=marker, markersize=2, alpha=alpha, picker=True)

            # top-view XY
            ln_xy, = ax_xy.plot(ddf["X_global"], ddf["Y_global"], **kw)
            # mark start / end
            ax_xy.scatter(ddf["X_global"].iloc[0],  ddf["Y_global"].iloc[0],
                          color=color, s=40, zorder=5, marker="D")
            ax_xy.scatter(ddf["X_global"].iloc[-1], ddf["Y_global"].iloc[-1],
                          color=color, s=40, zorder=5, marker="*")

            # Y vs time
            ln_y, = ax_y.plot(ddf["elapsed_time"], ddf["Y_global"], **kw)

            # X vs time
            ln_x, = ax_x.plot(ddf["elapsed_time"], ddf["X_global"], **kw)

            for ln in (ln_xy, ln_y, ln_x):
                trial_artists[tid].append(ln)
                pick_map[id(ln)] = tid
                hover_map[ln] = tid

            trial_meta[tid][drone_name] = {
                "x_start": round(float(ddf["X_global"].iloc[0]), 1),
                "x_end":   round(float(ddf["X_global"].iloc[-1]), 1),
                "y_end":   round(float(ddf["Y_global"].iloc[-1]), 1),
            }

        # Y-spacing panel: one line per pair, per trial
        spacing_df = trial_df[trial_df["drone_name"] == "drone_left"].dropna(
            subset=["elapsed_time"]
        ).sort_values("elapsed_time")

        for col, pkw in SPACING_PAIR_STYLES.items():
            if col not in spacing_df.columns:
                continue
            valid = spacing_df.dropna(subset=[col])
            if valid.empty:
                continue
            gkw = dict(
                linewidth=1.6 if t_idx == 0 else 1.1,
                color=pkw["color"],
                linestyle=pkw.get("linestyle", ls),
                alpha=0.85 if t_idx == 0 else 0.5,
                picker=True,
            )
            ln_g, = ax_gap.plot(valid["elapsed_time"], valid[col], **gkw)
            trial_artists[tid].append(ln_g)
            pick_map[id(ln_g)] = tid
            hover_map[ln_g] = tid

        trial_legend_handles.append(
            Line2D([0], [0], color="gray", linestyle=ls, linewidth=2,
                   label=f"Trial {tid}")
        )

    # ---- axes decoration ------------------------------------------------
    # XY
    ax_xy.set_title("Top View — Drone Trajectories (X–Y)", fontsize=12, fontweight="bold")
    ax_xy.set_xlabel("X global (cm)")
    ax_xy.set_ylabel("Y global (cm)")
    ax_xy.grid(True, linestyle="--", alpha=0.4)
    ax_xy.set_xlim(-30, 140)

    # Y vs time
    ax_y.set_title("Forward Progress — Y_global vs Time", fontsize=12, fontweight="bold")
    ax_y.set_xlabel("Elapsed time (s)")
    ax_y.set_ylabel("Y global (cm)")
    ax_y.axhline(250, color="gray", linewidth=0.8, linestyle=":", alpha=0.6,
                 label="target Y=250 cm")
    ax_y.grid(True, linestyle="--", alpha=0.4)

    # X vs time
    ax_x.set_title("Lane Adherence — X_global vs Time", fontsize=12, fontweight="bold")
    ax_x.set_xlabel("Elapsed time (s)")
    ax_x.set_ylabel("X global (cm)")
    ax_x.grid(True, linestyle="--", alpha=0.4)
    ax_x.legend(fontsize=8, loc="upper right")

    # Gap
    ax_gap.set_title("Formation Y-Spacing Between Drones", fontsize=12, fontweight="bold")
    ax_gap.set_xlabel("Elapsed time (s)")
    ax_gap.set_ylabel("ΔY between drones (cm)")
    ax_gap.grid(True, linestyle="--", alpha=0.4)

    # ---- legends --------------------------------------------------------
    # Drone colour legend (shared)
    drone_handles = [
        Line2D([0], [0], color=s["color"], marker=s["marker"], markersize=6,
               linewidth=2, label=s["label"])
        for s in DRONE_STYLES.values()
    ]
    spacing_handles = [
        Line2D([0], [0], color=pkw["color"], linewidth=2,
               linestyle=pkw.get("linestyle", "-"), label=pkw["label"])
        for pkw in SPACING_PAIR_STYLES.values()
    ]

    ax_xy.legend(handles=drone_handles, fontsize=8, loc="upper left",
                 title="Drone", title_fontsize=8)
    ax_y.legend(handles=drone_handles + [
        Line2D([0], [0], color="gray", linestyle=":", label="target Y")
    ], fontsize=8, loc="lower right")
    ax_gap.legend(
        handles=spacing_handles + [
            mpatches.Patch(color="gray", alpha=0.2, label=f"±{TOLERANCE_CM} cm tolerance")
        ],
        fontsize=8, loc="upper right"
    )

    # phase legend
    phase_handles = [
        mpatches.Patch(color=c, alpha=0.4, label=ph.replace("_", " "))
        for ph, c in PHASE_COLORS.items()
    ]

    fig.legend(
        handles=trial_legend_handles + phase_handles,
        loc="lower center", ncol=n_trials + len(PHASE_COLORS),
        fontsize=8, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"Front Formation — 3-Drone Position & Spacing Analysis\n{os.path.basename(csv_path)}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    # ---- interactivity --------------------------------------------------
    focus_trial = {"id": None}

    tooltip = fig.text(
        0.01, 0.99, "", va="top", ha="left", fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": "gray", "alpha": 0.85},
    )
    tooltip.set_visible(False)

    def set_focus(target_tid):
        if focus_trial["id"] == target_tid:
            return
        focus_trial["id"] = target_tid
        for tid, artists in trial_artists.items():
            visible = target_tid is None or tid == target_tid
            for a in artists:
                a.set_visible(visible)
        fig.canvas.draw_idle()

    def on_pick(event):
        tid = pick_map.get(id(event.artist))
        if tid is None:
            return
        set_focus(None if focus_trial["id"] == tid else tid)

    def on_hover(event):
        shown = False
        for artist, tid in hover_map.items():
            try:
                hit, _ = artist.contains(event)
            except Exception:
                continue
            if not hit:
                continue
            meta = trial_meta.get(tid, {})
            parts = [f"Trial {tid}"]
            for dn, dm in meta.items():
                parts.append(
                    f"  {dn}: X={dm['x_end']:.0f} Y={dm['y_end']:.0f} cm"
                )
            tooltip.set_text("\n".join(parts))
            tooltip.set_visible(True)
            shown = True
            break
        if not shown:
            tooltip.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    out_png = os.path.splitext(csv_path)[0] + "_formation_plot.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_png}")
    plt.show()


def main():
    csv_path = resolve_csv_path()
    df = load_csv(csv_path)
    build_plot(df, csv_path)


if __name__ == "__main__":
    main()
