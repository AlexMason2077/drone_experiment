import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CSV_FILE = "tello_trajectory_log_missionpad.csv"
#CSV_FILE = "swarm_missionpad_targectory_log.csv"
#CSV_FILE = "tello_trajectory_log_all.csv"
#CSV_FILE = "tello_trajectory_log_different_battery_same_level.csv"
#CSV_FILE = "tello_trajectory_log_same_battery_same_level.csv"
TARGET_HEIGHT = 80
TARGET_DISTANCE = 100

EXPECTED_COLUMNS = [
    "trial_id", "phase", "timestamp", "elapsed_time", "mid",
    "x", "y", "z",
    "X_global", "Y_global", "Z_global",
    "yaw", "vgx", "vgy", "vgz", "battery",
    "battery_number", "takeoff_battery"
]

def build_ideal_trajectory(height=80, distance=100, n=50):
    # takeoff
    z1 = [height * i / (n - 1) for i in range(n)]
    x1 = [0] * n
    y1 = [0] * n

    # forward
    y2 = [distance * i / (n - 1) for i in range(n)]
    x2 = [0] * n
    z2 = [height] * n

    # landing
    z3 = [height - height * i / (n - 1) for i in range(n)]
    x3 = [0] * n
    y3 = [distance] * n

    x = x1 + x2 + x3
    y = y1 + y2 + y3
    z = z1 + z2 + z3
    return x, y, z

def load_csv_robust(csv_file):
    try:
        df = pd.read_csv(
            csv_file,
            on_bad_lines="skip",
            engine="python"
        )
        if "trial_id" in df.columns:
            return df
    except Exception:
        pass

    print("Header not detected correctly. Re-reading CSV without header...")
    df = pd.read_csv(
        csv_file,
        header=None,
        names=EXPECTED_COLUMNS,
        on_bad_lines="skip",
        engine="python"
    )
    return df


def main():
    df = load_csv_robust(CSV_FILE)

    required_cols = ["trial_id", "X_global", "Y_global", "Z_global", "elapsed_time"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep only rows with valid global coordinates
    df = df.dropna(subset=["X_global", "Y_global", "Z_global"]).copy()

    if df.empty:
        print("No valid trajectory data found.")
        return

    # Convert to numeric
    for col in ["X_global", "Y_global", "Z_global", "elapsed_time", "trial_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["trial_id", "X_global", "Y_global", "Z_global", "elapsed_time"])

    if "takeoff_battery" in df.columns:
        df["takeoff_battery"] = pd.to_numeric(df["takeoff_battery"], errors="coerce")

    x_ideal, y_ideal, z_ideal = build_ideal_trajectory(
        height=TARGET_HEIGHT,
        distance=TARGET_DISTANCE,
        n=50
    )

    fig = plt.figure(figsize=(14, 10))

    # 1) 3D
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot(x_ideal, y_ideal, z_ideal, linestyle="--", linewidth=2, label="Ideal Trajectory")

    # 2) XY
    ax2 = fig.add_subplot(222)
    ax2.plot(x_ideal, y_ideal, linestyle="--", linewidth=2, label="Ideal")

    # 3) Z vs time
    ax3 = fig.add_subplot(223)
    ax3.axhline(TARGET_HEIGHT, linestyle="--", linewidth=2, label="Target Height")

    # 4) Y vs time
    ax4 = fig.add_subplot(224)
    ax4.axhline(TARGET_DISTANCE, linestyle="--", linewidth=2, label="Target Y=100")

    trial_ids = sorted(df["trial_id"].unique())
    trial_artists = {}
    trial_meta = {}

    for trial_id in trial_ids:
        traj = df[df["trial_id"] == trial_id].sort_values("elapsed_time")

        x_actual = traj["X_global"].tolist()
        y_actual = traj["Y_global"].tolist()
        z_actual = traj["Z_global"].tolist()
        t_actual = traj["elapsed_time"].tolist()

        if len(x_actual) == 0:
            continue

        battery_number = "Unknown"
        if "battery_number" in traj.columns:
            bn = traj["battery_number"].dropna().astype(str)
            if not bn.empty and bn.iloc[0].strip() != "":
                battery_number = bn.iloc[0]

        takeoff_battery = "Unknown"
        if "takeoff_battery" in traj.columns:
            tb = traj["takeoff_battery"].dropna()
            if not tb.empty:
                takeoff_battery = f"{int(tb.iloc[0])}%"
        elif "battery" in traj.columns:
            b = pd.to_numeric(traj["battery"], errors="coerce").dropna()
            if not b.empty:
                takeoff_battery = f"{int(b.iloc[0])}%"

        trial_meta[trial_id] = {
            "battery_number": battery_number,
            "takeoff_battery": takeoff_battery,
        }

        label = f"Trial {int(trial_id)}"
        line1 = ax1.plot(x_actual, y_actual, z_actual, marker="o", label=label)[0]
        start1 = ax1.scatter(x_actual[0], y_actual[0], z_actual[0], marker="o", s=60)
        end1 = ax1.scatter(x_actual[-1], y_actual[-1], z_actual[-1], marker="x", s=80)

        line2 = ax2.plot(x_actual, y_actual, marker="o", label=label)[0]
        line3 = ax3.plot(t_actual, z_actual, marker="o", label=label)[0]
        line4 = ax4.plot(t_actual, y_actual, marker="o", label=label)[0]

        trial_artists[trial_id] = [line1, start1, end1, line2, line3, line4]

    ax1.set_title("3D Trajectory: Actual vs Ideal")
    ax1.set_xlabel("X (cm)")
    ax1.set_ylabel("Y (cm)")
    ax1.set_zlabel("Z (cm)")
    leg1 = ax1.legend()

    ax2.scatter(0, 0, s=100, marker="x", label="Pad1 Center")
    ax2.scatter(0, 100, s=100, marker="x", label="Pad2 Center")
    ax2.set_title("Top View (X-Y)")
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Y (cm)")
    ax2.grid(True)
    ax2.axis("equal")
    leg2 = ax2.legend()

    ax3.set_title("Height vs Time")
    ax3.set_xlabel("Elapsed Time (s)")
    ax3.set_ylabel("Z (cm)")
    ax3.grid(True)
    leg3 = ax3.legend()

    ax4.set_title("Forward Progress vs Time")
    ax4.set_xlabel("Elapsed Time (s)")
    ax4.set_ylabel("Y (cm)")
    ax4.grid(True)
    leg4 = ax4.legend()

    focus_trial = {"id": None}

    def set_focus_trial(target_trial_id):
        if focus_trial["id"] == target_trial_id:
            return

        focus_trial["id"] = target_trial_id
        for tid, artists in trial_artists.items():
            show = (target_trial_id is None) or (tid == target_trial_id)
            for artist in artists:
                artist.set_visible(show)
        fig.canvas.draw_idle()

    trial_label_to_id = {f"Trial {int(tid)}": tid for tid in trial_artists.keys()}
    pick_artist_to_trial = {}
    hover_artist_to_trial = {}
    tooltip = fig.text(
        0.02,
        0.98,
        "",
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.9},
    )
    tooltip.set_visible(False)

    def register_legend_pick(legend):
        handles = getattr(legend, "legendHandles", None)
        if handles is None:
            handles = getattr(legend, "legend_handles", [])
        texts = legend.get_texts()
        n = min(len(handles), len(texts))

        for i in range(n):
            label = texts[i].get_text()
            tid = trial_label_to_id.get(label)
            if tid is None:
                continue

            texts[i].set_picker(True)
            pick_artist_to_trial[id(texts[i])] = tid
            hover_artist_to_trial[texts[i]] = tid

            try:
                handles[i].set_picker(True)
                pick_artist_to_trial[id(handles[i])] = tid
                hover_artist_to_trial[handles[i]] = tid
            except Exception:
                pass

    for lg in [leg1, leg2, leg3, leg4]:
        register_legend_pick(lg)

    def on_pick(event):
        tid = pick_artist_to_trial.get(id(event.artist))
        if tid is None:
            return
        if focus_trial["id"] == tid:
            set_focus_trial(None)
        else:
            set_focus_trial(tid)

    def on_hover(event):
        shown = False
        for artist, tid in hover_artist_to_trial.items():
            try:
                hit, _ = artist.contains(event)
            except Exception:
                continue
            if not hit:
                continue

            meta = trial_meta.get(tid, {})
            tooltip.set_text(
                f"Trial {int(tid)}\n"
                f"Battery No: {meta.get('battery_number', 'Unknown')}\n"
                f"Takeoff Battery: {meta.get('takeoff_battery', 'Unknown')}"
            )
            x = min(max(event.x / fig.bbox.width + 0.01, 0.01), 0.75)
            y = min(max(event.y / fig.bbox.height + 0.01, 0.01), 0.95)
            tooltip.set_position((x, y))
            tooltip.set_visible(True)
            shown = True
            break

        if not shown and tooltip.get_visible():
            tooltip.set_visible(False)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
