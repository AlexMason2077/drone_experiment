import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


LOG_DIR = "swarm_aruco_logs"

EXPECTED_COLUMNS = [
    "timestamp_iso",
    "elapsed_time_s",
    "phase",
    "leader_detected",
    "leader_x_cm",
    "leader_y_cm",
    "leader_z_cm",
    "leader_yaw_deg",
    "follower_detected",
    "follower_x_cm",
    "follower_y_cm",
    "follower_z_cm",
    "follower_yaw_deg",
    "spacing_cm",
    "spacing_error_cm",
    "vertical_gap_cm",
]

TARGET_SPACING_CM = 50.0
SMOOTHING_WINDOW = 5


def resolve_csv_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]

    if not os.path.isdir(LOG_DIR):
        raise FileNotFoundError(
            f"Log directory '{LOG_DIR}' not found. Pass a CSV path explicitly."
        )

    candidates = [
        os.path.join(LOG_DIR, name)
        for name in os.listdir(LOG_DIR)
        if name.startswith("swarm_aruco_trail_") and name.endswith(".csv")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No swarm ArUco trail CSV found in '{LOG_DIR}'."
        )

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def load_csv_robust(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        if "elapsed_time_s" in df.columns:
            return df
    except Exception:
        pass

    return pd.read_csv(
        csv_path,
        header=None,
        names=EXPECTED_COLUMNS,
        on_bad_lines="skip",
        engine="python",
    )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "elapsed_time_s",
        "leader_x_cm",
        "leader_y_cm",
        "follower_x_cm",
        "follower_y_cm",
        "spacing_cm",
        "phase",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    for col in [
        "elapsed_time_s",
        "leader_x_cm",
        "leader_y_cm",
        "leader_z_cm",
        "follower_x_cm",
        "follower_y_cm",
        "follower_z_cm",
        "spacing_cm",
        "spacing_error_cm",
        "vertical_gap_cm",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["leader_detected", "follower_detected"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().eq("true")

    df = df.sort_values("elapsed_time_s").copy()
    for prefix in ["leader", "follower"]:
        for axis in ["x_cm", "y_cm", "z_cm"]:
            col = f"{prefix}_{axis}"
            if col in df.columns:
                df[f"{col}_smooth"] = (
                    df[col].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
                )
    return df


def add_start_end_markers(ax, x_col: str, y_col: str, color: str, label_prefix: str, df: pd.DataFrame):
    valid = df.dropna(subset=[x_col, y_col])
    if valid.empty:
        return

    ax.scatter(valid.iloc[0][x_col], valid.iloc[0][y_col], color=color, marker="o", s=70, label=f"{label_prefix} start")
    ax.scatter(valid.iloc[-1][x_col], valid.iloc[-1][y_col], color=color, marker="x", s=90, label=f"{label_prefix} end")


def plot_trail(df: pd.DataFrame, csv_path: str) -> None:
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    leader_xy = df.dropna(subset=["leader_x_cm", "leader_y_cm"])
    follower_xy = df.dropna(subset=["follower_x_cm", "follower_y_cm"])
    spacing_df = df.dropna(subset=["elapsed_time_s", "spacing_cm"])
    z_df = df.dropna(subset=["elapsed_time_s", "leader_z_cm", "follower_z_cm"])
    forward_df = df[df["phase"].isin(["forward_1", "forward_2", "wait_for_global_progress"])].copy()

    if not leader_xy.empty:
        ax1.plot(
            leader_xy["leader_x_cm"],
            leader_xy["leader_y_cm"],
            color="tab:blue",
            linewidth=1,
            alpha=0.25,
            label="Leader raw",
        )
        ax1.plot(
            leader_xy["leader_x_cm_smooth"],
            leader_xy["leader_y_cm_smooth"],
            color="tab:blue",
            linewidth=2,
            label="Leader smooth",
        )
        add_start_end_markers(ax1, "leader_x_cm_smooth", "leader_y_cm_smooth", "tab:blue", "Leader", leader_xy)

    if not follower_xy.empty:
        ax1.plot(
            follower_xy["follower_x_cm"],
            follower_xy["follower_y_cm"],
            color="tab:orange",
            linewidth=1,
            alpha=0.25,
            label="Follower raw",
        )
        ax1.plot(
            follower_xy["follower_x_cm_smooth"],
            follower_xy["follower_y_cm_smooth"],
            color="tab:orange",
            linewidth=2,
            label="Follower smooth",
        )
        add_start_end_markers(ax1, "follower_x_cm_smooth", "follower_y_cm_smooth", "tab:orange", "Follower", follower_xy)

    if not forward_df.empty:
        leader_forward = forward_df.dropna(subset=["leader_x_cm", "leader_y_cm"])
        follower_forward = forward_df.dropna(subset=["follower_x_cm", "follower_y_cm"])

        if len(leader_forward) >= 2:
            ax1.plot(
                [leader_forward.iloc[0]["leader_x_cm"], leader_forward.iloc[-1]["leader_x_cm"]],
                [leader_forward.iloc[0]["leader_y_cm"], leader_forward.iloc[-1]["leader_y_cm"]],
                linestyle="--",
                color="tab:blue",
                alpha=0.6,
                label="Leader forward ref",
            )

        if len(follower_forward) >= 2:
            ax1.plot(
                [follower_forward.iloc[0]["follower_x_cm"], follower_forward.iloc[-1]["follower_x_cm"]],
                [follower_forward.iloc[0]["follower_y_cm"], follower_forward.iloc[-1]["follower_y_cm"]],
                linestyle="--",
                color="tab:orange",
                alpha=0.6,
                label="Follower forward ref",
            )

    ax1.set_title("Top View Trajectory")
    ax1.set_xlabel("X (cm)")
    ax1.set_ylabel("Y (cm)")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.legend()

    if not spacing_df.empty:
        ax2.plot(
            spacing_df["elapsed_time_s"],
            spacing_df["spacing_cm"],
            color="tab:green",
            linewidth=2,
            label="Measured spacing",
        )
        ax2.axhline(
            TARGET_SPACING_CM,
            linestyle="--",
            color="black",
            linewidth=1.5,
            label="Target spacing 50 cm",
        )

    ax2.set_title("Drone Spacing vs Time")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("Spacing (cm)")
    ax2.grid(True)
    ax2.legend()

    if not z_df.empty:
        ax3.plot(
            z_df["elapsed_time_s"],
            z_df["leader_z_cm"],
            color="tab:blue",
            linewidth=2,
            label="Leader Z",
        )
        ax3.plot(
            z_df["elapsed_time_s"],
            z_df["follower_z_cm"],
            color="tab:orange",
            linewidth=2,
            label="Follower Z",
        )

    ax3.set_title("Height vs Time")
    ax3.set_xlabel("Elapsed Time (s)")
    ax3.set_ylabel("Z (cm)")
    ax3.grid(True)
    ax3.legend()

    if not leader_xy.empty:
        ax4.plot(
            leader_xy["elapsed_time_s"],
            leader_xy["leader_y_cm"],
            color="tab:blue",
            alpha=0.25,
            linewidth=1,
            label="Leader Y raw",
        )
        ax4.plot(
            leader_xy["elapsed_time_s"],
            leader_xy["leader_y_cm_smooth"],
            color="tab:blue",
            linewidth=2,
            label="Leader Y smooth",
        )
    if not follower_xy.empty:
        ax4.plot(
            follower_xy["elapsed_time_s"],
            follower_xy["follower_y_cm"],
            color="tab:orange",
            alpha=0.25,
            linewidth=1,
            label="Follower Y raw",
        )
        ax4.plot(
            follower_xy["elapsed_time_s"],
            follower_xy["follower_y_cm_smooth"],
            color="tab:orange",
            linewidth=2,
            label="Follower Y smooth",
        )

    ax4.set_title("Forward Progress vs Time")
    ax4.set_xlabel("Elapsed Time (s)")
    ax4.set_ylabel("Y (cm)")
    ax4.grid(True)
    ax4.legend()

    fig.suptitle(f"Swarm ArUco Trail\n{os.path.basename(csv_path)}", fontsize=14)
    plt.tight_layout()

    output_png = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {output_png}")
    plt.show()


def main():
    csv_path = resolve_csv_path()
    print(f"Using CSV: {csv_path}")
    df = load_csv_robust(csv_path)
    df = prepare_dataframe(df)
    plot_trail(df, csv_path)


if __name__ == "__main__":
    main()
