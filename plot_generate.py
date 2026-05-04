import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "database"


def latest_file(folder, pattern):
    files = sorted(folder.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def read_csv(path):
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df.columns = df.columns.str.strip()
    for col in [
        "hover_elapsed_time", "elapsed_time", "X_global", "Y_global", "Z_global",
        "target_x", "target_y", "target_z", "battery", "battery_hover_start",
        "battery_hover_end", "battery_drop", "position_error_dist",
        "mean_spacing_error", "max_spacing_error",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_plots_dir(experiment_dir):
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_all_trajectory(coord_df, plots_dir):
    if coord_df.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_xy, ax_x, ax_y, ax_z = axes.flatten()
    for drone_name, group in coord_df.groupby("drone_name"):
        group = group.sort_values("hover_elapsed_time")
        ax_xy.plot(group["X_global"], group["Y_global"], marker="o", markersize=2, linewidth=1.4, label=drone_name)
        ax_x.plot(group["hover_elapsed_time"], group["X_global"], linewidth=1.4, label=drone_name)
        ax_y.plot(group["hover_elapsed_time"], group["Y_global"], linewidth=1.4, label=drone_name)
        ax_z.plot(group["hover_elapsed_time"], group["Z_global"], linewidth=1.4, label=drone_name)
        target = group.dropna(subset=["target_x", "target_y"]).head(1)
        if not target.empty:
            ax_xy.scatter(target["target_x"].iloc[0], target["target_y"].iloc[0], marker="x", s=80)

    ax_xy.set_title("All drones: top-view position")
    ax_xy.set_xlabel("X global (cm)")
    ax_xy.set_ylabel("Y global (cm)")
    ax_xy.axis("equal")
    ax_x.set_title("X global vs hover time")
    ax_y.set_title("Y global vs hover time")
    ax_z.set_title("Z global vs hover time")
    for ax in axes.flatten():
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / "all_position_overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_all_battery(coord_df, battery_df, plots_dir):
    if coord_df.empty and battery_df.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_curve, ax_consumed, ax_drop, ax_share = axes.flatten()
    if not coord_df.empty and "battery" in coord_df.columns:
        for drone_name, group in coord_df.groupby("drone_name"):
            group = group.sort_values("hover_elapsed_time")
            ax_curve.plot(
                group["hover_elapsed_time"],
                group["battery"],
                linewidth=1.8,
                marker="o",
                markersize=2,
                label=drone_name,
            )
            start_battery = group["battery"].dropna().iloc[0] if not group["battery"].dropna().empty else None
            if start_battery is not None:
                ax_consumed.plot(
                    group["hover_elapsed_time"],
                    start_battery - group["battery"],
                    linewidth=1.8,
                    marker="o",
                    markersize=2,
                    label=drone_name,
                )
    ax_curve.set_title("All five drones: battery percentage during 60s hover")
    ax_curve.set_xlabel("Hover time (s)")
    ax_curve.set_ylabel("Battery (%)")
    ax_curve.grid(True, linestyle="--", alpha=0.35)
    ax_curve.legend(fontsize=8)

    ax_consumed.set_title("All five drones: battery consumed during hover")
    ax_consumed.set_xlabel("Hover time (s)")
    ax_consumed.set_ylabel("Battery consumed (%)")
    ax_consumed.grid(True, linestyle="--", alpha=0.35)
    ax_consumed.legend(fontsize=8)

    if not battery_df.empty and "battery_drop" in battery_df.columns:
        battery_df = battery_df.sort_values("takeoff_order")
        ax_drop.bar(battery_df["drone_name"], battery_df["battery_drop"], color="#216c5f")
        total_drop = battery_df["battery_drop"].sum()
        shares = (battery_df["battery_drop"] / total_drop * 100.0) if total_drop else battery_df["battery_drop"] * 0
        ax_share.bar(battery_df["drone_name"], shares, color="#2f80a8")
        for idx, value in enumerate(battery_df["battery_drop"]):
            ax_drop.text(idx, value, f"{value:g}%", ha="center", va="bottom", fontsize=8)
        for idx, value in enumerate(shares):
            ax_share.text(idx, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_drop.set_title("Final battery drop comparison")
    ax_drop.set_ylabel("Battery drop (%)")
    ax_drop.tick_params(axis="x", rotation=25)
    ax_drop.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax_share.set_title("Share of total battery consumption")
    ax_share.set_ylabel("Share (%)")
    ax_share.tick_params(axis="x", rotation=25)
    ax_share.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out = plots_dir / "all_battery_overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_single_drone(coord_df, battery_df, plots_dir, drone_name):
    if coord_df.empty:
        return None
    coord_df = coord_df.sort_values("hover_elapsed_time")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_xy, ax_pos, ax_err, ax_battery = axes.flatten()

    ax_xy.plot(coord_df["X_global"], coord_df["Y_global"], marker="o", markersize=2, linewidth=1.3)
    target = coord_df.dropna(subset=["target_x", "target_y"]).head(1)
    if not target.empty:
        ax_xy.scatter(target["target_x"].iloc[0], target["target_y"].iloc[0], marker="x", s=90, label="target")
    ax_xy.set_title(f"{drone_name}: top-view position")
    ax_xy.set_xlabel("X global (cm)")
    ax_xy.set_ylabel("Y global (cm)")
    ax_xy.axis("equal")

    for col in ["X_global", "Y_global", "Z_global"]:
        if col in coord_df.columns:
            ax_pos.plot(coord_df["hover_elapsed_time"], coord_df[col], label=col)
    ax_pos.set_title("Position vs hover time")
    ax_pos.set_xlabel("Hover time (s)")
    ax_pos.legend(fontsize=8)

    if "position_error_dist" in coord_df.columns:
        ax_err.plot(coord_df["hover_elapsed_time"], coord_df["position_error_dist"], color="#9a5b18")
    ax_err.set_title("Position error distance")
    ax_err.set_xlabel("Hover time (s)")
    ax_err.set_ylabel("cm")

    if "battery" in coord_df.columns:
        ax_battery.plot(coord_df["hover_elapsed_time"], coord_df["battery"], color="#216c5f")
    if not battery_df.empty:
        label = f"drop={battery_df['battery_drop'].iloc[0]}%"
        ax_battery.text(0.02, 0.94, label, transform=ax_battery.transAxes)
    ax_battery.set_title("Battery during hover")
    ax_battery.set_xlabel("Hover time (s)")
    ax_battery.set_ylabel("Battery (%)")

    for ax in axes.flatten():
        ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    out = plots_dir / f"{drone_name}_overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def generate_for_experiment(experiment_id):
    experiment_dir = DATA_DIR / experiment_id
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {experiment_dir}")
    plots_dir = ensure_plots_dir(experiment_dir)
    coord_path = latest_file(experiment_dir, "*_all_coordination.csv")
    battery_path = latest_file(experiment_dir, "*_all_battery.csv")
    coord_df = read_csv(coord_path)
    battery_df = read_csv(battery_path)

    outputs = []
    for out in [plot_all_trajectory(coord_df, plots_dir), plot_all_battery(coord_df, battery_df, plots_dir)]:
        if out:
            outputs.append(out)

    drones_dir = experiment_dir / "drones"
    if drones_dir.exists():
        for drone_dir in sorted(path for path in drones_dir.iterdir() if path.is_dir()):
            drone_coord = read_csv(latest_file(drone_dir, "*_coordination.csv"))
            drone_battery = read_csv(latest_file(drone_dir, "*_battery.csv"))
            drone_name = drone_coord["drone_name"].dropna().iloc[0] if not drone_coord.empty else drone_dir.name
            out = plot_single_drone(drone_coord, drone_battery, plots_dir, drone_name)
            if out:
                outputs.append(out)
    return outputs


def experiment_dirs_for_condition(condition_key):
    return sorted(
        path for path in DATA_DIR.glob(f"{condition_key}_*")
        if path.is_dir() and path.name != f"{condition_key}_summary" and path.name.rsplit("_", 1)[-1].isdigit()
    )


def generate_for_condition(condition_key):
    trial_dirs = experiment_dirs_for_condition(condition_key)
    if not trial_dirs:
        raise FileNotFoundError(f"No trial folders found for condition: {condition_key}")

    summary_dir = DATA_DIR / f"{condition_key}_summary"
    plots_dir = summary_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    battery_frames = []
    coord_frames = []
    for trial_dir in trial_dirs:
        trial_id = trial_dir.name.rsplit("_", 1)[-1]
        battery_df = read_csv(latest_file(trial_dir, "*_all_battery.csv"))
        coord_df = read_csv(latest_file(trial_dir, "*_all_coordination.csv"))
        if not battery_df.empty:
            battery_df["trial_id"] = trial_id
            battery_frames.append(battery_df)
        if not coord_df.empty:
            coord_df["trial_id"] = trial_id
            coord_frames.append(coord_df)

    outputs = []
    if battery_frames:
        all_battery = pd.concat(battery_frames, ignore_index=True)
        outputs.append(plot_condition_battery_summary(all_battery, condition_key, plots_dir))
    if coord_frames:
        all_coord = pd.concat(coord_frames, ignore_index=True)
        outputs.append(plot_condition_position_summary(all_coord, condition_key, plots_dir))
    return [output for output in outputs if output]


def plot_condition_battery_summary(df, condition_key, plots_dir):
    df["battery_drop"] = pd.to_numeric(df["battery_drop"], errors="coerce")
    grouped = df.groupby("drone_name")["battery_drop"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index().sort_values("drone_name")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax_trials, ax_mean = axes
    for drone_name, group in df.groupby("drone_name"):
        group = group.sort_values("trial_id")
        ax_trials.plot(group["trial_id"], group["battery_drop"], marker="o", linewidth=1.6, label=drone_name)
    ax_trials.set_title(f"{condition_key}: battery drop across trials")
    ax_trials.set_xlabel("Trial")
    ax_trials.set_ylabel("Battery drop (%)")
    ax_trials.grid(True, linestyle="--", alpha=0.35)
    ax_trials.legend(fontsize=8)

    ax_mean.bar(summary["drone_name"], summary["mean"], yerr=summary["std"].fillna(0), capsize=4, color="#216c5f")
    ax_mean.set_title("Mean battery drop by drone")
    ax_mean.set_ylabel("Mean battery drop (%)")
    ax_mean.tick_params(axis="x", rotation=25)
    ax_mean.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out = plots_dir / "condition_battery_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_condition_position_summary(df, condition_key, plots_dir):
    for col in ["position_error_dist", "mean_spacing_error", "max_spacing_error"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax_error, ax_spacing = axes
    if "position_error_dist" in df.columns:
        summary = df.groupby(["trial_id", "drone_name"])["position_error_dist"].mean().reset_index()
        for drone_name, group in summary.groupby("drone_name"):
            group = group.sort_values("trial_id")
            ax_error.plot(group["trial_id"], group["position_error_dist"], marker="o", linewidth=1.5, label=drone_name)
    ax_error.set_title(f"{condition_key}: mean position error by trial")
    ax_error.set_xlabel("Trial")
    ax_error.set_ylabel("Mean position error (cm)")
    ax_error.grid(True, linestyle="--", alpha=0.35)
    ax_error.legend(fontsize=8)

    trial_spacing = df.groupby("trial_id")[["mean_spacing_error", "max_spacing_error"]].mean(numeric_only=True).reset_index()
    if "mean_spacing_error" in trial_spacing:
        ax_spacing.plot(trial_spacing["trial_id"], trial_spacing["mean_spacing_error"], marker="o", label="mean spacing error")
    if "max_spacing_error" in trial_spacing:
        ax_spacing.plot(trial_spacing["trial_id"], trial_spacing["max_spacing_error"], marker="s", label="max spacing error")
    ax_spacing.set_title("Formation spacing error across trials")
    ax_spacing.set_xlabel("Trial")
    ax_spacing.set_ylabel("Spacing error (cm)")
    ax_spacing.grid(True, linestyle="--", alpha=0.35)
    ax_spacing.legend(fontsize=8)

    fig.tight_layout()
    out = plots_dir / "condition_position_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate plots for an archived experiment.")
    parser.add_argument("--experiment-id")
    parser.add_argument("--condition-key")
    args = parser.parse_args()
    if args.condition_key:
        outputs = generate_for_condition(args.condition_key)
    elif args.experiment_id:
        outputs = generate_for_experiment(args.experiment_id)
    else:
        raise SystemExit("Pass --experiment-id or --condition-key")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
