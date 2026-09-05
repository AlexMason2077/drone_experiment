import argparse
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = BASE_DIR / "database"
REGISTRY_FILE = DATA_DIR / "experiment_registry.json"


def load_registry():
    if not REGISTRY_FILE.exists():
        return {"experiments": []}
    try:
        with REGISTRY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"experiments": []}
    if not isinstance(data, dict):
        return {"experiments": []}
    data.setdefault("experiments", [])
    return data


def outlier_experiment_ids():
    registry = load_registry()
    return {
        str(experiment.get("experiment_id", ""))
        for experiment in registry.get("experiments", [])
        if experiment.get("is_outlier")
    }


def latest_file(folder, pattern):
    # Archive filenames contain the run timestamp.  Sorting by name keeps the
    # newest run stable even when an older CSV is corrected and its mtime changes.
    files = sorted(folder.glob(pattern), key=lambda path: path.name, reverse=True)
    return files[0] if files else None


def read_csv(path):
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df.columns = df.columns.str.strip()
    for col in [
        "hover_elapsed_time", "node_elapsed_time", "elapsed_time", "X_global", "Y_global", "Z_global",
        "target_x", "target_y", "target_z", "battery", "battery_hover_start",
        "battery_hover_end", "battery_drop", "position_error_dist",
        "mean_spacing_error", "max_spacing_error", "templ", "temph",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def battery_summary_from_coordination(coord_df):
    required = {"drone_name", "battery"}
    if coord_df.empty or not required.issubset(coord_df.columns):
        return pd.DataFrame()

    rows = []
    for drone_name, group in coord_df.groupby("drone_name"):
        group = group.copy()
        group["battery"] = pd.to_numeric(group["battery"], errors="coerce")
        group = group[group["battery"].notna()]
        if group.empty:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        start_battery = first["battery"]
        end_battery = last["battery"]
        row = {}
        for col in [
            "run_id", "experiment_id", "formation", "wind_direction", "wind_speed",
            "inter_drone_distance_cm", "drone_name", "drone_ip", "battery_id",
            "takeoff_order", "drone_role", "mission_pad", "grid_column", "grid_row",
            "target_pad", "node_forward_distance_cm", "node_speed_cm_s",
        ]:
            if col in group.columns:
                row[col] = last.get(col, first.get(col, ""))
        if "timestamp" in group.columns:
            row["node_start_timestamp"] = first.get("timestamp", "")
            row["node_end_timestamp"] = last.get("timestamp", "")
        if "node_elapsed_time" in group.columns:
            node_times = pd.to_numeric(group["node_elapsed_time"], errors="coerce")
            row["node_duration_sec"] = node_times.max()
        row["battery_hover_start"] = start_battery
        row["battery_hover_end"] = end_battery
        row["battery_drop"] = start_battery - end_battery
        rows.append(row)
    return pd.DataFrame(rows)


def time_column(df):
    if "node_elapsed_time" in df.columns and df["node_elapsed_time"].notna().any():
        return "node_elapsed_time", "Node-to-node flight time (s)"
    return "hover_elapsed_time", "Hover time (s)"


def ensure_plots_dir(experiment_dir):
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def clean_label(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def battery_id_for_drone(battery_df, drone_name):
    if battery_df.empty or "battery_id" not in battery_df.columns or "drone_name" not in battery_df.columns:
        return ""
    matches = battery_df[battery_df["drone_name"] == drone_name]["battery_id"].dropna()
    return clean_label(matches.iloc[0]) if not matches.empty else ""


def drone_battery_label(drone_name, battery_id):
    return f"{drone_name} / {battery_id}" if battery_id else drone_name


def add_discharge_rate(df):
    df = df.copy()
    if "battery_drop" in df.columns:
        df["battery_drop"] = pd.to_numeric(df["battery_drop"], errors="coerce")
    duration_col = None
    for col in ["node_duration_sec", "hover_duration_sec"]:
        if col in df.columns:
            duration_col = col
            df[col] = pd.to_numeric(df[col], errors="coerce")
            break
    if duration_col:
        df["battery_drop_per_min"] = df["battery_drop"] / (df[duration_col] / 60.0)
        df.loc[df[duration_col] <= 0, "battery_drop_per_min"] = pd.NA
    return df


def plot_all_trajectory(coord_df, plots_dir):
    if coord_df.empty:
        return None
    t_col, t_label = time_column(coord_df)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_xy, ax_x, ax_y, ax_z = axes.flatten()
    for drone_name, group in coord_df.groupby("drone_name"):
        group = group.sort_values(t_col)
        ax_xy.plot(group["X_global"], group["Y_global"], marker="o", markersize=2, linewidth=1.4, label=drone_name)
        ax_x.plot(group[t_col], group["X_global"], linewidth=1.4, label=drone_name)
        ax_y.plot(group[t_col], group["Y_global"], linewidth=1.4, label=drone_name)
        ax_z.plot(group[t_col], group["Z_global"], linewidth=1.4, label=drone_name)
        target = group.dropna(subset=["target_x", "target_y"]).head(1)
        if not target.empty:
            ax_xy.scatter(target["target_x"].iloc[0], target["target_y"].iloc[0], marker="x", s=80)

    ax_xy.set_title("All drones: top-view position")
    ax_xy.set_xlabel("X global (cm)")
    ax_xy.set_ylabel("Y global (cm)")
    ax_xy.axis("equal")
    ax_x.set_title(f"X global vs {t_label}")
    ax_y.set_title(f"Y global vs {t_label}")
    ax_z.set_title(f"Z global vs {t_label}")
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
    t_col, t_label = time_column(coord_df) if not coord_df.empty else ("hover_elapsed_time", "Hover time (s)")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_curve, ax_consumed, ax_drop, ax_share = axes.flatten()
    if not coord_df.empty and "battery" in coord_df.columns:
        for drone_name, group in coord_df.groupby("drone_name"):
            group = group.sort_values(t_col)
            battery_id = battery_id_for_drone(battery_df, drone_name)
            label = drone_battery_label(drone_name, battery_id)
            ax_curve.plot(
                group[t_col],
                group["battery"],
                linewidth=1.8,
                marker="o",
                markersize=2,
                label=label,
            )
            start_battery = group["battery"].dropna().iloc[0] if not group["battery"].dropna().empty else None
            if start_battery is not None:
                ax_consumed.plot(
                    group[t_col],
                    start_battery - group["battery"],
                    linewidth=1.8,
                    marker="o",
                    markersize=2,
                    label=label,
                )
    ax_curve.set_title("All five drones: battery percentage during node-to-node flight")
    ax_curve.set_xlabel(t_label)
    ax_curve.set_ylabel("Battery (%)")
    ax_curve.grid(True, linestyle="--", alpha=0.35)
    ax_curve.legend(fontsize=8)

    ax_consumed.set_title("All five drones: battery consumed during node-to-node flight")
    ax_consumed.set_xlabel(t_label)
    ax_consumed.set_ylabel("Battery consumed (%)")
    ax_consumed.grid(True, linestyle="--", alpha=0.35)
    ax_consumed.legend(fontsize=8)

    if not battery_df.empty and "battery_drop" in battery_df.columns:
        battery_df = add_discharge_rate(battery_df).sort_values("takeoff_order")
        labels = [
            drone_battery_label(clean_label(row["drone_name"]), clean_label(row.get("battery_id", "")))
            for _, row in battery_df.iterrows()
        ]
        ax_drop.bar(labels, battery_df["battery_drop"], color="#216c5f")
        total_drop = battery_df["battery_drop"].sum()
        shares = (battery_df["battery_drop"] / total_drop * 100.0) if total_drop else battery_df["battery_drop"] * 0
        ax_share.bar(labels, shares, color="#2f80a8")
        for idx, value in enumerate(battery_df["battery_drop"]):
            ax_drop.text(idx, value, f"{value:g}%", ha="center", va="bottom", fontsize=8)
        if "battery_drop_per_min" in battery_df.columns:
            for idx, value in enumerate(battery_df["battery_drop_per_min"]):
                if pd.notna(value):
                    ax_drop.text(idx, 0, f"{value:.1f}%/min", ha="center", va="bottom", fontsize=7, rotation=90)
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


def plot_all_temperature(coord_df, plots_dir):
    required = {"drone_name", "templ", "temph"}
    if coord_df.empty or not required.issubset(coord_df.columns):
        return None
    t_col, t_label = time_column(coord_df)
    df = coord_df.copy()
    df["temperature_avg"] = (df["templ"] + df["temph"]) / 2
    df = df.dropna(subset=[t_col, "temperature_avg", "drone_name"])
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    for drone_name, group in df.groupby("drone_name"):
        group = group.sort_values(t_col)
        ax.plot(group[t_col], group["temperature_avg"], linewidth=1.8, label=drone_name)

    ax.set_title("All five drones: average temperature during node-to-node flight")
    ax.set_xlabel(t_label)
    ax.set_ylabel("Temperature (C)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = plots_dir / "all_temperature_overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_single_drone(coord_df, battery_df, plots_dir, drone_name):
    if coord_df.empty:
        return None
    t_col, t_label = time_column(coord_df)
    coord_df = coord_df.sort_values(t_col)
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
            ax_pos.plot(coord_df[t_col], coord_df[col], label=col)
    ax_pos.set_title(f"Position vs {t_label}")
    ax_pos.set_xlabel(t_label)
    ax_pos.legend(fontsize=8)

    if "position_error_dist" in coord_df.columns:
        ax_err.plot(coord_df[t_col], coord_df["position_error_dist"], color="#9a5b18")
    ax_err.set_title("Position error distance")
    ax_err.set_xlabel(t_label)
    ax_err.set_ylabel("cm")

    if "battery" in coord_df.columns:
        ax_battery.plot(coord_df[t_col], coord_df["battery"], color="#216c5f")
    if not battery_df.empty:
        battery_id = clean_label(battery_df["battery_id"].iloc[0]) if "battery_id" in battery_df.columns else ""
        label_parts = []
        if battery_id:
            label_parts.append(f"battery={battery_id}")
        label_parts.append(f"drop={battery_df['battery_drop'].iloc[0]}%")
        if "node_duration_sec" in battery_df.columns:
            duration = pd.to_numeric(battery_df["node_duration_sec"], errors="coerce").iloc[0]
            drop = pd.to_numeric(battery_df["battery_drop"], errors="coerce").iloc[0]
            if pd.notna(duration) and duration > 0 and pd.notna(drop):
                label_parts.append(f"rate={drop / (duration / 60.0):.1f}%/min")
        label = "\n".join(label_parts)
        ax_battery.text(0.02, 0.94, label, transform=ax_battery.transAxes)
    ax_battery.set_title("Battery during node-to-node flight")
    ax_battery.set_xlabel(t_label)
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
    for out in [
        plot_all_trajectory(coord_df, plots_dir),
        plot_all_battery(coord_df, battery_df, plots_dir),
        plot_all_temperature(coord_df, plots_dir),
    ]:
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
    outliers = outlier_experiment_ids()
    return sorted(
        path for path in DATA_DIR.glob(f"{condition_key}_*")
        if path.is_dir() and path.name != f"{condition_key}_summary" and path.name.rsplit("_", 1)[-1].isdigit()
        and path.name not in outliers
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
        if battery_df.empty and not coord_df.empty:
            battery_df = battery_summary_from_coordination(coord_df)
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
        outputs.append(plot_condition_battery_lines(all_coord, condition_key, plots_dir))
        outputs.append(plot_condition_temperature_lines(all_coord, condition_key, plots_dir))
        outputs.append(plot_condition_position_summary(all_coord, condition_key, plots_dir))
    return [output for output in outputs if output]


def plot_condition_battery_lines(df, condition_key, plots_dir):
    t_col, t_label = time_column(df)
    required = {"trial_id", "drone_name", t_col, "battery"}
    if df.empty or not required.issubset(df.columns):
        return None

    df = df.copy()
    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    df["battery"] = pd.to_numeric(df["battery"], errors="coerce")
    df = df.dropna(subset=[t_col, "battery", "drone_name", "trial_id"])
    if df.empty:
        return None

    drone_names = sorted(df["drone_name"].dropna().unique())
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, drone_name in zip(axes, drone_names):
        drone_df = df[df["drone_name"] == drone_name]
        for trial_id, group in drone_df.groupby("trial_id"):
            group = group.sort_values(t_col)
            ax.plot(
                group[t_col],
                group["battery"],
                linewidth=1.5,
                label=f"trial {trial_id}",
            )
        ax.set_title(f"{drone_name}: battery percentage")
        ax.set_xlabel(t_label)
        ax.set_ylabel("Battery (%)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

    for ax in axes[len(drone_names):]:
        ax.axis("off")

    fig.suptitle(f"{condition_key}: all drone battery percentage lines", fontsize=14)
    fig.tight_layout()
    out = plots_dir / "condition_battery_lines.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_condition_temperature_lines(df, condition_key, plots_dir):
    t_col, t_label = time_column(df)
    required = {"trial_id", "drone_name", t_col, "templ", "temph"}
    if df.empty or not required.issubset(df.columns):
        return None

    df = df.copy()
    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    df["templ"] = pd.to_numeric(df["templ"], errors="coerce")
    df["temph"] = pd.to_numeric(df["temph"], errors="coerce")
    df["temperature_avg"] = (df["templ"] + df["temph"]) / 2
    df = df.dropna(subset=[t_col, "temperature_avg", "drone_name", "trial_id"])
    if df.empty:
        return None

    drone_names = sorted(df["drone_name"].dropna().unique())
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, drone_name in zip(axes, drone_names):
        drone_df = df[df["drone_name"] == drone_name]
        for trial_id, group in drone_df.groupby("trial_id"):
            group = group.sort_values(t_col)
            ax.plot(
                group[t_col],
                group["temperature_avg"],
                linewidth=1.5,
                label=f"trial {trial_id}",
            )
        ax.set_title(f"{drone_name}: average temperature")
        ax.set_xlabel(t_label)
        ax.set_ylabel("Temperature (C)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

    for ax in axes[len(drone_names):]:
        ax.axis("off")

    fig.suptitle(f"{condition_key}: all drone temperature lines", fontsize=14)
    fig.tight_layout()
    out = plots_dir / "condition_temperature_lines.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_condition_battery_summary(df, condition_key, plots_dir):
    df = add_discharge_rate(df)
    df["battery_drop"] = pd.to_numeric(df["battery_drop"], errors="coerce")
    source_out = plots_dir / "condition_battery_summary_source.csv"
    df.to_csv(source_out, index=False)
    grouped = df.groupby("drone_name")["battery_drop"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index().sort_values("drone_name")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax_trials, ax_mean = axes
    for drone_name, group in df.groupby("drone_name"):
        group = group.sort_values("trial_id")
        labels = []
        for _, row in group.iterrows():
            battery_id = clean_label(row.get("battery_id", ""))
            labels.append(f"{row['trial_id']}\n{battery_id}" if battery_id else row["trial_id"])
        ax_trials.plot(labels, group["battery_drop"], marker="o", linewidth=1.6, label=drone_name)
    ax_trials.set_title(f"{condition_key}: battery drop across trials")
    ax_trials.set_xlabel("Trial / battery")
    ax_trials.set_ylabel("Battery drop (%)")
    ax_trials.tick_params(axis="x", rotation=30)
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


def plot_condition_battery_id_summary(df, condition_key, plots_dir):
    required = {"battery_id", "drone_name", "battery_drop"}
    if df.empty or not required.issubset(df.columns):
        return None

    df = add_discharge_rate(df)
    if "battery_drop_per_min" not in df.columns:
        return None
    df["battery_id"] = df["battery_id"].map(clean_label)
    df["drone_name"] = df["drone_name"].map(clean_label)
    df = df.dropna(subset=["battery_drop", "battery_drop_per_min"])
    df = df[(df["battery_id"] != "") & (df["drone_name"] != "")]
    if df.empty:
        return None

    batteries = sorted(df["battery_id"].unique())
    cols = 2
    rows = max(1, (len(batteries) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, max(5, rows * 4)), sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    drone_order = sorted(df["drone_name"].unique())
    x_lookup = {drone_name: idx for idx, drone_name in enumerate(drone_order)}

    for ax, battery_id in zip(axes, batteries):
        battery_df = df[df["battery_id"] == battery_id].sort_values(["drone_name", "trial_id"])
        for trial_id, group in battery_df.groupby("trial_id"):
            x_values = [x_lookup[name] for name in group["drone_name"]]
            ax.scatter(x_values, group["battery_drop_per_min"], s=55, label=f"trial {trial_id}")
            for x_value, (_, row) in zip(x_values, group.iterrows()):
                ax.text(
                    x_value,
                    row["battery_drop_per_min"],
                    f"{row['battery_drop']:g}%",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )

        means = battery_df.groupby("drone_name")["battery_drop_per_min"].mean().reindex(drone_order)
        ax.plot(range(len(drone_order)), means, color="#216c5f", linewidth=1.8, alpha=0.75, label="mean")
        ax.set_title(f"{battery_id}: discharge rate by drone")
        ax.set_xticks(range(len(drone_order)))
        ax.set_xticklabels(drone_order, rotation=25)
        ax.set_ylabel("Battery drop rate (%/min)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7)

    for ax in axes[len(batteries):]:
        ax.axis("off")

    fig.suptitle(f"{condition_key}: per-battery discharge rate consistency", fontsize=14)
    fig.tight_layout()
    out = plots_dir / "condition_battery_id_discharge_rates.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_condition_battery_id_drop_summary(df, condition_key, plots_dir):
    required = {"battery_id", "drone_name", "battery_drop"}
    if df.empty or not required.issubset(df.columns):
        return None

    df = df.copy()
    df["battery_drop"] = pd.to_numeric(df["battery_drop"], errors="coerce")
    df["battery_id"] = df["battery_id"].map(clean_label)
    df["drone_name"] = df["drone_name"].map(clean_label)
    df = df.dropna(subset=["battery_drop"])
    df = df[(df["battery_id"] != "") & (df["drone_name"] != "")]
    if df.empty:
        return None

    batteries = sorted(df["battery_id"].unique())
    cols = 2
    rows = max(1, (len(batteries) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, max(5, rows * 4)), sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    drone_order = sorted(df["drone_name"].unique())
    x_lookup = {drone_name: idx for idx, drone_name in enumerate(drone_order)}

    for ax, battery_id in zip(axes, batteries):
        battery_df = df[df["battery_id"] == battery_id].sort_values(["drone_name", "trial_id"])
        for trial_id, group in battery_df.groupby("trial_id"):
            x_values = [x_lookup[name] for name in group["drone_name"]]
            ax.scatter(x_values, group["battery_drop"], s=55, label=f"trial {trial_id}")
            for x_value, (_, row) in zip(x_values, group.iterrows()):
                ax.text(x_value, row["battery_drop"], f"{row['battery_drop']:g}%", fontsize=7, ha="center", va="bottom")

        means = battery_df.groupby("drone_name")["battery_drop"].mean().reindex(drone_order)
        ax.plot(range(len(drone_order)), means, color="#216c5f", linewidth=1.8, alpha=0.75, label="mean")
        ax.set_title(f"{battery_id}: battery drop by drone")
        ax.set_xticks(range(len(drone_order)))
        ax.set_xticklabels(drone_order, rotation=25)
        ax.set_ylabel("Battery drop (%)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7)

    for ax in axes[len(batteries):]:
        ax.axis("off")

    fig.suptitle(f"{condition_key}: per-battery consumption by drone", fontsize=14)
    fig.tight_layout()
    out = plots_dir / "condition_battery_id_drops.png"
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
