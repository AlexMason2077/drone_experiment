"""
Collect five-drone node-to-node flight data from an experiment record.

This script is designed to be started by app.py with an experiment ID:

    python3 -u data_collector.py --experiment-id EXP_ID

It reads database/experiment_registry.json, validates the saved drone IPs and
mission pad positions, connects to the Tello swarm, waits for GUI confirmation
at the takeoff prompt, then takes off, climbs over each configured start mission
pad, flies forward 250 cm along the lane at 10 cm/s, logs coordination and
battery data during the node-to-node flight, and lands.
"""

import argparse
import csv
import json
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from djitellopy import TelloSwarm


DATA_DIR = BASE_DIR / "database"
REGISTRY_FILE = DATA_DIR / "experiment_registry.json"

IP_PREFIX = "192.168.0."
ROW_SPACING_CM = 50
COLUMN_SPACING_CM = 50
TAKEOFF_HEIGHT_CM = 80
LOG_INTERVAL_SEC = 0.1
TAKEOFF_CLIMB_SPEED_CM_S = 20
NODE_FLIGHT_SPEED_CM_S = 10
NODE_FORWARD_DISTANCE_CM = 250
NODE_SEGMENT_DISTANCE_CM = ROW_SPACING_CM
LONG_GO_RESPONSE_TIMEOUT_SEC = 40
PRE_NODE_SETTLE_SEC = 2.0
TAKEOFF_HEIGHT_ADJUST_MIN_CM = 20
GROUP_PAD_WAIT_REPORT_INTERVAL_SEC = 3.0
SEGMENT_STAGGER_DELAY_SEC = 0.2
COLUMN_WIND_STAGGER_DELAY_SEC = 0.5
COLUMN_WIND_STAGGER_DELAYS_SEC = [0.0, 1.0, 1.7, 2.4, 3.0]
COLUMN_TARGET_SPACING_CM = 50.0
COLUMN_SAFETY_RELEASE_SPACING_CM = COLUMN_TARGET_SPACING_CM + 5.0
COLUMN_SAFETY_WAIT_TIMEOUT_SEC = 20.0
COLUMN_SAFETY_REPORT_INTERVAL_SEC = 2.0
VEE_COLUMN_DETECTION_TOLERANCE_CM = 25
SEGMENT_TARGET_REPORT_INTERVAL_SEC = 3.0
PAD_LOCK_MIN_HITS = 2
PAD_LOCK_GRACE_SEC = 0.75
START_PAD_LOCK_TIMEOUT_SEC = 15.0
TARGET_PAD_LOCK_TIMEOUT_SEC = 12.0
SEGMENT_TARGET_TOLERANCE_CM = 15
BATTERY_WINDOW_LOW_PERCENT = 20
BATTERY_WINDOW_HIGH_PERCENT = 75
DISCHARGE_CHECK_INTERVAL_SEC = 8.0
DISCHARGE_MAX_DURATION_SEC = 900.0
PAD_SEQUENCE = [1, 2, 3, 4, 5, 6, 7, 8]

MISSION_PAD_COLUMNS = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8, 1],
    [5, 6, 7, 8, 1, 2],
]
COLUMN_MISSION_PAD_COLUMNS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 3, 4],
    [],
    [],
    [],
    [],
]
VEE_MISSION_PAD_COLUMNS = [
    [1, 2, 3, 4, 5, 6],
    [5, 6, 7, 8, 1, 2],
    [3, 4, 5, 6, 7, 8],
    [5, 6, 7, 8, 1, 2],
    [1, 2, 3, 4, 7, 8],
]
VEE_COLUMN_ORIGINS_CM = [
    (0.0, 0.0),
    (25 * math.sqrt(2), 25 * math.sqrt(2)),
    (50 * math.sqrt(2), 50 * math.sqrt(2)),
    (75 * math.sqrt(2), 25 * math.sqrt(2)),
    (100 * math.sqrt(2), 0.0),
]
VEE_75_COLUMN_ORIGINS_CM = [
    (0.0, 0.0),
    (35 * math.sqrt(2), 35 * math.sqrt(2)),
    (75 * math.sqrt(2), 75 * math.sqrt(2)),
    (115 * math.sqrt(2), 35 * math.sqrt(2)),
    (150 * math.sqrt(2), 0.0),
]
ECHALON_COLUMN_ORIGINS_CM = [
    (0.0, 150 * math.sqrt(2)),
    (35 * math.sqrt(2), 115 * math.sqrt(2)),
    (75 * math.sqrt(2), 75 * math.sqrt(2)),
    (115 * math.sqrt(2), 35 * math.sqrt(2)),
    (150 * math.sqrt(2), 0.0),
]
DIAMOND_MISSION_PAD_COLUMNS = [
    [],
    [1, 2, 3, 4, 5, 6, 7, 8],
    [2, 3, 4, 5, 6, 7, 8, 1],
    [3, 4, 5, 6, 7, 8, 1, 2],
    [],
]

COORDINATION_COLUMNS = [
    "run_id",
    "experiment_id",
    "formation",
    "wind_direction",
    "wind_speed",
    "inter_drone_distance_cm",
    "drone_name",
    "drone_ip",
    "battery_id",
    "takeoff_order",
    "drone_role",
    "mission_pad",
    "grid_column",
    "grid_row",
    "phase",
    "timestamp",
    "elapsed_time",
    "hover_elapsed_time",
    "node_elapsed_time",
    "mid",
    "x",
    "y",
    "z",
    "X_global",
    "Y_global",
    "Z_global",
    "target_x",
    "target_y",
    "target_z",
    "target_pad",
    "node_forward_distance_cm",
    "node_speed_cm_s",
    "position_error_x",
    "position_error_y",
    "position_error_z",
    "position_error_dist",
    "mean_spacing_error",
    "max_spacing_error",
    "battery",
    "battery_hover_start",
    "battery_hover_end",
    "yaw",
    "pitch",
    "roll",
    "vgx",
    "vgy",
    "vgz",
    "agx",
    "agy",
    "agz",
    "templ",
    "temph",
    "tof",
    "h",
    "baro",
    "motor_time",
]

BATTERY_COLUMNS = [
    "run_id",
    "experiment_id",
    "formation",
    "wind_direction",
    "wind_speed",
    "inter_drone_distance_cm",
    "drone_name",
    "drone_ip",
    "battery_id",
    "takeoff_order",
    "drone_role",
    "mission_pad",
    "grid_column",
    "grid_row",
    "hover_start_timestamp",
    "hover_end_timestamp",
    "hover_duration_sec",
    "node_start_timestamp",
    "node_end_timestamp",
    "node_duration_sec",
    "target_pad",
    "node_forward_distance_cm",
    "node_speed_cm_s",
    "battery_hover_start",
    "battery_hover_end",
    "battery_drop",
]

BATTERY_TIMESERIES_COLUMNS = [
    "run_id",
    "experiment_id",
    "formation",
    "wind_direction",
    "wind_speed",
    "inter_drone_distance_cm",
    "drone_name",
    "drone_ip",
    "battery_id",
    "takeoff_order",
    "drone_role",
    "mission_pad",
    "target_pad",
    "phase",
    "timestamp",
    "elapsed_time",
    "node_elapsed_time",
    "battery",
    "battery_start",
    "battery_drop_from_start",
]


phase_lock = threading.Lock()
logging_active = False
current_phases = []
hover_start_batteries = {}
hover_end_batteries = {}


class ExperimentStopped(Exception):
    pass


def load_experiment(experiment_id, registry_file=REGISTRY_FILE):
    if not registry_file.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_file}")
    with registry_file.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    for experiment in registry.get("experiments", []):
        if experiment.get("experiment_id") == experiment_id:
            return experiment
    raise ValueError(f"Experiment ID not found in registry: {experiment_id}")


def normalize_ip(value):
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith(IP_PREFIX):
        return text
    return f"{IP_PREFIX}{text}"


def int_field(value, field_name):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value}") from exc


def experiment_inter_drone_distance_cm(experiment):
    return int_field(experiment.get("inter_drone_distance_cm", 50), "inter_drone_distance_cm")


def clamp(value, low, high):
    return max(low, min(high, value))


def mission_pad_columns_for_experiment(experiment):
    formation = str(experiment.get("formation", "")).strip().lower()
    if formation == "column":
        return COLUMN_MISSION_PAD_COLUMNS
    if formation == "diamond":
        return DIAMOND_MISSION_PAD_COLUMNS
    if is_vee_75cm_experiment(experiment):
        return MISSION_PAD_COLUMNS
    if formation == "vee":
        return VEE_MISSION_PAD_COLUMNS
    return MISSION_PAD_COLUMNS


def experiment_column_spacing_cm(experiment):
    formation = str(experiment.get("formation", "")).strip().lower()
    if formation in {"front", "vee"} and experiment_inter_drone_distance_cm(experiment) == 75:
        return 75
    return COLUMN_SPACING_CM


def is_echalon_formation(formation):
    return str(formation or "").strip().lower() in {"echalon", "echelon", "echolon"}


def is_vee_75cm_experiment(experiment):
    formation = str(experiment.get("formation", "")).strip().lower()
    return formation == "vee" and experiment_inter_drone_distance_cm(experiment) == 75


def is_vee_75cm_config(config):
    return (
        str(config.get("formation", "")).strip().lower() == "vee"
        and int(config.get("inter_drone_distance_cm") or 50) == 75
    )


def is_front_head_75cm_experiment(experiment):
    formation = str(experiment.get("formation", "")).strip().lower()
    wind_direction = str(experiment.get("wind_direction", "")).strip().lower()
    return (
        (formation == "front" or is_echalon_formation(formation))
        and wind_direction == "head wind"
        and experiment_inter_drone_distance_cm(experiment) == 75
    )


def config_column_spacing_cm(config):
    return int(config.get("column_spacing_cm") or COLUMN_SPACING_CM)


def position_at_column_row(formation, grid_column, row_idx, column_spacing_cm=COLUMN_SPACING_CM):
    if formation == "vee":
        origins = VEE_75_COLUMN_ORIGINS_CM if column_spacing_cm == 75 else VEE_COLUMN_ORIGINS_CM
        origin_x, origin_y = origins[grid_column]
        return origin_x, origin_y + row_idx * ROW_SPACING_CM
    if is_echalon_formation(formation):
        origin_x, origin_y = ECHALON_COLUMN_ORIGINS_CM[grid_column]
        return origin_x, origin_y + row_idx * ROW_SPACING_CM
    return grid_column * column_spacing_cm, row_idx * ROW_SPACING_CM


def lane_pad_sequence(grid_column, min_rows=None, columns=None):
    columns = columns or MISSION_PAD_COLUMNS
    configured_lane = list(columns[grid_column])
    row_count = min_rows or len(configured_lane)
    if row_count <= len(configured_lane):
        return configured_lane[:row_count]
    start_idx = PAD_SEQUENCE.index(configured_lane[-1])
    extension = [
        PAD_SEQUENCE[(start_idx + offset) % len(PAD_SEQUENCE)]
        for offset in range(1, row_count - len(configured_lane) + 1)
    ]
    return configured_lane + extension


def target_pad_for_start(mission_pad, forward_distance_cm=NODE_FORWARD_DISTANCE_CM):
    steps = int(round(forward_distance_cm / ROW_SPACING_CM))
    start_idx = PAD_SEQUENCE.index(mission_pad)
    return PAD_SEQUENCE[(start_idx + steps) % len(PAD_SEQUENCE)]


def node_direction_for_experiment(experiment):
    formation = str(experiment.get("formation", "")).strip().lower()
    wind_direction = str(experiment.get("wind_direction", "")).strip().lower()
    if formation == "column":
        return -1 if wind_direction == "head wind" else 1
    if is_front_head_75cm_experiment(experiment):
        return -1
    # Vee tail-wind runs keep the same pad layout and +Y flight path as head-wind runs.
    # The fan is physically moved to the tail side, so no coordinate reversal is needed.
    if formation == "diamond" and wind_direction == "tail wind":
        return -1
    if formation == "front" and wind_direction == "tail wind" and experiment_inter_drone_distance_cm(experiment) != 75:
        return -1
    return 1


def pad_at_physical_row(config, row_idx):
    if row_idx < 0:
        raise ValueError(f"row_idx cannot be negative: {row_idx}")
    lane = lane_pad_sequence(config["grid_column"], min_rows=row_idx + 1, columns=config.get("mission_pad_columns"))
    return lane[row_idx]


def pad_at_column_row(grid_column, row_idx, columns=None):
    if row_idx < 0:
        raise ValueError(f"row_idx cannot be negative: {row_idx}")
    lane = lane_pad_sequence(grid_column, min_rows=row_idx + 1, columns=columns)
    return lane[row_idx]


def build_tello_configs(experiment):
    drones = experiment.get("drones", [])
    if len(drones) != 5:
        raise ValueError(f"Expected exactly 5 drones in experiment record, got {len(drones)}.")

    formation = str(experiment.get("formation", "")).strip().lower()
    node_direction = node_direction_for_experiment(experiment)
    mission_pad_columns = mission_pad_columns_for_experiment(experiment)
    column_spacing_cm = experiment_column_spacing_cm(experiment)
    default_steps = int(round(NODE_FORWARD_DISTANCE_CM / ROW_SPACING_CM))
    configs = []
    seen_ips = set()
    seen_batteries = set()
    seen_positions = set()
    for idx, drone in enumerate(sorted(drones, key=lambda item: int_field(item.get("takeoff_order", 999), "takeoff_order"))):
        ip = normalize_ip(drone.get("ip") or drone.get("ip_suffix"))
        if not ip:
            raise ValueError(f"Drone {idx + 1} has no IP.")
        if ip in seen_ips:
            raise ValueError(f"Duplicate drone IP: {ip}")
        seen_ips.add(ip)

        battery_id = str(drone.get("battery_id") or "").strip().upper()
        if not battery_id:
            raise ValueError(f"Missing battery_id for {ip}. Select one charged battery for each drone.")
        if battery_id in seen_batteries:
            raise ValueError(f"Duplicate battery assignment: {battery_id}")
        seen_batteries.add(battery_id)

        grid_column = int_field(drone.get("grid_column"), "grid_column")
        grid_row = int_field(drone.get("grid_row"), "grid_row")
        if grid_column < 0 or grid_column >= len(mission_pad_columns):
            raise ValueError(f"grid_column out of range for {ip}: {grid_column}")
        if grid_row < 0 or grid_row >= len(mission_pad_columns[grid_column]):
            raise ValueError(f"grid_row out of range for {ip}: {grid_row}")
        if is_vee_75cm_experiment(experiment) and grid_row != 0:
            raise ValueError(
                f"vee + 75cm must start from the first row; {ip} is assigned to row {grid_row}."
            )

        mission_pad = int_field(drone.get("mission_pad"), "mission_pad")
        expected_pad = mission_pad_columns[grid_column][grid_row]
        if mission_pad != expected_pad:
            raise ValueError(
                f"Mission pad mismatch for {ip}: record pad={mission_pad}, "
                f"layout pad={expected_pad} at column={grid_column}, row={grid_row}"
            )

        position_key = (grid_column, grid_row)
        if position_key in seen_positions:
            raise ValueError(f"Two drones are assigned to the same grid position: {position_key}")
        seen_positions.add(position_key)
        start_x, start_y = position_at_column_row(formation, grid_column, grid_row, column_spacing_cm)
        if node_direction < 0:
            target_row = max(0, grid_row - default_steps)
        else:
            target_row = grid_row + default_steps
        target_pad = pad_at_column_row(grid_column, target_row, columns=mission_pad_columns)
        node_distance = abs(target_row - grid_row) * ROW_SPACING_CM
        target_x, target_y = position_at_column_row(formation, grid_column, target_row, column_spacing_cm)

        configs.append({
            "name": f"drone_{idx + 1}",
            "ip": ip,
            "battery_id": battery_id,
            "takeoff_order": int_field(drone.get("takeoff_order", idx + 1), "takeoff_order"),
            "role": str(drone.get("role") or f"drone_{idx + 1}"),
            "mission_pad": mission_pad,
            "mission_pad_columns": mission_pad_columns,
            "formation": formation,
            "wind_direction": str(experiment.get("wind_direction", "")).strip().lower(),
            "wind_speed": str(experiment.get("wind_speed", "")).strip().lower(),
            "inter_drone_distance_cm": experiment_inter_drone_distance_cm(experiment),
            "column_spacing_cm": column_spacing_cm,
            "target_pad": target_pad,
            "grid_column": grid_column,
            "grid_row": grid_row,
            "target_grid_row": target_row,
            "node_row_direction": node_direction,
            "node_segment_count": abs(target_row - grid_row),
            "start_x": start_x,
            "start_y": start_y,
            "target_x": target_x,
            "target_y": target_y,
            "target_z": TAKEOFF_HEIGHT_CM,
            "node_forward_distance_cm": node_distance,
            "node_speed_cm_s": NODE_FLIGHT_SPEED_CM_S,
        })
    return configs


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def drone_folder_name(config):
    suffix = config["ip"].replace(IP_PREFIX, "")
    return f"{safe_name(config['name'])}_{suffix}_pad{config['mission_pad']}"


def output_paths(experiment_id, configs):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = DATA_DIR / safe_name(experiment_id)
    drones_dir = experiment_dir / "drones"
    plots_dir = experiment_dir / "plots"
    drones_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    coordination_path = experiment_dir / f"{safe_name(experiment_id)}_{run_id}_all_coordination.csv"
    battery_path = experiment_dir / f"{safe_name(experiment_id)}_{run_id}_all_battery.csv"
    battery_timeseries_path = experiment_dir / f"{safe_name(experiment_id)}_{run_id}_all_battery_timeseries.csv"
    drone_paths = {}
    for config in configs:
        folder = drones_dir / drone_folder_name(config)
        folder.mkdir(parents=True, exist_ok=True)
        drone_paths[config["ip"]] = {
            "coordination": folder / f"{safe_name(experiment_id)}_{run_id}_{safe_name(config['name'])}_coordination.csv",
            "battery": folder / f"{safe_name(experiment_id)}_{run_id}_{safe_name(config['name'])}_battery.csv",
        }
    battery_plot_path = plots_dir / f"{safe_name(experiment_id)}_{run_id}_all_battery_lines.png"
    temperature_plot_path = plots_dir / f"{safe_name(experiment_id)}_{run_id}_all_temperature_lines.png"
    return (
        run_id,
        experiment_dir,
        coordination_path,
        battery_path,
        battery_timeseries_path,
        drone_paths,
        battery_plot_path,
        temperature_plot_path,
    )


def run_output_files(coordination_path, battery_path, battery_timeseries_path, drone_paths):
    paths = [coordination_path, battery_path, battery_timeseries_path]
    for items in drone_paths.values():
        paths.extend(items.values())
    return paths


def read_coordination_battery_series(path):
    series = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            drone_name = row.get("drone_name", "")
            if not drone_name:
                continue
            try:
                elapsed_value = row.get("node_elapsed_time") or row.get("hover_elapsed_time", "")
                node_time = float(elapsed_value)
                battery = float(row.get("battery", ""))
            except (TypeError, ValueError):
                continue
            series.setdefault(drone_name, []).append((node_time, battery))
    for values in series.values():
        values.sort(key=lambda item: item[0])
    return series


def generate_battery_line_plot(coordination_path, output_path, experiment_id, run_id):
    series = read_coordination_battery_series(coordination_path)
    if not series:
        print("Battery plot skipped: no battery time series found.", flush=True)
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    for drone_name, values in sorted(series.items()):
        times = [item[0] for item in values]
        batteries = [item[1] for item in values]
        if not times or not batteries:
            continue
        ax.plot(times, batteries, linewidth=1.8, label=drone_name)

    ax.set_title(f"{experiment_id} {run_id}: battery percentage during node-to-node flight")
    ax.set_xlabel("Node-to-node flight time (s)")
    ax.set_ylabel("Battery (%)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Battery line plot saved: {output_path}", flush=True)
    return output_path


def read_coordination_temperature_series(path):
    series = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            drone_name = row.get("drone_name", "")
            if not drone_name:
                continue
            try:
                elapsed_value = row.get("node_elapsed_time") or row.get("hover_elapsed_time", "")
                node_time = float(elapsed_value)
                temp_low = float(row.get("templ", ""))
                temp_high = float(row.get("temph", ""))
            except (TypeError, ValueError):
                continue
            temp_avg = (temp_low + temp_high) / 2
            series.setdefault(drone_name, []).append((node_time, temp_avg))
    for values in series.values():
        values.sort(key=lambda item: item[0])
    return series


def generate_temperature_line_plot(coordination_path, output_path, experiment_id, run_id):
    series = read_coordination_temperature_series(coordination_path)
    if not series:
        print("Temperature plot skipped: no temperature time series found.", flush=True)
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    for drone_name, values in sorted(series.items()):
        times = [item[0] for item in values]
        temperatures = [item[1] for item in values]
        if not times or not temperatures:
            continue
        ax.plot(times, temperatures, linewidth=1.8, label=drone_name)

    ax.set_title(f"{experiment_id} {run_id}: average temperature during node-to-node flight")
    ax.set_xlabel("Node-to-node flight time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Temperature line plot saved: {output_path}", flush=True)
    return output_path


def cleanup_failed_run_outputs(experiment_dir, output_files):
    print("Cleaning up incomplete run data...", flush=True)
    for path in output_files:
        try:
            if path.exists():
                path.unlink()
                print(f"  Deleted {path}", flush=True)
        except OSError as exc:
            print(f"  Warning: failed to delete {path}: {exc}", flush=True)

    drones_dir = experiment_dir / "drones"
    for folder in sorted([drones_dir, experiment_dir], key=lambda item: len(item.parts), reverse=True):
        try:
            if folder.exists():
                for child in sorted(folder.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                    if child.is_dir() and not any(child.iterdir()):
                        child.rmdir()
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
        except OSError as exc:
            print(f"  Warning: failed to remove empty folder {folder}: {exc}", flush=True)


def write_header(path, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)


def append_row(path, row):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def set_phase(index, phase):
    with phase_lock:
        current_phases[index] = phase


def set_phase_all(phase):
    with phase_lock:
        for idx in range(len(current_phases)):
            current_phases[idx] = phase


def get_phase(index):
    with phase_lock:
        return current_phases[index]


def reset_runtime_state(configs):
    global current_phases, hover_start_batteries, hover_end_batteries
    current_phases = ["idle"] * len(configs)
    hover_start_batteries = {cfg["ip"]: "" for cfg in configs}
    hover_end_batteries = {cfg["ip"]: "" for cfg in configs}


def get_state_safe(tello):
    state = tello.get_current_state()
    return {
        "mid": state.get("mid", -1),
        "x": state.get("x", 0),
        "y": state.get("y", 0),
        "z": state.get("z", 0),
        "yaw": state.get("yaw", 0),
        "pitch": state.get("pitch", 0),
        "roll": state.get("roll", 0),
        "vgx": state.get("vgx", 0),
        "vgy": state.get("vgy", 0),
        "vgz": state.get("vgz", 0),
        "agx": state.get("agx", 0.0),
        "agy": state.get("agy", 0.0),
        "agz": state.get("agz", 0.0),
        "templ": state.get("templ", 0),
        "temph": state.get("temph", 0),
        "tof": state.get("tof", 0),
        "h": state.get("h", 0),
        "baro": state.get("baro", 0.0),
        "motor_time": state.get("time", 0),
    }


def monitor_takeoff_health(swarm, configs, duration=2.5, interval=0.5):
    """Print early flight state so failed takeoffs are visible in the run log."""
    start = time.time()
    next_sample = start
    last_states = [None] * len(configs)
    print("Takeoff health monitor started.", flush=True)
    while time.time() - start < duration:
        now = time.time()
        if now < next_sample:
            time.sleep(min(0.05, next_sample - now))
            continue
        elapsed = now - start
        parts = []
        for idx, tello in enumerate(swarm.tellos):
            config = configs[idx]
            try:
                state = get_state_safe(tello)
                last_states[idx] = state
                parts.append(
                    f"{config['name']} ip={config['ip']} "
                    f"h={state['h']} tof={state['tof']} mid={state['mid']} "
                    f"motor_time={state['motor_time']} "
                    f"bat={tello.get_battery()}%"
                )
            except Exception as exc:
                parts.append(f"{config['name']} ip={config['ip']} state_error={exc}")
        print(f"  takeoff+{elapsed:.1f}s: " + " | ".join(parts), flush=True)
        next_sample += interval

    for idx, state in enumerate(last_states):
        if state is None:
            continue
        measured_height = max(int(state.get("h") or 0), int(state.get("tof") or 0))
        if measured_height < 20:
            print(
                f"  Warning: {configs[idx]['name']} still reports low height after takeoff "
                f"(h={state['h']} tof={state['tof']}, motor_time={state['motor_time']}). "
                "This usually means takeoff failed or the drone auto-stopped.",
                flush=True,
            )


def pad_origin_for_detection(config, mid, preferred_row=None):
    min_row = min(config["grid_row"], config["target_grid_row"]) - 1
    max_row = max(config["grid_row"], config["target_grid_row"]) + 1
    column = lane_pad_sequence(
        config["grid_column"],
        min_rows=max(max_row + 2, len(PAD_SEQUENCE)),
        columns=config.get("mission_pad_columns"),
    )
    candidate_rows = [
        row_idx
        for row_idx, pad_id in enumerate(column)
        if pad_id == mid and min_row <= row_idx <= max_row
    ]
    if not candidate_rows:
        return None, None
    if preferred_row is not None:
        detected_row = min(candidate_rows, key=lambda row_idx: abs(row_idx - preferred_row))
    else:
        detected_row = candidate_rows[0]
    return position_at_column_row(
        config.get("formation", ""),
        config["grid_column"],
        detected_row,
        config_column_spacing_cm(config),
    )


def to_global(config, state, preferred_row=None):
    mid = state["mid"]
    if mid == -1:
        return None, None, None
    origin_x, origin_y = pad_origin_for_detection(config, mid, preferred_row=preferred_row)
    if origin_x is None or origin_y is None:
        return None, None, None
    return origin_x + state["x"], origin_y + state["y"], state["z"]


def spacing_stats(configs, positions):
    errors = []
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            actual_a = positions.get(i)
            actual_b = positions.get(j)
            if actual_a is None or actual_b is None:
                continue
            target_dist = math.hypot(
                configs[i]["target_x"] - configs[j]["target_x"],
                configs[i]["target_y"] - configs[j]["target_y"],
            )
            actual_dist = math.hypot(actual_a[0] - actual_b[0], actual_a[1] - actual_b[1])
            errors.append(abs(actual_dist - target_dist))
    if not errors:
        return None, None
    return round(sum(errors) / len(errors), 3), round(max(errors), 3)


def wait_for_pad(
    tello,
    pad_id,
    timeout=8.0,
    interval=0.15,
    min_hits=PAD_LOCK_MIN_HITS,
    grace_sec=PAD_LOCK_GRACE_SEC,
    label="mission pad",
):
    deadline = time.time() + timeout
    hits = 0
    first_seen = None
    last_seen = None
    last_state = None
    last_report = 0.0
    while time.time() < deadline:
        state = get_state_safe(tello)
        last_state = state
        now = time.time()
        if state["mid"] == pad_id:
            hits += 1
            first_seen = first_seen or now
            last_seen = now
            if hits >= min_hits:
                return True
        elif last_seen is not None and now - last_seen <= grace_sec and hits >= min_hits:
            return True
        else:
            hits = 0
            first_seen = None

        if now - last_report >= GROUP_PAD_WAIT_REPORT_INTERVAL_SEC:
            print(
                f"  Waiting for {label} m{pad_id}; currently sees m{state['mid']} "
                f"x={state['x']} y={state['y']} z={state['z']} tof={state['tof']} h={state['h']}.",
                flush=True,
            )
            last_report = now
        time.sleep(interval)
    if last_state is not None:
        seen_duration = 0.0 if first_seen is None or last_seen is None else last_seen - first_seen
        print(
            f"  Timed out waiting for {label} m{pad_id}; last sees m{last_state['mid']} "
            f"x={last_state['x']} y={last_state['y']} z={last_state['z']} "
            f"tof={last_state['tof']} h={last_state['h']} "
            f"confirmed_hits={hits} seen_duration={seen_duration:.2f}s.",
            flush=True,
        )
    return False


def wait_for_expected_pad(tello, config, timeout=START_PAD_LOCK_TIMEOUT_SEC, interval=0.15):
    return wait_for_pad(
        tello,
        config["mission_pad"],
        timeout=timeout,
        interval=interval,
        label=f"{config['name']} start pad",
    )


def wait_for_all_expected_start_pads(swarm, configs, timeout=START_PAD_LOCK_TIMEOUT_SEC):
    errors = []
    errors_lock = threading.Lock()

    def worker(idx, tello):
        config = configs[idx]
        set_phase(idx, "acquire_start_pad")
        try:
            for _ in range(2):
                try:
                    tello.enable_mission_pads()
                    tello.set_mission_pad_detection_direction(0)
                    break
                except Exception as exc:
                    print(
                        f"  Warning: {config['name']} mission pad re-enable returned error: {exc}",
                        flush=True,
                    )
                    time.sleep(0.3)
            if not wait_for_expected_pad(tello, config, timeout=timeout):
                raise RuntimeError(
                    f"{config['name']} failed to detect expected mission pad {config['mission_pad']}."
                )
            print(f"{config['name']} detected start mission pad {config['mission_pad']}.", flush=True)
            hold_position(tello, config, "start pad lock")
        except Exception as exc:
            with errors_lock:
                errors.append((idx, exc))

    threads = []
    for idx, tello in enumerate(swarm.tellos):
        thread = threading.Thread(target=worker, args=(idx, tello), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        details = "; ".join(f"{configs[idx]['name']}: {exc}" for idx, exc in errors)
        raise RuntimeError(f"Start mission pad acquisition failed: {details}")


def is_no_valid_marker_error(exc):
    return "no valid marker" in str(exc).lower()


def run_swarm_parallel_checked(swarm, label, command_func):
    errors = []
    errors_lock = threading.Lock()

    def worker(idx, tello):
        try:
            command_func(idx, tello)
        except Exception as exc:
            with errors_lock:
                errors.append((idx, exc))

    threads = []
    for idx, tello in enumerate(swarm.tellos):
        thread = threading.Thread(target=worker, args=(idx, tello), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        details = "; ".join(f"drone_{idx + 1}: {exc}" for idx, exc in errors)
        raise RuntimeError(f"{label} failed: {details}")


def coordinate_climb_on_start_pads(swarm, configs):
    if any(config.get("formation") == "vee" for config in configs):
        print(
            f"Stabilizing vee takeoff height near {TAKEOFF_HEIGHT_CM} cm without mission-pad-relative climb commands...",
            flush=True,
        )
        set_phase_all("coordinate_climb")

        def adjust_height(_idx, tello):
            state = get_state_safe(tello)
            measured_height = state["tof"] or state["h"]
            if measured_height <= 0:
                raise RuntimeError("No valid downward height reading available for vee takeoff stabilization.")
            current_height = int(round(measured_height))
            height_delta = TAKEOFF_HEIGHT_CM - current_height
            if abs(height_delta) < TAKEOFF_HEIGHT_ADJUST_MIN_CM:
                return
            distance = max(TAKEOFF_HEIGHT_ADJUST_MIN_CM, min(500, abs(height_delta)))
            if height_delta > 0:
                tello.move_up(distance)
            else:
                tello.move_down(distance)

        run_swarm_parallel_checked(swarm, "Takeoff height stabilization", adjust_height)
        time.sleep(1.0)
        return

    print(f"Climbing all drones to {TAKEOFF_HEIGHT_CM} cm above their start mission pads...", flush=True)
    set_phase_all("coordinate_climb")

    def climb_on_start_pad(i, tello):
        config = configs[i]
        last_error = None
        for attempt in range(3):
            if attempt > 0:
                if not wait_for_expected_pad(tello, config, timeout=5.0):
                    last_error = RuntimeError(
                        f"{config['name']} could not reacquire start pad m{config['mission_pad']}."
                    )
                    continue
            try:
                tello.go_xyz_speed_mid(
                    0,
                    0,
                    TAKEOFF_HEIGHT_CM,
                    TAKEOFF_CLIMB_SPEED_CM_S,
                    config["mission_pad"],
                )
                return
            except Exception as exc:
                last_error = exc
                if not is_no_valid_marker_error(exc):
                    raise
                print(
                    f"  {config['name']} lost start pad m{config['mission_pad']} during climb; retrying.",
                    flush=True,
                )
                hold_position(tello, config, "coordinate climb retry")
                time.sleep(0.5)
        raise RuntimeError(f"{config['name']} climb failed after marker retries: {last_error}")

    run_swarm_parallel_checked(
        swarm,
        "Coordinate climb",
        climb_on_start_pad,
    )
    time.sleep(1.0)


def is_valid_column_detection(config, x_global):
    tolerance = COLUMN_SPACING_CM * 0.75
    if config.get("formation") == "vee":
        tolerance = VEE_COLUMN_DETECTION_TOLERANCE_CM
    return abs(x_global - config["target_x"]) <= tolerance


def hold_position(tello, config, context, state=None):
    try:
        tello.send_rc_control(0, 0, 0, 0)
    except Exception as exc:
        print(
            f"  Warning: {config['name']} hover hold during {context} returned error: {exc}",
            flush=True,
        )
    return state


def is_front_segment_target_reached(config, state, target_x, target_y, target_row, tolerance):
    if state["mid"] == -1:
        return False, None, None
    x_global, y_global, _ = to_global(config, state, preferred_row=target_row)
    if x_global is None or y_global is None:
        return False, x_global, y_global
    reached = (
        is_valid_column_detection(config, x_global)
        and abs(target_x - x_global) <= tolerance
        and abs(target_y - y_global) <= tolerance
    )
    return reached, x_global, y_global


def wait_for_segment_target(
    tello,
    config,
    target_pad,
    target_x,
    target_y,
    target_row=None,
    tolerance=SEGMENT_TARGET_TOLERANCE_CM,
    interval=0.15,
):
    last_report = 0.0
    while True:
        hold_position(tello, config, "target pad verification")
        state = get_state_safe(tello)
        if state["mid"] != target_pad:
            now = time.time()
            if now - last_report >= SEGMENT_TARGET_REPORT_INTERVAL_SEC:
                print(
                    f"    {config['name']} waiting for target pad m{target_pad}; "
                    f"currently sees m{state['mid']}. Press Stop to abort/land.",
                    flush=True,
                )
                last_report = now
            time.sleep(interval)
            continue
        x_global, y_global, _ = to_global(config, state, preferred_row=target_row)
        if (
            x_global is not None
            and y_global is not None
            and is_valid_column_detection(config, x_global)
            and abs(target_x - x_global) <= tolerance
            and abs(target_y - y_global) <= tolerance
        ):
            return True
        now = time.time()
        if now - last_report >= SEGMENT_TARGET_REPORT_INTERVAL_SEC:
            print(
                f"    {config['name']} sees target pad m{target_pad} but position is "
                f"global=({x_global},{y_global}), target=({target_x},{target_y}). "
                "Press Stop to abort/land.",
                flush=True,
            )
            last_report = now
        time.sleep(interval)


def advance_to_target_pad(
    tello,
    config,
    speed=NODE_FLIGHT_SPEED_CM_S,
    tolerance=SEGMENT_TARGET_TOLERANCE_CM,
    max_corrections=3,
):
    step = config["node_row_direction"]
    stop_row = config["target_grid_row"] + step
    for row_idx in range(config["grid_row"], stop_row, step):
        target_x, pad_y = position_at_column_row(
            config.get("formation", ""),
            config["grid_column"],
            row_idx,
            config_column_spacing_cm(config),
        )
        expected_pad = pad_at_physical_row(config, row_idx)
        for _ in range(max_corrections):
            state = get_state_safe(tello)
            if state["mid"] == -1:
                time.sleep(0.3)
                continue
            x_global, y_global, _ = to_global(config, state)
            if x_global is None or y_global is None:
                time.sleep(0.3)
                continue
            if not is_valid_column_detection(config, x_global):
                time.sleep(0.3)
                continue
            if abs(pad_y - y_global) <= tolerance and abs(target_x - x_global) <= tolerance:
                break
            origin_x, origin_y = pad_origin_for_detection(config, state["mid"])
            if origin_x is None or origin_y is None:
                time.sleep(0.3)
                continue
            local_x = max(-500, min(500, int(round(target_x - origin_x))))
            local_y = max(-500, min(500, int(round(pad_y - origin_y))))
            local_z = max(20, min(500, int(round(TAKEOFF_HEIGHT_CM))))
            try:
                tello.go_xyz_speed_mid(local_x, local_y, local_z, max(10, min(100, speed)), int(state["mid"]))
            except Exception as exc:
                if is_no_valid_marker_error(exc):
                    print(
                        f"  {config['name']} lost marker m{state['mid']} during correction; reacquiring marker.",
                        flush=True,
                    )
                    time.sleep(0.3)
                    continue
                raise
            time.sleep(0.5)
        if row_idx != config["grid_row"] and not wait_for_pad(
            tello,
            expected_pad,
            timeout=TARGET_PAD_LOCK_TIMEOUT_SEC,
            label=f"{config['name']} row {row_idx} pad",
        ):
            raise RuntimeError(
                f"{config['name']} failed to detect mission pad {expected_pad} at row {row_idx}."
            )


def fly_synchronized_node_segment(
    tello,
    config,
    current_row,
    speed=NODE_FLIGHT_SPEED_CM_S,
    tolerance=SEGMENT_TARGET_TOLERANCE_CM,
    max_corrections=3,
):
    next_row = current_row + config["node_row_direction"]
    next_pad_x, next_pad_y = position_at_column_row(
        config.get("formation", ""),
        config["grid_column"],
        next_row,
        config_column_spacing_cm(config),
    )
    current_pad = pad_at_physical_row(config, current_row)
    expected_pad = pad_at_physical_row(config, next_row)
    allowed_pads = {current_pad, expected_pad}

    for _ in range(max_corrections):
        state = get_state_safe(tello)
        if state["mid"] == -1:
            time.sleep(0.3)
            continue
        if state["mid"] not in allowed_pads:
            time.sleep(0.3)
            continue
        preferred_row = next_row if state["mid"] == expected_pad else current_row
        x_global, y_global, _ = to_global(config, state, preferred_row=preferred_row)
        if x_global is None or y_global is None:
            time.sleep(0.3)
            continue
        if not is_valid_column_detection(config, x_global):
            time.sleep(0.3)
            continue

        if abs(next_pad_y - y_global) <= tolerance and abs(next_pad_x - x_global) <= tolerance:
            break

        origin_x, origin_y = pad_origin_for_detection(config, state["mid"], preferred_row=preferred_row)
        if origin_x is None or origin_y is None:
            time.sleep(0.3)
            continue

        local_x = max(-500, min(500, int(round(next_pad_x - origin_x))))
        local_y = max(-500, min(500, int(round(next_pad_y - origin_y))))
        local_z = max(20, min(500, int(round(TAKEOFF_HEIGHT_CM))))
        if local_y * config["node_row_direction"] < 0:
            time.sleep(0.3)
            continue
        try:
            tello.go_xyz_speed_mid(local_x, local_y, local_z, max(10, min(100, speed)), int(state["mid"]))
        except Exception as exc:
            if is_no_valid_marker_error(exc):
                print(
                    f"  {config['name']} lost marker m{state['mid']} during segment; reacquiring marker.",
                    flush=True,
                )
                time.sleep(0.3)
                continue
            raise
        time.sleep(0.5)

    if not wait_for_segment_target(
        tello,
        config,
        expected_pad,
        next_pad_x,
        next_pad_y,
        target_row=next_row,
        tolerance=tolerance,
    ):
        raise RuntimeError(
            f"{config['name']} failed to detect mission pad {expected_pad} after synchronized segment."
        )


def wait_for_all_segment_start_pads(swarm, configs, current_rows):
    last_report = 0.0
    hover_error_reported = set()

    while True:
        waiting = []
        for index, tello in enumerate(swarm.tellos):
            config = configs[index]
            expected_pad = pad_at_physical_row(config, current_rows[index])
            state = get_state_safe(tello)
            if state["mid"] != expected_pad:
                waiting.append((config["name"], expected_pad, state["mid"]))
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception as exc:
                if index not in hover_error_reported:
                    print(
                        f"  Warning: {config['name']} hover hold while waiting for group pad lock returned error: {exc}",
                        flush=True,
                    )
                    hover_error_reported.add(index)

        if not waiting:
            print("  All drones detected their current mission pads; starting segment together.", flush=True)
            return

        now = time.time()
        if now - last_report >= GROUP_PAD_WAIT_REPORT_INTERVAL_SEC:
            waiting_text = "; ".join(
                f"{name} needs m{expected_pad} sees m{seen_pad}"
                for name, expected_pad, seen_pad in waiting
            )
            print(f"  Waiting for all current pads: {waiting_text}", flush=True)
            last_report = now
        time.sleep(0.2)


def fly_continuous_node_to_node(tello, config, speed=NODE_FLIGHT_SPEED_CM_S):
    signed_y_distance = int(round(config["target_y"] - config["start_y"]))
    command = "go {} {} {} {} m{}".format(
        0,
        signed_y_distance,
        TAKEOFF_HEIGHT_CM,
        max(10, min(100, speed)),
        int(config["mission_pad"]),
    )
    tello.send_control_command(command, timeout=LONG_GO_RESPONSE_TIMEOUT_SEC)


def segment_launch_order(configs, current_rows):
    return sorted(
        range(len(configs)),
        key=lambda index: (
            current_rows[index] * configs[index]["node_row_direction"],
            -configs[index]["takeoff_order"],
        ),
        reverse=True,
    )


def segment_stagger_delay(configs):
    if not configs:
        return SEGMENT_STAGGER_DELAY_SEC
    formation = str(configs[0].get("formation", "")).strip().lower()
    wind_direction = str(configs[0].get("wind_direction", "")).strip().lower()
    wind_speed = str(configs[0].get("wind_speed", "")).strip().lower()
    has_wind = wind_direction and wind_direction != "no wind" and wind_speed != "no wind"
    if formation == "column" and has_wind:
        return COLUMN_WIND_STAGGER_DELAY_SEC
    return SEGMENT_STAGGER_DELAY_SEC


def segment_stagger_delays(configs, launch_count):
    if not configs:
        return [release_order * SEGMENT_STAGGER_DELAY_SEC for release_order in range(launch_count)]
    formation = str(configs[0].get("formation", "")).strip().lower()
    if formation == "front":
        return [0.0] * launch_count
    wind_direction = str(configs[0].get("wind_direction", "")).strip().lower()
    wind_speed = str(configs[0].get("wind_speed", "")).strip().lower()
    has_wind = wind_direction and wind_direction != "no wind" and wind_speed != "no wind"
    if formation == "column" and has_wind:
        delays = list(COLUMN_WIND_STAGGER_DELAYS_SEC[:launch_count])
        while len(delays) < launch_count:
            delays.append(delays[-1] + COLUMN_WIND_STAGGER_DELAY_SEC)
        return delays
    stagger_delay = segment_stagger_delay(configs)
    return [release_order * stagger_delay for release_order in range(launch_count)]


def is_column_spacing_gate_enabled(configs):
    if not configs:
        return False
    return str(configs[0].get("formation", "")).strip().lower() == "column"


def observed_segment_y(tello, config, current_row):
    state = get_state_safe(tello)
    if state["mid"] == -1:
        return None

    next_row = current_row + config["node_row_direction"]
    current_pad = pad_at_physical_row(config, current_row)
    next_pad = pad_at_physical_row(config, next_row)
    if state["mid"] == next_pad:
        preferred_row = next_row
    elif state["mid"] == current_pad:
        preferred_row = current_row
    else:
        preferred_row = current_row

    x_global, y_global, _ = to_global(config, state, preferred_row=preferred_row)
    if x_global is None or y_global is None:
        return None
    if not is_valid_column_detection(config, x_global):
        return None
    return y_global


def wait_for_column_spacing_gate(swarm, configs, current_rows, launch_order, release_order, follower_index):
    if not is_column_spacing_gate_enabled(configs) or release_order == 0:
        return

    leader_index = launch_order[release_order - 1]
    leader_config = configs[leader_index]
    follower_config = configs[follower_index]
    direction = follower_config["node_row_direction"]
    deadline = time.time() + COLUMN_SAFETY_WAIT_TIMEOUT_SEC
    last_report = 0.0

    while True:
        leader_y = observed_segment_y(swarm.tellos[leader_index], leader_config, current_rows[leader_index])
        follower_y = observed_segment_y(swarm.tellos[follower_index], follower_config, current_rows[follower_index])

        if leader_y is not None and follower_y is not None:
            spacing = direction * (leader_y - follower_y)
            if spacing >= COLUMN_SAFETY_RELEASE_SPACING_CM:
                return
        else:
            spacing = None

        try:
            swarm.tellos[follower_index].send_rc_control(0, 0, 0, 0)
        except Exception:
            pass

        now = time.time()
        if now >= deadline:
            spacing_text = "unknown" if spacing is None else f"{spacing:.1f}cm"
            raise RuntimeError(
                f"{follower_config['name']} was not released because "
                f"{leader_config['name']} did not clear "
                f"{COLUMN_SAFETY_RELEASE_SPACING_CM:.0f}cm in column; "
                f"spacing={spacing_text}."
            )
        if now - last_report >= COLUMN_SAFETY_REPORT_INTERVAL_SEC:
            spacing_text = "unknown" if spacing is None else f"{spacing:.1f}cm"
            print(
                f"    {follower_config['name']} holding: waiting for "
                f"{leader_config['name']} to clear "
                f"{COLUMN_SAFETY_RELEASE_SPACING_CM:.0f}cm in column; "
                f"spacing={spacing_text}.",
                flush=True,
            )
            last_report = now
        time.sleep(0.1)


def run_staggered_node_segment(swarm, configs, current_rows, segment_index):
    errors = []
    errors_lock = threading.Lock()
    work_done_count = [0]
    work_done_lock = threading.Lock()
    all_work_done = threading.Event()
    launch_order = segment_launch_order(configs, current_rows)
    stagger_delays = segment_stagger_delays(configs, len(launch_order))
    order_text = " -> ".join(configs[index]["name"] for index in launch_order)
    delay_text = ", ".join(
        f"{configs[index]['name']}={stagger_delays[release_order]:.1f}s"
        for release_order, index in enumerate(launch_order)
    )
    print(
        f"  Segment {segment_index + 1} launch order: {order_text}",
        flush=True,
    )
    print(
        f"  Segment {segment_index + 1} launch delays: {delay_text}",
        flush=True,
    )

    def worker(release_order, index):
        config = configs[index]
        delay = stagger_delays[release_order]
        deadline = time.time() + delay
        while time.time() < deadline:
            try:
                swarm.tellos[index].send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
            time.sleep(0.05)
        try:
            wait_for_column_spacing_gate(swarm, configs, current_rows, launch_order, release_order, index)
            fly_synchronized_node_segment(
                swarm.tellos[index],
                config,
                current_rows[index],
                speed=NODE_FLIGHT_SPEED_CM_S,
            )
        except Exception as exc:
            with errors_lock:
                errors.append((index, exc))
        with work_done_lock:
            work_done_count[0] += 1
            if work_done_count[0] >= len(launch_order):
                all_work_done.set()
        while not all_work_done.wait(timeout=0.1):
            try:
                swarm.tellos[index].send_rc_control(0, 0, 0, 0)
            except Exception:
                pass

    threads = []
    for release_order, index in enumerate(launch_order):
        thread = threading.Thread(target=worker, args=(release_order, index), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        details = "; ".join(f"{configs[index]['name']}: {exc}" for index, exc in errors)
        raise RuntimeError(f"Node segment {segment_index + 1} failed: {details}")


def fly_front_segment(
    tello,
    config,
    current_row,
    speed=NODE_FLIGHT_SPEED_CM_S,
    tolerance=SEGMENT_TARGET_TOLERANCE_CM,
    max_corrections=3,
):
    next_row = current_row + config["node_row_direction"]
    next_pad_x, next_pad_y = position_at_column_row(
        config.get("formation", ""),
        config["grid_column"],
        next_row,
        config_column_spacing_cm(config),
    )
    expected_pad = pad_at_physical_row(config, next_row)
    last_report = 0.0
    last_nudge = 0.0
    command_failures = 0

    while True:
        for _ in range(max_corrections):
            state = get_state_safe(tello)
            if state["mid"] == -1:
                time.sleep(0.3)
                continue
            x_global, y_global, _ = to_global(config, state)
            if x_global is None or y_global is None:
                time.sleep(0.3)
                continue
            if not is_valid_column_detection(config, x_global):
                time.sleep(0.3)
                continue
            if abs(next_pad_y - y_global) <= tolerance and abs(next_pad_x - x_global) <= tolerance:
                print(
                    f"  {config['name']} reached front row {next_row} by position "
                    f"global=({x_global:.1f},{y_global:.1f}) while seeing m{state['mid']}.",
                    flush=True,
                )
                return
            origin_x, origin_y = pad_origin_for_detection(config, state["mid"])
            if origin_x is None or origin_y is None:
                time.sleep(0.3)
                continue
            local_x = max(-500, min(500, int(round(next_pad_x - origin_x))))
            local_y = max(-500, min(500, int(round(next_pad_y - origin_y))))
            local_z = max(20, min(500, int(round(TAKEOFF_HEIGHT_CM))))
            try:
                go_cmd = "go {} {} {} {} m{}".format(
                    local_x, local_y, local_z,
                    max(10, min(100, speed)),
                    int(state["mid"]),
                )
                tello.send_control_command(go_cmd, timeout=LONG_GO_RESPONSE_TIMEOUT_SEC)
            except Exception as exc:
                command_failures += 1
                now = time.time()
                hold_position(tello, config, "front segment command failure")
                state_after_error = get_state_safe(tello)
                reached, x_after_error, y_after_error = is_front_segment_target_reached(
                    config,
                    state_after_error,
                    next_pad_x,
                    next_pad_y,
                    next_row,
                    tolerance,
                )
                if reached:
                    print(
                        f"  {config['name']} command returned {exc}, but front row {next_row} "
                        f"is reached at global=({x_after_error:.1f},{y_after_error:.1f}); continuing.",
                        flush=True,
                    )
                    return
                if now - last_report >= 3.0:
                    print(
                        f"  {config['name']} go command failed ({exc}); "
                        f"now sees m{state_after_error['mid']} "
                        f"global=({x_after_error},{y_after_error}).",
                        flush=True,
                    )
                    last_report = now
                if command_failures >= 2:
                    direction = config["node_row_direction"]
                    try:
                        tello.send_rc_control(0, 15 * direction, 0, 0)
                        time.sleep(0.6)
                        tello.send_rc_control(0, 0, 0, 0)
                    except Exception:
                        pass
                time.sleep(0.8)
                continue
            time.sleep(0.5)

        if wait_for_pad(
            tello,
            expected_pad,
            timeout=TARGET_PAD_LOCK_TIMEOUT_SEC,
            label=f"{config['name']} front row {next_row} pad",
        ):
            return

        state = get_state_safe(tello)
        reached, x_global, y_global = is_front_segment_target_reached(
            config,
            state,
            next_pad_x,
            next_pad_y,
            next_row,
            tolerance,
        )
        if reached:
            print(
                f"  {config['name']} reached front row {next_row} by position "
                f"global=({x_global:.1f},{y_global:.1f}) while seeing m{state['mid']}.",
                flush=True,
            )
            return

        now = time.time()
        if now - last_report >= 3.0:
            print(
                f"  {config['name']} waiting for front row {next_row}; needs pad m{expected_pad} "
                f"or target position ({next_pad_x},{next_pad_y}), currently sees m{state['mid']} "
                f"global=({x_global},{y_global}).",
                flush=True,
            )
            last_report = now

        # When stuck between pads (mid=-1), nudge toward the expected pad so the
        # drone drifts out of the detection dead zone rather than hovering indefinitely.
        if state["mid"] == -1 and now - last_nudge >= 2.0:
            direction = config["node_row_direction"]
            try:
                tello.send_rc_control(0, 20 * direction, 0, 0)
                time.sleep(0.4)
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
            last_nudge = now
        else:
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
        time.sleep(0.2)


def fly_node_to_node(swarm, configs):
    set_phase_all("node_to_node")
    segments = configs[0]["node_segment_count"]
    if any(config["node_segment_count"] != segments for config in configs):
        raise RuntimeError("All drones must have the same node segment count.")
    if segments <= 0:
        raise RuntimeError("Node-to-node flight has zero segments; check start and target mission pads.")
    direction_text = "negative y" if configs[0]["node_row_direction"] < 0 else "positive y"
    print(
        f"Flying all drones in {segments} mission-pad segments "
        f"({NODE_SEGMENT_DISTANCE_CM} cm each) along {direction_text} at "
        f"{NODE_FLIGHT_SPEED_CM_S} cm/s.",
        flush=True,
    )
    for config in configs:
        path_rows = range(
            config["grid_row"],
            config["target_grid_row"] + config["node_row_direction"],
            config["node_row_direction"],
        )
        path_pads = [str(pad_at_physical_row(config, row_idx)) for row_idx in path_rows]
        print(
            f"  {config['name']}: pad {config['mission_pad']} -> pad {config['target_pad']} "
            f"path={' -> '.join(path_pads)} "
            f"target=({config['target_x']},{config['target_y']},{config['target_z']})",
            flush=True,
        )

    formation = str(configs[0].get("formation", "")).strip().lower()
    is_front = formation == "front" or is_echalon_formation(formation) or all(is_vee_75cm_config(config) for config in configs)
    current_rows = [config["grid_row"] for config in configs]
    for segment_index in range(segments):
        set_phase_all(f"node_segment_{segment_index + 1}_of_{segments}")
        print(f"  Segment {segment_index + 1}/{segments}", flush=True)
        if is_front:
            run_swarm_parallel_checked(
                swarm,
                f"Node segment {segment_index + 1}",
                lambda i, tello: fly_front_segment(
                    tello, configs[i], current_rows[i], speed=NODE_FLIGHT_SPEED_CM_S,
                )
            )
        else:
            wait_for_all_segment_start_pads(swarm, configs, current_rows)
            run_staggered_node_segment(swarm, configs, current_rows, segment_index)
        for i, config in enumerate(configs):
            current_rows[i] += config["node_row_direction"]
            print(
                f"    {config['name']} detected pad {pad_at_physical_row(config, current_rows[i])}.",
                flush=True,
            )
    print("  Segmented node-to-node flight complete.", flush=True)

    set_phase_all("arrived_target_node")


def logger_loop(
    swarm,
    configs,
    experiment,
    run_id,
    coordination_path,
    battery_timeseries_path,
    drone_paths,
    experiment_start_time,
    node_start_time,
):
    global logging_active
    last_live_battery_report = 0.0
    while logging_active:
        now = time.time()
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        elapsed = round(now - experiment_start_time, 3)
        node_elapsed = round(now - node_start_time, 3)

        snapshots = []
        positions = {}
        for idx, tello in enumerate(swarm.tellos):
            state = get_state_safe(tello)
            x_global, y_global, z_global = to_global(configs[idx], state)
            snapshots.append((state, x_global, y_global, z_global, tello.get_battery()))
            if x_global is not None and y_global is not None:
                positions[idx] = (x_global, y_global)

        mean_spacing_error, max_spacing_error = spacing_stats(configs, positions)

        if now - last_live_battery_report >= 3.0:
            status_parts = []
            for idx, (_, _, _, _, battery) in enumerate(snapshots):
                config = configs[idx]
                status_parts.append(f"{config['name']} battery_id={config['battery_id']} battery={battery}%")
            print("Live battery status: " + " | ".join(status_parts), flush=True)
            last_live_battery_report = now

        for idx, (state, x_global, y_global, z_global, battery) in enumerate(snapshots):
            config = configs[idx]
            err_x = config["target_x"] - x_global if x_global is not None else None
            err_y = config["target_y"] - y_global if y_global is not None else None
            err_z = config["target_z"] - z_global if z_global is not None else None
            err_dist = (
                round(math.sqrt(err_x ** 2 + err_y ** 2 + err_z ** 2), 3)
                if err_x is not None and err_y is not None and err_z is not None
                else None
            )
            row = [
                run_id,
                experiment["experiment_id"],
                experiment.get("formation", ""),
                experiment.get("wind_direction", ""),
                experiment.get("wind_speed", ""),
                experiment_inter_drone_distance_cm(experiment),
                config["name"],
                config["ip"],
                config["battery_id"],
                config["takeoff_order"],
                config["role"],
                config["mission_pad"],
                config["grid_column"],
                config["grid_row"],
                get_phase(idx),
                timestamp,
                elapsed,
                node_elapsed,
                node_elapsed,
                state["mid"],
                state["x"],
                state["y"],
                state["z"],
                x_global,
                y_global,
                z_global,
                config["target_x"],
                config["target_y"],
                config["target_z"],
                config["target_pad"],
                config["node_forward_distance_cm"],
                config["node_speed_cm_s"],
                err_x,
                err_y,
                err_z,
                err_dist,
                mean_spacing_error,
                max_spacing_error,
                battery,
                hover_start_batteries.get(config["ip"], ""),
                hover_end_batteries.get(config["ip"], ""),
                state["yaw"],
                state["pitch"],
                state["roll"],
                state["vgx"],
                state["vgy"],
                state["vgz"],
                state["agx"],
                state["agy"],
                state["agz"],
                state["templ"],
                state["temph"],
                state["tof"],
                state["h"],
                state["baro"],
                state["motor_time"],
            ]
            append_row(coordination_path, row)
            append_row(drone_paths[config["ip"]]["coordination"], row)
            battery_start = hover_start_batteries.get(config["ip"], "")
            try:
                battery_drop_from_start = int(battery_start) - int(battery)
            except (TypeError, ValueError):
                battery_drop_from_start = ""
            append_row(battery_timeseries_path, [
                run_id,
                experiment["experiment_id"],
                experiment.get("formation", ""),
                experiment.get("wind_direction", ""),
                experiment.get("wind_speed", ""),
                experiment_inter_drone_distance_cm(experiment),
                config["name"],
                config["ip"],
                config["battery_id"],
                config["takeoff_order"],
                config["role"],
                config["mission_pad"],
                config["target_pad"],
                get_phase(idx),
                timestamp,
                elapsed,
                node_elapsed,
                battery,
                battery_start,
                battery_drop_from_start,
            ])
        time.sleep(LOG_INTERVAL_SEC)


def save_battery_rows(path, drone_paths, configs, experiment, run_id, node_start_timestamp, node_end_timestamp, duration):
    for config in configs:
        start = hover_start_batteries.get(config["ip"], "")
        end = hover_end_batteries.get(config["ip"], "")
        try:
            drop = int(start) - int(end)
        except (TypeError, ValueError):
            drop = ""
        row = [
            run_id,
            experiment["experiment_id"],
            experiment.get("formation", ""),
            experiment.get("wind_direction", ""),
            experiment.get("wind_speed", ""),
            experiment_inter_drone_distance_cm(experiment),
            config["name"],
            config["ip"],
            config["battery_id"],
            config["takeoff_order"],
            config["role"],
            config["mission_pad"],
            config["grid_column"],
            config["grid_row"],
            node_start_timestamp,
            node_end_timestamp,
            duration,
            node_start_timestamp,
            node_end_timestamp,
            duration,
            config["target_pad"],
            config["node_forward_distance_cm"],
            config["node_speed_cm_s"],
            start,
            end,
            drop,
        ]
        append_row(path, row)
        append_row(drone_paths[config["ip"]]["battery"], row)


def read_battery_window_status(swarm, configs, label):
    print(f"\n{label}", flush=True)
    readings = {}
    high_battery = []
    for idx, tello in enumerate(swarm.tellos):
        config = configs[idx]
        battery_percent = tello.get_battery()
        try:
            battery_value = int(battery_percent)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{config['name']} ({config['ip']}) returned invalid battery value: {battery_percent}"
            ) from exc
        readings[config["ip"]] = battery_value
        print(
            f"  {config['name']} ({config['ip']}, battery {config['battery_id']}): {battery_value}%",
            flush=True,
        )
        if battery_value > BATTERY_WINDOW_HIGH_PERCENT:
            high_battery.append((idx, config, battery_value))

    return readings, high_battery


def connect_and_check(swarm, configs):
    for idx, tello in enumerate(swarm.tellos):
        config = configs[idx]
        print(f"  Connecting {config['name']} ({config['ip']}, battery {config['battery_id']})...", flush=True)
        try:
            tello.connect()
        except Exception as exc:
            raise RuntimeError(f"Failed to connect {config['name']} at {config['ip']}: {exc}") from exc

        battery_percent = tello.get_battery()
        print(f"  OK - battery: {battery_percent}%", flush=True)
        try:
            int(battery_percent)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{config['name']} ({config['ip']}) returned invalid battery value: {battery_percent}"
            ) from exc

    return read_battery_window_status(
        swarm,
        configs,
        f"Battery window check ({BATTERY_WINDOW_LOW_PERCENT}-{BATTERY_WINDOW_HIGH_PERCENT}% required):",
    )


def run_selected_parallel(indexed_tellos, label, command_func):
    errors = []
    errors_lock = threading.Lock()

    def worker(idx, tello):
        try:
            command_func(idx, tello)
        except Exception as exc:
            with errors_lock:
                errors.append((idx, exc))

    threads = []
    for idx, tello in indexed_tellos:
        thread = threading.Thread(target=worker, args=(idx, tello), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        details = "; ".join(f"drone_index={idx}: {exc}" for idx, exc in errors)
        raise RuntimeError(f"{label} failed: {details}")


def discharge_high_battery_drones(swarm, configs, high_battery):
    high_indices = [idx for idx, _, _ in high_battery]
    indexed_tellos = [(idx, swarm.tellos[idx]) for idx in high_indices]
    print(
        "\nBattery discharge mode: these drones are above "
        f"{BATTERY_WINDOW_HIGH_PERCENT}% and will hover until they reach the window:",
        flush=True,
    )
    for _, config, battery_value in high_battery:
        print(
            f"  {config['name']} ip={config['ip']} battery_id={config['battery_id']} current={battery_value}%",
            flush=True,
        )

    set_phase_all("battery_discharge_idle")
    for idx in high_indices:
        set_phase(idx, "battery_discharge_takeoff")

    print(
        "Taking off high-battery drones for discharge hover. "
        "Mission pads are not required in discharge mode, and no experiment data will be recorded.",
        flush=True,
    )
    run_selected_parallel(indexed_tellos, "Battery discharge takeoff", lambda idx, tello: tello.takeoff())
    time.sleep(2.5)

    for idx in high_indices:
        set_phase(idx, "battery_discharge_hover")
    run_selected_parallel(
        indexed_tellos,
        "Battery discharge hover hold",
        lambda idx, tello: tello.send_rc_control(0, 0, 0, 0),
    )
    time.sleep(1.0)

    active = set(high_indices)
    deadline = time.time() + DISCHARGE_MAX_DURATION_SEC
    while active and time.time() < deadline:
        finished = []
        print("  Discharge battery status:", flush=True)
        for idx in sorted(active):
            tello = swarm.tellos[idx]
            config = configs[idx]
            battery_value = int(tello.get_battery())
            print(
                f"    {config['name']} battery_id={config['battery_id']} battery={battery_value}%",
                flush=True,
            )
            if battery_value <= BATTERY_WINDOW_HIGH_PERCENT:
                finished.append(idx)
        for idx in finished:
            config = configs[idx]
            set_phase(idx, "battery_discharge_landing")
            print(
                f"  {config['name']} reached {BATTERY_WINDOW_HIGH_PERCENT}%. Landing this drone.",
                flush=True,
            )
            try:
                swarm.tellos[idx].land()
            except Exception as exc:
                print(f"  Warning: {config['name']} discharge landing returned error: {exc}", flush=True)
            active.remove(idx)
            time.sleep(0.5)
        if active:
            for idx in active:
                set_phase(idx, "battery_discharge_hover")
                try:
                    swarm.tellos[idx].send_rc_control(0, 0, 0, 0)
                except Exception as exc:
                    print(f"  Warning: {configs[idx]['name']} hover hold returned error: {exc}", flush=True)
            time.sleep(DISCHARGE_CHECK_INTERVAL_SEC)

    if active:
        for idx in sorted(active):
            config = configs[idx]
            print(
                f"  Warning: {config['name']} did not reach {BATTERY_WINDOW_HIGH_PERCENT}% before timeout; landing.",
                flush=True,
            )
            try:
                swarm.tellos[idx].land()
            except Exception as exc:
                print(f"  Warning: {config['name']} timeout landing returned error: {exc}", flush=True)
            time.sleep(0.5)
        raise RuntimeError("Battery discharge timed out before all high-battery drones reached the experiment window.")

    print("Battery discharge hover complete. All high-battery drones landed.", flush=True)


def safe_land_all(swarm, configs):
    for idx, tello in enumerate(swarm.tellos):
        try:
            set_phase(idx, "emergency_landing")
            tello.land()
            time.sleep(0.5)
        except Exception:
            pass


def land_all_with_tolerance(swarm, configs):
    errors = []
    errors_lock = threading.Lock()

    def worker(idx, tello):
        set_phase(idx, "landing")
        try:
            tello.land()
            print(f"  {configs[idx]['name']} landing command accepted.", flush=True)
        except Exception as exc:
            print(f"  Warning: {configs[idx]['name']} landing command returned error: {exc}", flush=True)
            with errors_lock:
                errors.append((idx, exc))

    threads = []
    for idx, tello in enumerate(swarm.tellos):
        thread = threading.Thread(target=worker, args=(idx, tello), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


def wait_for_each_drone_target_pad_and_record_battery(swarm, configs):
    target_batteries = {}
    errors = []
    lock = threading.Lock()

    def worker(idx, tello):
        config = configs[idx]
        try:
            set_phase(idx, "verify_target_pad")
            while not wait_for_pad(
                tello,
                config["target_pad"],
                timeout=TARGET_PAD_LOCK_TIMEOUT_SEC,
                label=f"{config['name']} final target pad",
            ):
                hold_position(tello, config, "final target pad verification")
                print(
                    f"  {config['name']} waiting for final target mission pad "
                    f"{config['target_pad']}. Press Stop to abort/land.",
                    flush=True,
                )

            print(
                f"{config['name']} detected target mission pad {config['target_pad']}; "
                "recording battery before automatic landing.",
                flush=True,
            )
            try:
                battery = str(tello.get_battery())
            except Exception as exc:
                battery = ""
                print(f"  Warning: {config['name']} battery read at target pad failed: {exc}", flush=True)

            set_phase(idx, "target_pad_hold")
            hold_position(tello, config, "target pad hold")
            with lock:
                target_batteries[config["ip"]] = battery
        except Exception as exc:
            with lock:
                errors.append((idx, exc))

    threads = []
    for idx, tello in enumerate(swarm.tellos):
        thread = threading.Thread(target=worker, args=(idx, tello), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        details = "; ".join(f"{configs[idx]['name']}: {exc}" for idx, exc in errors)
        raise RuntimeError(f"Target-pad verification failed: {details}")
    return target_batteries


def hold_all_until_stop(swarm, configs, reason):
    set_phase_all("holding_until_stop")
    print(
        f"{reason} All drones will keep hovering. Press Stop in the GUI to land.",
        flush=True,
    )
    while True:
        for idx, tello in enumerate(swarm.tellos):
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception as exc:
                print(f"  Warning: {configs[idx]['name']} hover hold returned error: {exc}", flush=True)
        time.sleep(0.5)


def print_plan(configs, experiment):
    print("\nExperiment loaded:", flush=True)
    print(f"  experiment_id : {experiment['experiment_id']}", flush=True)
    print(f"  formation     : {experiment.get('formation', '')}", flush=True)
    print(f"  wind          : {experiment.get('wind_direction', '')} / {experiment.get('wind_speed', '')}", flush=True)
    print(f"  distance      : {experiment_inter_drone_distance_cm(experiment)} cm", flush=True)
    print(f"  x spacing     : {experiment_column_spacing_cm(experiment)} cm", flush=True)
    for config in configs:
        print(
            f"  order={config['takeoff_order']} ip={config['ip']} battery={config['battery_id']} role={config['role']} "
            f"pad={config['mission_pad']} grid=({config['grid_column']},{config['grid_row']}) "
            f"target_pad={config['target_pad']} target_grid_row={config['target_grid_row']} "
            f"target=({config['target_x']},{config['target_y']},{config['target_z']})",
            flush=True,
        )


def run_collection(experiment_id):
    global logging_active
    DATA_DIR.mkdir(exist_ok=True)

    experiment = load_experiment(experiment_id)
    configs = build_tello_configs(experiment)
    reset_runtime_state(configs)
    (
        run_id,
        experiment_dir,
        coordination_path,
        battery_path,
        battery_timeseries_path,
        drone_paths,
        battery_plot_path,
        temperature_plot_path,
    ) = output_paths(experiment["experiment_id"], configs)
    output_files = run_output_files(coordination_path, battery_path, battery_timeseries_path, drone_paths) + [
        battery_plot_path,
        temperature_plot_path,
    ]
    run_completed = False
    cleanup_done = False
    write_header(coordination_path, COORDINATION_COLUMNS)
    write_header(battery_path, BATTERY_COLUMNS)
    write_header(battery_timeseries_path, BATTERY_TIMESERIES_COLUMNS)
    for paths in drone_paths.values():
        write_header(paths["coordination"], COORDINATION_COLUMNS)
        write_header(paths["battery"], BATTERY_COLUMNS)

    print_plan(configs, experiment)
    print(f"\nExperiment archive : {experiment_dir}", flush=True)
    print(f"Coordination output: {coordination_path}", flush=True)
    print(f"Battery output     : {battery_path}", flush=True)
    print(f"Battery time series: {battery_timeseries_path}", flush=True)

    swarm = TelloSwarm.fromIps([config["ip"] for config in configs])
    logger_thread = None
    experiment_start_time = time.time()
    takeoff_started = False

    try:
        print("\nPreflight: connecting and checking all drones...", flush=True)
        _, high_battery = connect_and_check(swarm, configs)
        for tello in swarm.tellos:
            tello.enable_mission_pads()
            tello.set_mission_pad_detection_direction(0)

        while high_battery:
            print(
                "\nSome drones are above the experiment battery window "
                f"({BATTERY_WINDOW_LOW_PERCENT}-{BATTERY_WINDOW_HIGH_PERCENT}%).",
                flush=True,
            )
            for _, config, battery_value in high_battery:
                print(
                    f"  {config['name']} ip={config['ip']} battery_id={config['battery_id']} "
                    f"battery={battery_value}% > {BATTERY_WINDOW_HIGH_PERCENT}%",
                    flush=True,
                )
            print(
                f"Press Enter to discharge high-battery drones to {BATTERY_WINDOW_HIGH_PERCENT}% before the experiment, or type skip to take off directly...",
                flush=True,
            )
            discharge_choice = input().strip().lower()
            if discharge_choice in {"skip", "no", "direct"}:
                print(
                    "Battery discharge hover skipped by user. Proceeding to formal takeoff confirmation.",
                    flush=True,
                )
                high_battery = []
                break
            takeoff_started = True
            discharge_high_battery_drones(swarm, configs, high_battery)
            _, high_battery = read_battery_window_status(
                swarm,
                configs,
                f"Post-discharge battery window check ({BATTERY_WINDOW_LOW_PERCENT}-{BATTERY_WINDOW_HIGH_PERCENT}% required):",
            )

        print("Preflight checks passed. All drones are inside the battery window.", flush=True)

        print("Press Enter to take off all five drones...", flush=True)
        input()

        set_phase_all("takeoff")
        print("Taking off all five drones...", flush=True)
        takeoff_started = True
        swarm.takeoff()
        monitor_takeoff_health(swarm, configs, duration=2.5)

        set_phase_all("acquire_start_pad")
        wait_for_all_expected_start_pads(swarm, configs)

        set_phase_all("coordinate_climb")
        coordinate_climb_on_start_pads(swarm, configs)

        set_phase_all("pre_node_settle")
        print(f"Settling for {PRE_NODE_SETTLE_SEC:.1f} seconds before node-to-node flight...", flush=True)
        time.sleep(PRE_NODE_SETTLE_SEC)

        node_start_timestamp = datetime.now().isoformat(timespec="milliseconds")
        for idx, tello in enumerate(swarm.tellos):
            hover_start_batteries[configs[idx]["ip"]] = str(tello.get_battery())
        print("Node-to-node battery baselines captured.", flush=True)

        set_phase_all("node_logging")
        node_start_time = time.time()
        logging_active = True
        logger_thread = threading.Thread(
            target=logger_loop,
            args=(
                swarm,
                configs,
                experiment,
                run_id,
                coordination_path,
                battery_timeseries_path,
                drone_paths,
                experiment_start_time,
                node_start_time,
            ),
            daemon=True,
        )
        logger_thread.start()

        print("Node-to-node logging started.", flush=True)
        fly_node_to_node(swarm, configs)

        formation = str(experiment.get("formation", "")).strip().lower()
        if formation == "front" or is_echalon_formation(formation) or is_vee_75cm_experiment(experiment):
            set_phase_all("verify_target_pad")
            print("Verifying final target mission pads before automatic landing.", flush=True)
            for idx, tello in enumerate(swarm.tellos):
                config = configs[idx]
                if not wait_for_pad(
                    tello,
                    config["target_pad"],
                    timeout=TARGET_PAD_LOCK_TIMEOUT_SEC,
                    label=f"{config['name']} final target pad",
                ):
                    raise RuntimeError(
                        f"{config['name']} failed to detect target mission pad {config['target_pad']}."
                    )
                print(f"{config['name']} detected target mission pad {config['target_pad']}.", flush=True)
            logging_active = False
            if logger_thread:
                logger_thread.join(timeout=2.0)
            node_end_timestamp = datetime.now().isoformat(timespec="milliseconds")
            node_duration = round(time.time() - node_start_time, 3)
            for idx, tello in enumerate(swarm.tellos):
                hover_end_batteries[configs[idx]["ip"]] = str(tello.get_battery())
        else:
            print("Verifying final target mission pads before automatic landing.", flush=True)
            target_batteries = wait_for_each_drone_target_pad_and_record_battery(swarm, configs)
            logging_active = False
            if logger_thread:
                logger_thread.join(timeout=2.0)
            node_end_timestamp = datetime.now().isoformat(timespec="milliseconds")
            node_duration = round(time.time() - node_start_time, 3)
            for config in configs:
                hover_end_batteries[config["ip"]] = target_batteries.get(config["ip"], "")
        save_battery_rows(
            battery_path,
            drone_paths,
            configs,
            experiment,
            run_id,
            node_start_timestamp,
            node_end_timestamp,
            node_duration,
        )
        print("Node-to-node flight complete. Battery summary saved.", flush=True)
        generate_battery_line_plot(
            coordination_path,
            battery_plot_path,
            experiment["experiment_id"],
            run_id,
        )
        generate_temperature_line_plot(
            coordination_path,
            temperature_plot_path,
            experiment["experiment_id"],
            run_id,
        )

        time.sleep(1.5)
        run_completed = True
        print("Experiment data saved. Landing all drones now.", flush=True)
        land_all_with_tolerance(swarm, configs)
        print("Experiment finished.", flush=True)

    except (ExperimentStopped, KeyboardInterrupt) as exc:
        logging_active = False
        print(f"\nExperiment stopped: {exc}", flush=True)
        if takeoff_started:
            safe_land_all(swarm, configs)
        if run_completed:
            print("Experiment data was already saved; keeping output files.", flush=True)
            return True
        cleanup_failed_run_outputs(experiment_dir, output_files)
        cleanup_done = True
        return False

    except Exception as exc:
        logging_active = False
        print(f"\nERROR: {exc}", flush=True)
        if takeoff_started:
            try:
                hold_all_until_stop(swarm, configs, "Error occurred.")
            except (ExperimentStopped, KeyboardInterrupt) as stop_exc:
                print(f"\nExperiment stopped after error: {stop_exc}", flush=True)
                safe_land_all(swarm, configs)
                cleanup_failed_run_outputs(experiment_dir, output_files)
                cleanup_done = True
                return False
        cleanup_failed_run_outputs(experiment_dir, output_files)
        cleanup_done = True
        raise

    finally:
        logging_active = False
        if logger_thread:
            logger_thread.join(timeout=2.0)
        if not takeoff_started:
            for tello in swarm.tellos:
                try:
                    tello.end()
                except Exception:
                    pass
        if not run_completed and not cleanup_done:
            cleanup_failed_run_outputs(experiment_dir, output_files)

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Collect five-drone node-to-node experiment data.")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID from database/experiment_registry.json")
    return parser.parse_args()


def main():
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(ExperimentStopped("GUI stop requested")))
    args = parse_args()
    ok = run_collection(args.experiment_id)
    if not ok:
        sys.exit(130)


if __name__ == "__main__":
    main()
