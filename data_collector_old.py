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
BATTERY_WINDOW_LOW_PERCENT = 40
BATTERY_WINDOW_HIGH_PERCENT = 75
DISCHARGE_HOVER_HEIGHT_CM = 80
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

COORDINATION_COLUMNS = [
    "run_id",
    "experiment_id",
    "formation",
    "wind_direction",
    "wind_speed",
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


def clamp(value, low, high):
    return max(low, min(high, value))


def lane_pad_sequence(grid_column, min_rows=None):
    start_pad = MISSION_PAD_COLUMNS[grid_column][0]
    start_idx = PAD_SEQUENCE.index(start_pad)
    row_count = min_rows or len(PAD_SEQUENCE)
    return [PAD_SEQUENCE[(start_idx + offset) % len(PAD_SEQUENCE)] for offset in range(row_count)]


def target_pad_for_start(mission_pad, forward_distance_cm=NODE_FORWARD_DISTANCE_CM):
    steps = int(round(forward_distance_cm / ROW_SPACING_CM))
    start_idx = PAD_SEQUENCE.index(mission_pad)
    return PAD_SEQUENCE[(start_idx + steps) % len(PAD_SEQUENCE)]


def node_direction_for_experiment(experiment):
    formation = str(experiment.get("formation", "")).strip().lower()
    wind_direction = str(experiment.get("wind_direction", "")).strip().lower()
    if formation == "front" and wind_direction == "tail wind":
        return -1
    return 1


def pad_at_physical_row(config, row_idx):
    if row_idx < 0:
        raise ValueError(f"row_idx cannot be negative: {row_idx}")
    lane = lane_pad_sequence(config["grid_column"], min_rows=row_idx + 1)
    return lane[row_idx]


def pad_at_column_row(grid_column, row_idx):
    if row_idx < 0:
        raise ValueError(f"row_idx cannot be negative: {row_idx}")
    lane = lane_pad_sequence(grid_column, min_rows=row_idx + 1)
    return lane[row_idx]


def build_tello_configs(experiment):
    drones = experiment.get("drones", [])
    if len(drones) != 5:
        raise ValueError(f"Expected exactly 5 drones in experiment record, got {len(drones)}.")

    node_direction = node_direction_for_experiment(experiment)
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
        if grid_column < 0 or grid_column >= len(MISSION_PAD_COLUMNS):
            raise ValueError(f"grid_column out of range for {ip}: {grid_column}")
        if grid_row < 0 or grid_row >= len(MISSION_PAD_COLUMNS[grid_column]):
            raise ValueError(f"grid_row out of range for {ip}: {grid_row}")

        mission_pad = int_field(drone.get("mission_pad"), "mission_pad")
        expected_pad = MISSION_PAD_COLUMNS[grid_column][grid_row]
        if mission_pad != expected_pad:
            raise ValueError(
                f"Mission pad mismatch for {ip}: record pad={mission_pad}, "
                f"layout pad={expected_pad} at column={grid_column}, row={grid_row}"
            )

        position_key = (grid_column, grid_row)
        if position_key in seen_positions:
            raise ValueError(f"Two drones are assigned to the same grid position: {position_key}")
        seen_positions.add(position_key)
        start_x = grid_column * COLUMN_SPACING_CM
        start_y = grid_row * ROW_SPACING_CM
        if node_direction < 0:
            target_row = max(0, grid_row - default_steps)
        else:
            target_row = grid_row + default_steps
        target_pad = pad_at_column_row(grid_column, target_row)
        node_distance = abs(target_row - grid_row) * ROW_SPACING_CM

        configs.append({
            "name": f"drone_{idx + 1}",
            "ip": ip,
            "battery_id": battery_id,
            "takeoff_order": int_field(drone.get("takeoff_order", idx + 1), "takeoff_order"),
            "role": str(drone.get("role") or f"drone_{idx + 1}"),
            "mission_pad": mission_pad,
            "target_pad": target_pad,
            "grid_column": grid_column,
            "grid_row": grid_row,
            "target_grid_row": target_row,
            "node_row_direction": node_direction,
            "node_segment_count": abs(target_row - grid_row),
            "start_x": start_x,
            "start_y": start_y,
            "target_x": start_x,
            "target_y": target_row * ROW_SPACING_CM,
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


def pad_origin_for_detection(config, mid):
    min_row = min(config["grid_row"], config["target_grid_row"]) - 1
    max_row = max(config["grid_row"], config["target_grid_row"]) + 1
    column = lane_pad_sequence(config["grid_column"], min_rows=max(max_row + 2, len(PAD_SEQUENCE)))
    candidate_rows = [
        row_idx
        for row_idx, pad_id in enumerate(column)
        if pad_id == mid and min_row <= row_idx <= max_row
    ]
    if not candidate_rows:
        return None, None
    detected_row = candidate_rows[0]
    return config["grid_column"] * COLUMN_SPACING_CM, detected_row * ROW_SPACING_CM


def to_global(config, state):
    mid = state["mid"]
    if mid == -1:
        return None, None, None
    origin_x, origin_y = pad_origin_for_detection(config, mid)
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


def wait_for_pad(tello, pad_id, timeout=8.0, interval=0.15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = get_state_safe(tello)
        if state["mid"] == pad_id:
            return True
        time.sleep(interval)
    return False


def wait_for_expected_pad(tello, config, timeout=8.0, interval=0.15):
    return wait_for_pad(tello, config["mission_pad"], timeout=timeout, interval=interval)


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
    print(f"Climbing all drones to {TAKEOFF_HEIGHT_CM} cm above their start mission pads...", flush=True)
    set_phase_all("coordinate_climb")
    run_swarm_parallel_checked(
        swarm,
        "Coordinate climb",
        lambda i, tello: tello.go_xyz_speed_mid(
            0,
            0,
            TAKEOFF_HEIGHT_CM,
            TAKEOFF_CLIMB_SPEED_CM_S,
            configs[i]["mission_pad"],
        )
    )
    time.sleep(1.0)


def is_valid_column_detection(config, x_global):
    return abs(x_global - config["target_x"]) <= COLUMN_SPACING_CM * 0.75


def advance_to_target_pad(tello, config, speed=NODE_FLIGHT_SPEED_CM_S, tolerance=8, max_corrections=3):
    step = config["node_row_direction"]
    stop_row = config["target_grid_row"] + step
    for row_idx in range(config["grid_row"], stop_row, step):
        pad_y = row_idx * ROW_SPACING_CM
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
            if abs(pad_y - y_global) <= tolerance and abs(config["target_x"] - x_global) <= tolerance:
                break
            origin_x, origin_y = pad_origin_for_detection(config, state["mid"])
            if origin_x is None or origin_y is None:
                time.sleep(0.3)
                continue
            local_x = max(-500, min(500, int(round(config["target_x"] - origin_x))))
            local_y = max(-500, min(500, int(round(pad_y - origin_y))))
            local_z = max(20, min(500, int(round(TAKEOFF_HEIGHT_CM))))
            tello.go_xyz_speed_mid(local_x, local_y, local_z, max(10, min(100, speed)), int(state["mid"]))
            time.sleep(0.5)
        if row_idx != config["grid_row"] and not wait_for_pad(tello, expected_pad, timeout=4.0):
            raise RuntimeError(
                f"{config['name']} failed to detect mission pad {expected_pad} at row {row_idx}."
            )


def fly_synchronized_node_segment(tello, config, current_row, speed=NODE_FLIGHT_SPEED_CM_S, tolerance=8, max_corrections=3):
    next_row = current_row + config["node_row_direction"]
    next_pad_y = next_row * ROW_SPACING_CM
    expected_pad = pad_at_physical_row(config, next_row)

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

        if abs(next_pad_y - y_global) <= tolerance and abs(config["target_x"] - x_global) <= tolerance:
            break

        origin_x, origin_y = pad_origin_for_detection(config, state["mid"])
        if origin_x is None or origin_y is None:
            time.sleep(0.3)
            continue

        local_x = max(-500, min(500, int(round(config["target_x"] - origin_x))))
        local_y = max(-500, min(500, int(round(next_pad_y - origin_y))))
        local_z = max(20, min(500, int(round(TAKEOFF_HEIGHT_CM))))
        tello.go_xyz_speed_mid(local_x, local_y, local_z, max(10, min(100, speed)), int(state["mid"]))
        time.sleep(0.5)

    if not wait_for_pad(tello, expected_pad, timeout=4.0):
        raise RuntimeError(
            f"{config['name']} failed to detect mission pad {expected_pad} after synchronized segment."
        )


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

    current_rows = [config["grid_row"] for config in configs]
    for segment_index in range(segments):
        set_phase_all(f"node_segment_{segment_index + 1}_of_{segments}")
        print(f"  Segment {segment_index + 1}/{segments}", flush=True)
        run_swarm_parallel_checked(
            swarm,
            f"Node segment {segment_index + 1}",
            lambda i, tello: fly_synchronized_node_segment(
                tello,
                configs[i],
                current_rows[i],
                speed=NODE_FLIGHT_SPEED_CM_S,
            )
        )
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
    low_battery = []
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
        if battery_value < BATTERY_WINDOW_LOW_PERCENT:
            low_battery.append((config, battery_value))
        elif battery_value > BATTERY_WINDOW_HIGH_PERCENT:
            high_battery.append((idx, config, battery_value))

    if low_battery:
        print("\nPreflight battery window check failed:", flush=True)
        for config, battery_value in low_battery:
            print(
                f"  {config['name']} ip={config['ip']} battery_id={config['battery_id']} "
                f"battery={battery_value}% < {BATTERY_WINDOW_LOW_PERCENT}%",
                flush=True,
            )
        raise RuntimeError(
            f"Takeoff blocked: {len(low_battery)} drone(s) are below "
            f"{BATTERY_WINDOW_LOW_PERCENT}% battery."
        )
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

    print("Taking off high-battery drones for discharge hover. No experiment data will be recorded.", flush=True)
    run_selected_parallel(indexed_tellos, "Battery discharge takeoff", lambda idx, tello: tello.takeoff())
    time.sleep(2.5)

    for idx in high_indices:
        set_phase(idx, "battery_discharge_climb")
    run_selected_parallel(
        indexed_tellos,
        "Battery discharge climb",
        lambda idx, tello: tello.go_xyz_speed_mid(
            0,
            0,
            DISCHARGE_HOVER_HEIGHT_CM,
            TAKEOFF_CLIMB_SPEED_CM_S,
            configs[idx]["mission_pad"],
        ),
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
    for idx, tello in enumerate(swarm.tellos):
        set_phase(idx, "landing")
        try:
            tello.land()
            print(f"  {configs[idx]['name']} landing command accepted.", flush=True)
        except Exception as exc:
            print(f"  Warning: {configs[idx]['name']} landing command returned error: {exc}", flush=True)
        finally:
            time.sleep(0.5)


def print_plan(configs, experiment):
    print("\nExperiment loaded:", flush=True)
    print(f"  experiment_id : {experiment['experiment_id']}", flush=True)
    print(f"  formation     : {experiment.get('formation', '')}", flush=True)
    print(f"  wind          : {experiment.get('wind_direction', '')} / {experiment.get('wind_speed', '')}", flush=True)
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
    landing_attempted = False
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
        time.sleep(2.5)

        set_phase_all("acquire_start_pad")
        for idx, tello in enumerate(swarm.tellos):
            config = configs[idx]
            if not wait_for_expected_pad(tello, config):
                raise RuntimeError(
                    f"{config['name']} failed to detect expected mission pad {config['mission_pad']}."
                )
            print(f"{config['name']} detected start mission pad {config['mission_pad']}.", flush=True)

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

        set_phase_all("verify_target_pad")
        for idx, tello in enumerate(swarm.tellos):
            config = configs[idx]
            if not wait_for_pad(tello, config["target_pad"], timeout=6.0):
                raise RuntimeError(
                    f"{config['name']} failed to detect target mission pad {config['target_pad']} "
                    f"after node-to-node flight."
                )
            print(f"{config['name']} detected target mission pad {config['target_pad']}.", flush=True)

        logging_active = False
        if logger_thread:
            logger_thread.join(timeout=2.0)

        node_end_timestamp = datetime.now().isoformat(timespec="milliseconds")
        node_duration = round(time.time() - node_start_time, 3)
        for idx, tello in enumerate(swarm.tellos):
            hover_end_batteries[configs[idx]["ip"]] = str(tello.get_battery())
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

        set_phase_all("landing")
        print("Landing all five drones...", flush=True)
        landing_attempted = True
        land_all_with_tolerance(swarm, configs)
        time.sleep(1.5)
        run_completed = True
        print("Experiment finished.", flush=True)

    except (ExperimentStopped, KeyboardInterrupt) as exc:
        logging_active = False
        print(f"\nExperiment stopped: {exc}", flush=True)
        if takeoff_started:
            landing_attempted = True
            safe_land_all(swarm, configs)
        cleanup_failed_run_outputs(experiment_dir, output_files)
        cleanup_done = True
        return False

    except Exception as exc:
        logging_active = False
        print(f"\nERROR: {exc}", flush=True)
        if takeoff_started:
            landing_attempted = True
            safe_land_all(swarm, configs)
        cleanup_failed_run_outputs(experiment_dir, output_files)
        cleanup_done = True
        raise

    finally:
        logging_active = False
        if logger_thread:
            logger_thread.join(timeout=2.0)
        if not landing_attempted:
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
