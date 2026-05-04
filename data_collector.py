"""
Collect five-drone hover data from an experiment record.

This script is designed to be started by app.py with an experiment ID:

    python3 -u data_collector.py --experiment-id EXP_ID

It reads database/experiment_registry.json, validates the saved drone IPs and
mission pad positions, connects to the Tello swarm, waits for GUI confirmation
at the takeoff prompt, then takes off, stabilizes over each configured start
mission pad, hovers for 60 seconds while logging coordination and battery data,
and lands.
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

from djitellopy import TelloSwarm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "database"
REGISTRY_FILE = DATA_DIR / "experiment_registry.json"

IP_PREFIX = "192.168.0."
ROW_SPACING_CM = 50
COLUMN_SPACING_CM = 50
TAKEOFF_HEIGHT_CM = 80
HOVER_DURATION_SEC = 60
LOG_INTERVAL_SEC = 0.1
GO_SPEED = 20
PRE_HOVER_SETTLE_SEC = 2.0
HOVER_COMMAND_INTERVAL_SEC = 0.2
HOVER_RECENTER_INTERVAL_SEC = 3.0
MIN_BATTERY_PERCENT = 15
TAKEOFF_BATTERY_MIN_PERCENT = 30

MISSION_PAD_COLUMNS = [
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 1],
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
    "battery_hover_start",
    "battery_hover_end",
    "battery_drop",
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


def build_tello_configs(experiment):
    drones = experiment.get("drones", [])
    if len(drones) != 5:
        raise ValueError(f"Expected exactly 5 drones in experiment record, got {len(drones)}.")

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

        configs.append({
            "name": f"drone_{idx + 1}",
            "ip": ip,
            "battery_id": battery_id,
            "takeoff_order": int_field(drone.get("takeoff_order", idx + 1), "takeoff_order"),
            "role": str(drone.get("role") or f"drone_{idx + 1}"),
            "mission_pad": mission_pad,
            "grid_column": grid_column,
            "grid_row": grid_row,
            "target_x": grid_column * COLUMN_SPACING_CM,
            "target_y": grid_row * ROW_SPACING_CM,
            "target_z": TAKEOFF_HEIGHT_CM,
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
    drone_paths = {}
    for config in configs:
        folder = drones_dir / drone_folder_name(config)
        folder.mkdir(parents=True, exist_ok=True)
        drone_paths[config["ip"]] = {
            "coordination": folder / f"{safe_name(experiment_id)}_{run_id}_{safe_name(config['name'])}_coordination.csv",
            "battery": folder / f"{safe_name(experiment_id)}_{run_id}_{safe_name(config['name'])}_battery.csv",
        }
    return run_id, experiment_dir, coordination_path, battery_path, drone_paths


def run_output_files(coordination_path, battery_path, drone_paths):
    paths = [coordination_path, battery_path]
    for items in drone_paths.values():
        paths.extend(items.values())
    return paths


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
    column = MISSION_PAD_COLUMNS[config["grid_column"]]
    if mid not in column:
        return None, None
    detected_row = column.index(mid)
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


def wait_for_expected_pad(tello, config, timeout=8.0, interval=0.15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = get_state_safe(tello)
        if state["mid"] == config["mission_pad"]:
            return True
        time.sleep(interval)
    return False


def recenter_on_mission_pads(swarm, configs, phase):
    set_phase_all(phase)
    swarm.parallel(
        lambda i, tello: tello.go_xyz_speed_mid(
            0,
            0,
            TAKEOFF_HEIGHT_CM,
            GO_SPEED,
            configs[i]["mission_pad"],
        )
    )
    time.sleep(1.0)


def coordinate_climb_on_start_pads(swarm, configs):
    print(f"Climbing all drones to {TAKEOFF_HEIGHT_CM} cm above their start mission pads...", flush=True)
    recenter_on_mission_pads(swarm, configs, "coordinate_climb")


def active_hover_hold(swarm, configs, duration_sec):
    deadline = time.time() + duration_sec
    last_reported = None
    last_recenter = 0.0

    while True:
        now = time.time()
        if now >= deadline:
            break

        if now - last_recenter >= HOVER_RECENTER_INTERVAL_SEC:
            print("Re-centering all drones above their configured mission pads...", flush=True)
            recenter_on_mission_pads(swarm, configs, "hover_recenter")
            last_recenter = time.time()
        else:
            for idx, tello in enumerate(swarm.tellos):
                try:
                    set_phase(idx, "hover_hold")
                    tello.send_rc_control(0, 0, 0, 0)
                except Exception as exc:
                    print(
                        f"  Warning: {configs[idx]['name']} hover hold command returned error: {exc}",
                        flush=True,
                    )

        remaining = max(0, int(deadline - time.time()))
        if remaining % 10 == 0 and remaining != last_reported:
            print(f"Hover remaining: {remaining}s", flush=True)
            last_reported = remaining

        sleep_for = min(HOVER_COMMAND_INTERVAL_SEC, max(0, deadline - time.time()))
        if sleep_for <= 0:
            break
        time.sleep(sleep_for)

    for idx, tello in enumerate(swarm.tellos):
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception as exc:
            print(
                f"  Warning: {configs[idx]['name']} final hover hold command returned error: {exc}",
                flush=True,
            )


def logger_loop(swarm, configs, experiment, run_id, coordination_path, drone_paths, experiment_start_time, hover_start_time):
    global logging_active
    while logging_active:
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        elapsed = round(time.time() - experiment_start_time, 3)
        hover_elapsed = round(time.time() - hover_start_time, 3)

        snapshots = []
        positions = {}
        for idx, tello in enumerate(swarm.tellos):
            state = get_state_safe(tello)
            x_global, y_global, z_global = to_global(configs[idx], state)
            snapshots.append((state, x_global, y_global, z_global, tello.get_battery()))
            if x_global is not None and y_global is not None:
                positions[idx] = (x_global, y_global)

        mean_spacing_error, max_spacing_error = spacing_stats(configs, positions)

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
                hover_elapsed,
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
        time.sleep(LOG_INTERVAL_SEC)


def save_battery_rows(path, drone_paths, configs, experiment, run_id, hover_start_timestamp, hover_end_timestamp, duration):
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
            hover_start_timestamp,
            hover_end_timestamp,
            duration,
            start,
            end,
            drop,
        ]
        append_row(path, row)
        append_row(drone_paths[config["ip"]]["battery"], row)


def connect_and_check(swarm, configs):
    low_battery = []
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
            battery_value = int(battery_percent)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{config['name']} ({config['ip']}) returned invalid battery value: {battery_percent}"
            ) from exc
        if battery_value < TAKEOFF_BATTERY_MIN_PERCENT:
            low_battery.append((config, battery_value))

    if low_battery:
        print("\nPreflight battery check failed:", flush=True)
        for config, battery_value in low_battery:
            print(
                f"  {config['name']} ip={config['ip']} battery_id={config['battery_id']} "
                f"battery={battery_value}% < {TAKEOFF_BATTERY_MIN_PERCENT}%",
                flush=True,
            )
        raise RuntimeError(
            f"Takeoff blocked: {len(low_battery)} drone(s) are below "
            f"{TAKEOFF_BATTERY_MIN_PERCENT}% battery."
        )


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
            f"target=({config['target_x']},{config['target_y']},{config['target_z']})",
            flush=True,
        )


def run_collection(experiment_id, hover_duration=HOVER_DURATION_SEC):
    global logging_active
    DATA_DIR.mkdir(exist_ok=True)

    experiment = load_experiment(experiment_id)
    configs = build_tello_configs(experiment)
    reset_runtime_state(configs)
    run_id, experiment_dir, coordination_path, battery_path, drone_paths = output_paths(experiment["experiment_id"], configs)
    output_files = run_output_files(coordination_path, battery_path, drone_paths)
    run_completed = False
    cleanup_done = False
    write_header(coordination_path, COORDINATION_COLUMNS)
    write_header(battery_path, BATTERY_COLUMNS)
    for paths in drone_paths.values():
        write_header(paths["coordination"], COORDINATION_COLUMNS)
        write_header(paths["battery"], BATTERY_COLUMNS)

    print_plan(configs, experiment)
    print(f"\nExperiment archive : {experiment_dir}", flush=True)
    print(f"Coordination output: {coordination_path}", flush=True)
    print(f"Battery output     : {battery_path}", flush=True)

    swarm = TelloSwarm.fromIps([config["ip"] for config in configs])
    logger_thread = None
    experiment_start_time = time.time()
    landing_attempted = False
    takeoff_started = False

    try:
        print("\nPreflight: connecting and checking all drones...", flush=True)
        connect_and_check(swarm, configs)
        for tello in swarm.tellos:
            tello.enable_mission_pads()
            tello.set_mission_pad_detection_direction(0)
        print("Preflight checks passed.", flush=True)

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

        set_phase_all("pre_hover_settle")
        print(f"Settling for {PRE_HOVER_SETTLE_SEC:.1f} seconds before timed hover...", flush=True)
        time.sleep(PRE_HOVER_SETTLE_SEC)

        hover_start_timestamp = datetime.now().isoformat(timespec="milliseconds")
        for idx, tello in enumerate(swarm.tellos):
            hover_start_batteries[configs[idx]["ip"]] = str(tello.get_battery())
        print("Hover battery baselines captured.", flush=True)

        set_phase_all("hover_logging")
        hover_start_time = time.time()
        logging_active = True
        logger_thread = threading.Thread(
            target=logger_loop,
            args=(swarm, configs, experiment, run_id, coordination_path, drone_paths, experiment_start_time, hover_start_time),
            daemon=True,
        )
        logger_thread.start()

        print(f"Hover logging started for {hover_duration} seconds.", flush=True)
        active_hover_hold(swarm, configs, hover_duration)

        logging_active = False
        if logger_thread:
            logger_thread.join(timeout=2.0)

        hover_end_timestamp = datetime.now().isoformat(timespec="milliseconds")
        for idx, tello in enumerate(swarm.tellos):
            hover_end_batteries[configs[idx]["ip"]] = str(tello.get_battery())
        save_battery_rows(
            battery_path,
            drone_paths,
            configs,
            experiment,
            run_id,
            hover_start_timestamp,
            hover_end_timestamp,
            hover_duration,
        )
        print("Hover complete. Battery summary saved.", flush=True)

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
    parser = argparse.ArgumentParser(description="Collect five-drone hover experiment data.")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID from database/experiment_registry.json")
    parser.add_argument("--hover-duration", type=int, default=HOVER_DURATION_SEC, help="Hover logging duration in seconds")
    return parser.parse_args()


def main():
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(ExperimentStopped("GUI stop requested")))
    args = parse_args()
    ok = run_collection(args.experiment_id, hover_duration=args.hover_duration)
    if not ok:
        sys.exit(130)


if __name__ == "__main__":
    main()
