"""
Collect single-drone battery baseline data.

This script is started by app.py:

    python3 -u single_drone_baseline.py --drone-number 1 --battery-id B11 --mode hover

It keeps baseline data separate from formal five-drone experiment data under:

    database/baselines/drone_1_B11/
"""

import argparse
import csv
import json
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

from djitellopy import Tello


DATA_DIR = BASE_DIR / "database"
BASELINE_DIR = DATA_DIR / "baselines"
IP_PREFIX = "192.168.0."
DRONE_NUMBER_TO_IP_SUFFIX = {
    "1": "101",
    "2": "109",
    "3": "103",
    "4": "106",
    "5": "107",
}

TAKEOFF_HEIGHT_CM = 80
TAKEOFF_CLIMB_SPEED_CM_S = 20
ROW_SPACING_CM = 50
FORWARD_DISTANCE_CM = 250
FLIGHT_SPEED_CM_S = 10
HOVER_DURATION_SEC = 150
PRE_NODE_SETTLE_SEC = 2.0
HOVER_LANDING_BATTERY_PERCENT = 10
BASELINE_BATTERY_WINDOW_LOW_PERCENT = 40
BASELINE_BATTERY_WINDOW_HIGH_PERCENT = 75
LOG_INTERVAL_SEC = 1.0
RC_INTERVAL_SEC = 0.2
LONG_GO_RESPONSE_TIMEOUT_SEC = 40
DISCHARGE_HOVER_HEIGHT_CM = 80
WINDOWED_MOVEMENT_MODES = {"head_forward_250", "tail_forward_250"}
PAD_SEQUENCE = [1, 2, 3, 4, 5, 6, 7, 8]

MODE_LABELS = {
    "hover": "hover baseline",
    "head_forward_250": "head wind forward 250cm",
    "tail_forward_250": "tail wind forward 250cm",
    "side_forward_250": "side wind lateral 250cm",
}

MISSION_PAD_COLUMNS = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8, 1],
    [5, 6, 7, 8, 1, 2],
]

BASELINE_COLUMNS = [
    "run_id",
    "baseline_id",
    "drone_name",
    "drone_number",
    "drone_ip",
    "battery_id",
    "mode",
    "direction",
    "baseline_path",
    "phase",
    "timestamp",
    "elapsed_time",
    "mid",
    "x",
    "y",
    "z",
    "battery",
    "battery_start",
    "battery_drop_from_start",
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

SUMMARY_COLUMNS = [
    "run_id",
    "baseline_id",
    "drone_name",
    "drone_number",
    "drone_ip",
    "battery_id",
    "mode",
    "direction",
    "baseline_path",
    "start_timestamp",
    "end_timestamp",
    "duration_sec",
    "battery_start",
    "battery_end",
    "battery_drop",
    "templ_start",
    "templ_end",
    "temph_start",
    "temph_end",
    "end_reason",
    "notes",
]


logging_active = False
current_phase = "idle"
phase_lock = threading.Lock()


class BaselineStopped(Exception):
    pass


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def normalize_battery_id(value):
    text = str(value or "").strip().upper().replace(" ", "")
    if not text:
        return ""
    if text.isdigit():
        return f"B{int(text):02d}"
    if text.startswith("B") and text[1:].isdigit():
        return f"B{int(text[1:]):02d}"
    return text


def baseline_path(start_col, start_row, direction):
    if start_col is None or start_row is None:
        return []
    step = -1 if direction == "down" else 1
    pads = []
    row = start_row
    while 0 <= row < len(MISSION_PAD_COLUMNS[start_col]):
        pads.append(MISSION_PAD_COLUMNS[start_col][row])
        row += step
    return pads


def lane_pad_sequence(start_col, min_rows=None):
    start_pad = MISSION_PAD_COLUMNS[start_col][0]
    start_idx = PAD_SEQUENCE.index(start_pad)
    row_count = min_rows or len(PAD_SEQUENCE)
    return [PAD_SEQUENCE[(start_idx + offset) % len(PAD_SEQUENCE)] for offset in range(row_count)]


def pad_at_physical_row(start_col, row_idx):
    if row_idx < 0:
        raise ValueError(f"row_idx cannot be negative: {row_idx}")
    lane = lane_pad_sequence(start_col, min_rows=row_idx + 1)
    return lane[row_idx]


def node_direction_from_baseline(direction):
    return -1 if direction == "down" else 1


def movement_config(start_col, start_row, direction):
    step = node_direction_from_baseline(direction)
    segment_count = int(round(FORWARD_DISTANCE_CM / ROW_SPACING_CM))
    target_row = start_row + step * segment_count
    if target_row < 0:
        raise ValueError("Selected start pad does not leave 250cm of travel in the down direction.")
    start_pad = pad_at_physical_row(start_col, start_row)
    target_pad = pad_at_physical_row(start_col, target_row)
    path_rows = range(start_row, target_row + step, step)
    path_pads = [pad_at_physical_row(start_col, row_idx) for row_idx in path_rows]
    return {
        "node_row_direction": step,
        "node_segment_count": segment_count,
        "target_row": target_row,
        "target_pad": target_pad,
        "path_pads": path_pads,
        "path_text": " -> ".join(str(pad) for pad in path_pads),
        "start_pad": start_pad,
        "start_x": start_col * ROW_SPACING_CM,
        "start_y": start_row * ROW_SPACING_CM,
        "target_x": start_col * ROW_SPACING_CM,
        "target_y": target_row * ROW_SPACING_CM,
        "target_z": TAKEOFF_HEIGHT_CM,
        "movement_distance_cm": FORWARD_DISTANCE_CM,
    }


def write_header(path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)


def append_row(path, row):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def get_phase():
    with phase_lock:
        return current_phase


def set_phase(phase):
    global current_phase
    with phase_lock:
        current_phase = phase


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


def wait_for_pad(tello, pad_id=None, timeout=8.0, interval=0.15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = get_state_safe(tello)
        if pad_id is None and state["mid"] != -1:
            return int(state["mid"])
        if pad_id is not None and int(state["mid"]) == int(pad_id):
            return int(state["mid"])
        time.sleep(interval)
    return None


def logger_loop(tello, config, data_path, start_time, battery_start):
    global logging_active
    last_live_report = 0.0
    while logging_active:
        now = time.time()
        state = get_state_safe(tello)
        battery = tello.get_battery()
        try:
            battery_drop = int(battery_start) - int(battery)
        except (TypeError, ValueError):
            battery_drop = ""

        if now - last_live_report >= 3.0:
            print(
                f"Live battery status: {config['drone_name']} battery_id={config['battery_id']} battery={battery}%",
                flush=True,
            )
            last_live_report = now

        append_row(data_path, [
            config["run_id"],
            config["baseline_id"],
            config["drone_name"],
            config["drone_number"],
            config["ip"],
            config["battery_id"],
            config["mode"],
            config["direction"],
            config["baseline_path"],
            get_phase(),
            datetime.now().isoformat(timespec="milliseconds"),
            round(now - start_time, 3),
            state["mid"],
            state["x"],
            state["y"],
            state["z"],
            battery,
            battery_start,
            battery_drop,
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
        ])
        time.sleep(LOG_INTERVAL_SEC)


def active_hover_until_battery(tello, threshold_percent):
    next_rc_time = time.time()
    next_check_time = time.time()
    print(f"Hover-discharge running until battery reaches {threshold_percent}%.", flush=True)
    while True:
        now = time.time()
        if now >= next_rc_time:
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception as exc:
                print(f"Warning: send_rc_control failed: {exc}", flush=True)
            next_rc_time += RC_INTERVAL_SEC
        if now >= next_check_time:
            battery = tello.get_battery()
            height = ""
            try:
                height = tello.get_height()
            except Exception:
                pass
            print(f"Hover baseline status: battery={battery}% height={height}cm", flush=True)
            try:
                if int(battery) <= threshold_percent:
                    return f"battery reached {battery}%"
            except (TypeError, ValueError):
                pass
            next_check_time += 1.0
        time.sleep(0.02)


def battery_value_or_error(tello, drone_name, ip):
    battery_percent = tello.get_battery()
    try:
        return int(battery_percent)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{drone_name} ({ip}) returned invalid battery value: {battery_percent}"
        ) from exc


def run_window_discharge(tello, config):
    drone_name = config["drone_name"]
    set_phase("battery_discharge_takeoff")
    print("Taking off baseline drone for battery discharge hover. No baseline data will be recorded.", flush=True)
    tello.takeoff()
    time.sleep(2.5)
    set_phase("battery_discharge_climb")
    tello.go_xyz_speed_mid(
        0,
        0,
        DISCHARGE_HOVER_HEIGHT_CM,
        TAKEOFF_CLIMB_SPEED_CM_S,
        int(config["start_pad"]),
    )
    time.sleep(1.0)
    set_phase("battery_discharge_hover")
    active_hover_until_battery(tello, BASELINE_BATTERY_WINDOW_HIGH_PERCENT)
    set_phase("battery_discharge_landing")
    print(f"{drone_name} reached {BASELINE_BATTERY_WINDOW_HIGH_PERCENT}%. Landing discharge hover.", flush=True)
    tello.land()
    time.sleep(1.5)
    set_phase("battery_discharge_complete")


def check_windowed_baseline_battery(tello, config):
    drone_name = config["drone_name"]
    ip = config["ip"]
    battery_id = config["battery_id"]
    battery_value = battery_value_or_error(tello, drone_name, ip)
    print(
        f"Battery window check ({BASELINE_BATTERY_WINDOW_LOW_PERCENT}-{BASELINE_BATTERY_WINDOW_HIGH_PERCENT}% required):",
        flush=True,
    )
    print(f"  {drone_name} ({ip}, battery {battery_id}): {battery_value}%", flush=True)
    if battery_value < BASELINE_BATTERY_WINDOW_LOW_PERCENT:
        raise RuntimeError(
            f"Takeoff blocked: {drone_name} battery={battery_value}% < "
            f"{BASELINE_BATTERY_WINDOW_LOW_PERCENT}%."
        )
    if battery_value <= BASELINE_BATTERY_WINDOW_HIGH_PERCENT:
        return

    while battery_value > BASELINE_BATTERY_WINDOW_HIGH_PERCENT:
        print(
            "\nBaseline drone is above the experiment battery window "
            f"({BASELINE_BATTERY_WINDOW_LOW_PERCENT}-{BASELINE_BATTERY_WINDOW_HIGH_PERCENT}%).",
            flush=True,
        )
        print(
            f"  {drone_name} ip={ip} battery_id={battery_id} "
            f"battery={battery_value}% > {BASELINE_BATTERY_WINDOW_HIGH_PERCENT}%",
            flush=True,
        )
        print(
            f"Press Enter to discharge high-battery drones to {BASELINE_BATTERY_WINDOW_HIGH_PERCENT}% before the experiment, or type skip to take off directly...",
            flush=True,
        )
        discharge_choice = input().strip().lower()
        if discharge_choice in {"skip", "no", "direct"}:
            print("Battery discharge hover skipped by user. Proceeding to formal takeoff confirmation.", flush=True)
            return
        run_window_discharge(tello, config)
        battery_value = battery_value_or_error(tello, drone_name, ip)
        print(
            f"Post-discharge battery window check ({BASELINE_BATTERY_WINDOW_LOW_PERCENT}-{BASELINE_BATTERY_WINDOW_HIGH_PERCENT}% required):",
            flush=True,
        )
        print(f"  {drone_name} ({ip}, battery {battery_id}): {battery_value}%", flush=True)
        if battery_value < BASELINE_BATTERY_WINDOW_LOW_PERCENT:
            raise RuntimeError(
                f"Takeoff blocked: {drone_name} battery={battery_value}% < "
                f"{BASELINE_BATTERY_WINDOW_LOW_PERCENT}%."
            )


def generate_plots(data_path, plots_dir, baseline_id):
    rows = []
    with data_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    def floats(column):
        values = []
        for row in rows:
            try:
                values.append(float(row.get(column, "")))
            except (TypeError, ValueError):
                values.append(None)
        return values

    t = floats("elapsed_time")
    battery = floats("battery")
    templ = floats("templ")
    temph = floats("temph")
    x_vals = floats("x")
    y_vals = floats("y")
    z_vals = floats("z")
    valid_t_battery = [(ti, bi) for ti, bi in zip(t, battery) if ti is not None and bi is not None]
    valid_t_temp_l = [(ti, vi) for ti, vi in zip(t, templ) if ti is not None and vi is not None]
    valid_t_temp_h = [(ti, vi) for ti, vi in zip(t, temph) if ti is not None and vi is not None]
    valid_xyz = [
        (xi, yi, zi)
        for xi, yi, zi in zip(x_vals, y_vals, z_vals)
        if xi is not None and yi is not None and zi is not None
    ]

    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    phase_colors = {
        "pre_takeoff": "#dce7f2",
        "takeoff": "#dff0e7",
        "climb_to_80cm": "#dff0e7",
        "hover_to_10_percent": "#fff1c7",
        "hover": "#fff1c7",
        "head_forward_250": "#eadff5",
        "tail_forward_250": "#eadff5",
        "side_forward_250": "#eadff5",
        "landing": "#f2dddd",
        "complete": "#e7ecef",
    }

    def phase_ranges():
        ranges = []
        current_phase = None
        start = None
        last_t = None
        for row in rows:
            try:
                row_t = float(row.get("elapsed_time", ""))
            except (TypeError, ValueError):
                continue
            phase = row.get("phase", "") or "unknown"
            if current_phase is None:
                current_phase = phase
                start = row_t
            elif phase != current_phase:
                ranges.append((current_phase, start, last_t if last_t is not None else row_t))
                current_phase = phase
                start = row_t
            last_t = row_t
        if current_phase is not None and start is not None and last_t is not None:
            ranges.append((current_phase, start, last_t))
        return ranges

    def shade_phases(ax):
        used_labels = set()
        for phase, start, end in phase_ranges():
            if end <= start:
                continue
            color = phase_colors.get(phase, "#edf0f2")
            label = phase if phase not in used_labels else None
            ax.axvspan(start, end, color=color, alpha=0.35, linewidth=0, label=label)
            used_labels.add(phase)

    if valid_t_battery:
        out = plots_dir / f"{baseline_id}_battery_line.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        shade_phases(ax)
        ax.plot([item[0] for item in valid_t_battery], [item[1] for item in valid_t_battery], linewidth=2)
        ax.set_title(f"{baseline_id}: battery percentage")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Battery (%)")
        ax.axhline(HOVER_LANDING_BATTERY_PERCENT, color="#a23b3b", linestyle="--", linewidth=1, label="landing threshold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)

    if valid_t_temp_l or valid_t_temp_h:
        out = plots_dir / f"{baseline_id}_temperature_line.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        shade_phases(ax)
        if valid_t_temp_l:
            ax.plot([item[0] for item in valid_t_temp_l], [item[1] for item in valid_t_temp_l], label="templ")
        if valid_t_temp_h:
            ax.plot([item[0] for item in valid_t_temp_h], [item[1] for item in valid_t_temp_h], label="temph")
        ax.set_title(f"{baseline_id}: temperature")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Temperature (C)")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)

    if valid_xyz:
        out = plots_dir / f"{baseline_id}_position_trace.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].plot([item[0] for item in valid_xyz], [item[1] for item in valid_xyz], linewidth=1.8)
        axes[0].set_title("Mission pad local X/Y")
        axes[0].set_xlabel("x (cm)")
        axes[0].set_ylabel("y (cm)")
        axes[0].axis("equal")
        if t:
            valid_t_z = [(ti, zi) for ti, zi in zip(t, z_vals) if ti is not None and zi is not None]
            if valid_t_z:
                shade_phases(axes[1])
                axes[1].plot([item[0] for item in valid_t_z], [item[1] for item in valid_t_z], linewidth=1.8)
        axes[1].set_title("Height")
        axes[1].set_xlabel("Elapsed time (s)")
        axes[1].set_ylabel("z (cm)")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"{baseline_id}: position")
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)

    return outputs


def cleanup_incomplete(paths):
    print("Baseline stopped before normal completion. Preserving incomplete baseline data for review.", flush=True)
    for path in paths:
        if path.exists():
            print(f"  Kept {path}", flush=True)


def run_baseline(args):
    global logging_active
    battery_id = normalize_battery_id(args.battery_id)
    drone_number = str(args.drone_number)
    if drone_number not in DRONE_NUMBER_TO_IP_SUFFIX:
        raise ValueError(f"Unknown drone number: {drone_number}")
    if args.mode not in MODE_LABELS:
        raise ValueError(f"Unknown baseline mode: {args.mode}")
    if args.direction not in {"up", "down"}:
        raise ValueError(f"Unknown baseline direction: {args.direction}")

    start_col = args.start_col
    start_row = args.start_row
    if args.mode != "hover":
        if start_col is None or start_row is None or args.start_pad is None:
            raise ValueError("Movement baseline requires a selected start mission pad cell.")
        if start_col < 0 or start_col >= len(MISSION_PAD_COLUMNS):
            raise ValueError(f"start_col out of range: {start_col}")
        if start_row < 0 or start_row >= len(MISSION_PAD_COLUMNS[start_col]):
            raise ValueError(f"start_row out of range: {start_row}")
        expected_pad = MISSION_PAD_COLUMNS[start_col][start_row]
        if int(args.start_pad) != expected_pad:
            raise ValueError(f"Start pad mismatch: pad={args.start_pad}, board cell pad={expected_pad}.")

    move_config = {}
    if args.mode == "hover":
        path_pads = []
        path_text = ""
        movement_distance_cm = 0
    else:
        move_config = movement_config(start_col, start_row, args.direction)
        path_pads = move_config["path_pads"]
        path_text = move_config["path_text"]
        movement_distance_cm = move_config["movement_distance_cm"]

    ip = f"{IP_PREFIX}{DRONE_NUMBER_TO_IP_SUFFIX[drone_number]}"
    drone_name = f"drone_{drone_number}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_id = f"{drone_name}_{battery_id}_{safe_name(args.mode)}_{run_id}"
    baseline_dir = BASELINE_DIR / f"{drone_name}_{battery_id}"
    plots_dir = baseline_dir / "plots"
    data_path = baseline_dir / f"{baseline_id}_timeseries.csv"
    summary_path = baseline_dir / f"{baseline_id}_summary.csv"
    metadata_path = baseline_dir / f"{baseline_id}_metadata.json"
    output_paths = [data_path, summary_path, metadata_path]

    baseline_dir.mkdir(parents=True, exist_ok=True)
    write_header(data_path, BASELINE_COLUMNS)
    write_header(summary_path, SUMMARY_COLUMNS)
    metadata = {
        "run_id": run_id,
        "baseline_id": baseline_id,
        "drone_number": drone_number,
        "drone_name": drone_name,
        "drone_ip": ip,
        "battery_id": battery_id,
        "mode": args.mode,
        "mode_label": MODE_LABELS[args.mode],
        "start_pad": args.start_pad,
        "start_col": start_col,
        "start_row": start_row,
        "direction": args.direction,
        "baseline_path": path_pads,
        "movement_distance_cm": movement_distance_cm if args.mode != "hover" else 0,
        "target_pad": move_config.get("target_pad"),
        "target_row": move_config.get("target_row"),
        "landing_battery_percent": HOVER_LANDING_BATTERY_PERCENT if args.mode == "hover" else None,
        "notes": args.notes,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    config = {
        "run_id": run_id,
        "baseline_id": baseline_id,
        "drone_number": drone_number,
        "drone_name": drone_name,
        "ip": ip,
        "battery_id": battery_id,
        "mode": args.mode,
        "direction": args.direction,
        "baseline_path": path_text,
        "start_pad": args.start_pad,
        "target_pad": move_config.get("target_pad"),
        "target_row": move_config.get("target_row"),
        "movement_distance_cm": movement_distance_cm,
    }

    tello = Tello(host=ip)
    start_timestamp = ""
    end_timestamp = ""
    start_time = time.time()
    battery_start = ""
    battery_end = ""
    first_state = {}
    final_state = {}
    landed = False
    logger = None
    end_reason = ""

    print("\nSingle-drone baseline loaded:", flush=True)
    print(f"  baseline_id : {baseline_id}", flush=True)
    print(f"  drone       : {drone_name} ({ip})", flush=True)
    print(f"  battery     : {battery_id}", flush=True)
    print(f"  mode        : {MODE_LABELS[args.mode]}", flush=True)
    if args.mode == "hover":
        print(f"  hover rule  : no mission pad; land at {HOVER_LANDING_BATTERY_PERCENT}% battery", flush=True)
    else:
        print(f"  direction   : {args.direction}", flush=True)
        print(f"  path        : {path_text}", flush=True)
        print(f"  target pad  : {move_config['target_pad']} ({movement_distance_cm}cm)", flush=True)
    print(f"  data output : {data_path}", flush=True)

    try:
        print("\nPreflight: connecting baseline drone...", flush=True)
        print(f"  Connecting {drone_name} ({ip}, battery {battery_id})...", flush=True)
        tello.connect()
        battery = tello.get_battery()
        print(f"  OK - battery: {battery}%", flush=True)
        if args.mode != "hover":
            tello.enable_mission_pads()
            tello.set_mission_pad_detection_direction(0)
        else:
            print("  Hover baseline does not use mission pads.", flush=True)
        if args.mode in WINDOWED_MOVEMENT_MODES:
            check_windowed_baseline_battery(tello, config)
        print("Preflight checks passed.", flush=True)
        print("Press Enter to take off single baseline drone...", flush=True)
        input()
        print("[GUI] Takeoff confirmed.", flush=True)

        start_time = time.time()
        set_phase("pre_takeoff")

        set_phase("takeoff")
        tello.takeoff()

        if args.mode == "hover":
            battery_start = str(tello.get_battery())
            first_state = get_state_safe(tello)
            start_timestamp = datetime.now().isoformat(timespec="seconds")
            print(f"Baseline battery start captured before takeoff: {battery_start}%", flush=True)
            logging_active = True
            logger = threading.Thread(
                target=logger_loop,
                args=(tello, config, data_path, start_time, battery_start),
                daemon=True,
            )
            logger.start()
            print("Stabilising for 3 seconds before long hover-discharge.", flush=True)
            time.sleep(3.0)
            set_phase("hover_to_10_percent")
            end_reason = active_hover_until_battery(tello, HOVER_LANDING_BATTERY_PERCENT)
        else:
            time.sleep(2.5)
            set_phase("acquire_start_pad")
            detected_pad = wait_for_pad(tello, int(args.start_pad), timeout=8.0)
            if detected_pad is None:
                raise RuntimeError(f"{drone_name} failed to detect expected mission pad {args.start_pad}.")
            print(f"{drone_name} detected start mission pad {detected_pad}.", flush=True)

            set_phase("coordinate_climb")
            print(f"Climbing {drone_name} to {TAKEOFF_HEIGHT_CM} cm above start mission pad {args.start_pad}...", flush=True)
            tello.go_xyz_speed_mid(0, 0, TAKEOFF_HEIGHT_CM, TAKEOFF_CLIMB_SPEED_CM_S, int(args.start_pad))
            time.sleep(1.0)

            set_phase("pre_node_settle")
            print(f"Settling for {PRE_NODE_SETTLE_SEC:.1f} seconds before node-to-node flight...", flush=True)
            time.sleep(PRE_NODE_SETTLE_SEC)

            battery_start = str(tello.get_battery())
            first_state = get_state_safe(tello)
            start_timestamp = datetime.now().isoformat(timespec="seconds")
            print(f"Node-to-node battery baseline captured: {battery_start}%", flush=True)

            logging_active = True
            start_time = time.time()
            logger = threading.Thread(
                target=logger_loop,
                args=(tello, config, data_path, start_time, battery_start),
                daemon=True,
            )
            logger.start()

            set_phase("continuous_node_to_node")
            signed_distance = movement_distance_cm * move_config["node_row_direction"]
            command = f"go 0 {signed_distance} {TAKEOFF_HEIGHT_CM} {FLIGHT_SPEED_CM_S} m{int(args.start_pad)}"
            print(
                f"Flying {drone_name} continuously {movement_distance_cm} cm at {FLIGHT_SPEED_CM_S} cm/s. "
                "Intermediate mission pads are pass-through only.",
                flush=True,
            )
            print(f"Running baseline movement command: {command}", flush=True)
            tello.send_control_command(command, timeout=LONG_GO_RESPONSE_TIMEOUT_SEC)
            set_phase("verify_target_pad")
            if wait_for_pad(tello, int(move_config["target_pad"]), timeout=6.0) is None:
                raise RuntimeError(
                    f"{drone_name} failed to detect target mission pad {move_config['target_pad']} "
                    "after node-to-node flight."
                )
            print(f"{drone_name} detected target mission pad {move_config['target_pad']}.", flush=True)
            logging_active = False
            if logger:
                logger.join(timeout=2.0)
            end_reason = f"node-to-node complete; target pad {move_config['target_pad']} detected"

        set_phase("landing")
        tello.land()
        landed = True
        time.sleep(1.5)
        battery_end = str(tello.get_battery())
        final_state = get_state_safe(tello)
        end_timestamp = datetime.now().isoformat(timespec="seconds")
        duration = round(time.time() - start_time, 3)
        set_phase("complete")
        time.sleep(0.5)
        logging_active = False
        if logger:
            logger.join(timeout=2.0)

        try:
            battery_drop = int(battery_start) - int(battery_end)
        except (TypeError, ValueError):
            battery_drop = ""
        append_row(summary_path, [
            run_id,
            baseline_id,
            drone_name,
            drone_number,
            ip,
            battery_id,
            args.mode,
            args.direction,
            path_text,
            start_timestamp,
            end_timestamp,
            duration,
            battery_start,
            battery_end,
            battery_drop,
            first_state.get("templ", ""),
            final_state.get("templ", ""),
            first_state.get("temph", ""),
            final_state.get("temph", ""),
            end_reason,
            args.notes,
        ])
        generated = generate_plots(data_path, plots_dir, baseline_id)
        print("\nBaseline complete.", flush=True)
        print(f"  Summary: {summary_path}", flush=True)
        for path in generated:
            print(f"  Plot: {path}", flush=True)
        return True

    except (KeyboardInterrupt, BaselineStopped, Exception) as exc:
        logging_active = False
        print(f"\nERROR: {exc}", flush=True)
        try:
            if not landed:
                set_phase("landing")
                tello.land()
        except Exception as land_exc:
            print(f"  Warning: landing command failed: {land_exc}", flush=True)
        cleanup_incomplete(output_paths)
        return False
    finally:
        logging_active = False
        try:
            tello.end()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Collect single-drone battery baseline data.")
    parser.add_argument("--drone-number", required=True, choices=sorted(DRONE_NUMBER_TO_IP_SUFFIX))
    parser.add_argument("--battery-id", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODE_LABELS))
    parser.add_argument("--start-pad", type=int, choices=range(1, 9), default=None)
    parser.add_argument("--start-col", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=None)
    parser.add_argument("--direction", choices=["up", "down"], default="up")
    parser.add_argument("--hover-duration", type=float, default=HOVER_DURATION_SEC)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main():
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(BaselineStopped("GUI stop requested")))
    ok = run_baseline(parse_args())
    if not ok:
        sys.exit(130)


if __name__ == "__main__":
    main()
