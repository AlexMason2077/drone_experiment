"""
Hover column experiment for three Tello drones.

Experiment plan
1. Use one mission-pad column only: 1, 2, 3, 4, 5, 6, 7, 8.
2. Drone 102 starts on mission pad 1.
3. Drone 101 starts on mission pad 2.
4. Drone 103 starts on mission pad 3.
5. All three drones take off together, move to 80 cm hover height, and hold for 60 seconds.
6. After the hover window ends, all three drones land together.
7. The script logs mission pad position, global position, battery, and summary statistics.
"""

from djitellopy import TelloSwarm
import csv
import math
import os
import threading
import time
from datetime import datetime


# =========================================================================
# Constants
# =========================================================================
AVAILABLE_IP_SUFFIXES = ["101", "102", "103"]
IP_PREFIX = "192.168.0."

ROW_SPACING_CM = 50
LANE_X = 0
TAKEOFF_HEIGHT_CM = 80
HOVER_DURATION_SEC = 60
GO_SPEED = 20
POSITION_TOLERANCE_CM = 8
MAX_POSITION_CORRECTIONS = 3
FINAL_STABLE_SAMPLES = 3
LOG_INTERVAL = 0.1
PRE_HOVER_SETTLE_SEC = 2.0
HOVER_COMMAND_INTERVAL_SEC = 0.25
START_PAD_TIMEOUT_SEC = 12.0
ENABLE_HORIZONTAL_FORMATION_HOLD = False
TAKEOFF_VERIFY_TIMEOUT_SEC = 5.0
TAKEOFF_RETRY_COUNT = 2
HOVER_KP_X = 0.18
HOVER_KP_Y = 0.18
HOVER_KP_Z = 0.28
HOVER_MAX_RC_XY = 8
HOVER_MAX_RC_Z = 10
HOVER_DEADBAND_XY_CM = 8
HOVER_DEADBAND_Z_CM = 8
HOVER_CONFIRM_CYCLES = 3
HOVER_SMOOTHING = 0.7
MIN_INTER_DRONE_Y_CM = 35

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "hover_column_summary.csv")

PAD_SEQUENCE = [1, 2, 3, 4, 5, 6, 7, 8]
DRONE_LAYOUT = [
    {"name": "drone_front", "role": "front", "suffix": "102", "start_pad": 1},
    {"name": "drone_middle", "role": "middle", "suffix": "101", "start_pad": 2},
    {"name": "drone_back", "role": "back", "suffix": "103", "start_pad": 3},
]

EXPERIMENT_TYPE = "hover_column"

CSV_COLUMNS = [
    "trial_id", "experiment_type", "hover_duration_sec",
    "drone_name", "drone_ip", "drone_role",
    "start_pad", "target_pad",
    "phase", "timestamp", "elapsed_time", "hover_elapsed_time",
    "mid", "x", "y", "z",
    "X_global", "Y_global", "Z_global",
    "yaw", "pitch", "roll",
    "vgx", "vgy", "vgz",
    "agx", "agy", "agz",
    "battery", "battery_hover_start", "battery_end", "battery_drop",
    "templ", "temph", "tof", "h", "baro", "motor_time",
    "target_x", "target_y", "target_z",
    "position_error_x", "position_error_y", "position_error_z", "position_error_dist",
]

SUMMARY_COLUMNS = [
    "trial_id", "experiment_type", "hover_duration_sec",
    "drone_name", "drone_ip", "drone_role",
    "start_pad", "target_pad",
    "target_x", "target_y", "target_z",
    "battery_hover_start", "battery_end", "battery_drop",
    "hover_start_timestamp", "hover_end_timestamp",
    "hover_duration_logged_sec",
    "normalized_battery_start", "normalized_battery_end", "normalized_battery_drop",
]


# =========================================================================
# Global mutable state
# =========================================================================
phase_lock = threading.Lock()
logging_active = False
current_phases = []
hover_reference_time = None
hover_start_timestamp = ""
hover_end_timestamp = ""
run_hover_start_batteries = {}
run_end_batteries = {}
TELLO_CONFIGS = []


# =========================================================================
# Config helpers
# =========================================================================
def build_tello_configs():
    configs = []
    for item in DRONE_LAYOUT:
        if item["suffix"] not in AVAILABLE_IP_SUFFIXES:
            raise ValueError(f"Unexpected suffix in config: {item['suffix']}")
        target_y = pad_origin_y(item["start_pad"])
        configs.append({
            "name": item["name"],
            "role": item["role"],
            "ip": f"{IP_PREFIX}{item['suffix']}",
            "start_pad": item["start_pad"],
            "target_pad": item["start_pad"],
            "target_x": LANE_X,
            "target_y": target_y,
            "target_z": TAKEOFF_HEIGHT_CM,
        })
    return configs


def reset_runtime_state():
    global current_phases, run_hover_start_batteries, run_end_batteries
    current_phases = ["idle"] * len(TELLO_CONFIGS)
    run_hover_start_batteries = {cfg["ip"]: "" for cfg in TELLO_CONFIGS}
    run_end_batteries = {cfg["ip"]: "" for cfg in TELLO_CONFIGS}


# =========================================================================
# Phase helpers
# =========================================================================
def set_phase(idx, phase):
    with phase_lock:
        current_phases[idx] = phase


def set_phase_all(phase):
    with phase_lock:
        for i in range(len(current_phases)):
            current_phases[i] = phase


def get_phase(idx):
    with phase_lock:
        return current_phases[idx]


# =========================================================================
# Trial ID
# =========================================================================
def get_next_trial_id():
    max_id = 0
    try:
        for fname in os.listdir(OUTPUT_DIR):
            if not fname.endswith(".csv") or "summary" in fname:
                continue
            fpath = os.path.join(OUTPUT_DIR, fname)
            try:
                with open(fpath, "r", newline="") as f:
                    for row in csv.DictReader(f):
                        try:
                            max_id = max(max_id, int(row.get("trial_id", 0)))
                        except (ValueError, TypeError):
                            pass
            except Exception:
                pass
    except Exception:
        pass
    return max_id + 1


# =========================================================================
# Telemetry helpers
# =========================================================================
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


def pad_origin_y(mid):
    if mid not in PAD_SEQUENCE:
        return None
    return PAD_SEQUENCE.index(mid) * ROW_SPACING_CM


def to_global(mid, x, y, z):
    origin_y = pad_origin_y(mid)
    if origin_y is None or mid == -1:
        return None, None, None
    return LANE_X + x, origin_y + y, z


def battery_drop_for_ip(ip):
    start = run_hover_start_batteries.get(ip, "")
    end = run_end_batteries.get(ip, "")
    try:
        return int(start) - int(end)
    except (TypeError, ValueError):
        return None


# =========================================================================
# CSV helpers
# =========================================================================
def init_summary_if_needed():
    rewrite_header = False

    if not os.path.exists(SUMMARY_FILE):
        rewrite_header = True
    else:
        try:
            with open(SUMMARY_FILE, "r", newline="") as f:
                first_row = next(csv.reader(f), [])
            if first_row != SUMMARY_COLUMNS:
                rewrite_header = True
        except Exception:
            rewrite_header = True

    if rewrite_header:
        existing_rows = []
        if os.path.exists(SUMMARY_FILE):
            try:
                with open(SUMMARY_FILE, "r", newline="") as f:
                    reader = csv.reader(f)
                    existing_rows = [row for row in reader if row]
            except Exception:
                existing_rows = []

        if existing_rows and existing_rows[0] == SUMMARY_COLUMNS:
            existing_rows = existing_rows[1:]

        with open(SUMMARY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(SUMMARY_COLUMNS)
            if existing_rows:
                writer.writerows(existing_rows)


def append_log_row(log_file, row):
    with open(log_file, "a", newline="") as f:
        csv.writer(f).writerow(row)


def append_summary_rows(trial_id):
    rows = []
    hover_logged = ""
    if hover_reference_time is not None and hover_end_timestamp:
        hover_logged = HOVER_DURATION_SEC

    for cfg in TELLO_CONFIGS:
        start = run_hover_start_batteries.get(cfg["ip"], "")
        end = run_end_batteries.get(cfg["ip"], "")
        drop = battery_drop_for_ip(cfg["ip"])
        normalized_end = 100 - drop if drop is not None else ""
        rows.append([
            trial_id, EXPERIMENT_TYPE, HOVER_DURATION_SEC,
            cfg["name"], cfg["ip"], cfg["role"],
            cfg["start_pad"], cfg["target_pad"],
            cfg["target_x"], cfg["target_y"], cfg["target_z"],
            start, end, drop if drop is not None else "",
            hover_start_timestamp, hover_end_timestamp,
            hover_logged,
            100, normalized_end, drop if drop is not None else "",
        ])

    with open(SUMMARY_FILE, "a", newline="") as f:
        csv.writer(f).writerows(rows)


# =========================================================================
# Logger thread
# =========================================================================
def logger_loop(swarm, trial_id, start_time, log_file):
    global logging_active
    while logging_active:
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        elapsed_time = round(time.time() - start_time, 3)
        hover_elapsed_time = ""
        if hover_reference_time is not None:
            hover_elapsed_time = round(max(0.0, time.time() - hover_reference_time), 3)

        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            state = get_state_safe(tello)
            x_global, y_global, z_global = to_global(
                state["mid"], state["x"], state["y"], state["z"]
            )
            pos_err_x = round(cfg["target_x"] - x_global, 3) if x_global is not None else None
            pos_err_y = round(cfg["target_y"] - y_global, 3) if y_global is not None else None
            pos_err_z = round(cfg["target_z"] - z_global, 3) if z_global is not None else None
            pos_err_dist = (
                round(math.sqrt(pos_err_x ** 2 + pos_err_y ** 2 + pos_err_z ** 2), 3)
                if None not in (pos_err_x, pos_err_y, pos_err_z)
                else None
            )
            battery = tello.get_battery()
            battery_drop = battery_drop_for_ip(cfg["ip"])

            row = [
                trial_id, EXPERIMENT_TYPE, HOVER_DURATION_SEC,
                cfg["name"], cfg["ip"], cfg["role"],
                cfg["start_pad"], cfg["target_pad"],
                get_phase(i), timestamp, elapsed_time, hover_elapsed_time,
                state["mid"], state["x"], state["y"], state["z"],
                x_global, y_global, z_global,
                state["yaw"], state["pitch"], state["roll"],
                state["vgx"], state["vgy"], state["vgz"],
                state["agx"], state["agy"], state["agz"],
                battery,
                run_hover_start_batteries.get(cfg["ip"], ""),
                run_end_batteries.get(cfg["ip"], ""),
                battery_drop if battery_drop is not None else "",
                state["templ"], state["temph"], state["tof"], state["h"], state["baro"], state["motor_time"],
                cfg["target_x"], cfg["target_y"], cfg["target_z"],
                pos_err_x, pos_err_y, pos_err_z, pos_err_dist,
            ]
            append_log_row(log_file, row)

        time.sleep(LOG_INTERVAL)


# =========================================================================
# Flight helpers
# =========================================================================
def wait_for_expected_pad(tello, expected_mid, timeout=6.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if get_state_safe(tello)["mid"] == expected_mid:
            return True
        time.sleep(interval)
    return False


def move_to_target_height_without_pad(tello, cfg):
    state = get_state_safe(tello)
    tof = state.get("tof", 0)
    if not isinstance(tof, (int, float)) or tof <= 0:
        return False

    delta = int(round(cfg["target_z"] - tof))
    if abs(delta) <= POSITION_TOLERANCE_CM:
        return True

    try:
        if delta > 0:
            tello.move_up(max(20, min(500, delta)))
        else:
            tello.move_down(max(20, min(500, abs(delta))))
        time.sleep(1.0)
        return True
    except Exception:
        return False


def is_probably_airborne(tello):
    state = get_state_safe(tello)
    tof = state.get("tof", 0)
    h = state.get("h", 0)
    z = state.get("z", 0)

    for value in (tof, h, z):
        if isinstance(value, (int, float)) and value >= 20:
            return True
    return False


def takeoff_all_with_verification(swarm):
    print("  Sending parallel takeoff command...")
    swarm.parallel(lambda i, tello: tello.takeoff())
    time.sleep(2.5)

    for attempt in range(1, TAKEOFF_RETRY_COUNT + 1):
        grounded = []
        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            if is_probably_airborne(tello):
                print(f"  {cfg['name']} appears airborne.")
            else:
                grounded.append((i, tello, cfg))

        if not grounded:
            return True

        if attempt > TAKEOFF_RETRY_COUNT:
            break

        print(f"  Warning: {len(grounded)} drone(s) still look grounded after takeoff. Retrying attempt {attempt}/{TAKEOFF_RETRY_COUNT}...")
        for i, tello, cfg in grounded:
            try:
                set_phase(i, "takeoff_retry")
                tello.takeoff()
            except Exception as exc:
                print(f"  Warning: {cfg['name']} takeoff retry returned error: {exc}")
            time.sleep(0.3)

        deadline = time.time() + TAKEOFF_VERIFY_TIMEOUT_SEC
        while time.time() < deadline:
            if all(is_probably_airborne(tello) for _, tello, _ in grounded):
                break
            time.sleep(0.2)

    failed = [TELLO_CONFIGS[i]["name"] for i, tello in enumerate(swarm.tellos) if not is_probably_airborne(tello)]
    if failed:
        raise RuntimeError(f"These drones did not confirm airborne state after takeoff: {', '.join(failed)}")
    return True


def move_to_hover_target(tello, cfg, speed=GO_SPEED, max_corrections=MAX_POSITION_CORRECTIONS):
    for _ in range(max_corrections):
        state = get_state_safe(tello)
        current_mid = state["mid"]
        if current_mid == -1:
            if move_to_target_height_without_pad(tello, cfg):
                return True
            time.sleep(0.3)
            continue

        tello.go_xyz_speed_mid(
            0,
            0,
            cfg["target_z"],
            speed,
            cfg["target_pad"],
        )
        time.sleep(1.0)

        stable, _ = is_hover_target_stable(tello, cfg)
        if stable:
            return True

    return False


def is_hover_target_stable(tello, cfg, tolerance=POSITION_TOLERANCE_CM, stable_samples=FINAL_STABLE_SAMPLES):
    consecutive_ok = 0
    last_state = None
    for _ in range(stable_samples * 4):
        state = get_state_safe(tello)
        x_global, y_global, z_global = to_global(state["mid"], state["x"], state["y"], state["z"])
        if None in (x_global, y_global, z_global):
            consecutive_ok = 0
            time.sleep(0.15)
            continue

        err_x = cfg["target_x"] - x_global
        err_y = cfg["target_y"] - y_global
        err_z = cfg["target_z"] - z_global
        last_state = (state["mid"], x_global, y_global, z_global, err_x, err_y, err_z)

        if all(abs(v) <= tolerance for v in (err_x, err_y, err_z)):
            consecutive_ok += 1
            if consecutive_ok >= stable_samples:
                return True, last_state
        else:
            consecutive_ok = 0
        time.sleep(0.15)

    return False, last_state


def safe_land_all(swarm):
    for i, tello in enumerate(swarm.tellos):
        try:
            set_phase(i, "emergency_landing")
            tello.land()
        except Exception as exc:
            cfg = TELLO_CONFIGS[i] if i < len(TELLO_CONFIGS) else {"name": f"drone_{i}"}
            print(f"  Warning: {cfg['name']} land command returned error: {exc}")
        finally:
            time.sleep(0.5)


def land_all_with_tolerance(swarm):
    for i, tello in enumerate(swarm.tellos):
        cfg = TELLO_CONFIGS[i]
        set_phase(i, "landing")
        try:
            tello.land()
            print(f"  {cfg['name']} landing command accepted.")
        except Exception as exc:
            print(f"  Warning: {cfg['name']} landing command returned error: {exc}")
        finally:
            time.sleep(0.5)


def active_hover_hold(swarm, duration_sec):
    deadline = time.time() + duration_sec

    while True:
        if time.time() >= deadline:
            break

        for i, tello in enumerate(swarm.tellos):
            try:
                set_phase(i, "hover_hold")
                tello.send_rc_control(0, 0, 0, 0)
            except Exception as exc:
                cfg = TELLO_CONFIGS[i] if i < len(TELLO_CONFIGS) else {"name": f"drone_{i}"}
                print(f"  Warning: {cfg['name']} hover hold command returned error: {exc}")

        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(HOVER_COMMAND_INTERVAL_SEC, remaining))

    for i, tello in enumerate(swarm.tellos):
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception as exc:
            cfg = TELLO_CONFIGS[i] if i < len(TELLO_CONFIGS) else {"name": f"drone_{i}"}
            print(f"  Warning: {cfg['name']} final hover hold command returned error: {exc}")


def clamp(value, low, high):
    return max(low, min(high, value))


def compute_altitude_hold_command(tello, cfg):
    state = get_state_safe(tello)
    tof = state.get("tof", 0)
    if isinstance(tof, (int, float)) and tof > 0:
        current_height = tof
    else:
        current_height = state.get("h", 0)

    err_z = cfg["target_z"] - current_height
    ud = 0
    if abs(err_z) > HOVER_DEADBAND_Z_CM:
        ud = int(round(clamp(HOVER_KP_Z * err_z, -HOVER_MAX_RC_Z, HOVER_MAX_RC_Z)))

    return ud, {
        "mid": state["mid"],
        "height": current_height,
        "err_z": err_z,
    }


def active_hover_hold_altitude_only(swarm, duration_sec):
    deadline = time.time() + duration_sec
    last_status_print = 0.0
    previous_ud = {cfg["name"]: 0.0 for cfg in TELLO_CONFIGS}
    confirm_counts = {cfg["name"]: 0 for cfg in TELLO_CONFIGS}

    while True:
        now = time.time()
        if now >= deadline:
            break

        snapshots = []
        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            name = cfg["name"]
            try:
                set_phase(i, "hover_hold_altitude_only")
                ud_candidate, snap = compute_altitude_hold_command(tello, cfg)
                snapshots.append((name, ud_candidate, snap))

                if ud_candidate == 0:
                    confirm_counts[name] = 0
                    final_ud = 0
                else:
                    confirm_counts[name] += 1
                    if confirm_counts[name] >= HOVER_CONFIRM_CYCLES:
                        smoothed = previous_ud[name] * HOVER_SMOOTHING + ud_candidate * (1.0 - HOVER_SMOOTHING)
                        final_ud = int(round(smoothed))
                    else:
                        final_ud = 0

                previous_ud[name] = final_ud
                tello.send_rc_control(0, 0, final_ud, 0)
            except Exception as exc:
                print(f"  Warning: {name} altitude-only hover command returned error: {exc}")

        if now - last_status_print >= 2.0 and snapshots:
            print("  Hover hold status:")
            print("    Safety mode: horizontal correction disabled; only altitude hold is active")
            for name, ud_candidate, snap in snapshots:
                print(
                    f"    {name}: mid={snap['mid']} H={snap['height']:.1f} "
                    f"err_z={snap['err_z']:.1f} raw_ud={ud_candidate} "
                    f"applied_ud={int(round(previous_ud[name]))}"
                )
            last_status_print = now

        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(HOVER_COMMAND_INTERVAL_SEC, remaining))

    for i, tello in enumerate(swarm.tellos):
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception as exc:
            cfg = TELLO_CONFIGS[i] if i < len(TELLO_CONFIGS) else {"name": f"drone_{i}"}
            print(f"  Warning: {cfg['name']} final altitude-only hover command returned error: {exc}")


def compute_hover_command(tello, cfg):
    state = get_state_safe(tello)
    x_global, y_global, _ = to_global(state["mid"], state["x"], state["y"], state["z"])

    err_x = None if x_global is None else cfg["target_x"] - x_global
    err_y = None if y_global is None else cfg["target_y"] - y_global

    tof = state.get("tof", 0)
    if isinstance(tof, (int, float)) and tof > 0:
        current_height = tof
    else:
        current_height = state.get("h", 0)
    err_z = cfg["target_z"] - current_height

    lr = 0
    fb = 0
    ud = 0

    # Keep only one horizontal correction axis active at a time to reduce oscillation.
    dominant_horizontal = None
    if err_x is not None and err_y is not None:
        dominant_horizontal = "x" if abs(err_x) >= abs(err_y) else "y"
    elif err_x is not None:
        dominant_horizontal = "x"
    elif err_y is not None:
        dominant_horizontal = "y"

    if dominant_horizontal == "x" and err_x is not None and abs(err_x) > HOVER_DEADBAND_XY_CM:
        lr = int(round(clamp(HOVER_KP_X * err_x, -HOVER_MAX_RC_XY, HOVER_MAX_RC_XY)))

    if dominant_horizontal == "y" and err_y is not None and abs(err_y) > HOVER_DEADBAND_XY_CM:
        fb = int(round(clamp(HOVER_KP_Y * err_y, -HOVER_MAX_RC_XY, HOVER_MAX_RC_XY)))

    if abs(err_z) > HOVER_DEADBAND_Z_CM:
        ud = int(round(clamp(HOVER_KP_Z * err_z, -HOVER_MAX_RC_Z, HOVER_MAX_RC_Z)))

    return lr, fb, ud, {
        "mid": state["mid"],
        "x_global": x_global,
        "y_global": y_global,
        "height": current_height,
        "err_x": err_x,
        "err_y": err_y,
        "err_z": err_z,
    }


def active_hover_hold_with_formation(swarm, duration_sec):
    deadline = time.time() + duration_sec
    last_status_print = 0.0
    previous_commands = {
        cfg["name"]: {"lr": 0.0, "fb": 0.0, "ud": 0.0}
        for cfg in TELLO_CONFIGS
    }
    confirm_counts = {
        cfg["name"]: {"lr": 0, "fb": 0, "ud": 0}
        for cfg in TELLO_CONFIGS
    }

    while True:
        now = time.time()
        if now >= deadline:
            break

        raw_commands = []
        snapshots = []
        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            try:
                set_phase(i, "hover_hold_closed_loop")
                lr, fb, ud, snap = compute_hover_command(tello, cfg)
                raw_commands.append((i, tello, cfg, lr, fb, ud, snap))
                snapshots.append((cfg["name"], lr, fb, ud, snap))
            except Exception as exc:
                print(f"  Warning: {cfg['name']} closed-loop hover command returned error: {exc}")

        # Safety guard: if front/middle/back spacing collapses too much, freeze horizontal motion.
        valid_y = []
        for _, _, cfg, _, _, _, snap in raw_commands:
            if snap["y_global"] is not None:
                valid_y.append((cfg["name"], snap["y_global"]))
        valid_y.sort(key=lambda item: item[1])
        freeze_horizontal = False
        for idx in range(len(valid_y) - 1):
            if abs(valid_y[idx + 1][1] - valid_y[idx][1]) < MIN_INTER_DRONE_Y_CM:
                freeze_horizontal = True
                break

        for i, tello, cfg, lr, fb, ud, snap in raw_commands:
            name = cfg["name"]

            if freeze_horizontal:
                lr = 0
                fb = 0

            # Require the same axis to request correction for several cycles before acting.
            axis_candidates = {"lr": lr, "fb": fb, "ud": ud}
            final_axes = {}
            for axis_name, candidate in axis_candidates.items():
                if candidate == 0:
                    confirm_counts[name][axis_name] = 0
                    final_axes[axis_name] = 0
                else:
                    confirm_counts[name][axis_name] += 1
                    if confirm_counts[name][axis_name] >= HOVER_CONFIRM_CYCLES:
                        prev = previous_commands[name][axis_name]
                        smoothed = prev * HOVER_SMOOTHING + candidate * (1.0 - HOVER_SMOOTHING)
                        final_axes[axis_name] = int(round(smoothed))
                    else:
                        final_axes[axis_name] = 0

            previous_commands[name]["lr"] = final_axes["lr"]
            previous_commands[name]["fb"] = final_axes["fb"]
            previous_commands[name]["ud"] = final_axes["ud"]

            try:
                tello.send_rc_control(final_axes["lr"], final_axes["fb"], final_axes["ud"], 0)
            except Exception as exc:
                print(f"  Warning: {name} closed-loop hover send returned error: {exc}")

        if now - last_status_print >= 2.0 and snapshots:
            print("  Hover hold status:")
            if freeze_horizontal:
                print(f"    Safety: horizontal correction frozen because spacing dropped below {MIN_INTER_DRONE_Y_CM} cm")
            for name, lr, fb, ud, snap in snapshots:
                applied = previous_commands[name]
                xg = "NA" if snap["x_global"] is None else f"{snap['x_global']:.1f}"
                yg = "NA" if snap["y_global"] is None else f"{snap['y_global']:.1f}"
                ex = "NA" if snap["err_x"] is None else f"{snap['err_x']:.1f}"
                ey = "NA" if snap["err_y"] is None else f"{snap['err_y']:.1f}"
                print(
                    f"    {name}: mid={snap['mid']} X={xg} Y={yg} H={snap['height']:.1f} "
                    f"err=({ex},{ey},{snap['err_z']:.1f}) raw=({lr},{fb},{ud}) "
                    f"applied=({applied['lr']},{applied['fb']},{applied['ud']})"
                )
            last_status_print = now

        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(HOVER_COMMAND_INTERVAL_SEC, remaining))

    for i, tello in enumerate(swarm.tellos):
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception as exc:
            cfg = TELLO_CONFIGS[i] if i < len(TELLO_CONFIGS) else {"name": f"drone_{i}"}
            print(f"  Warning: {cfg['name']} final closed-loop hover command returned error: {exc}")


def connect_swarm(swarm):
    for i, tello in enumerate(swarm.tellos):
        cfg = TELLO_CONFIGS[i]
        print(f"  Connecting {cfg['name']} ({cfg['ip']})...")
        tello.connect()
        print(f"  OK - battery: {tello.get_battery()}%")


# =========================================================================
# Main
# =========================================================================
def main():
    global logging_active
    global TELLO_CONFIGS
    global hover_reference_time
    global hover_start_timestamp
    global hover_end_timestamp

    TELLO_CONFIGS = build_tello_configs()
    reset_runtime_state()
    init_summary_if_needed()

    trial_id = get_next_trial_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"hover_column_{timestamp}.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)

    print("\nHover column experiment")
    print("  drone_front  -> 192.168.0.102 -> pad 1")
    print("  drone_middle -> 192.168.0.101 -> pad 2")
    print("  drone_back   -> 192.168.0.103 -> pad 3")
    print(f"  hover height: {TAKEOFF_HEIGHT_CM} cm")
    print(f"  hover time  : {HOVER_DURATION_SEC} s")
    print(f"\nTrial {trial_id} - logging to: {os.path.basename(log_file)}")

    swarm = TelloSwarm.fromIps([cfg["ip"] for cfg in TELLO_CONFIGS])
    logger_thread = None

    try:
        print("\nConnecting swarm...")
        connect_swarm(swarm)

        for tello in swarm.tellos:
            tello.enable_mission_pads()
            tello.set_mission_pad_detection_direction(0)
        time.sleep(1.0)

        input("\nPress Enter to take off all three drones...")
        set_phase_all("takeoff")
        takeoff_all_with_verification(swarm)

        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            set_phase(i, "acquire_origin_pad")
            if wait_for_expected_pad(tello, cfg["start_pad"], timeout=START_PAD_TIMEOUT_SEC):
                print(f"  {cfg['name']} locked pad {cfg['start_pad']}.")
            else:
                print(
                    f"  Warning: {cfg['name']} did not detect start pad {cfg['start_pad']} "
                    f"within {START_PAD_TIMEOUT_SEC:.0f}s. Continuing with height fallback."
                )

        set_phase_all("move_to_hover_target")
        print(f"\nMoving all drones to {TAKEOFF_HEIGHT_CM} cm hover...")
        swarm.parallel(lambda i, tello: move_to_hover_target(tello, TELLO_CONFIGS[i]))

        print("\nStability check at hover targets...")
        stability_ok = [False] * len(swarm.tellos)
        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            set_phase(i, "stability_check")
            ok, last = is_hover_target_stable(tello, cfg)
            stability_ok[i] = ok
            if last is not None:
                mid, x_global, y_global, z_global, err_x, err_y, err_z = last
                print(
                    f"  {cfg['name']} mid={mid} X={x_global:.1f} Y={y_global:.1f} Z={z_global:.1f} "
                    f"err=({err_x:.1f}, {err_y:.1f}, {err_z:.1f})"
                )
            if not ok:
                print(f"  Warning: {cfg['name']} did not pass stability check, but hover will continue.")

        set_phase_all("pre_hover_settle")
        print(f"\nSettling for {PRE_HOVER_SETTLE_SEC:.1f} seconds before timed hover...")
        time.sleep(PRE_HOVER_SETTLE_SEC)

        for i, tello in enumerate(swarm.tellos):
            run_hover_start_batteries[TELLO_CONFIGS[i]["ip"]] = str(tello.get_battery())

        start_time = time.time()
        hover_reference_time = start_time
        hover_start_timestamp = datetime.now().isoformat(timespec="milliseconds")
        logging_active = True
        logger_thread = threading.Thread(
            target=logger_loop,
            args=(swarm, trial_id, start_time, log_file),
            daemon=True,
        )
        logger_thread.start()

        if ENABLE_HORIZONTAL_FORMATION_HOLD:
            print(f"\nHovering for {HOVER_DURATION_SEC} seconds with closed-loop formation hold...")
            active_hover_hold_with_formation(swarm, HOVER_DURATION_SEC)
        else:
            print(f"\nHovering for {HOVER_DURATION_SEC} seconds in safety mode (altitude hold only)...")
            active_hover_hold_altitude_only(swarm, HOVER_DURATION_SEC)
        hover_end_timestamp = datetime.now().isoformat(timespec="milliseconds")

        print("\nLanding all drones...")
        land_all_with_tolerance(swarm)
        time.sleep(1.5)

        for i, tello in enumerate(swarm.tellos):
            run_end_batteries[TELLO_CONFIGS[i]["ip"]] = str(tello.get_battery())

        logging_active = False
        if logger_thread:
            logger_thread.join(timeout=2.0)
            logger_thread = None

        append_summary_rows(trial_id)

        print(f"\nTrial {trial_id} complete.")
        print(f"  Trajectory log : {log_file}")
        print(f"  Summary file   : {SUMMARY_FILE}")

        for cfg in TELLO_CONFIGS:
            start = run_hover_start_batteries.get(cfg["ip"], "")
            end = run_end_batteries.get(cfg["ip"], "")
            drop = battery_drop_for_ip(cfg["ip"])
            print(
                f"  {cfg['name']}: hover_start={start}% end={end}% "
                f"drop={drop if drop is not None else 'N/A'}%"
            )

    except Exception as exc:
        print(f"\nERROR: {exc}")
        safe_land_all(swarm)

    finally:
        logging_active = False
        if logger_thread:
            logger_thread.join(timeout=2.0)
        for tello in swarm.tellos:
            try:
                tello.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
