from djitellopy import TelloSwarm
import csv
import os
import threading
import time
from datetime import datetime


# =========================
# Experiment settings
# =========================
AVAILABLE_IP_SUFFIXES = ["101", "104", "102"]
IP_PREFIX = "192.168.0."

ROW_SPACING_CM = 50
LANE_SPACING_CM = 50  # Adjust this if the physical spacing between columns is not 50 cm.
TARGET_FORWARD_Y_CM = 250
TAKEOFF_HEIGHT_CM = 80
FINAL_APPROACH_HEIGHT_CM = 40
GO_SPEED = 10
FINE_CORRECTION_SPEED = 10
POSITION_TOLERANCE = 8
FINAL_POSITION_TOLERANCE = 5
MAX_POSITION_CORRECTIONS = 3
FINAL_STABLE_SAMPLES = 2
LOG_INTERVAL = 0.1
ABORT_IF_NOT_WITHIN_TOLERANCE = False

LOG_FILE = "front_experiment_trajectory_log.csv"
SUMMARY_FILE = "front_experiment_battery_summary.csv"

LANE_ORDER = ["left", "middle", "right"]
LANE_X_OFFSETS = {
    "left": 0,
    "middle": LANE_SPACING_CM,
    "right": LANE_SPACING_CM * 2,
}

PAD_LAYOUTS = {
    "left": [1, 2, 3, 4, 5, 6, 7, 8],
    "middle": [2, 3, 4, 5, 6, 7, 8, 1],
    "right": [3, 4, 5, 6, 7, 8, 1, 2],
}

BASE_POSITIONS = [
    {
        "name": "drone_left",
        "lane": "left",
        "start_pad": 1,
        "target_pad": 6,
        "target_x": LANE_X_OFFSETS["left"],
        "target_y": TARGET_FORWARD_Y_CM,
    },
    {
        "name": "drone_middle",
        "lane": "middle",
        "start_pad": 2,
        "target_pad": 7,
        "target_x": LANE_X_OFFSETS["middle"],
        "target_y": TARGET_FORWARD_Y_CM,
    },
    {
        "name": "drone_right",
        "lane": "right",
        "start_pad": 3,
        "target_pad": 8,
        "target_x": LANE_X_OFFSETS["right"],
        "target_y": TARGET_FORWARD_Y_CM,
    },
]

TELLO_CONFIGS = []

CSV_COLUMNS = [
    "trial_id",
    "drone_name",
    "drone_ip",
    "lane",
    "start_pad",
    "target_pad",
    "phase",
    "timestamp",
    "elapsed_time",
    "mid",
    "x",
    "y",
    "z",
    "X_global",
    "Y_global",
    "Z_global",
    "yaw",
    "vgx",
    "vgy",
    "vgz",
    "battery",
    "battery_takeoff",
    "battery_end",
    "temph",
    "y_distance_diff_mean",
    "y_distance_diff_12_cm",
    "y_distance_diff_23_cm",
    "y_distance_diff_13_cm",
]

SUMMARY_COLUMNS = [
    "trial_id",
    "drone_name",
    "drone_ip",
    "lane",
    "start_pad",
    "target_pad",
    "battery_takeoff",
    "battery_end",
    "battery_drop",
]


phase_lock = threading.Lock()
logging_active = False
current_phases = []
run_takeoff_batteries = {}
run_end_batteries = {}


def build_tello_configs(order_input):
    suffixes = [part.strip() for part in order_input.split(",") if part.strip()]
    if len(suffixes) != 3:
        raise ValueError("Please enter exactly three IP suffixes, for example: 103,104,102")
    if sorted(suffixes) != sorted(AVAILABLE_IP_SUFFIXES):
        raise ValueError("The order must include exactly these drones: 101, 104, 102")

    configs = []
    for i, suffix in enumerate(suffixes):
        base = BASE_POSITIONS[i]
        configs.append(
            {
                "name": base["name"],
                "lane": base["lane"],
                "ip": f"{IP_PREFIX}{suffix}",
                "start_pad": base["start_pad"],
                "target_pad": base["target_pad"],
                "target_x": base["target_x"],
                "target_y": base["target_y"],
            }
        )
    return configs


def reset_runtime_state():
    global current_phases, run_takeoff_batteries, run_end_batteries
    current_phases = ["idle" for _ in TELLO_CONFIGS]
    run_takeoff_batteries = {cfg["ip"]: "" for cfg in TELLO_CONFIGS}
    run_end_batteries = {cfg["ip"]: "" for cfg in TELLO_CONFIGS}


def set_phase(drone_index, phase):
    with phase_lock:
        current_phases[drone_index] = phase


def set_phase_all(phase):
    with phase_lock:
        for i in range(len(current_phases)):
            current_phases[i] = phase


def get_phase(drone_index):
    with phase_lock:
        return current_phases[drone_index]


def get_next_trial_id(log_file):
    if not os.path.exists(log_file):
        return 1

    try:
        with open(log_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            trial_ids = []
            for row in reader:
                try:
                    trial_ids.append(int(row["trial_id"]))
                except Exception:
                    pass
        return max(trial_ids) + 1 if trial_ids else 1
    except Exception:
        return 1


def init_csv_if_needed(file_path, header):
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return

    with open(file_path, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        with open(file_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return

    if rows[0] == header:
        return

    idx_map = {name: i for i, name in enumerate(rows[0])}
    converted_rows = []
    for old_row in rows[1:]:
        converted_rows.append(
            [
                old_row[idx_map[col]] if idx_map.get(col, -1) < len(old_row) and idx_map.get(col, -1) >= 0 else ""
                for col in header
            ]
        )

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(converted_rows)


def get_state_safe(tello):
    state = tello.get_current_state()
    return {
        "mid": state.get("mid", -1),
        "x": state.get("x", 0),
        "y": state.get("y", 0),
        "z": state.get("z", 0),
        "yaw": state.get("yaw", 0),
        "vgx": state.get("vgx", 0),
        "vgy": state.get("vgy", 0),
        "vgz": state.get("vgz", 0),
        "temph": state.get("temph", 0),
    }


def get_pad_origin(lane, mid):
    row_sequence = PAD_LAYOUTS.get(lane, [])
    if mid not in row_sequence:
        return None, None

    row_index = row_sequence.index(mid)
    pad_origin_x = LANE_X_OFFSETS[lane]
    pad_origin_y = row_index * ROW_SPACING_CM
    return pad_origin_x, pad_origin_y


def to_global(lane, mid, x, y, z):
    if mid == -1:
        return None, None, None

    pad_origin_x, pad_origin_y = get_pad_origin(lane, mid)
    if pad_origin_x is None or pad_origin_y is None:
        return None, None, None

    return pad_origin_x + x, pad_origin_y + y, z


def append_log_row(log_file, row):
    with open(log_file, "a", newline="") as f:
        csv.writer(f).writerow(row)


def append_summary_rows(summary_file, trial_id):
    rows = []
    for config in TELLO_CONFIGS:
        start_battery = run_takeoff_batteries.get(config["ip"], "")
        end_battery = run_end_batteries.get(config["ip"], "")
        start_val = int(start_battery) if str(start_battery).strip() != "" else None
        end_val = int(end_battery) if str(end_battery).strip() != "" else None
        drop = (start_val - end_val) if start_val is not None and end_val is not None else ""

        rows.append(
            [
                trial_id,
                config["name"],
                config["ip"],
                config["lane"],
                config["start_pad"],
                config["target_pad"],
                start_battery,
                end_battery,
                drop,
            ]
        )

    with open(summary_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def logger_loop(swarm, trial_id, start_time, log_file, interval=0.1):
    global logging_active

    while logging_active:
        snapshot = []
        y_globals = []

        for i, tello in enumerate(swarm.tellos):
            config = TELLO_CONFIGS[i]
            s = get_state_safe(tello)
            Xg, Yg, Zg = to_global(config["lane"], s["mid"], s["x"], s["y"], s["z"])
            snapshot.append((s, Xg, Yg, Zg))
            y_globals.append(Yg)

        y_distance_diff_12 = None
        y_distance_diff_23 = None
        y_distance_diff_13 = None
        pairwise_diffs = []

        if len(y_globals) >= 2 and y_globals[0] is not None and y_globals[1] is not None:
            y_distance_diff_12 = round(abs(y_globals[0] - y_globals[1]), 3)
            pairwise_diffs.append(y_distance_diff_12)
        if len(y_globals) >= 3 and y_globals[1] is not None and y_globals[2] is not None:
            y_distance_diff_23 = round(abs(y_globals[1] - y_globals[2]), 3)
            pairwise_diffs.append(y_distance_diff_23)
        if len(y_globals) >= 3 and y_globals[0] is not None and y_globals[2] is not None:
            y_distance_diff_13 = round(abs(y_globals[0] - y_globals[2]), 3)
            pairwise_diffs.append(y_distance_diff_13)

        y_distance_diff_mean = round(sum(pairwise_diffs) / len(pairwise_diffs), 3) if pairwise_diffs else None

        for i, tello in enumerate(swarm.tellos):
            config = TELLO_CONFIGS[i]
            s, Xg, Yg, Zg = snapshot[i]

            row = [
                trial_id,
                config["name"],
                config["ip"],
                config["lane"],
                config["start_pad"],
                config["target_pad"],
                get_phase(i),
                datetime.now().isoformat(timespec="milliseconds"),
                round(time.time() - start_time, 3),
                s["mid"],
                s["x"],
                s["y"],
                s["z"],
                Xg,
                Yg,
                Zg,
                s["yaw"],
                s["vgx"],
                s["vgy"],
                s["vgz"],
                tello.get_battery(),
                run_takeoff_batteries.get(config["ip"], ""),
                run_end_batteries.get(config["ip"], ""),
                s["temph"],
                y_distance_diff_mean,
                y_distance_diff_12,
                y_distance_diff_23,
                y_distance_diff_13,
            ]

            append_log_row(log_file, row)
            print(row)

        time.sleep(interval)


def is_valid_lane_detection(Xg, lane):
    """Reject pad detections from adjacent lanes (cross-lane contamination)."""
    return abs(Xg - LANE_X_OFFSETS[lane]) <= LANE_SPACING_CM * 0.75


def advance_to_target_pad(tello, lane, target_x, target_y, target_z, speed=10, tolerance=8, max_corrections=3):
    """Navigate forward pad-by-pad, correcting both X (lane alignment) and Y (forward) at each step."""
    pad_sequence = PAD_LAYOUTS[lane]
    for step_idx, _ in enumerate(pad_sequence):
        pad_y = step_idx * ROW_SPACING_CM
        if pad_y > target_y + tolerance:
            break
        for _ in range(max_corrections):
            s = get_state_safe(tello)
            cur_mid = s["mid"]
            if cur_mid == -1:
                time.sleep(0.3)
                continue
            Xg, Yg, _ = to_global(lane, cur_mid, s["x"], s["y"], s["z"])
            if Xg is None or Yg is None:
                time.sleep(0.3)
                continue
            if not is_valid_lane_detection(Xg, lane):
                print(f"[advance] Cross-lane pad detected lane={lane} Xg={Xg:.1f}, ignoring")
                time.sleep(0.3)
                continue
            err_x = target_x - Xg
            err_y = pad_y - Yg
            if abs(err_y) <= tolerance and abs(err_x) <= tolerance:
                break
            pad_origin_x, pad_origin_y = get_pad_origin(lane, cur_mid)
            local_x = int(round(target_x - pad_origin_x))
            local_y = int(round(pad_y - pad_origin_y))
            local_z = int(round(target_z))
            local_x = max(-500, min(500, local_x))
            local_y = max(-500, min(500, local_y))
            local_z = max(20, min(500, local_z))
            tello.go_xyz_speed_mid(local_x, local_y, local_z, max(10, min(100, speed)), cur_mid)
            time.sleep(0.5)


def wait_for_expected_pad(tello, expected_mid, timeout=5.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if get_state_safe(tello)["mid"] == expected_mid:
            return True
        time.sleep(interval)
    return False


def go_to_global_target(tello, lane, target_x, target_y, target_z, speed=10, tolerance=8, max_corrections=3):
    for i in range(max_corrections):
        s = get_state_safe(tello)
        Xg, Yg, _ = to_global(lane, s["mid"], s["x"], s["y"], s["z"])

        if Xg is None or Yg is None:
            print(f"[Correction {i + 1}] No valid mission pad detected in lane {lane}.")
            time.sleep(0.3)
            continue

        if not is_valid_lane_detection(Xg, lane):
            print(f"[Correction {i + 1}] Cross-lane pad detected lane={lane} Xg={Xg:.1f}, ignoring.")
            time.sleep(0.3)
            continue

        err_x = target_x - Xg
        err_y = target_y - Yg
        print(
            f"[Correction {i + 1}] lane={lane}, current(X={Xg:.1f}, Y={Yg:.1f}), "
            f"target(X={target_x}, Y={target_y}), err(X={err_x:.1f}, Y={err_y:.1f})"
        )

        if abs(err_x) <= tolerance and abs(err_y) <= tolerance:
            return True

        mid = int(s["mid"])
        pad_origin_x, pad_origin_y = get_pad_origin(lane, mid)
        if pad_origin_x is None or pad_origin_y is None:
            time.sleep(0.3)
            continue

        local_target_x = int(round(target_x - pad_origin_x))
        local_target_y = int(round(target_y - pad_origin_y))
        local_target_z = int(round(target_z))

        local_target_x = max(-500, min(500, local_target_x))
        local_target_y = max(-500, min(500, local_target_y))
        local_target_z = max(20, min(500, local_target_z))
        local_speed = max(10, min(100, int(round(speed))))

        tello.go_xyz_speed_mid(local_target_x, local_target_y, local_target_z, local_speed, mid)
        time.sleep(0.5)

    return False


def is_global_target_stable(tello, lane, target_x, target_y, tolerance=5, stable_samples=2, sample_interval=0.15):
    consecutive_ok = 0
    last_state = None

    for _ in range(stable_samples * 4):
        s = get_state_safe(tello)
        Xg, Yg, _ = to_global(lane, s["mid"], s["x"], s["y"], s["z"])
        if Xg is None or Yg is None:
            consecutive_ok = 0
            time.sleep(sample_interval)
            continue

        if not is_valid_lane_detection(Xg, lane):
            consecutive_ok = 0
            time.sleep(sample_interval)
            continue

        err_x = target_x - Xg
        err_y = target_y - Yg
        last_state = (Xg, Yg, s["mid"], err_x, err_y)

        if abs(err_x) <= tolerance and abs(err_y) <= tolerance:
            consecutive_ok += 1
            if consecutive_ok >= stable_samples:
                return True, last_state
        else:
            consecutive_ok = 0

        time.sleep(sample_interval)

    return False, last_state


def safe_land_all(swarm):
    for i, tello in enumerate(swarm.tellos):
        try:
            set_phase(i, "emergency_landing")
            tello.land()
            time.sleep(0.5)
        except Exception:
            pass


def connect_swarm_with_ip_reporting(swarm):
    for i, tello in enumerate(swarm.tellos):
        config = TELLO_CONFIGS[i]
        ip = config["ip"]
        print(f"Connecting {config['name']} at {ip}...")
        try:
            tello.connect()
            battery = tello.get_battery()
            print(f"Connected {config['name']} at {ip}, battery: {battery}%")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to {config['name']} at {ip}: {e}") from e


def main():
    global logging_active, TELLO_CONFIGS

    init_csv_if_needed(LOG_FILE, CSV_COLUMNS)
    init_csv_if_needed(SUMMARY_FILE, SUMMARY_COLUMNS)
    trial_id = get_next_trial_id(LOG_FILE)
    logger_thread = None

    while True:
        order_input = input(
            "Please enter the drone IP suffix order from left to right (for example: 103,104,102): "
        ).strip()
        try:
            TELLO_CONFIGS = build_tello_configs(order_input)
            break
        except ValueError as e:
            print(e)

    reset_runtime_state()
    swarm = TelloSwarm.fromIps([cfg["ip"] for cfg in TELLO_CONFIGS])
    final_checks = [False for _ in TELLO_CONFIGS]

    try:
        print("Connecting swarm...")
        connect_swarm_with_ip_reporting(swarm)

        for i, tello in enumerate(swarm.tellos):
            config = TELLO_CONFIGS[i]
            tello.enable_mission_pads()
            tello.set_mission_pad_detection_direction(0)

        input("Press Enter to take off all three drones...")
        set_phase_all("takeoff")
        swarm.takeoff()
        time.sleep(2.5)

        for i, tello in enumerate(swarm.tellos):
            config = TELLO_CONFIGS[i]
            set_phase(i, "acquire_origin_pad")
            if not wait_for_expected_pad(tello, config["start_pad"], timeout=5.0):
                raise RuntimeError(
                    f"{config['name']} failed to detect start pad {config['start_pad']}."
                )
            print(f"{config['name']} locked to start pad {config['start_pad']}.")

        set_phase_all("coordinate_climb")
        print(f"Climbing all drones to {TAKEOFF_HEIGHT_CM} cm...")
        swarm.parallel(
            lambda i, tello: tello.go_xyz_speed_mid(
                0, 0, TAKEOFF_HEIGHT_CM, GO_SPEED, TELLO_CONFIGS[i]["start_pad"]
            )
        )
        time.sleep(1.0)

        for i, tello in enumerate(swarm.tellos):
            run_takeoff_batteries[TELLO_CONFIGS[i]["ip"]] = str(tello.get_battery())

        start_time = time.time()
        logging_active = True
        logger_thread = threading.Thread(
            target=logger_loop,
            args=(swarm, trial_id, start_time, LOG_FILE, LOG_INTERVAL),
            daemon=True,
        )
        logger_thread.start()

        set_phase_all("forward_to_y_250")
        print(f"Advancing all drones to Y={TARGET_FORWARD_Y_CM} cm pad-by-pad at {GO_SPEED} cm/s...")
        swarm.parallel(
            lambda i, tello: advance_to_target_pad(
                tello,
                lane=TELLO_CONFIGS[i]["lane"],
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=TAKEOFF_HEIGHT_CM,
                speed=GO_SPEED,
                tolerance=POSITION_TOLERANCE,
                max_corrections=MAX_POSITION_CORRECTIONS,
            )
        )
        time.sleep(1.0)

        print("Running lane-aware global correction on the mission pad matrix...")
        set_phase_all("coordinate_correction")
        swarm.parallel(
            lambda i, tello: go_to_global_target(
                tello,
                lane=TELLO_CONFIGS[i]["lane"],
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=TAKEOFF_HEIGHT_CM,
                speed=FINE_CORRECTION_SPEED,
                tolerance=POSITION_TOLERANCE,
                max_corrections=MAX_POSITION_CORRECTIONS,
            )
        )

        set_phase_all("final_descent_prepare")
        swarm.parallel(
            lambda i, tello: go_to_global_target(
                tello,
                lane=TELLO_CONFIGS[i]["lane"],
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=FINAL_APPROACH_HEIGHT_CM,
                speed=FINE_CORRECTION_SPEED,
                tolerance=POSITION_TOLERANCE,
                max_corrections=MAX_POSITION_CORRECTIONS,
            )
        )

        print("Running final global checks before landing...")
        for i, tello in enumerate(swarm.tellos):
            config = TELLO_CONFIGS[i]
            set_phase(i, "final_global_check")
            ok, last = is_global_target_stable(
                tello,
                lane=config["lane"],
                target_x=config["target_x"],
                target_y=config["target_y"],
                tolerance=FINAL_POSITION_TOLERANCE,
                stable_samples=FINAL_STABLE_SAMPLES,
            )
            final_checks[i] = ok

            if last is not None:
                Xg, Yg, mid, err_x, err_y = last
                print(
                    f"[FinalCheck] {config['name']} lane={config['lane']} mid={mid}, "
                    f"X={Xg:.1f}, Y={Yg:.1f}, errX={err_x:.1f}, errY={err_y:.1f}, ok={ok}"
                )
            else:
                print(f"[FinalCheck] {config['name']} has no valid pad observation.")

        if ABORT_IF_NOT_WITHIN_TOLERANCE and not all(final_checks):
            for i, ok in enumerate(final_checks):
                if not ok:
                    set_phase(i, "landing_aborted_not_in_tolerance")
            print("Landing aborted because at least one drone is outside tolerance.")
            return

        set_phase_all("landing")
        print("Landing all three drones...")
        swarm.land()
        time.sleep(2.0)

        for i, tello in enumerate(swarm.tellos):
            run_end_batteries[TELLO_CONFIGS[i]["ip"]] = str(tello.get_battery())

        logging_active = False
        if logger_thread is not None:
            logger_thread.join(timeout=2.0)
            logger_thread = None

        append_summary_rows(SUMMARY_FILE, trial_id)
        print(f"Experiment finished. Trial {trial_id} saved to {LOG_FILE}")
        print(f"Battery summary saved to {SUMMARY_FILE}")

    except Exception as e:
        print("ERROR:", e)
        safe_land_all(swarm)

    finally:
        logging_active = False
        if logger_thread is not None:
            logger_thread.join(timeout=2.0)

        for tello in swarm.tellos:
            try:
                tello.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()