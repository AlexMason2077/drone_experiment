from djitellopy import TelloSwarm
import time
import csv
import os
import threading
from datetime import datetime

# =========================
# Experiment settings
# =========================
AVAILABLE_IP_SUFFIXES = ["101", "103", "102"]
IP_PREFIX = "192.168.0."
POSITION_TARGETS = [
    {"name": "drone_1", "start_pad": 1, "target_pad": 5, "target_x": 0, "target_y": 250},
    {"name": "drone_2", "start_pad": 2, "target_pad": 6, "target_x": 0, "target_y": 300},
    {"name": "drone_3", "start_pad": 3, "target_pad": 7, "target_x": 0, "target_y": 350},
]
TELLO_CONFIGS = []

GO_SPEED = 20
Z_APPROACH = 80
Z_FINAL_APPROACH = 50
POSITION_TOLERANCE = 5
FINAL_POSITION_TOLERANCE = 3
MAX_POSITION_CORRECTIONS = 2
FINE_CORRECTION_SPEED = 20
FINAL_STABLE_SAMPLES = 1
ABORT_IF_NOT_WITHIN_TOLERANCE = True
LOG_INTERVAL = 0.1
BATTERY_EQUALIZE_THRESHOLD = 80
HOVER_CHECK_INTERVAL_S = 3.0
HOVER_POSITION_TOLERANCE = 8
HOVER_CORRECTION_SPEED = 20

LOG_FILE = "swarm_missionpad_trajectory_log_50.csv"
SUMMARY_FILE = "swarm_missionpad_battery_summary_50.csv"

PAD_GLOBAL_Y = {
    1: 0,
    2: 50,
    3: 100,
    4: 150,
    5: 200,
    6: 250,
    7: 300,
}

CSV_COLUMNS = [
    "trial_id",
    "drone_name",
    "drone_ip",
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
    "battery_103",
    "battery_103_takeoff",
    "battery_103_end",
    "temph",
    "y_distance_diff_cm",
    "y_distance_diff_12_cm",
    "y_distance_diff_23_cm",
    "y_distance_diff_13_cm",
]


# Shared state for logging thread
phase_lock = threading.Lock()
logging_active = False
current_phases = ["idle" for _ in TELLO_CONFIGS]
run_takeoff_batteries = {}
run_end_batteries = {}
run_battery_id_103 = ""


def build_tello_configs(order_input):
    suffixes = [part.strip() for part in order_input.split(",") if part.strip()]
    if len(suffixes) != 3:
        raise ValueError("Please enter exactly three IP suffixes, for example: 103,104,102")
    if sorted(suffixes) != sorted(AVAILABLE_IP_SUFFIXES):
        raise ValueError("The order must include exactly these drones: 103, 104, 102")

    configs = []
    for i, suffix in enumerate(suffixes):
        base = POSITION_TARGETS[i]
        configs.append({
            "name": base["name"],
            "ip": f"{IP_PREFIX}{suffix}",
            "start_pad": base["start_pad"],
            "target_pad": base["target_pad"],
            "target_x": base["target_x"],
            "target_y": base["target_y"],
        })
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


def init_csv_if_needed(log_file):
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
        return

    with open(log_file, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
        return

    existing_header = rows[0]
    if existing_header == CSV_COLUMNS:
        return

    idx_map = {name: i for i, name in enumerate(existing_header)}
    converted_rows = []
    for old_row in rows[1:]:
        new_row = []
        for col in CSV_COLUMNS:
            idx = idx_map.get(col, -1)
            value = old_row[idx] if (idx >= 0 and idx < len(old_row)) else ""
            new_row.append(value)
        converted_rows.append(new_row)

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(converted_rows)

    print("Upgraded swarm missionpad CSV header.")


def init_summary_if_needed(summary_file):
    summary_columns = [
        "trial_id",
        "battery_id_103",
        "drone_103_start_battery",
        "drone_103_end_battery",
        "drone_103_drop",
    ]

    if os.path.exists(summary_file):
        return

    with open(summary_file, "w", newline="") as f:
        csv.writer(f).writerow(summary_columns)


def append_summary_row(summary_file, trial_id):
    target_ip = f"{IP_PREFIX}103"
    start_battery = run_takeoff_batteries.get(target_ip, "")
    end_battery = run_end_batteries.get(target_ip, "")
    start_val = int(start_battery) if str(start_battery).strip() != "" else None
    end_val = int(end_battery) if str(end_battery).strip() != "" else None
    drop = (start_val - end_val) if start_val is not None and end_val is not None else ""
    row = [trial_id, run_battery_id_103, start_battery, end_battery, drop]

    with open(summary_file, "a", newline="") as f:
        csv.writer(f).writerow(row)


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


def to_global(mid, x, y, z):
    if mid in PAD_GLOBAL_Y:
        return x, PAD_GLOBAL_Y[mid] + y, z
    return None, None, None


def append_log_row(log_file, row):
    with open(log_file, "a", newline="") as f:
        csv.writer(f).writerow(row)


def wait_until_equalized_battery(swarm, threshold=BATTERY_EQUALIZE_THRESHOLD, interval_s=HOVER_CHECK_INTERVAL_S):
    target_ip = f"{IP_PREFIX}103"
    print(f"Hovering until drone {target_ip} reaches <= {threshold}% battery...")
    set_phase_all("battery_equalization_hover")

    while True:
        batteries = [tello.get_battery() for tello in swarm.tellos]
        battery_text = ", ".join(
            f"{TELLO_CONFIGS[i]['ip']}: {b}%"
            for i, b in enumerate(batteries)
        )
        print(f"[Battery Equalization] {battery_text}")

        target_index = next(
            i for i, cfg in enumerate(TELLO_CONFIGS)
            if cfg["ip"] == target_ip
        )
        if batteries[target_index] <= threshold:
            print(f"Drone {target_ip} reached the target battery level.")
            return batteries

        print("Maintaining hover positions over the start pads...")
        swarm.parallel(
            lambda i, tello: maintain_hover_position(
                tello,
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=PAD_GLOBAL_Y[TELLO_CONFIGS[i]["start_pad"]],
                target_z=Z_APPROACH,
            )
        )
        time.sleep(interval_s)


def maintain_hover_position(tello, target_x, target_y, target_z):
    s = get_state_safe(tello)
    Xg, Yg, _ = to_global(s["mid"], s["x"], s["y"], s["z"])

    if Xg is None or Yg is None:
        tello.send_rc_control(0, 0, 0, 0)
        return

    err_x = target_x - Xg
    err_y = target_y - Yg
    if abs(err_x) <= HOVER_POSITION_TOLERANCE and abs(err_y) <= HOVER_POSITION_TOLERANCE:
        tello.send_rc_control(0, 0, 0, 0)
        return

    go_to_global_target(
        tello,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
        speed=HOVER_CORRECTION_SPEED,
        tolerance=HOVER_POSITION_TOLERANCE,
        max_corrections=1,
    )


def logger_loop(swarm, trial_id, start_time, log_file, interval=0.1):
    global logging_active

    while logging_active:
        snapshot = []
        y_globals = []
        battery_103_current = ""
        battery_103_takeoff = run_takeoff_batteries.get(f"{IP_PREFIX}103", "")
        battery_103_end = run_end_batteries.get(f"{IP_PREFIX}103", "")

        for i, tello in enumerate(swarm.tellos):
            s = get_state_safe(tello)
            Xg, Yg, Zg = to_global(s["mid"], s["x"], s["y"], s["z"])
            snapshot.append((s, Xg, Yg, Zg))
            y_globals.append(Yg)
            if TELLO_CONFIGS[i]["ip"] == f"{IP_PREFIX}103":
                battery_103_current = tello.get_battery()

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

        y_distance_diff = None
        if pairwise_diffs:
            y_distance_diff = round(sum(pairwise_diffs) / len(pairwise_diffs), 3)

        for i, tello in enumerate(swarm.tellos):
            s, Xg, Yg, Zg = snapshot[i]
            config = TELLO_CONFIGS[i]

            row = [
                trial_id,
                config["name"],
                config["ip"],
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
                battery_103_current,
                battery_103_takeoff,
                battery_103_end,
                s["temph"],
                y_distance_diff,
                y_distance_diff_12,
                y_distance_diff_23,
                y_distance_diff_13,
            ]

            append_log_row(log_file, row)
            print(row)

        time.sleep(interval)


def wait_for_expected_pad(tello, expected_mid, timeout=4.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = get_state_safe(tello)
        if s["mid"] == expected_mid:
            return True
        time.sleep(interval)
    return False


def go_to_global_target(
    tello,
    target_x,
    target_y,
    target_z,
    speed=20,
    tolerance=5,
    max_corrections=4,
):
    for i in range(max_corrections):
        s = get_state_safe(tello)
        Xg, Yg, _ = to_global(s["mid"], s["x"], s["y"], s["z"])

        if Xg is None or Yg is None:
            print(f"[Correction {i + 1}] No valid mission pad, skip correction this round.")
            time.sleep(0.3)
            continue

        err_x = target_x - Xg
        err_y = target_y - Yg
        print(
            f"[Correction {i + 1}] current(X={Xg:.1f}, Y={Yg:.1f}), "
            f"target(X={target_x}, Y={target_y}), err(X={err_x:.1f}, Y={err_y:.1f})"
        )

        if abs(err_x) <= tolerance and abs(err_y) <= tolerance:
            print("Target tolerance satisfied.")
            return True

        mid = int(s["mid"])
        local_target_y = int(round(target_y - PAD_GLOBAL_Y[mid]))
        local_target_y = max(-500, min(500, local_target_y))
        local_target_x = int(round(target_x))
        local_target_x = max(-500, min(500, local_target_x))
        local_target_z = max(20, min(500, int(round(target_z))))
        local_speed = max(10, min(100, int(round(speed))))

        print(
            f"[Correction {i + 1}] go_xyz_speed_mid(x={local_target_x}, y={local_target_y}, "
            f"z={local_target_z}, speed={local_speed}, mid={mid})"
        )
        tello.go_xyz_speed_mid(local_target_x, local_target_y, local_target_z, local_speed, mid)
        time.sleep(0.6)

    print("Correction loop finished; tolerance not fully satisfied.")
    return False


def is_global_target_stable(
    tello,
    target_x,
    target_y,
    tolerance=5,
    stable_samples=3,
    sample_interval=0.15,
):
    consecutive_ok = 0
    last_state = None

    for _ in range(stable_samples * 4):
        s = get_state_safe(tello)
        Xg, Yg, _ = to_global(s["mid"], s["x"], s["y"], s["z"])
        if Xg is None or Yg is None:
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


def main():
    global logging_active, run_takeoff_batteries, run_end_batteries, run_battery_id_103, TELLO_CONFIGS

    init_csv_if_needed(LOG_FILE)
    init_summary_if_needed(SUMMARY_FILE)
    trial_id = get_next_trial_id(LOG_FILE)
    logger_thread = None

    while True:
        order_input = input(
            "Please enter the drone order on mission pads 1, 2, 3 using IP suffixes "
            "(for example: 103,104,102): "
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
        swarm.connect()

        for i, tello in enumerate(swarm.tellos):
            battery = tello.get_battery()
            print(f"{TELLO_CONFIGS[i]['name']} ({TELLO_CONFIGS[i]['ip']}) battery: {battery}%")
            tello.enable_mission_pads()
            tello.set_mission_pad_detection_direction(0)

        while True:
            battery_id = input("Please enter the battery ID for drone 192.168.0.103: ").strip()
            if battery_id:
                run_battery_id_103 = battery_id
                break
            print("Battery ID cannot be empty. Please try again.")

        input("Press Enter to take off all three drones...")
        set_phase_all("pre_takeoff")

        set_phase_all("takeoff")
        print("Takeoff all three drones...")
        swarm.takeoff()
        time.sleep(2.5)

        for i, tello in enumerate(swarm.tellos):
            set_phase(i, "acquire_origin_pad")
            expected_pad = TELLO_CONFIGS[i]["start_pad"]
            ok = wait_for_expected_pad(tello, expected_pad, timeout=4.0)
            if not ok:
                raise RuntimeError(
                    f"{TELLO_CONFIGS[i]['name']} failed to detect start pad {expected_pad}."
                )
            print(f"{TELLO_CONFIGS[i]['name']} locked to start pad {expected_pad}.")

        set_phase_all("coordinate_climb")
        print("Climbing all three drones to working height in their local pad frames...")
        swarm.parallel(
            lambda i, tello: tello.go_xyz_speed_mid(0, 0, Z_APPROACH, GO_SPEED, TELLO_CONFIGS[i]["start_pad"])
        )
        time.sleep(1.0)

        equalized_batteries = wait_until_equalized_battery(swarm)

        for i, battery in enumerate(equalized_batteries):
            run_takeoff_batteries[TELLO_CONFIGS[i]["ip"]] = str(battery)

        start_time = time.time()
        logging_active = True
        logger_thread = threading.Thread(
            target=logger_loop,
            args=(swarm, trial_id, start_time, LOG_FILE, LOG_INTERVAL),
            daemon=True,
        )
        logger_thread.start()

        set_phase_all("go_to_target_coordinate")
        print("Moving all three drones forward 200cm together...")
        swarm.parallel(
            lambda i, tello: tello.go_xyz_speed_mid(0, 200, Z_APPROACH, GO_SPEED, TELLO_CONFIGS[i]["start_pad"])
        )
        time.sleep(1.0)

        for i, tello in enumerate(swarm.tellos):
            if TELLO_CONFIGS[i]["ip"] == f"{IP_PREFIX}103":
                run_end_batteries[TELLO_CONFIGS[i]["ip"]] = str(tello.get_battery())
        print(
            "Forward-flight battery end point: "
            + ", ".join(
                f"{cfg['ip']}: {run_end_batteries.get(cfg['ip'], 'n/a')}%"
                for cfg in TELLO_CONFIGS
            )
        )

        logging_active = False
        if logger_thread is not None:
            logger_thread.join(timeout=2.0)
            logger_thread = None

        append_summary_row(SUMMARY_FILE, trial_id)
        print(f"Battery summary saved to {SUMMARY_FILE}")

        print("Running parallel global correction to target pads...")
        set_phase_all("coordinate_correction")
        swarm.parallel(
            lambda i, tello: go_to_global_target(
                tello,
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=Z_APPROACH,
                speed=GO_SPEED,
                tolerance=POSITION_TOLERANCE,
                max_corrections=MAX_POSITION_CORRECTIONS,
            )
        )

        print("Descending all three drones in parallel for final alignment...")
        set_phase_all("final_descent_prepare")
        swarm.parallel(
            lambda i, tello: go_to_global_target(
                tello,
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=Z_FINAL_APPROACH,
                speed=FINE_CORRECTION_SPEED,
                tolerance=POSITION_TOLERANCE,
                max_corrections=MAX_POSITION_CORRECTIONS + 1,
            )
        )

        print("Running final global checks...")
        for i, tello in enumerate(swarm.tellos):
            set_phase(i, "final_global_check")
            ok, last = is_global_target_stable(
                tello,
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                tolerance=FINAL_POSITION_TOLERANCE,
                stable_samples=FINAL_STABLE_SAMPLES,
            )
            final_checks[i] = ok

            if last is not None:
                Xg, Yg, mid, err_x, err_y = last
                print(
                    f"[FinalCheck] {TELLO_CONFIGS[i]['name']} mid={mid}, "
                    f"X={Xg:.1f}, Y={Yg:.1f}, errX={err_x:.1f}, errY={err_y:.1f}, ok={ok}"
                )
            else:
                print(f"[FinalCheck] {TELLO_CONFIGS[i]['name']} has no valid pad observation.")

        if ABORT_IF_NOT_WITHIN_TOLERANCE and not all(final_checks):
            for i, ok in enumerate(final_checks):
                if not ok:
                    set_phase(i, "landing_aborted_not_in_tolerance")
            print("Landing aborted: at least one drone is not within tolerance.")
            return

        set_phase_all("pre_landing_hover")
        time.sleep(0.8)

        set_phase_all("landing")
        print("Landing all three drones...")
        swarm.land()
        time.sleep(2.0)

        print(f"Experiment finished. Trial {trial_id} saved to {LOG_FILE}")

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
