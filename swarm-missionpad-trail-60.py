from djitellopy import TelloSwarm
import time
import csv
import os
import threading
from datetime import datetime

# =========================
# Experiment settings
# =========================
PAD_SPACING_CM = 60
FORWARD_DISTANCE_CM = PAD_SPACING_CM * 2

TELLO_CONFIGS = [
    {
        "name": "drone_1",
        "ip": "192.168.0.103",
        "start_pad": 1,
        "target_pad": 3,
        "target_x": 0,
        "target_y": 120,
    },
    {
        "name": "drone_2",
        "ip": "192.168.0.102",
        "start_pad": 2,
        "target_pad": 4,
        "target_x": 0,
        "target_y": 180,
    },
]

GO_SPEED = 20
Z_APPROACH = 80
Z_FINAL_APPROACH = 35
POSITION_TOLERANCE = 3
FINAL_POSITION_TOLERANCE = 1
MAX_POSITION_CORRECTIONS = 4
FINE_CORRECTION_SPEED = 12
FINAL_STABLE_SAMPLES = 2
ABORT_IF_NOT_WITHIN_TOLERANCE = True
LOG_INTERVAL = 0.1

LOG_FILE = "swarm_missionpad_trajectory_log_60.csv"

PAD_GLOBAL_Y = {
    1: 0,
    2: 60,
    3: 120,
    4: 180,
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
    "battery",
    "takeoff_battery",
    "temph",
    "y_distance_diff_cm",
]


phase_lock = threading.Lock()
logging_active = False
current_phases = ["idle" for _ in TELLO_CONFIGS]
run_takeoff_batteries = ["" for _ in TELLO_CONFIGS]


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


def logger_loop(swarm, trial_id, start_time, log_file, interval=0.1):
    global logging_active

    while logging_active:
        snapshot = []
        y_globals = []

        for tello in swarm.tellos:
            s = get_state_safe(tello)
            Xg, Yg, Zg = to_global(s["mid"], s["x"], s["y"], s["z"])
            snapshot.append((s, Xg, Yg, Zg))
            y_globals.append(Yg)

        y_distance_diff = None
        if len(y_globals) >= 2 and y_globals[0] is not None and y_globals[1] is not None:
            y_distance_diff = round(abs(y_globals[0] - y_globals[1]), 3)

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
                tello.get_battery(),
                run_takeoff_batteries[i],
                s["temph"],
                y_distance_diff,
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
    global logging_active, run_takeoff_batteries

    init_csv_if_needed(LOG_FILE)
    trial_id = get_next_trial_id(LOG_FILE)
    swarm = TelloSwarm.fromIps([cfg["ip"] for cfg in TELLO_CONFIGS])
    logger_thread = None
    final_checks = [False for _ in TELLO_CONFIGS]

    try:
        print("Connecting swarm...")
        swarm.connect()

        for i, tello in enumerate(swarm.tellos):
            battery = tello.get_battery()
            run_takeoff_batteries[i] = str(battery)
            print(f"{TELLO_CONFIGS[i]['name']} ({TELLO_CONFIGS[i]['ip']}) battery: {battery}%")
            tello.enable_mission_pads()
            tello.set_mission_pad_detection_direction(0)

        input("Press Enter to take off both drones...")

        start_time = time.time()
        set_phase_all("pre_takeoff")
        logging_active = True
        logger_thread = threading.Thread(
            target=logger_loop,
            args=(swarm, trial_id, start_time, LOG_FILE, LOG_INTERVAL),
            daemon=True,
        )
        logger_thread.start()

        time.sleep(1.0)

        set_phase_all("takeoff")
        print("Takeoff both drones...")
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
        print("Climbing both drones to working height in their local pad frames...")
        swarm.parallel(
            lambda i, tello: tello.go_xyz_speed_mid(0, 0, Z_APPROACH, GO_SPEED, TELLO_CONFIGS[i]["start_pad"])
        )
        time.sleep(1.0)

        set_phase_all("go_to_target_coordinate")
        print(f"Moving both drones forward {FORWARD_DISTANCE_CM}cm together...")
        swarm.parallel(
            lambda i, tello: tello.go_xyz_speed_mid(0, FORWARD_DISTANCE_CM, Z_APPROACH, GO_SPEED, TELLO_CONFIGS[i]["start_pad"])
        )
        time.sleep(1.0)

        print("Running per-drone global correction to target pads...")
        for i, tello in enumerate(swarm.tellos):
            set_phase(i, "coordinate_correction")
            go_to_global_target(
                tello,
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=Z_APPROACH,
                speed=GO_SPEED,
                tolerance=POSITION_TOLERANCE,
                max_corrections=MAX_POSITION_CORRECTIONS,
            )

        print("Descending both drones for final alignment...")
        for i, tello in enumerate(swarm.tellos):
            set_phase(i, "final_descent_prepare")
            go_to_global_target(
                tello,
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=Z_FINAL_APPROACH,
                speed=FINE_CORRECTION_SPEED,
                tolerance=POSITION_TOLERANCE,
                max_corrections=MAX_POSITION_CORRECTIONS + 2,
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
        print("Landing both drones...")
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
