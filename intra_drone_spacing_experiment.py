from djitellopy import TelloSwarm
import csv
import os
import threading
import time
import math
from datetime import datetime


# =========================================================================
# Constants
# =========================================================================
AVAILABLE_IP_SUFFIXES = ["101", "104", "102"]
IP_PREFIX = "192.168.0."

ROW_SPACING_CM           = 50
LANE_SPACING_CM          = 50
TARGET_FORWARD_Y_CM      = 250
TAKEOFF_HEIGHT_CM        = 80
FINAL_APPROACH_HEIGHT_CM = 40
GO_SPEED                 = 20
FINE_CORRECTION_SPEED    = 10
POSITION_TOLERANCE       = 8
FINAL_POSITION_TOLERANCE = 5
MAX_POSITION_CORRECTIONS = 3
FINAL_STABLE_SAMPLES     = 2
LOG_INTERVAL             = 0.1
ABORT_IF_NOT_WITHIN_TOLERANCE = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "all_experiments_summary.csv")

LANE_X_OFFSETS = {
    "left":   0,
    "middle": LANE_SPACING_CM,
    "right":  LANE_SPACING_CM * 2,
}

PAD_LAYOUTS = {
    "left":   [1, 2, 3, 4, 5, 6, 7, 8],
    "middle": [2, 3, 4, 5, 6, 7, 8, 1],
    "right":  [3, 4, 5, 6, 7, 8, 1, 2],
}

BASE_POSITIONS = [
    {"name": "drone_left",   "lane": "left",   "start_pad": 1},
    {"name": "drone_middle", "lane": "middle", "start_pad": 2},
    {"name": "drone_right",  "lane": "right",  "start_pad": 3},
]

# Y-offset multipliers for [left_drone, middle_drone, right_drone].
# Actual target_y = TARGET_FORWARD_Y_CM + (delta_y_mult * intra_distance_cm)
#
# Formation shapes (top-down view, drones moving north / +Y):
#
#   front   : L M R  — side-by-side, all same Y
#   column  : L      — left leads, file staggered north→south (each in own lane)
#             M
#             R
#   vee     :  M     — middle leads, wings behind symmetrically (V-shape)
#            L   R
#   echelon :     R  — right leads, diagonal step left-and-back
#             M
#           L
#   diamond : L   R  — left+right forward, middle trails (inverted-V / wedge)
#               M
FORMATIONS = {
    "front": [
        {"role": "left",       "delta_y_mult":  0},
        {"role": "middle",     "delta_y_mult":  0},
        {"role": "right",      "delta_y_mult":  0},
    ],
    "column": [
        {"role": "leader",     "delta_y_mult":  0},
        {"role": "follower",   "delta_y_mult": -1},
        {"role": "trailer",    "delta_y_mult": -2},
    ],
    "vee": [
        {"role": "left_wing",  "delta_y_mult": -1},
        {"role": "leader",     "delta_y_mult":  0},
        {"role": "right_wing", "delta_y_mult": -1},
    ],
    "echelon": [
        {"role": "trailer",    "delta_y_mult": -2},
        {"role": "follower",   "delta_y_mult": -1},
        {"role": "leader",     "delta_y_mult":  0},
    ],
    "diamond": [
        {"role": "left_lead",  "delta_y_mult":  0},
        {"role": "trailer",    "delta_y_mult": -1},
        {"role": "right_lead", "delta_y_mult":  0},
    ],
}

CSV_COLUMNS = [
    "trial_id", "formation_type", "intra_distance_cm",
    "drone_name", "drone_ip", "drone_role",
    "lane", "start_pad", "target_pad",
    "phase", "timestamp", "elapsed_time",
    # Mission pad + local position
    "mid", "x", "y", "z",
    # Global position
    "X_global", "Y_global", "Z_global",
    # Orientation
    "yaw", "pitch", "roll",
    # Velocity
    "vgx", "vgy", "vgz",
    # Acceleration
    "agx", "agy", "agz",
    # Power / sensors
    "battery", "battery_takeoff", "battery_end",
    "templ", "temph", "tof", "h", "baro", "motor_time",
    # Intended formation target
    "target_x", "target_y",
    # Formation error (how far from ideal position)
    "formation_error_x", "formation_error_y", "formation_error_dist",
    # Pairwise Euclidean distance between drones (cm)
    "dist_12_cm", "dist_23_cm", "dist_13_cm",
    # X/Y components of pairwise distances
    "dist_12_x", "dist_12_y",
    "dist_23_x", "dist_23_y",
    "dist_13_x", "dist_13_y",
    "mean_intra_dist",
]

SUMMARY_COLUMNS = [
    "trial_id", "formation_type", "intra_distance_cm",
    "drone_name", "drone_ip", "drone_role",
    "lane", "start_pad", "target_pad",
    "battery_takeoff", "battery_end", "battery_drop",
]


# =========================================================================
# Global mutable state
# =========================================================================
phase_lock            = threading.Lock()
logging_active        = False
current_phases        = []
run_takeoff_batteries = {}
run_end_batteries     = {}
TELLO_CONFIGS         = []
FORMATION_TYPE        = ""
INTRA_DISTANCE_CM     = 0


# =========================================================================
# Config
# =========================================================================
def build_tello_configs(order_input, formation_type, distance_cm):
    suffixes = [p.strip() for p in order_input.split(",") if p.strip()]
    if len(suffixes) != 3:
        raise ValueError("Enter exactly three IP suffixes, e.g. 101,104,102")
    if sorted(suffixes) != sorted(AVAILABLE_IP_SUFFIXES):
        raise ValueError(f"Suffixes must be exactly: {', '.join(sorted(AVAILABLE_IP_SUFFIXES))}")
    if formation_type not in FORMATIONS:
        raise ValueError(f"Unknown formation '{formation_type}'. Choose: {', '.join(FORMATIONS)}")
    if distance_cm <= 0:
        raise ValueError("Distance must be > 0 cm.")

    max_pad_y = (len(PAD_LAYOUTS["left"]) - 1) * ROW_SPACING_CM
    formation = FORMATIONS[formation_type]
    configs = []

    for i, suffix in enumerate(suffixes):
        base     = BASE_POSITIONS[i]
        target_y = TARGET_FORWARD_Y_CM + formation[i]["delta_y_mult"] * distance_cm

        if not (0 <= target_y <= max_pad_y):
            raise ValueError(
                f"Formation '{formation_type}' distance={distance_cm} cm places "
                f"{base['name']} at Y={target_y} cm — outside testbed (0–{max_pad_y} cm). "
                f"Reduce distance."
            )

        target_x  = LANE_X_OFFSETS[base["lane"]]
        pad_seq   = PAD_LAYOUTS[base["lane"]]
        row_idx   = max(0, min(len(pad_seq) - 1, round(target_y / ROW_SPACING_CM)))
        target_pad = pad_seq[row_idx]

        configs.append({
            "name":       base["name"],
            "role":       formation[i]["role"],
            "lane":       base["lane"],
            "ip":         f"{IP_PREFIX}{suffix}",
            "start_pad":  base["start_pad"],
            "target_pad": target_pad,
            "target_x":   target_x,
            "target_y":   target_y,
        })
    return configs


def reset_runtime_state():
    global current_phases, run_takeoff_batteries, run_end_batteries
    current_phases        = ["idle"] * len(TELLO_CONFIGS)
    run_takeoff_batteries = {cfg["ip"]: "" for cfg in TELLO_CONFIGS}
    run_end_batteries     = {cfg["ip"]: "" for cfg in TELLO_CONFIGS}


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
# Trial ID (global across all CSVs in OUTPUT_DIR)
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
# Telemetry
# =========================================================================
def get_state_safe(tello):
    s = tello.get_current_state()
    return {
        "mid":        s.get("mid",   -1),
        "x":          s.get("x",      0),
        "y":          s.get("y",      0),
        "z":          s.get("z",      0),
        "yaw":        s.get("yaw",    0),
        "pitch":      s.get("pitch",  0),
        "roll":       s.get("roll",   0),
        "vgx":        s.get("vgx",    0),
        "vgy":        s.get("vgy",    0),
        "vgz":        s.get("vgz",    0),
        "agx":        s.get("agx",    0.0),
        "agy":        s.get("agy",    0.0),
        "agz":        s.get("agz",    0.0),
        "templ":      s.get("templ",  0),
        "temph":      s.get("temph",  0),
        "tof":        s.get("tof",    0),
        "h":          s.get("h",      0),
        "baro":       s.get("baro",   0.0),
        "motor_time": s.get("time",   0),
    }


def get_pad_origin(lane, mid):
    row_seq = PAD_LAYOUTS.get(lane, [])
    if mid not in row_seq:
        return None, None
    return LANE_X_OFFSETS[lane], row_seq.index(mid) * ROW_SPACING_CM


def to_global(lane, mid, x, y, z):
    if mid == -1:
        return None, None, None
    ox, oy = get_pad_origin(lane, mid)
    if ox is None:
        return None, None, None
    return ox + x, oy + y, z


def is_valid_lane_detection(Xg, lane):
    return abs(Xg - LANE_X_OFFSETS[lane]) <= LANE_SPACING_CM * 0.75


# =========================================================================
# CSV helpers
# =========================================================================
def init_summary_if_needed():
    if not os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "w", newline="") as f:
            csv.writer(f).writerow(SUMMARY_COLUMNS)


def append_log_row(log_file, row):
    with open(log_file, "a", newline="") as f:
        csv.writer(f).writerow(row)


def append_summary_rows(trial_id):
    rows = []
    for cfg in TELLO_CONFIGS:
        start = run_takeoff_batteries.get(cfg["ip"], "")
        end   = run_end_batteries.get(cfg["ip"], "")
        sv    = int(start) if str(start).strip() != "" else None
        ev    = int(end)   if str(end).strip()   != "" else None
        drop  = (sv - ev)  if sv is not None and ev is not None else ""
        rows.append([
            trial_id, FORMATION_TYPE, INTRA_DISTANCE_CM,
            cfg["name"], cfg["ip"], cfg["role"],
            cfg["lane"], cfg["start_pad"], cfg["target_pad"],
            start, end, drop,
        ])
    with open(SUMMARY_FILE, "a", newline="") as f:
        csv.writer(f).writerows(rows)


# =========================================================================
# Logger thread
# =========================================================================
def _pdist(snaps, a, b):
    xa, ya = snaps[a]["Xg"], snaps[a]["Yg"]
    xb, yb = snaps[b]["Xg"], snaps[b]["Yg"]
    if None in (xa, ya, xb, yb):
        return None, None, None
    dx = round(xb - xa, 3)
    dy = round(yb - ya, 3)
    return dx, dy, round(math.hypot(dx, dy), 3)


def logger_loop(swarm, trial_id, start_time, log_file):
    global logging_active

    while logging_active:
        ts   = datetime.now().isoformat(timespec="milliseconds")
        elap = round(time.time() - start_time, 3)

        snaps = []
        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            s   = get_state_safe(tello)
            Xg, Yg, Zg = to_global(cfg["lane"], s["mid"], s["x"], s["y"], s["z"])
            snaps.append({"s": s, "Xg": Xg, "Yg": Yg, "Zg": Zg,
                          "batt": tello.get_battery()})

        d12_x, d12_y, d12 = _pdist(snaps, 0, 1)
        d23_x, d23_y, d23 = _pdist(snaps, 1, 2)
        d13_x, d13_y, d13 = _pdist(snaps, 0, 2)
        valid_d = [d for d in (d12, d23, d13) if d is not None]
        mean_dist = round(sum(valid_d) / len(valid_d), 3) if valid_d else None

        for i in range(len(swarm.tellos)):
            cfg  = TELLO_CONFIGS[i]
            snap = snaps[i]
            s, Xg, Yg, Zg, batt = (
                snap["s"], snap["Xg"], snap["Yg"], snap["Zg"], snap["batt"]
            )

            fe_x = round(cfg["target_x"] - Xg, 3) if Xg is not None else None
            fe_y = round(cfg["target_y"] - Yg, 3) if Yg is not None else None
            fe_d = (round(math.hypot(fe_x, fe_y), 3)
                    if fe_x is not None and fe_y is not None else None)

            row = [
                trial_id, FORMATION_TYPE, INTRA_DISTANCE_CM,
                cfg["name"], cfg["ip"], cfg["role"],
                cfg["lane"], cfg["start_pad"], cfg["target_pad"],
                get_phase(i), ts, elap,
                s["mid"], s["x"], s["y"], s["z"],
                Xg, Yg, Zg,
                s["yaw"], s["pitch"], s["roll"],
                s["vgx"], s["vgy"], s["vgz"],
                s["agx"], s["agy"], s["agz"],
                batt,
                run_takeoff_batteries.get(cfg["ip"], ""),
                run_end_batteries.get(cfg["ip"], ""),
                s["templ"], s["temph"], s["tof"], s["h"], s["baro"], s["motor_time"],
                cfg["target_x"], cfg["target_y"],
                fe_x, fe_y, fe_d,
                d12, d23, d13,
                d12_x, d12_y,
                d23_x, d23_y,
                d13_x, d13_y,
                mean_dist,
            ]
            append_log_row(log_file, row)

        time.sleep(LOG_INTERVAL)


# =========================================================================
# Navigation helpers  (identical logic to front_experiment.py)
# =========================================================================
def advance_to_target_pad(tello, lane, target_x, target_y, target_z,
                          speed=GO_SPEED, tolerance=POSITION_TOLERANCE,
                          max_corrections=MAX_POSITION_CORRECTIONS):
    pad_sequence = PAD_LAYOUTS[lane]
    for step_idx, _ in enumerate(pad_sequence):
        pad_y = step_idx * ROW_SPACING_CM
        if pad_y > target_y + tolerance:
            break
        for _ in range(max_corrections):
            s = get_state_safe(tello)
            if s["mid"] == -1:
                time.sleep(0.3)
                continue
            Xg, Yg, _ = to_global(lane, s["mid"], s["x"], s["y"], s["z"])
            if Xg is None or Yg is None:
                time.sleep(0.3)
                continue
            if not is_valid_lane_detection(Xg, lane):
                time.sleep(0.3)
                continue
            if abs(pad_y - Yg) <= tolerance and abs(target_x - Xg) <= tolerance:
                break
            ox, oy = get_pad_origin(lane, s["mid"])
            lx = max(-500, min(500, int(round(target_x - ox))))
            ly = max(-500, min(500, int(round(pad_y    - oy))))
            lz = max(20,   min(500, int(round(target_z))))
            tello.go_xyz_speed_mid(lx, ly, lz, max(10, min(100, speed)), s["mid"])
            time.sleep(0.5)


def wait_for_expected_pad(tello, expected_mid, timeout=5.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if get_state_safe(tello)["mid"] == expected_mid:
            return True
        time.sleep(interval)
    return False


def go_to_global_target(tello, lane, target_x, target_y, target_z,
                        speed=FINE_CORRECTION_SPEED, tolerance=POSITION_TOLERANCE,
                        max_corrections=MAX_POSITION_CORRECTIONS):
    for attempt in range(max_corrections):
        s = get_state_safe(tello)
        Xg, Yg, _ = to_global(lane, s["mid"], s["x"], s["y"], s["z"])
        if Xg is None or Yg is None:
            print(f"  [Corr {attempt+1}] No valid pad — lane={lane}")
            time.sleep(0.3)
            continue
        if not is_valid_lane_detection(Xg, lane):
            print(f"  [Corr {attempt+1}] Cross-lane pad lane={lane} Xg={Xg:.1f}, skip")
            time.sleep(0.3)
            continue
        ex, ey = target_x - Xg, target_y - Yg
        print(f"  [Corr {attempt+1}] lane={lane} cur=({Xg:.1f},{Yg:.1f}) "
              f"tgt=({target_x},{target_y}) err=({ex:.1f},{ey:.1f})")
        if abs(ex) <= tolerance and abs(ey) <= tolerance:
            return True
        mid = int(s["mid"])
        ox, oy = get_pad_origin(lane, mid)
        if ox is None:
            time.sleep(0.3)
            continue
        lx = max(-500, min(500, int(round(target_x - ox))))
        ly = max(-500, min(500, int(round(target_y - oy))))
        lz = max(20,   min(500, int(round(target_z))))
        tello.go_xyz_speed_mid(lx, ly, lz, max(10, min(100, int(speed))), mid)
        time.sleep(0.5)
    return False


def is_global_target_stable(tello, lane, target_x, target_y,
                             tolerance=FINAL_POSITION_TOLERANCE,
                             stable_samples=FINAL_STABLE_SAMPLES,
                             sample_interval=0.15):
    consecutive_ok = 0
    last_state = None
    for _ in range(stable_samples * 4):
        s = get_state_safe(tello)
        Xg, Yg, _ = to_global(lane, s["mid"], s["x"], s["y"], s["z"])
        if Xg is None or Yg is None or not is_valid_lane_detection(Xg, lane):
            consecutive_ok = 0
            time.sleep(sample_interval)
            continue
        ex, ey = target_x - Xg, target_y - Yg
        last_state = (Xg, Yg, s["mid"], ex, ey)
        if abs(ex) <= tolerance and abs(ey) <= tolerance:
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


def connect_swarm(swarm):
    for i, tello in enumerate(swarm.tellos):
        cfg = TELLO_CONFIGS[i]
        print(f"  Connecting {cfg['name']} ({cfg['ip']})...")
        try:
            tello.connect()
            print(f"  OK — battery: {tello.get_battery()}%")
        except Exception as e:
            raise RuntimeError(f"Failed to connect {cfg['name']} at {cfg['ip']}: {e}") from e


# =========================================================================
# Main
# =========================================================================
def main():
    global logging_active, TELLO_CONFIGS, FORMATION_TYPE, INTRA_DISTANCE_CM

    # --- user inputs ---
    print(f"\nAvailable formations: {', '.join(FORMATIONS)}")
    while True:
        ft = input("Formation type: ").strip().lower()
        if ft in FORMATIONS:
            FORMATION_TYPE = ft
            break
        print(f"  Not recognised. Choose from: {', '.join(FORMATIONS)}")

    while True:
        raw = input("Intra-drone distance (cm): ").strip()
        try:
            dist = float(raw)
            if dist > 0:
                INTRA_DISTANCE_CM = dist
                break
        except ValueError:
            pass
        print("  Enter a positive number.")

    while True:
        order_input = input(
            "IP suffix order left→middle→right (e.g. 101,104,102): "
        ).strip()
        try:
            TELLO_CONFIGS = build_tello_configs(order_input, FORMATION_TYPE, INTRA_DISTANCE_CM)
            break
        except ValueError as e:
            print(f"  {e}")

    # --- show formation summary ---
    print(f"\nFormation: {FORMATION_TYPE}  |  Distance: {INTRA_DISTANCE_CM} cm")
    for cfg in TELLO_CONFIGS:
        print(f"  {cfg['name']} ({cfg['role']})  lane={cfg['lane']}  "
              f"target=({cfg['target_x']}, {cfg['target_y']}) cm  "
              f"pad {cfg['start_pad']} → {cfg['target_pad']}")

    # --- init files ---
    init_summary_if_needed()
    trial_id  = get_next_trial_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(
        OUTPUT_DIR,
        f"{FORMATION_TYPE}_{int(INTRA_DISTANCE_CM)}cm_{timestamp}.csv"
    )
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)
    print(f"\nTrial {trial_id} — logging to: {os.path.basename(log_file)}")

    reset_runtime_state()
    swarm         = TelloSwarm.fromIps([cfg["ip"] for cfg in TELLO_CONFIGS])
    final_checks  = [False] * len(TELLO_CONFIGS)
    logger_thread = None

    try:
        print("\nConnecting swarm...")
        connect_swarm(swarm)

        for tello in swarm.tellos:
            tello.enable_mission_pads()
            tello.set_mission_pad_detection_direction(0)

        input("\nPress Enter to take off all three drones...")
        set_phase_all("takeoff")
        swarm.takeoff()
        time.sleep(2.5)

        # acquire start pads
        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            set_phase(i, "acquire_origin_pad")
            if not wait_for_expected_pad(tello, cfg["start_pad"], timeout=5.0):
                raise RuntimeError(
                    f"{cfg['name']} failed to detect start pad {cfg['start_pad']}."
                )
            print(f"  {cfg['name']} locked start pad {cfg['start_pad']}.")

        # climb
        set_phase_all("coordinate_climb")
        print(f"\nClimbing to {TAKEOFF_HEIGHT_CM} cm...")
        swarm.parallel(
            lambda i, tello: tello.go_xyz_speed_mid(
                0, 0, TAKEOFF_HEIGHT_CM, GO_SPEED, TELLO_CONFIGS[i]["start_pad"]
            )
        )
        time.sleep(1.0)

        for i, tello in enumerate(swarm.tellos):
            run_takeoff_batteries[TELLO_CONFIGS[i]["ip"]] = str(tello.get_battery())

        # start logging
        start_time    = time.time()
        logging_active = True
        logger_thread  = threading.Thread(
            target=logger_loop,
            args=(swarm, trial_id, start_time, log_file),
            daemon=True,
        )
        logger_thread.start()

        # advance pad-by-pad
        set_phase_all("advance_to_target")
        print(f"\nAdvancing to formation targets...")
        swarm.parallel(
            lambda i, tello: advance_to_target_pad(
                tello,
                lane=TELLO_CONFIGS[i]["lane"],
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=TAKEOFF_HEIGHT_CM,
            )
        )
        time.sleep(1.0)

        # coarse global correction
        set_phase_all("coordinate_correction")
        print("Running global correction...")
        swarm.parallel(
            lambda i, tello: go_to_global_target(
                tello,
                lane=TELLO_CONFIGS[i]["lane"],
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=TAKEOFF_HEIGHT_CM,
            )
        )

        # descent to final approach height
        set_phase_all("final_descent_prepare")
        swarm.parallel(
            lambda i, tello: go_to_global_target(
                tello,
                lane=TELLO_CONFIGS[i]["lane"],
                target_x=TELLO_CONFIGS[i]["target_x"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=FINAL_APPROACH_HEIGHT_CM,
            )
        )

        # final position check
        print("\nFinal position checks...")
        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            set_phase(i, "final_global_check")
            ok, last = is_global_target_stable(
                tello, lane=cfg["lane"],
                target_x=cfg["target_x"], target_y=cfg["target_y"],
            )
            final_checks[i] = ok
            if last is not None:
                Xg, Yg, mid, ex, ey = last
                print(f"  {cfg['name']} ({cfg['role']}) mid={mid} "
                      f"X={Xg:.1f} Y={Yg:.1f} errX={ex:.1f} errY={ey:.1f} ok={ok}")
            else:
                print(f"  {cfg['name']} — no valid pad observation.")

        if ABORT_IF_NOT_WITHIN_TOLERANCE and not all(final_checks):
            for i, ok in enumerate(final_checks):
                if not ok:
                    set_phase(i, "landing_aborted_not_in_tolerance")
            print("Landing aborted — drone(s) outside tolerance.")
            return

        # land
        set_phase_all("landing")
        print("\nLanding all drones...")
        swarm.land()
        time.sleep(2.0)

        for i, tello in enumerate(swarm.tellos):
            run_end_batteries[TELLO_CONFIGS[i]["ip"]] = str(tello.get_battery())

        logging_active = False
        if logger_thread:
            logger_thread.join(timeout=2.0)
            logger_thread = None

        append_summary_rows(trial_id)
        print(f"\nTrial {trial_id} complete.")
        print(f"  Trajectory : {log_file}")
        print(f"  Summary    : {SUMMARY_FILE}")

    except Exception as e:
        print(f"\nERROR: {e}")
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
