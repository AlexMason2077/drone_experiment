"""
True column formation experiment.
All three drones fly in the leftmost lane, each advancing 250 cm forward:
  leader:   pad 1 (Y=0cm)   → pad 6 (Y=250cm)
  follower: pad 2 (Y=50cm)  → pad 7 (Y=300cm)
  trailer:  pad 3 (Y=100cm) → pad 8 (Y=350cm)

The 50 cm intra-drone gap is maintained throughout the flight.
"""

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
AVAILABLE_IP_SUFFIXES    = ["101", "102", "103"]
IP_PREFIX                = "192.168.0."

ROW_SPACING_CM           = 50
LANE_X                   = 0        # leftmost lane, X = 0 cm throughout
TARGET_FORWARD_Y_CM      = 250      # leader target Y
TAKEOFF_HEIGHT_CM        = 80
FINAL_APPROACH_HEIGHT_CM = 40
GO_SPEED                 = 10
FINE_CORRECTION_SPEED    = 10
POSITION_TOLERANCE       = 8
FINAL_POSITION_TOLERANCE = 5
MAX_POSITION_CORRECTIONS = 3
FINAL_STABLE_SAMPLES     = 2
LOG_INTERVAL             = 0.1
ABORT_IF_NOT_WITHIN_TOLERANCE = False

OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "all_experiments_summary.csv")

# Left lane: pad IDs at row 0–7 (Y = 0, 50, 100, …, 350 cm)
PAD_SEQUENCE = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_PAD_Y    = (len(PAD_SEQUENCE) - 1) * ROW_SPACING_CM  # 350 cm

# Column roles (index = swarm order: 0=leader, 1=follower, 2=trailer)
ROLES      = ["leader", "follower", "trailer"]
START_PADS = [1, 2, 3]   # leader Y=0, follower Y=50, trailer Y=100
# Each drone advances TARGET_FORWARD_Y_CM from its start → fixed 50 cm gap throughout
INTRA_DISTANCE_CM = ROW_SPACING_CM  # always 50 cm (one pad apart)

FORMATION_TYPE = "column"

CSV_COLUMNS = [
    "trial_id", "formation_type", "intra_distance_cm",
    "drone_name", "drone_ip", "drone_role",
    "lane", "start_pad", "target_pad",
    "phase", "timestamp", "elapsed_time",
    "mid", "x", "y", "z",
    "X_global", "Y_global", "Z_global",
    "yaw", "pitch", "roll",
    "vgx", "vgy", "vgz",
    "agx", "agy", "agz",
    "battery", "battery_takeoff", "battery_end",
    "templ", "temph", "tof", "h", "baro", "motor_time",
    "target_x", "target_y",
    "formation_error_x", "formation_error_y", "formation_error_dist",
    "dist_12_cm", "dist_23_cm", "dist_13_cm",
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


# =========================================================================
# Config
# =========================================================================
def build_tello_configs(order_input):
    suffixes = [p.strip() for p in order_input.split(",") if p.strip()]
    if len(suffixes) != 3:
        raise ValueError("Enter exactly three IP suffixes, e.g. 101,104,102")
    if sorted(suffixes) != sorted(AVAILABLE_IP_SUFFIXES):
        raise ValueError(f"Suffixes must be exactly: {', '.join(sorted(AVAILABLE_IP_SUFFIXES))}")

    configs = []
    for i, suffix in enumerate(suffixes):
        start_y   = PAD_SEQUENCE.index(START_PADS[i]) * ROW_SPACING_CM
        target_y  = start_y + TARGET_FORWARD_Y_CM   # each drone advances 250 cm
        row_idx   = max(0, min(len(PAD_SEQUENCE) - 1, round(target_y / ROW_SPACING_CM)))
        target_pad = PAD_SEQUENCE[row_idx]
        configs.append({
            "name":       f"drone_{ROLES[i]}",
            "role":       ROLES[i],
            "lane":       "left",
            "ip":         f"{IP_PREFIX}{suffix}",
            "start_pad":  START_PADS[i],
            "start_y":    start_y,
            "target_pad": target_pad,
            "target_x":   LANE_X,
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


def pad_origin_y(mid):
    """Y origin (cm) of a pad in the left lane."""
    if mid not in PAD_SEQUENCE:
        return None
    return PAD_SEQUENCE.index(mid) * ROW_SPACING_CM


def to_global(mid, x, y, z):
    oy = pad_origin_y(mid)
    if oy is None or mid == -1:
        return None, None, None
    return LANE_X + x, oy + y, z


def is_valid_pad(Xg):
    """Drone must be within 37.5 cm of X=0 (left lane)."""
    return abs(Xg - LANE_X) <= ROW_SPACING_CM * 0.75


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
            s = get_state_safe(tello)
            Xg, Yg, Zg = to_global(s["mid"], s["x"], s["y"], s["z"])
            snaps.append({"s": s, "Xg": Xg, "Yg": Yg, "Zg": Zg,
                          "batt": tello.get_battery()})

        # dist_12 = leader→follower, dist_23 = follower→trailer, dist_13 = leader→trailer
        d12_x, d12_y, d12 = _pdist(snaps, 0, 1)
        d23_x, d23_y, d23 = _pdist(snaps, 1, 2)
        d13_x, d13_y, d13 = _pdist(snaps, 0, 2)
        valid_d   = [d for d in (d12, d23, d13) if d is not None]
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
# Navigation helpers
# =========================================================================
def advance_to_target(tello, start_pad, target_y, target_z,
                      speed=GO_SPEED, tolerance=POSITION_TOLERANCE,
                      max_corrections=MAX_POSITION_CORRECTIONS):
    """
    Advance through pad rows from start_pad up to target_y.
    Skips rows behind the drone's start position to prevent backward movement.
    """
    start_row = PAD_SEQUENCE.index(start_pad) if start_pad in PAD_SEQUENCE else 0

    for step_idx in range(len(PAD_SEQUENCE)):
        if step_idx < start_row:
            continue                        # don't go backward
        pad_y = step_idx * ROW_SPACING_CM
        if pad_y > target_y + tolerance:
            break

        for _ in range(max_corrections):
            s = get_state_safe(tello)
            if s["mid"] == -1:
                time.sleep(0.3)
                continue
            Xg, Yg, _ = to_global(s["mid"], s["x"], s["y"], s["z"])
            if Xg is None or Yg is None:
                time.sleep(0.3)
                continue
            if not is_valid_pad(Xg):
                time.sleep(0.3)
                continue
            if abs(pad_y - Yg) <= tolerance:
                break
            oy = pad_origin_y(s["mid"])
            if oy is None:
                time.sleep(0.3)
                continue
            ly = max(-500, min(500, int(round(pad_y - oy))))
            lz = max(20,   min(500, int(round(target_z))))
            tello.go_xyz_speed_mid(0, ly, lz, max(10, min(100, speed)), s["mid"])
            time.sleep(0.5)


def wait_for_expected_pad(tello, expected_mid, timeout=5.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if get_state_safe(tello)["mid"] == expected_mid:
            return True
        time.sleep(interval)
    return False


def go_to_global_target(tello, target_y, target_z,
                        speed=FINE_CORRECTION_SPEED, tolerance=POSITION_TOLERANCE,
                        max_corrections=MAX_POSITION_CORRECTIONS):
    for attempt in range(max_corrections):
        s = get_state_safe(tello)
        Xg, Yg, _ = to_global(s["mid"], s["x"], s["y"], s["z"])
        if Xg is None or Yg is None:
            print(f"  [Corr {attempt+1}] No valid pad")
            time.sleep(0.3)
            continue
        if not is_valid_pad(Xg):
            print(f"  [Corr {attempt+1}] Off-lane Xg={Xg:.1f}, skip")
            time.sleep(0.3)
            continue
        ex, ey = LANE_X - Xg, target_y - Yg
        print(f"  [Corr {attempt+1}] cur=({Xg:.1f},{Yg:.1f}) tgt=(0,{target_y}) err=({ex:.1f},{ey:.1f})")
        if abs(ex) <= tolerance and abs(ey) <= tolerance:
            return True
        oy = pad_origin_y(s["mid"])
        if oy is None:
            time.sleep(0.3)
            continue
        ly = max(-500, min(500, int(round(target_y - oy))))
        lz = max(20,   min(500, int(round(target_z))))
        tello.go_xyz_speed_mid(0, ly, lz, max(10, min(100, int(speed))), int(s["mid"]))
        time.sleep(0.5)
    return False


def is_global_target_stable(tello, target_y,
                             tolerance=FINAL_POSITION_TOLERANCE,
                             stable_samples=FINAL_STABLE_SAMPLES,
                             sample_interval=0.15):
    consecutive_ok = 0
    last_state = None
    for _ in range(stable_samples * 4):
        s = get_state_safe(tello)
        Xg, Yg, _ = to_global(s["mid"], s["x"], s["y"], s["z"])
        if Xg is None or Yg is None or not is_valid_pad(Xg):
            consecutive_ok = 0
            time.sleep(sample_interval)
            continue
        ex, ey = LANE_X - Xg, target_y - Yg
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
    global logging_active, TELLO_CONFIGS

    print("\nColumn formation — all drones fly in the LEFT lane.")
    print("  leader:   pad 1 (Y=  0cm) → pad 6 (Y=250cm)")
    print("  follower: pad 2 (Y= 50cm) → pad 7 (Y=300cm)")
    print("  trailer:  pad 3 (Y=100cm) → pad 8 (Y=350cm)\n")

    while True:
        order_input = input(
            "IP suffix order  leader→follower→trailer  (e.g. 101,104,102): "
        ).strip()
        try:
            TELLO_CONFIGS = build_tello_configs(order_input)
            break
        except ValueError as e:
            print(f"  {e}")

    print(f"\nFormation: {FORMATION_TYPE}  |  Distance: {INTRA_DISTANCE_CM} cm")
    for cfg in TELLO_CONFIGS:
        print(f"  {cfg['name']} ({cfg['role']})  start=pad{cfg['start_pad']} (Y={cfg['start_y']}cm)"
              f"  target=pad{cfg['target_pad']} (Y={cfg['target_y']}cm)")

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

        # Acquire start pads
        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            set_phase(i, "acquire_origin_pad")
            if not wait_for_expected_pad(tello, cfg["start_pad"], timeout=5.0):
                raise RuntimeError(
                    f"{cfg['name']} failed to detect start pad {cfg['start_pad']}."
                )
            print(f"  {cfg['name']} locked start pad {cfg['start_pad']}.")

        # Climb
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

        # Start logging
        start_time    = time.time()
        logging_active = True
        logger_thread  = threading.Thread(
            target=logger_loop,
            args=(swarm, trial_id, start_time, log_file),
            daemon=True,
        )
        logger_thread.start()

        # Advance pad-by-pad (each drone skips rows behind its own start pad)
        set_phase_all("advance_to_target")
        print("\nAdvancing to column targets...")
        swarm.parallel(
            lambda i, tello: advance_to_target(
                tello,
                start_pad=TELLO_CONFIGS[i]["start_pad"],
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=TAKEOFF_HEIGHT_CM,
            )
        )
        time.sleep(1.0)

        # Fine correction
        set_phase_all("coordinate_correction")
        print("Running fine correction...")
        swarm.parallel(
            lambda i, tello: go_to_global_target(
                tello,
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=TAKEOFF_HEIGHT_CM,
            )
        )

        # Descend to final approach height
        set_phase_all("final_descent_prepare")
        swarm.parallel(
            lambda i, tello: go_to_global_target(
                tello,
                target_y=TELLO_CONFIGS[i]["target_y"],
                target_z=FINAL_APPROACH_HEIGHT_CM,
            )
        )

        # Final position check
        print("\nFinal position checks...")
        for i, tello in enumerate(swarm.tellos):
            cfg = TELLO_CONFIGS[i]
            set_phase(i, "final_global_check")
            ok, last = is_global_target_stable(tello, target_y=cfg["target_y"])
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

        # Land
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
