#!/usr/bin/env python3
# test_battery.py
# Battery calibration test for DJI Tello
# Drone IP: 192.168.0.103
#
# Procedure:
# 1. Enter the battery code, e.g. B01
# 2. The script checks connection and current battery level
# 3. Press Enter to take off
# 4. Hover for 90 seconds while logging battery usage
# 5. Stop logging, then land
#
# CSV output will be saved in: battery_tests/

from djitellopy import Tello
from datetime import datetime#!/usr/bin/env python3
# test_battery.py
# Single-drone battery calibration test for DJI Tello
#
# Drone IP: 192.168.0.103
#
# This version uses active hover:
# during the 90-second hover period, it repeatedly sends rc control (0, 0, 0, 0)
# to keep the drone in hover mode. It does NOT use mission pads.

from djitellopy import Tello
from datetime import datetime
from pathlib import Path
import csv
import time
import traceback


DRONE_IP = "192.168.0.103"
HOVER_SECONDS = 90.0
LOG_INTERVAL_SEC = 1.0
RC_INTERVAL_SEC = 0.2
OUTPUT_DIR = Path("battery_tests")
TAKEOFF_BATTERY_MIN_PERCENT = 20


COLUMNS = [
    "battery_code",
    "drone_ip",
    "timestamp",
    "elapsed_time_s",
    "battery_percent",
    "height_cm",
    "yaw",
    "pitch",
    "roll",
    "vgx",
    "vgy",
    "vgz",
    "agx",
    "agy",
    "agz",
    "tof",
    "baro",
    "templ",
    "temph",
    "motor_time",
]


def get_state_safe(tello: Tello) -> dict:
    """Return Tello state safely. Missing values are replaced with empty strings."""
    try:
        state = tello.get_current_state()
        if not isinstance(state, dict):
            return {}
        return state
    except Exception:
        return {}


def state_value(state: dict, key: str):
    """Read a state value safely."""
    return state.get(key, "")


def write_csv(path: Path, rows: list[dict]):
    """Write logged rows to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def active_hover_and_log(tello: Tello, battery_code: str, duration: float) -> list[dict]:
    """
    Actively hold hover while logging battery.

    Important:
    - send_rc_control(0, 0, 0, 0) is sent repeatedly.
    - Battery is logged once per second.
    - The function returns immediately after duration is reached.
    - Landing is handled outside this function.
    """
    rows = []

    hover_start_time = time.time()
    next_log_time = hover_start_time
    next_rc_time = hover_start_time

    while True:
        now = time.time()
        elapsed = now - hover_start_time

        if elapsed >= duration:
            break

        # Keep sending zero RC command to maintain active hover.
        if now >= next_rc_time:
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception as exc:
                print(f"Warning: send_rc_control failed: {exc}", flush=True)
            next_rc_time += RC_INTERVAL_SEC

        # Log once per second.
        if now >= next_log_time:
            state = get_state_safe(tello)

            try:
                battery = tello.get_battery()
            except Exception:
                battery = ""

            try:
                height = tello.get_height()
            except Exception:
                height = ""

            row = {
                "battery_code": battery_code,
                "drone_ip": DRONE_IP,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "elapsed_time_s": round(elapsed, 3),
                "battery_percent": battery,
                "height_cm": height,
                "yaw": state_value(state, "yaw"),
                "pitch": state_value(state, "pitch"),
                "roll": state_value(state, "roll"),
                "vgx": state_value(state, "vgx"),
                "vgy": state_value(state, "vgy"),
                "vgz": state_value(state, "vgz"),
                "agx": state_value(state, "agx"),
                "agy": state_value(state, "agy"),
                "agz": state_value(state, "agz"),
                "tof": state_value(state, "tof"),
                "baro": state_value(state, "baro"),
                "templ": state_value(state, "templ"),
                "temph": state_value(state, "temph"),
                "motor_time": state_value(state, "time"),
            }
            rows.append(row)

            print(
                f"t={elapsed:6.1f}s | battery={battery}% | height={height}cm | "
                f"vg=({row['vgx']},{row['vgy']},{row['vgz']})",
                flush=True,
            )

            next_log_time += LOG_INTERVAL_SEC

        time.sleep(0.02)

    # Send one final zero RC command before leaving hover phase.
    try:
        tello.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    battery_code = input("Enter battery code, e.g. B01: ").strip().upper()
    if not battery_code:
        print("Battery code cannot be empty.")
        return

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"battery_test_{battery_code}_{DRONE_IP.replace('.', '_')}_{run_timestamp}.csv"

    tello = Tello(host=DRONE_IP)

    takeoff_started = False
    rows = []

    try:
        print(f"\nConnecting to Tello at {DRONE_IP} ...", flush=True)
        tello.connect()

        battery_before_takeoff = tello.get_battery()
        print(f"Connected.", flush=True)
        print(f"Battery code: {battery_code}", flush=True)
        print(f"Current drone battery: {battery_before_takeoff}%", flush=True)

        try:
            battery_value = int(battery_before_takeoff)
            if battery_value < TAKEOFF_BATTERY_MIN_PERCENT:
                print(
                    f"\nTakeoff blocked: battery is below {TAKEOFF_BATTERY_MIN_PERCENT}%.",
                    flush=True,
                )
                return
        except (TypeError, ValueError):
            print("Warning: battery value is invalid. Continue only if you are sure.", flush=True)

        print("\nPreflight check:")
        print("1. Confirm the battery code is correct.")
        print("2. Confirm the drone is placed safely.")
        print("3. Confirm there is enough open space.")
        print("4. This script does NOT use mission pads.")
        input("\nPress Enter to take off and start the 90-second active hover test...")

        print("\nTaking off...", flush=True)
        tello.takeoff()
        takeoff_started = True

        # Give the drone a short time to stabilise after takeoff.
        print("Stabilising for 3 seconds before logging...", flush=True)
        time.sleep(3.0)

        hover_start_battery = tello.get_battery()
        hover_start_timestamp = datetime.now().isoformat(timespec="milliseconds")
        print(f"\nHover logging started.", flush=True)
        print(f"Hover start battery: {hover_start_battery}%", flush=True)
        print(f"Duration: {HOVER_SECONDS:.0f} seconds", flush=True)

        rows = active_hover_and_log(tello, battery_code, HOVER_SECONDS)

        hover_end_timestamp = datetime.now().isoformat(timespec="milliseconds")
        hover_end_battery = tello.get_battery()

        print("\nHover time reached. Logging stopped.", flush=True)
        print(f"Hover end battery: {hover_end_battery}%", flush=True)

        print("Landing...", flush=True)
        tello.land()
        takeoff_started = False
        time.sleep(1.5)

        if rows:
            write_csv(csv_path, rows)

        try:
            battery_drop = int(hover_start_battery) - int(hover_end_battery)
        except (TypeError, ValueError):
            battery_drop = ""

        print("\nTest summary")
        print("-" * 45)
        print(f"Battery code       : {battery_code}")
        print(f"Drone IP           : {DRONE_IP}")
        print(f"Hover start time   : {hover_start_timestamp}")
        print(f"Hover end time     : {hover_end_timestamp}")
        print(f"Hover duration     : {HOVER_SECONDS:.0f} seconds")
        print(f"Start battery      : {hover_start_battery}%")
        print(f"End battery        : {hover_end_battery}%")
        print(f"Battery drop       : {battery_drop}%")
        print(f"Rows recorded      : {len(rows)}")
        print(f"CSV saved to       : {csv_path}")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.", flush=True)
        if takeoff_started:
            print("Trying to land...", flush=True)
            try:
                tello.land()
            except Exception:
                print("Landing failed. Trying emergency stop...", flush=True)
                try:
                    tello.emergency()
                except Exception:
                    pass

    except Exception as exc:
        print("\nERROR:", exc, flush=True)
        traceback.print_exc()

        if takeoff_started:
            print("Trying to land for safety...", flush=True)
            try:
                tello.land()
            except Exception:
                print("Landing failed. Trying emergency stop...", flush=True)
                try:
                    tello.emergency()
                except Exception:
                    pass

    finally:
        # Save partial data if something failed during hover.
        if rows:
            try:
                write_csv(csv_path, rows)
                print(f"\nPartial/complete CSV saved to: {csv_path}", flush=True)
            except Exception as exc:
                print(f"Warning: failed to save CSV: {exc}", flush=True)

        try:
            tello.end()
        except Exception:
            pass

        print("\nProgram ended.", flush=True)


if __name__ == "__main__":
    main()

from pathlib import Path
import csv
import time
import traceback


DRONE_IP = "192.168.0.103"
HOVER_SECONDS = 90
TARGET_HEIGHT_CM = 80
LOG_INTERVAL_SECONDS = 1.0
OUTPUT_DIR = Path("battery_tests")


def safe_get_state_value(state: dict, key: str):
    """Safely read a value from Tello state dictionary."""
    value = state.get(key, None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    battery_code = input("Enter battery code, e.g. B01: ").strip().upper()
    if not battery_code:
        print("Battery code cannot be empty.")
        return

    trial_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"battery_test_{battery_code}_{trial_time}.csv"

    tello = Tello(host=DRONE_IP)

    is_flying = False
    log_rows = []

    try:
        print(f"\nConnecting to Tello at {DRONE_IP} ...")
        tello.connect()

        start_battery_check = tello.get_battery()
        print(f"Connected.")
        print(f"Battery code: {battery_code}")
        print(f"Current battery: {start_battery_check}%")

        print("\nPlease check:")
        print("1. Correct battery is installed.")
        print("2. Drone is on a flat surface.")
        print("3. Area is clear.")
        print("4. Mission pad / fan condition is as planned if you use them.")
        input("\nPress Enter to take off and start the 90-second hover test...")

        print("\nTaking off...")
        tello.takeoff()
        is_flying = True

        print(f"Moving to target height: {TARGET_HEIGHT_CM} cm")
        # Tello takeoff height is usually around 50-80 cm already.
        # We use go_xyz_speed_mid only if mission pad positioning is needed.
        # For a simple battery calibration, basic height adjustment is safer.
        try:
            current_height = tello.get_height()
            diff = TARGET_HEIGHT_CM - current_height
            if diff > 20:
                tello.move_up(diff)
            elif diff < -20:
                tello.move_down(abs(diff))
        except Exception as e:
            print(f"Warning: height adjustment skipped: {e}")

        print("\nHover test started. Logging battery for 90 seconds...")
        hover_start_time = time.time()
        first_logged_battery = None
        last_logged_battery = None

        while True:
            now = time.time()
            elapsed = now - hover_start_time

            if elapsed > HOVER_SECONDS:
                break

            state = tello.get_current_state()
            battery = tello.get_battery()

            if first_logged_battery is None:
                first_logged_battery = battery
            last_logged_battery = battery

            row = {
                "battery_code": battery_code,
                "drone_ip": DRONE_IP,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_time_s": round(elapsed, 2),
                "battery_percent": battery,
                "height_cm": tello.get_height(),
                "mid": safe_get_state_value(state, "mid"),
                "x": safe_get_state_value(state, "x"),
                "y": safe_get_state_value(state, "y"),
                "z": safe_get_state_value(state, "z"),
                "yaw": safe_get_state_value(state, "yaw"),
                "vgx": safe_get_state_value(state, "vgx"),
                "vgy": safe_get_state_value(state, "vgy"),
                "vgz": safe_get_state_value(state, "vgz"),
                "templ": safe_get_state_value(state, "templ"),
                "temph": safe_get_state_value(state, "temph"),
                "tof": safe_get_state_value(state, "tof"),
                "baro": state.get("baro", None),
            }
            log_rows.append(row)

            print(
                f"t={elapsed:5.1f}s | battery={battery:3d}% | "
                f"height={row['height_cm']}cm | mid={row['mid']}"
            )

            time.sleep(LOG_INTERVAL_SECONDS)

        print("\nHover test finished. Battery logging stopped.")

        end_battery_before_land = tello.get_battery()
        print(f"Battery before landing: {end_battery_before_land}%")

        print("Landing...")
        tello.land()
        is_flying = False

        # Save CSV after landing for safety.
        if log_rows:
            fieldnames = list(log_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(log_rows)

            drop = None
            if first_logged_battery is not None and last_logged_battery is not None:
                drop = first_logged_battery - last_logged_battery

            print("\nTest summary")
            print("-" * 40)
            print(f"Battery code: {battery_code}")
            print(f"Drone IP: {DRONE_IP}")
            print(f"Hover duration: {HOVER_SECONDS} seconds")
            print(f"First logged battery: {first_logged_battery}%")
            print(f"Last logged battery: {last_logged_battery}%")
            print(f"Battery drop during logged hover: {drop}%")
            print(f"CSV saved to: {csv_path}")
        else:
            print("No log rows were recorded. CSV was not saved.")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.")
        if is_flying:
            print("Emergency landing...")
            try:
                tello.land()
            except Exception:
                print("Landing failed. Trying emergency stop...")
                try:
                    tello.emergency()
                except Exception:
                    pass

    except Exception as e:
        print("\nError occurred:")
        print(e)
        traceback.print_exc()

        if is_flying:
            print("Trying to land for safety...")
            try:
                tello.land()
            except Exception:
                print("Landing failed. Trying emergency stop...")
                try:
                    tello.emergency()
                except Exception:
                    pass

    finally:
        try:
            tello.end()
        except Exception:
            pass
        print("\nProgram ended.")


if __name__ == "__main__":
    main()
