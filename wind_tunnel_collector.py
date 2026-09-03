"""Collect a five-drone, fixed-mission-pad wind-tunnel hover run."""

import argparse
import math
import signal
import sys
import threading
import time
from datetime import datetime

import data_collector as dc


TARGET_BATTERY_PERCENT = 20
LOW_BATTERY_CONFIRMATION_HITS = 8
MAX_PLAUSIBLE_BATTERY_DROP_PER_SAMPLE = 3
LARGE_BATTERY_DROP_CONFIRMATION_HITS = 3
INITIAL_POSITION_CORRECTION_DURATION_SEC = 6.0
FIXED_PAD_CONTROL_INTERVAL_SEC = 0.1
FIXED_PAD_XY_TOLERANCE_CM = 8
FIXED_PAD_Z_TOLERANCE_CM = 8
FIXED_PAD_XY_KP = 0.4
FIXED_PAD_Z_KP = 0.3
FIXED_PAD_MIN_XY_CONTROL = 6
FIXED_PAD_MAX_XY_CONTROL = 12
FIXED_PAD_MAX_Z_CONTROL = 8
GROUND_HEIGHT_THRESHOLD_CM = 15
GROUND_CONFIRMATION_HITS = 8
WIND_FLOW_DESCRIPTIONS = {
    "head wind": "source at +Y; airflow +Y -> -Y",
    "tail wind": "source at -Y; airflow -Y -> +Y",
    "side wind": "source at +X; airflow +X -> -X",
}

# Wind Tunnel Vee geometry from left/rear to centre/front to right/rear.
# Each adjacent pair is exactly 75 cm apart along a 45-degree arm.
VEE_75_ARM_PROJECTION_CM = 75.0 / math.sqrt(2.0)
WIND_TUNNEL_VEE_75_POSITIONS_CM = [
    (0.0, 0.0),
    (VEE_75_ARM_PROJECTION_CM, VEE_75_ARM_PROJECTION_CM),
    (2.0 * VEE_75_ARM_PROJECTION_CM, 2.0 * VEE_75_ARM_PROJECTION_CM),
    (3.0 * VEE_75_ARM_PROJECTION_CM, VEE_75_ARM_PROJECTION_CM),
    (4.0 * VEE_75_ARM_PROJECTION_CM, 0.0),
]
WIND_TUNNEL_ECHALON_75_POSITIONS_CM = [
    (0.0, 4.0 * VEE_75_ARM_PROJECTION_CM),
    (VEE_75_ARM_PROJECTION_CM, 3.0 * VEE_75_ARM_PROJECTION_CM),
    (2.0 * VEE_75_ARM_PROJECTION_CM, 2.0 * VEE_75_ARM_PROJECTION_CM),
    (3.0 * VEE_75_ARM_PROJECTION_CM, VEE_75_ARM_PROJECTION_CM),
    (4.0 * VEE_75_ARM_PROJECTION_CM, 0.0),
]
WIND_TUNNEL_COLUMN_75_POSITIONS_CM = [
    (0.0, 300.0),
    (0.0, 225.0),
    (0.0, 150.0),
    (0.0, 75.0),
    (0.0, 0.0),
]
WIND_TUNNEL_DIAMOND_75_POSITIONS_CM = [
    (75.0, 0.0),    # Drone 1 / Pad 5: rear
    (0.0, 75.0),    # Drone 2 / Pad 6: left
    (75.0, 75.0),   # Drone 3 / Pad 7: centre
    (150.0, 75.0),  # Drone 4 / Pad 8: right
    (75.0, 150.0),  # Drone 5 / Pad 1: front
]


def build_configs(experiment):
    drones = sorted(experiment.get("drones", []), key=lambda item: int(item.get("takeoff_order", 999)))
    if len(drones) != 5:
        raise ValueError(f"Wind Tunnel requires exactly five drones; found {len(drones)}.")
    formation = str(experiment.get("formation", "front")).strip().lower()
    spacing = dc.experiment_inter_drone_distance_cm(experiment)
    wind_direction = str(experiment.get("wind_direction", "")).strip().lower()
    front_75 = formation == "front" and spacing == 75
    vee_75 = formation == "vee" and spacing == 75
    echalon_75 = dc.is_echalon_formation(formation) and spacing == 75
    column_75 = formation == "column" and spacing == 75
    diamond_75 = formation == "diamond" and spacing == 75
    wind_tunnel_pads = [5, 6, 7, 8, 1]
    fixed_positions = []
    for idx in range(len(wind_tunnel_pads)):
        if front_75:
            # Physical order Pad 5 -> 6 -> 7 -> 8 -> 1 follows global +X.
            # The wind direction changes the fan side, not these pad centres.
            fixed_positions.append((idx * 75, 0))
        elif vee_75:
            # Physical Vee layout: pads 5,6,7,8,1. Pad 7 is the +Y apex;
            # printed Mission Pad arrows face +X and every aircraft nose faces +Y.
            fixed_positions.append(WIND_TUNNEL_VEE_75_POSITIONS_CM[idx])
        elif echalon_75:
            # Physical echelon layout: pads 5,6,7,8,1 descend along +X/-Y;
            # printed Mission Pad arrows face +X and every aircraft nose faces +Y.
            fixed_positions.append(WIND_TUNNEL_ECHALON_75_POSITIONS_CM[idx])
        elif column_75:
            # Physical column layout: pads 5,6,7,8,1 descend along -Y;
            # all pad centres share the same global X coordinate.
            fixed_positions.append(WIND_TUNNEL_COLUMN_75_POSITIONS_CM[idx])
        elif diamond_75:
            # Physical diamond layout: Pad 7 is the centre, with Pads 5,6,8,1
            # respectively 75 cm to its -Y, -X, +X, and +Y sides.
            fixed_positions.append(WIND_TUNNEL_DIAMOND_75_POSITIONS_CM[idx])
        else:
            fixed_positions.append(
                dc.position_at_column_row(formation, idx, 0, spacing, dc.ROW_SPACING_CM)
            )
    pad_origins_cm = {
        pad_id: fixed_positions[idx]
        for idx, pad_id in enumerate(wind_tunnel_pads)
    }

    configs = []
    for idx, drone in enumerate(drones):
        number = idx + 1
        mission_pad = wind_tunnel_pads[idx]
        if int(drone.get("drone_number", 0)) != number or int(drone.get("mission_pad", 0)) != mission_pad:
            raise ValueError(f"Wind Tunnel requires drone {number} on Mission Pad {mission_pad}.")
        suffix = dc.DRONE_NUMBER_TO_IP_SUFFIX[str(number)]
        start_x, start_y = fixed_positions[idx]
        configs.append({
            "name": f"drone_{number}",
            "ip": f"{dc.IP_PREFIX}{suffix}",
            "battery_id": str(drone.get("battery_id", "")).strip().upper(),
            "takeoff_order": number,
            "role": str(drone.get("role") or f"wind_tunnel_position_{number}"),
            "mission_pad": mission_pad,
            "mission_pad_columns": [[5], [6], [7], [8], [1]],
            "pad_origins_cm": pad_origins_cm,
            "formation": formation,
            "wind_direction": wind_direction,
            "wind_speed": str(experiment.get("wind_speed", "")).strip().lower(),
            "soc_mode": "",
            "front_continuous_protocol": False,
            "inter_drone_distance_cm": spacing,
            "column_spacing_cm": spacing,
            "row_spacing_cm": dc.ROW_SPACING_CM,
            "target_pad": mission_pad,
            "grid_column": idx,
            "grid_row": 0,
            "target_grid_row": 0,
            "node_row_direction": 1,
            "node_segment_count": 0,
            "start_x": start_x,
            "start_y": start_y,
            "target_x": start_x,
            "target_y": start_y,
            "target_z": dc.TAKEOFF_HEIGHT_CM,
            "node_forward_distance_cm": 0,
            "node_speed_cm_s": 0,
        })
    return configs


def align_wind_tunnel_start(swarm, configs):
    """Briefly run the same fixed-pad hover controller used for the experiment."""
    dc.set_phase_all("wind_tunnel_centering")
    run_fixed_pad_hover_control(
        swarm,
        configs,
        duration_sec=INITIAL_POSITION_CORRECTION_DURATION_SEC,
    )


def _minimum_effective_control(value, error):
    if value == 0 or error == 0:
        return 0
    if abs(value) >= FIXED_PAD_MIN_XY_CONTROL:
        return value
    return FIXED_PAD_MIN_XY_CONTROL if error > 0 else -FIXED_PAD_MIN_XY_CONTROL


def fixed_pad_hover_command(config, state):
    """Return one small correction toward the assigned Mission Pad centre."""
    observed_pad = int(state.get("mid", -1))
    assigned_pad = int(config["mission_pad"])
    pad_origins = config.get("pad_origins_cm") or {}
    if observed_pad == -1 or observed_pad not in pad_origins or assigned_pad not in pad_origins:
        return [0, 0, 0, 0]

    local_x = float(state.get("x") or 0.0)
    local_y = float(state.get("y") or 0.0)
    pad_z = float(state.get("z") or 0.0)
    pad_yaw = state.get("mission_pad_yaw")
    if pad_yaw is None:
        return [0, 0, 0, 0]

    observed_origin_x, observed_origin_y = pad_origins[observed_pad]
    assigned_origin_x, assigned_origin_y = pad_origins[assigned_pad]
    current_global_x = float(observed_origin_x) + local_x
    current_global_y = float(observed_origin_y) + local_y
    error_x = float(assigned_origin_x) - current_global_x
    error_y = float(assigned_origin_y) - current_global_y

    pad_command_x = 0
    if abs(error_x) > FIXED_PAD_XY_TOLERANCE_CM:
        pad_command_x = int(round(dc.clamp(
            FIXED_PAD_XY_KP * error_x,
            -FIXED_PAD_MAX_XY_CONTROL,
            FIXED_PAD_MAX_XY_CONTROL,
        )))
        pad_command_x = _minimum_effective_control(pad_command_x, error_x)

    pad_command_y = 0
    if abs(error_y) > FIXED_PAD_XY_TOLERANCE_CM:
        pad_command_y = int(round(dc.clamp(
            FIXED_PAD_XY_KP * error_y,
            -FIXED_PAD_MAX_XY_CONTROL,
            FIXED_PAD_MAX_XY_CONTROL,
        )))
        pad_command_y = _minimum_effective_control(pad_command_y, error_y)

    # Tello's downward-facing Mission Pad attitude reports approximately
    # +/-180 degrees when the aircraft's RC axes are aligned with the pad
    # axes.  Remove that camera-frame baseline before rotating the pad-frame
    # correction into the aircraft body frame.  Using mpry yaw directly here
    # reverses both x/y corrections and pushes a displaced drone farther away.
    control_yaw_degrees = (float(pad_yaw) % 360.0) - 180.0
    yaw_radians = math.radians(control_yaw_degrees)
    cos_yaw = math.cos(yaw_radians)
    sin_yaw = math.sin(yaw_radians)
    left_right = int(round(dc.clamp(
        pad_command_x * cos_yaw - pad_command_y * sin_yaw,
        -FIXED_PAD_MAX_XY_CONTROL,
        FIXED_PAD_MAX_XY_CONTROL,
    )))
    forward_back = int(round(dc.clamp(
        pad_command_x * sin_yaw + pad_command_y * cos_yaw,
        -FIXED_PAD_MAX_XY_CONTROL,
        FIXED_PAD_MAX_XY_CONTROL,
    )))

    up_down = 0
    if pad_z > 0 and abs(dc.TAKEOFF_HEIGHT_CM - pad_z) > FIXED_PAD_Z_TOLERANCE_CM:
        up_down = int(round(dc.clamp(
            FIXED_PAD_Z_KP * (dc.TAKEOFF_HEIGHT_CM - pad_z),
            -FIXED_PAD_MAX_Z_CONTROL,
            FIXED_PAD_MAX_Z_CONTROL,
        )))
    return [left_right, forward_back, up_down, 0]


def run_fixed_pad_hover_control(
    swarm,
    configs,
    stop_event=None,
    landed=None,
    duration_sec=None,
):
    """Continuously keep each drone above only its own assigned Mission Pad."""
    start = time.monotonic()
    next_tick = start
    last_report = -999.0
    if landed is None:
        landed = [False] * len(configs)

    while True:
        now = time.monotonic()
        elapsed = now - start
        if stop_event is not None and stop_event.is_set():
            break
        active_indices = [idx for idx in range(len(configs)) if not landed[idx]]
        if not active_indices:
            break
        if duration_sec is not None and elapsed >= duration_sec:
            break
        if now < next_tick:
            time.sleep(min(0.02, next_tick - now))
            continue

        try:
            states = [dc.get_state_safe(tello) for tello in swarm.tellos]
            commands = [
                fixed_pad_hover_command(configs[idx], states[idx])
                if idx in active_indices
                else [0, 0, 0, 0]
                for idx in range(len(configs))
            ]
            for idx in active_indices:
                if not landed[idx]:
                    swarm.tellos[idx].send_rc_control(*commands[idx])

            if elapsed - last_report >= 1.0:
                status = []
                for idx in active_indices:
                    state = states[idx]
                    if int(state.get("mid", -1)) == int(configs[idx]["mission_pad"]):
                        status.append(
                            f"{configs[idx]['name']} m{state['mid']} "
                            f"local=({state['x']},{state['y']},{state['z']}) "
                            f"rc={tuple(commands[idx][:3])}"
                        )
                    elif int(state.get("mid", -1)) in configs[idx].get("pad_origins_cm", {}):
                        status.append(
                            f"{configs[idx]['name']} expects m{configs[idx]['mission_pad']} "
                            f"sees m{state['mid']} RECOVER rc={tuple(commands[idx][:3])}"
                        )
                    else:
                        status.append(
                            f"{configs[idx]['name']} expects m{configs[idx]['mission_pad']} "
                            f"sees m{state['mid']} HOLD"
                        )
                print("  STATION KEEPING: " + " | ".join(status), flush=True)
                last_report = elapsed
        except Exception as exc:
            print(
                f"  RECOVERABLE STATION-KEEPING ERROR: {exc}. Holding all active "
                "aircraft; no landing command sent.",
                flush=True,
            )
            for idx in active_indices:
                try:
                    swarm.tellos[idx].send_rc_control(0, 0, 0, 0)
                except Exception:
                    pass
            time.sleep(0.2)
        next_tick += FIXED_PAD_CONTROL_INTERVAL_SEC

    for idx, tello in enumerate(swarm.tellos):
        if not landed[idx]:
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass


def run(experiment_id):
    experiment = dc.load_experiment(experiment_id)
    if experiment.get("protocol") != "wind_tunnel":
        raise ValueError("This collector only runs records whose protocol is wind_tunnel.")
    configs = build_configs(experiment)
    dc.reset_runtime_state(configs)
    paths = dc.output_paths(experiment_id, configs)
    run_id, experiment_dir, coordination_path, battery_path, timeseries_path, drone_paths, battery_plot, temp_plot = paths
    dc.write_header(coordination_path, dc.COORDINATION_COLUMNS)
    dc.write_header(battery_path, dc.BATTERY_COLUMNS)
    dc.write_header(timeseries_path, dc.BATTERY_TIMESERIES_COLUMNS)
    for item in drone_paths.values():
        dc.write_header(item["coordination"], dc.COORDINATION_COLUMNS)
        dc.write_header(item["battery"], dc.BATTERY_COLUMNS)

    print(f"Wind Tunnel experiment: {experiment_id}", flush=True)
    print(f"Formation={experiment['formation']}, distance={experiment['inter_drone_distance_cm']}cm, wind={experiment['wind_direction']} / {experiment['wind_speed']}", flush=True)
    print(
        "Physical wind direction: "
        + WIND_FLOW_DESCRIPTIONS.get(
            str(experiment.get("wind_direction", "")).strip().lower(),
            "unknown; verify fan placement before takeoff",
        ),
        flush=True,
    )
    for config in configs:
        print(
            f"  {config['name']} {config['ip']} battery={config['battery_id']} "
            f"-> Mission Pad {config['mission_pad']} global=({config['start_x']},{config['start_y']})cm",
            flush=True,
        )

    swarm = dc.TelloSwarm.fromIps([config["ip"] for config in configs])
    logger_thread = None
    takeoff_started = False
    landed = [False] * 5
    start_timestamp = ""
    start_time = None
    outputs_finalized = False
    explicit_stop_requested = False
    abort_event = threading.Event()
    workers = []
    controller_thread = None
    try:
        print("Preflight: connecting and enabling downward Mission Pad detection...", flush=True)
        dc.connect_and_check(swarm, configs, experiment=None)
        dc.prepare_formal_takeoff_state(swarm, configs)
        print("Place drone 1-5 above Mission Pads 5,6,7,8,1 respectively. Press Enter to take off all five drones...", flush=True)
        input()

        # Start the formal time series immediately before the takeoff command.
        # It therefore includes takeoff, pad acquisition, centering, sustained
        # hover, safety corrections, and every independent landing.
        start_readings = dc.read_all_batteries(swarm, configs)
        for config in configs:
            dc.hover_start_batteries[config["ip"]] = str(start_readings[config["ip"]])
        start_timestamp = datetime.now().isoformat(timespec="milliseconds")
        start_time = time.time()
        dc.set_phase_all("wind_tunnel_takeoff")
        dc.logging_active = True
        logger_thread = threading.Thread(
            target=dc.logger_loop,
            args=(swarm, configs, experiment, run_id, coordination_path, timeseries_path, drone_paths, start_time, start_time),
            daemon=True,
        )
        logger_thread.start()
        print("Full-flight telemetry recording started before takeoff.", flush=True)

        takeoff_started = True
        swarm.takeoff()
        # The SDK owns each aircraft during automatic takeoff, so RC correction
        # must not race the takeoff command. Start station keeping immediately
        # after takeoff returns, before waiting for the group pad-lock check.
        controller_thread = threading.Thread(
            target=run_fixed_pad_hover_control,
            args=(swarm, configs),
            kwargs={"stop_event": abort_event, "landed": landed},
            daemon=True,
        )
        controller_thread.start()
        # Acquire every assigned marker, then use short feedback-controlled RC
        # corrections.  Do not use a blocking go_xyz_speed_mid command here:
        # if a marker changes during that command, the aircraft can keep moving
        # on a stale coordinate frame.
        dc.set_phase_all("wind_tunnel_acquire_pad")
        try:
            dc.wait_for_all_expected_start_pads(swarm, configs)
        except Exception as alignment_exc:
            # One failed lock must not disable station keeping for the other
            # four aircraft. The controller remains active and can also guide
            # an aircraft back from another known pad to its assigned pad.
            print(
                f"RECOVERABLE ALIGNMENT WARNING: {alignment_exc}. "
                "Continuing station keeping for every aircraft; no landing command sent.",
                flush=True,
            )
        dc.set_phase_all("wind_tunnel_centering")
        time.sleep(INITIAL_POSITION_CORRECTION_DURATION_SEC)
        dc.set_phase_all("wind_tunnel_hover")

        monitor_start_readings = dc.read_all_batteries(swarm, configs)
        last_valid_batteries = [int(monitor_start_readings[config["ip"]]) for config in configs]
        low_battery_hits = [0] * len(configs)
        last_battery_warning = [0.0] * len(configs)
        large_drop_candidates = [None] * len(configs)
        large_drop_confirmation_hits = [0] * len(configs)
        ground_confirmation_hits = [0] * len(configs)

        def hover_worker_session(idx, tello):
            config = configs[idx]
            try:
                while True:
                    if abort_event.is_set():
                        return
                    current_state = dc.get_state_safe(tello)
                    raw_battery = tello.get_battery()
                    battery = int(raw_battery)

                    measured_height = max(
                        int(current_state.get("tof") or 0),
                        int(current_state.get("h") or 0),
                    )
                    if measured_height <= GROUND_HEIGHT_THRESHOLD_CM:
                        ground_confirmation_hits[idx] += 1
                    else:
                        ground_confirmation_hits[idx] = 0
                    if ground_confirmation_hits[idx] >= GROUND_CONFIRMATION_HITS:
                        dc.hover_end_batteries[config["ip"]] = str(battery)
                        landed[idx] = True
                        # Keep djitellopy.end() from issuing a redundant land()
                        # after telemetry has already confirmed the aircraft is
                        # physically on the ground.
                        tello.is_flying = False
                        dc.set_phase(idx, "wind_tunnel_uncommanded_landed")
                        print(
                            f"UNCOMMANDED LANDING DETECTED: {config['name']} remained at "
                            f"ground height (latest tof={current_state.get('tof')} "
                            f"h={current_state.get('h')}, battery={battery}%). "
                            "No program land command was sent; stopping commands to this drone.",
                            flush=True,
                        )
                        return

                    previous_battery = last_valid_batteries[idx]
                    plausible = 1 <= battery <= 100
                    if plausible and battery < previous_battery - MAX_PLAUSIBLE_BATTERY_DROP_PER_SAMPLE:
                        candidate = large_drop_candidates[idx]
                        if candidate is None or abs(battery - candidate) > MAX_PLAUSIBLE_BATTERY_DROP_PER_SAMPLE:
                            large_drop_candidates[idx] = battery
                            large_drop_confirmation_hits[idx] = 1
                        else:
                            large_drop_candidates[idx] = battery
                            large_drop_confirmation_hits[idx] += 1
                        plausible = (
                            large_drop_confirmation_hits[idx]
                            >= LARGE_BATTERY_DROP_CONFIRMATION_HITS
                        )
                    else:
                        large_drop_candidates[idx] = None
                        large_drop_confirmation_hits[idx] = 0
                    if not plausible:
                        low_battery_hits[idx] = 0
                        now = time.time()
                        if now - last_battery_warning[idx] >= 2.0:
                            print(
                                f"  IGNORED BATTERY GLITCH: {config['name']} returned {raw_battery}; "
                                f"last valid reading was {previous_battery}%. No landing command sent.",
                                flush=True,
                            )
                            last_battery_warning[idx] = now
                        # The shared station-keeping controller continues the
                        # flight heartbeat while this battery sample is ignored.
                        time.sleep(0.25)
                        continue
                    large_drop_candidates[idx] = None
                    large_drop_confirmation_hits[idx] = 0
                    last_valid_batteries[idx] = battery
                    if battery <= TARGET_BATTERY_PERCENT:
                        low_battery_hits[idx] += 1
                    else:
                        low_battery_hits[idx] = 0

                    if low_battery_hits[idx] >= LOW_BATTERY_CONFIRMATION_HITS:
                        dc.hover_end_batteries[config["ip"]] = str(battery)
                        dc.set_phase(idx, "wind_tunnel_landing_20_percent")
                        # Remove this drone from the shared controller before
                        # issuing land, so no later RC command can overwrite it.
                        landed[idx] = True
                        tello.send_rc_control(0, 0, 0, 0)
                        print(
                            f"PROGRAM LAND COMMAND: {config['name']} confirmed battery <= "
                            f"{TARGET_BATTERY_PERCENT}% for {LOW_BATTERY_CONFIRMATION_HITS} "
                            f"consecutive readings (latest {battery}%). Sending land now.",
                            flush=True,
                        )
                        tello.land()
                        dc.set_phase(idx, "wind_tunnel_landed")
                        return
                    time.sleep(0.1)
            except Exception as exc:
                raise exc

        def hover_worker(idx, tello):
            """Keep a drone airborne through recoverable telemetry/control errors."""
            config = configs[idx]
            while not abort_event.is_set() and not landed[idx]:
                try:
                    hover_worker_session(idx, tello)
                    return
                except Exception as exc:
                    dc.set_phase(idx, "wind_tunnel_recoverable_error")
                    print(
                        f"  RECOVERABLE: {config['name']} control/telemetry error ({exc}). "
                        "No landing command will be sent; holding and resuming monitoring.",
                        flush=True,
                    )
                    try:
                        tello.send_rc_control(0, 0, 0, 0)
                    except Exception:
                        pass
                    time.sleep(1.0)

        for idx, tello in enumerate(swarm.tellos):
            thread = threading.Thread(target=hover_worker, args=(idx, tello), daemon=True)
            workers.append(thread)
            thread.start()
        for thread in workers:
            thread.join()
        abort_event.set()
        if controller_thread:
            controller_thread.join(timeout=2.0)

        # Keep a short tail so the final landed phase/state is present in the
        # time series instead of stopping on the same instant as the last land.
        time.sleep(0.5)
        dc.logging_active = False
        if logger_thread:
            logger_thread.join(timeout=2.0)
        end_timestamp = datetime.now().isoformat(timespec="milliseconds")
        duration = round(time.time() - start_time, 3)
        dc.save_battery_rows(battery_path, drone_paths, configs, experiment, run_id, start_timestamp, end_timestamp, duration)
        outputs_finalized = True
        dc.generate_battery_line_plot(coordination_path, battery_plot, experiment_id, run_id)
        dc.generate_temperature_line_plot(coordination_path, temp_plot, experiment_id, run_id)
        print(f"Wind Tunnel experiment finished; data saved in {experiment_dir}.", flush=True)
        return True
    except (Exception, KeyboardInterrupt) as exc:
        explicit_stop_requested = isinstance(exc, (dc.ExperimentStopped, KeyboardInterrupt))
        abort_event.set()
        for thread in workers:
            thread.join(timeout=1.0)
        if controller_thread:
            controller_thread.join(timeout=2.0)
        # Wind Tunnel runs can be long and expensive. Never delete a partial
        # run: stop the logger cleanly, retain every time-series row already
        dc.logging_active = False
        if logger_thread:
            logger_thread.join(timeout=2.0)
        if start_time is not None and not outputs_finalized:
            end_timestamp = datetime.now().isoformat(timespec="milliseconds")
            duration = round(time.time() - start_time, 3)
            for idx, config in enumerate(configs):
                if dc.hover_end_batteries.get(config["ip"], "") != "":
                    continue
                try:
                    dc.hover_end_batteries[config["ip"]] = str(swarm.tellos[idx].get_battery())
                except Exception:
                    dc.hover_end_batteries[config["ip"]] = ""
            try:
                dc.save_battery_rows(
                    battery_path,
                    drone_paths,
                    configs,
                    experiment,
                    run_id,
                    start_timestamp,
                    end_timestamp,
                    duration,
                )
                outputs_finalized = True
            except Exception as summary_exc:
                print(f"Warning: partial battery summary could not be completed: {summary_exc}", flush=True)
            try:
                dc.generate_battery_line_plot(coordination_path, battery_plot, experiment_id, run_id)
                dc.generate_temperature_line_plot(coordination_path, temp_plot, experiment_id, run_id)
            except Exception as plot_exc:
                print(f"Warning: partial-run plots could not be completed: {plot_exc}", flush=True)
        print(
            f"PARTIAL RUN PRESERVED after error: {exc}. Existing data remains in {experiment_dir}.",
            flush=True,
        )
        raise
    finally:
        abort_event.set()
        for thread in workers:
            thread.join(timeout=1.0)
        dc.logging_active = False
        if logger_thread:
            logger_thread.join(timeout=2.0)
        if explicit_stop_requested and takeoff_started and not all(landed):
            for idx, tello in enumerate(swarm.tellos):
                if landed[idx]:
                    continue
                try:
                    tello.land()
                except Exception:
                    pass
        # djitellopy Tello.end() calls land() when is_flying is true. Therefore
        # only close Tello objects after every drone is known to be landed, or
        # after the user explicitly requested LAND ALL & STOP.
        if all(landed) or explicit_stop_requested or not takeoff_started:
            for tello in swarm.tellos:
                try:
                    tello.end()
                except Exception:
                    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Collect a five-drone Wind Tunnel hover experiment.")
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main():
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(dc.ExperimentStopped("GUI stop requested")))
    try:
        run(parse_args().experiment_id)
    except (dc.ExperimentStopped, KeyboardInterrupt) as exc:
        print(f"Wind Tunnel experiment stopped: {exc}", flush=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
