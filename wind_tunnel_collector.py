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
MISSION_PAD_CAMERA_YAW_BASELINE_DEG = 180.0
MISSION_PAD_HEADING_TOLERANCE_DEG = 35.0
GROUND_HEIGHT_THRESHOLD_CM = 15
GROUND_CONFIRMATION_HITS = 8
WIND_FLOW_DESCRIPTIONS = {
    "head wind": "source at +X; airflow +X -> -X (against the nose)",
    "tail wind": "source at -X; airflow -X -> +X (from behind)",
    "side wind": "source at +Y; airflow +Y -> -Y (down the pad line)",
}

# Default front head/side-wind frame shown on the Wind Tunnel floor plan:
#   * Pad 5 -> 6 -> 7 -> 8 -> 1 runs from global -Y to +Y.
#   * The official Mission Pad guide defines the printed rocket as +pad X.
#   * Each printed rocket therefore points global +X.
#   * Each aircraft nose also points global +X.
# Consequently +pad Y is global +Y, while aircraft body-right is global -Y.
# Front tail wind instead uses pads along +X and noses along +Y, as confirmed
# on site. The printed pad axes remain +X/+Y; only origins and body mapping differ.
FRONT_PAD_X_AXIS_GLOBAL = (1.0, 0.0)
FRONT_PAD_Y_AXIS_GLOBAL = (0.0, 1.0)

# Wind Tunnel Vee geometry from left/rear to centre/front to right/rear.
# These are the existing 75 cm reference layouts. build_configs scales their
# coordinates for 50 cm runs without changing pad order or controller axes.
# Each adjacent pair is exactly 75 cm apart along a 45-degree arm.
VEE_75_ARM_PROJECTION_CM = 75.0 / math.sqrt(2.0)
WIND_TUNNEL_VEE_75_POSITIONS_CM = [
    (0.0, 0.0),
    (VEE_75_ARM_PROJECTION_CM, VEE_75_ARM_PROJECTION_CM),
    (2.0 * VEE_75_ARM_PROJECTION_CM, 2.0 * VEE_75_ARM_PROJECTION_CM),
    (3.0 * VEE_75_ARM_PROJECTION_CM, VEE_75_ARM_PROJECTION_CM),
    (4.0 * VEE_75_ARM_PROJECTION_CM, 0.0),
]
WIND_TUNNEL_VEE_75_SIDE_POSITIONS_CM = [
    (0.0, 0.0),
    (VEE_75_ARM_PROJECTION_CM, VEE_75_ARM_PROJECTION_CM),
    (2.0 * VEE_75_ARM_PROJECTION_CM, 2.0 * VEE_75_ARM_PROJECTION_CM),
    (VEE_75_ARM_PROJECTION_CM, 3.0 * VEE_75_ARM_PROJECTION_CM),
    (0.0, 4.0 * VEE_75_ARM_PROJECTION_CM),
]
WIND_TUNNEL_ECHALON_75_POSITIONS_CM = [
    (0.0, 4.0 * VEE_75_ARM_PROJECTION_CM),
    (VEE_75_ARM_PROJECTION_CM, 3.0 * VEE_75_ARM_PROJECTION_CM),
    (2.0 * VEE_75_ARM_PROJECTION_CM, 2.0 * VEE_75_ARM_PROJECTION_CM),
    (3.0 * VEE_75_ARM_PROJECTION_CM, VEE_75_ARM_PROJECTION_CM),
    (4.0 * VEE_75_ARM_PROJECTION_CM, 0.0),
]
WIND_TUNNEL_ECHALON_75_SIDE_POSITIONS_CM = [
    (4.0 * VEE_75_ARM_PROJECTION_CM, 0.0),      # Drone 1 / Pad 5: farthest
    (3.0 * VEE_75_ARM_PROJECTION_CM, VEE_75_ARM_PROJECTION_CM),
    (2.0 * VEE_75_ARM_PROJECTION_CM, 2.0 * VEE_75_ARM_PROJECTION_CM),
    (VEE_75_ARM_PROJECTION_CM, 3.0 * VEE_75_ARM_PROJECTION_CM),
    (0.0, 4.0 * VEE_75_ARM_PROJECTION_CM),      # Drone 5 / Pad 1: nearest fan
]
WIND_TUNNEL_COLUMN_75_POSITIONS_CM = [
    (0.0, 300.0),
    (0.0, 225.0),
    (0.0, 150.0),
    (0.0, 75.0),
    (0.0, 0.0),
]
WIND_TUNNEL_COLUMN_75_SIDE_POSITIONS_CM = [
    (0.0, 0.0),
    (75.0, 0.0),
    (150.0, 0.0),
    (225.0, 0.0),
    (300.0, 0.0),
]
WIND_TUNNEL_DIAMOND_75_POSITIONS_CM = [
    (75.0, 0.0),    # Drone 1 / Pad 5: rear
    (0.0, 75.0),    # Drone 2 / Pad 6: left
    (75.0, 75.0),   # Drone 3 / Pad 7: centre
    (150.0, 75.0),  # Drone 4 / Pad 8: right
    (75.0, 150.0),  # Drone 5 / Pad 1: front
]
WIND_TUNNEL_DIAMOND_75_SIDE_POSITIONS_CM = [
    (0.0, 75.0),    # Drone 1 / Pad 5: left
    (75.0, 0.0),    # Drone 2 / Pad 6: bottom
    (75.0, 75.0),   # Drone 3 / Pad 7: centre
    (75.0, 150.0),  # Drone 4 / Pad 8: top
    (150.0, 75.0),  # Drone 5 / Pad 1: right/front
]


def build_configs(experiment):
    drones = sorted(experiment.get("drones", []), key=lambda item: int(item.get("takeoff_order", 999)))
    if len(drones) != 5:
        raise ValueError(f"Wind Tunnel requires exactly five drones; found {len(drones)}.")
    formation = str(experiment.get("formation", "front")).strip().lower()
    spacing = dc.experiment_inter_drone_distance_cm(experiment)
    wind_direction = str(experiment.get("wind_direction", "")).strip().lower()
    front = formation == "front"
    front_tail = front and wind_direction == "tail wind"
    front_side = front and wind_direction == "side wind"
    vee = formation == "vee"
    echalon = dc.is_echalon_formation(formation)
    column = formation == "column"
    diamond = formation == "diamond"
    vee_side = vee and wind_direction == "side wind"
    echalon_side = echalon and wind_direction == "side wind"
    column_side = column and wind_direction == "side wind"
    diamond_side = diamond and wind_direction == "side wind"
    side_rocket_x_layout = (
        vee_side or echalon_side or column_side or diamond_side
    )
    wind_tunnel_pads = [5, 6, 7, 8, 1]
    fixed_positions = []
    for idx in range(len(wind_tunnel_pads)):
        if front_tail:
            # Pad 5 -> 6 -> 7 -> 8 -> 1 follows the printed arrows (+X).
            # Noses face +Y; the fan behind the drones blows from -Y to +Y.
            fixed_positions.append((idx * spacing, 0))
        elif front:
            # Actual floor layout: Pad 5 is at the global -Y end and Pad 1 is
            # at the +Y end.  Rocket/nose direction is global +X (right).
            fixed_positions.append((0, idx * spacing))
        elif vee_side:
            # Pad 7 is the +X apex. Each arm step is 75 cm and the included
            # angle between Pad 7->6 and Pad 7->8 is exactly 90 degrees.
            fixed_positions.append(WIND_TUNNEL_VEE_75_SIDE_POSITIONS_CM[idx])
        elif vee:
            # Physical Vee layout: pads 5,6,7,8,1. Pad 7 is the +Y apex;
            # printed Mission Pad arrows face +X and every aircraft nose faces +Y.
            fixed_positions.append(WIND_TUNNEL_VEE_75_POSITIONS_CM[idx])
        elif echalon_side:
            # Pad 1 is nearest the +Y fan and Pad 5 is farthest. Adjacent pad
            # centres are 75 cm apart on a 45-degree diagonal.
            fixed_positions.append(WIND_TUNNEL_ECHALON_75_SIDE_POSITIONS_CM[idx])
        elif echalon:
            # Physical echelon layout: pads 5,6,7,8,1 descend along +X/-Y;
            # printed Mission Pad arrows face +X and every aircraft nose faces +Y.
            fixed_positions.append(WIND_TUNNEL_ECHALON_75_POSITIONS_CM[idx])
        elif column_side:
            # Pad 5 -> 6 -> 7 -> 8 -> 1 runs left-to-right along global +X.
            fixed_positions.append(WIND_TUNNEL_COLUMN_75_SIDE_POSITIONS_CM[idx])
        elif column:
            # Physical column layout: pads 5,6,7,8,1 descend along -Y;
            # all pad centres share the same global X coordinate.
            fixed_positions.append(WIND_TUNNEL_COLUMN_75_POSITIONS_CM[idx])
        elif diamond_side:
            # Pad 7 centre; Pad 8 top, Pad 6 bottom, Pad 5 left, Pad 1 right.
            fixed_positions.append(WIND_TUNNEL_DIAMOND_75_SIDE_POSITIONS_CM[idx])
        elif diamond:
            # Physical diamond layout: Pad 7 is the centre, with Pads 5,6,8,1
            # respectively 75 cm to its -Y, -X, +X, and +Y sides.
            fixed_positions.append(WIND_TUNNEL_DIAMOND_75_POSITIONS_CM[idx])
        else:
            fixed_positions.append(
                dc.position_at_column_row(formation, idx, 0, spacing, dc.ROW_SPACING_CM)
            )
    if vee or echalon or column or diamond:
        # Keep the established 75 cm geometry exactly; only scale distances.
        scale = spacing / 75.0
        fixed_positions = [(x * scale, y * scale) for x, y in fixed_positions]
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
            "mission_pad_axes_global": (
                (FRONT_PAD_X_AXIS_GLOBAL, FRONT_PAD_Y_AXIS_GLOBAL)
                if front
                else ((1.0, 0.0), (0.0, 1.0))
            ),
            "mission_pad_yaw_baseline_deg": MISSION_PAD_CAMERA_YAW_BASELINE_DEG,
            "pad_x_aligned_with_body_forward": (front and not front_tail) or side_rocket_x_layout,
            "lateral_only_cross_pad_recovery": front_side,
            "mission_pad_heading_tolerance_deg": (
                MISSION_PAD_HEADING_TOLERANCE_DEG
                if front or side_rocket_x_layout
                else None
            ),
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


def _signed_angle_degrees(value):
    """Normalize an angle to [-180, 180)."""
    return (float(value) + 180.0) % 360.0 - 180.0


def fixed_pad_hover_command(config, state):
    """Return one small correction toward the assigned Mission Pad centre."""
    observed_pad = int(state.get("mid", -1))
    assigned_pad = int(config["mission_pad"])
    pad_origins = config.get("pad_origins_cm") or {}
    if observed_pad == -1 or observed_pad not in pad_origins or assigned_pad not in pad_origins:
        return [0, 0, 0, 0]

    pad_z = float(state.get("z") or 0.0)
    pad_yaw = state.get("mission_pad_yaw")
    if pad_yaw is None:
        return [0, 0, 0, 0]

    assigned_origin_x, assigned_origin_y = pad_origins[assigned_pad]
    current_global_x, current_global_y, _ = dc.to_global(config, state)
    if current_global_x is None or current_global_y is None:
        return [0, 0, 0, 0]
    error_x = float(assigned_origin_x) - current_global_x
    error_y = float(assigned_origin_y) - current_global_y

    # Project the global error back into the printed pad frame.  For the
    # front layout this correctly maps global side-wind displacement (+/-Y)
    # to the aircraft's left/right RC channel.
    pad_x_axis, pad_y_axis = config.get(
        "mission_pad_axes_global",
        ((1.0, 0.0), (0.0, 1.0)),
    )
    pad_error_x = error_x * float(pad_x_axis[0]) + error_y * float(pad_x_axis[1])
    pad_error_y = error_x * float(pad_y_axis[0]) + error_y * float(pad_y_axis[1])
    lateral_only_recovery = (
        observed_pad != assigned_pad
        and config.get("lateral_only_cross_pad_recovery", False)
    )
    if lateral_only_recovery:
        # A pad-ID transition can make its longitudinal local X jump by tens of
        # centimetres. Do not turn that discontinuity into a forward/backward
        # command. First return along global Y to the assigned pad; normal
        # two-axis centering resumes as soon as that pad is detected again.
        pad_error_x = 0.0

    pad_command_x = 0
    if abs(pad_error_x) > FIXED_PAD_XY_TOLERANCE_CM:
        pad_command_x = int(round(dc.clamp(
            FIXED_PAD_XY_KP * pad_error_x,
            -FIXED_PAD_MAX_XY_CONTROL,
            FIXED_PAD_MAX_XY_CONTROL,
        )))
        pad_command_x = _minimum_effective_control(pad_command_x, pad_error_x)

    pad_command_y = 0
    if abs(pad_error_y) > FIXED_PAD_XY_TOLERANCE_CM:
        pad_command_y = int(round(dc.clamp(
            FIXED_PAD_XY_KP * pad_error_y,
            -FIXED_PAD_MAX_XY_CONTROL,
            FIXED_PAD_MAX_XY_CONTROL,
        )))
        pad_command_y = _minimum_effective_control(pad_command_y, pad_error_y)

    # Tello's downward-facing Mission Pad attitude reports approximately
    # +/-180 degrees when the aircraft's RC axes are aligned with the pad
    # axes.  Remove that camera-frame baseline before rotating the pad-frame
    # correction into the aircraft body frame.  Using mpry yaw directly here
    # reverses both x/y corrections and pushes a displaced drone farther away.
    yaw_baseline = float(config.get(
        "mission_pad_yaw_baseline_deg",
        MISSION_PAD_CAMERA_YAW_BASELINE_DEG,
    ))
    control_yaw_degrees = _signed_angle_degrees(float(pad_yaw) - yaw_baseline)
    heading_tolerance = config.get("mission_pad_heading_tolerance_deg")
    if heading_tolerance is not None and abs(control_yaw_degrees) > float(heading_tolerance):
        # A rotated aircraft would map correction onto the wrong RC axis. Hold
        # instead of issuing a potentially collision-inducing command.
        return [0, 0, 0, 0]
    yaw_radians = math.radians(control_yaw_degrees)
    cos_yaw = math.cos(yaw_radians)
    sin_yaw = math.sin(yaw_radians)
    if config.get("pad_x_aligned_with_body_forward", False):
        # With rocket/nose at global +X:
        #   pad +X = body forward, pad +Y = body left = -body right.
        # control_yaw is the small aircraft heading error after removing the
        # downward-camera's 180-degree baseline.
        left_right = int(round(dc.clamp(
            -pad_command_x * sin_yaw - pad_command_y * cos_yaw,
            -FIXED_PAD_MAX_XY_CONTROL,
            FIXED_PAD_MAX_XY_CONTROL,
        )))
        forward_back = int(round(dc.clamp(
            pad_command_x * cos_yaw - pad_command_y * sin_yaw,
            -FIXED_PAD_MAX_XY_CONTROL,
            FIXED_PAD_MAX_XY_CONTROL,
        )))
    else:
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
    if lateral_only_recovery:
        forward_back = 0

    up_down = 0
    if pad_z > 0 and abs(dc.TAKEOFF_HEIGHT_CM - pad_z) > FIXED_PAD_Z_TOLERANCE_CM:
        up_down = int(round(dc.clamp(
            FIXED_PAD_Z_KP * (dc.TAKEOFF_HEIGHT_CM - pad_z),
            -FIXED_PAD_MAX_Z_CONTROL,
            FIXED_PAD_MAX_Z_CONTROL,
        )))
    return [left_right, forward_back, up_down, 0]


class PadRecoveryGeometry:
    """Translate the assigned global centre into a currently visible pad frame.

    All configured pad axes must match the physical layout; no body-yaw or
    remembered position is used. Unknown pads cannot provide localization.
    """

    @staticmethod
    def target(config, observed_pad):
        origins = config.get("pad_origins_cm") or {}
        own = int(config["mission_pad"])
        if observed_pad not in origins or own not in origins:
            return None
        u, v = config.get("mission_pad_axes_global", ((1., 0.), (0., 1.)))
        if (abs(sum(a*a for a in u)-1) > 1e-6
                or abs(sum(a*a for a in v)-1) > 1e-6
                or abs(sum(a*b for a, b in zip(u, v))) > 1e-6):
            raise ValueError("Pad recovery requires known orthonormal global pad axes")
        dx = origins[own][0] - origins[observed_pad][0]
        dy = origins[own][1] - origins[observed_pad][1]
        target = (dx*u[0] + dy*u[1], dx*v[0] + dy*v[1], dc.TAKEOFF_HEIGHT_CM)
        if not all(math.isfinite(a) for a in target) or max(abs(target[0]), abs(target[1])) > 500:
            return None
        return target


def check_and_recenter_assigned_pad(tello, config, expected_pad=None):
    """One checked pad-centre move; caller owns this aircraft's command lock."""
    aligned, state = dc.start_pad_alignment_state(
        tello, config, tolerance=dc.START_ALIGNMENT_TOLERANCE_CM,
    )
    observed_pad = int(state["mid"])
    target = PadRecoveryGeometry.target(config, observed_pad)
    if target is None or (expected_pad is not None and observed_pad != expected_pad):
        # Never move using an unseen/unknown pad or a changed launch reference.
        tello.send_rc_control(0, 0, 0, 0)
        return "waiting_pad", state
    if aligned:
        # Heartbeat while idle only; never interrupt a pad-relative go command.
        tello.send_rc_control(0, 0, 0, 0)
        return "aligned", state

    print(
        f"PAD RECENTER: {config['name']} own=m{config['mission_pad']} reference=m{observed_pad} "
        f"from ({state['x']},{state['y']},{state['z']}) "
        f"-> {target} at {dc.TAKEOFF_CLIMB_SPEED_CM_S} cm/s",
        flush=True,
    )
    # Use the original 2.5 m startup positioning command. The supervising
    # controller latches failures; neither SDK nor controller blindly retries.
    previous_retries = tello.retry_count
    try:
        tello.retry_count = 1
        tello.go_xyz_speed_mid(
            *(int(round(a)) for a in target),
            dc.TAKEOFF_CLIMB_SPEED_CM_S, observed_pad,
        )
    finally:
        tello.retry_count = previous_retries
    aligned, state = dc.start_pad_alignment_state(
        tello, config, tolerance=dc.START_ALIGNMENT_TOLERANCE_CM,
    )
    # An SDK "ok" is not proof of arrival: inspect the measured position.
    return ("aligned" if aligned else "not_centered"), state


class PadObservationGuard:
    """Conservative packet/pose gates; a fresh packet does not prove a fresh pose."""

    def __init__(self, pad, config=None):
        self.pad = int(pad)
        self.config = config
        self.global_position = None
        self.packet = None
        self.packet_at = None
        self.pose = None
        self.pose_at = None
        self.stable_at = None
        self.hits = 0
        self.outside_key = None
        self.outside_at = None
        self.outside_hits = 0

    def observe(self, raw, now):
        # djitellopy replaces its state dict on every UDP state packet. Keep the
        # reference (not id(raw)) so stale cached packets cannot count as samples.
        fresh = raw is not self.packet
        pose = (tuple(int(raw[k]) for k in ("mid", "x", "y", "z"))
                if all(k in raw for k in ("mid", "x", "y", "z"))
                else (-1, 0, 0, 0))
        previous = self.pose
        target = (PadRecoveryGeometry.target(self.config, pose[0]) if self.config is not None
                  else ((0, 0, dc.TAKEOFF_HEIGHT_CM) if pose[0] == self.pad else None))
        known = target is not None
        # A pad switch changes the local origin. Compare physical/global points,
        # not the unrelated local x/y values on opposite sides of that switch.
        global_position = None
        if known and self.config is not None:
            global_position = dc.to_global(self.config, dict(zip(("mid", "x", "y", "z"), pose)))
        jump = (previous is not None and previous[0] == pose[0]
                and max(abs(a-b) for a, b in zip(previous[1:], pose[1:])) > 20)
        if global_position is not None and self.global_position is not None:
            jump = max(abs(a-b) for a, b in zip(global_position, self.global_position)) > 20
        if fresh:
            self.packet, self.packet_at = raw, now
            if pose != self.pose:
                self.pose_at = now
            stable = (previous is not None and previous[0] == pose[0]
                      and max(abs(a-b) for a, b in zip(previous[1:], pose[1:])) <= 5)
            if not stable or not known:
                self.hits, self.stable_at = 0, now
            self.hits += 1
            self.pose = pose
            self.global_position = global_position
        age = float("inf") if self.packet_at is None else now - self.packet_at
        offsets = tuple(pose[i+1] - target[i] for i in range(3)) if known else (0, 0, 0)
        error = max(abs(a) for a in offsets) if known else float("inf")
        aligned = pose[0] == self.pad and pose[3] > 0 and error <= dc.START_ALIGNMENT_TOLERANCE_CM
        if fresh:
            key = tuple((1 if v > 0 else -1) if abs(v) > dc.START_ALIGNMENT_TOLERANCE_CM else 0
                        for v in offsets)
            if aligned or not known or key != self.outside_key or previous is None or previous[0] != pose[0]:
                self.outside_hits, self.outside_at = 0, now
            self.outside_key = key
            if not aligned:
                self.outside_hits += 1
        frozen = (not aligned and self.pose_at is not None and now-self.pose_at >= 1.5)
        valid = known and pose[3] > 0 and age <= 0.6 and not frozen
        stable = valid and self.hits >= 3 and now-self.stable_at >= 0.3
        if not aligned:
            stable = (stable and self.outside_hits >= 3
                      and self.outside_at is not None and now-self.outside_at >= 0.3)
        return dict(pose=pose, fresh=fresh, valid=valid, stable=stable,
                    aligned=aligned, error=error, jump=jump, frozen=frozen, age=age,
                    target=target, offsets=offsets)


class IndependentTakeoff:
    """Simultaneous dispatch, but each aircraft releases its own control gate."""

    def __init__(self, swarm, configs, stop_event, command_locks, event_sink=None):
        self.swarm, self.configs = swarm, configs
        self.stop_event, self.command_locks = stop_event, command_locks
        self.event_sink = event_sink
        self.ready = [threading.Event() for _ in configs]
        self.errors = [None] * len(configs)
        self.release = threading.Event()
        self.threads = []

    def report(self, idx, kind, detail):
        print(f"{kind}: {self.configs[idx]['name']}: {detail}", flush=True)
        if self.event_sink is not None:
            try:
                self.event_sink(idx, kind, detail)
            except Exception as exc:
                print(f"TAKEOFF EVENT LOG ERROR: {exc}", flush=True)

    def launch(self, idx):
        while not self.release.wait(0.05):
            if self.stop_event.is_set():
                return
        tello = self.swarm.tellos[idx]
        try:
            with self.command_locks[idx]:
                if self.stop_event.is_set():
                    return
                self.report(idx, "TAKEOFF_REQUESTED", "single attempt; awaiting this aircraft's reply")
                previous_retries = tello.retry_count
                try:
                    tello.retry_count = 1
                    tello.takeoff()
                finally:
                    tello.retry_count = previous_retries
            # An interrupted/failed takeoff must never enable positioning.
            if not self.stop_event.is_set():
                dc.set_phase(idx, "wind_tunnel_acquire_pad")
                self.ready[idx].set()
                self.report(idx, "TAKEOFF_CONFIRMED", "independent pad control enabled")
        except Exception as exc:
            self.errors[idx] = str(exc)
            if not self.stop_event.is_set():
                dc.set_phase(idx, "wind_tunnel_takeoff_error")
            self.report(idx, "TAKEOFF_FAILED", str(exc))

    def start(self):
        try:
            for idx in range(len(self.configs)):
                thread = threading.Thread(target=self.launch, args=(idx,), daemon=True)
                self.threads.append(thread)
                thread.start()
        except Exception:
            self.stop_event.set()
            raise
        finally:
            self.release.set()

    def wait(self):
        for thread in self.threads:
            thread.join()
        failures = [f"{self.configs[i]['name']}: {error}" for i, error in enumerate(self.errors)
                    if error is not None]
        if failures:
            raise RuntimeError("Takeoff not confirmed; partial data retained: " + "; ".join(failures))


def run_fixed_pad_hover_control(
    swarm, configs, stop_event=None, landed=None, duration_sec=None,
    command_locks=None, event_sink=None, ready_events=None,
):
    """Monitor each single-attempt go concurrently; uncertain results latch off."""
    stop_event = stop_event if stop_event is not None else threading.Event()
    landed = landed if landed is not None else [False] * len(configs)
    command_locks = (command_locks if command_locks is not None
                     else [threading.Lock() for _ in configs])
    start = time.monotonic()

    def active(idx):
        return (not stop_event.is_set() and not landed[idx]
                and (duration_sec is None or time.monotonic() - start < duration_sec))

    def worker(idx):
        tello, config = swarm.tellos[idx], configs[idx]
        if ready_events is not None:
            while active(idx) and not ready_events[idx].is_set():
                # No RC/go is allowed while this aircraft is still taking off.
                stop_event.wait(0.05)
            if not active(idx):
                return
        guard = PadObservationGuard(config["mission_pad"], config)
        pending = None
        fault = None
        last_report = -999.0
        next_move = 0.0

        def report_event(kind, detail):
            if event_sink is not None:
                try:
                    event_sink(idx, kind, str(detail))
                except Exception as exc:
                    print(f"PAD EVENT LOG ERROR: {config['name']}: {exc}", flush=True)

        def latch(reason):
            nonlocal fault
            if fault is not None:
                return
            fault = reason
            # stop is an SDK motion brake, NOT emergency (motor cut).
            # Its untagged ACK may satisfy the outstanding go wait: ignore that
            # result forever after a fault, never treat it as permission to retry.
            print(f"PAD CONTROL FAULT: {config['name']}: {reason}. "
                  "Automatic recentering disabled for this run; operator attention required.",
                  flush=True)
            try:
                tello.send_command_without_return("stop")
            except Exception as exc:
                print(f"PAD BRAKE SEND FAILED: {config['name']}: {exc}", flush=True)
            report_event("FAULT", f"{reason}; last_pose={guard.pose}; brake not confirmed")

        def execute_move(task):
            try:
                with command_locks[idx]:
                    if not active(idx) or task["cancel"].is_set():
                        return
                    task["result"] = check_and_recenter_assigned_pad(tello, config, task["reference_pad"])
            except Exception as exc:
                task["error"] = str(exc)
            finally:
                task["done"].set()

        try:
            while active(idx):
                now = time.monotonic()
                try:
                    observation = guard.observe(tello.get_current_state(), now)
                    if pending is not None and fault is None:
                        reason = None
                        if not observation["valid"]:
                            reason = ("no known Pad detected, invalid height, "
                                      "stale state or frozen off-target pose")
                        elif observation["jump"]:
                            reason = "position jumped by more than 20 cm between checks"
                        elif observation["error"] > pending["initial_error"] + 15:
                            reason = "position error increased by more than 15 cm during recenter"
                        elif pending["done"].is_set() and pending.get("error"):
                            reason = "go failed: " + pending["error"]
                        elif now >= pending["deadline"]:
                            reason = "recenter deadline exceeded without verified completion"
                        if reason:
                            pending["cancel"].set()
                            latch(reason)
                        elif pending["done"].is_set() and pending.get("result", (None,))[0] == "waiting_pad":
                            # No go was sent: reconfirm the newly observed reference.
                            pending = None
                            next_move = now + 0.3
                        elif pending["done"].is_set():
                            # Require NEW post-command observations, not just SDK ok.
                            if observation["fresh"] and observation["aligned"]:
                                pending["arrival_hits"] += 1
                            else:
                                pending["arrival_hits"] = 0
                            if pending["arrival_hits"] >= 3:
                                report_event("ARRIVAL_VERIFIED", observation["pose"])
                                pending = None
                                next_move = now + 1.0

                    if fault is not None:
                        status = "control_fault"
                    elif pending is not None:
                        status = ("global_pad_recovery" if pending["reference_pad"] != config["mission_pad"]
                                  else "centering")
                    elif not observation["valid"]:
                        status = "acquire_pad"
                    elif observation["aligned"]:
                        status = "hover"
                    elif observation["stable"] and not observation["jump"] and now >= next_move:
                        distance = math.sqrt(sum(v*v for v in observation["offsets"]))
                        pending = dict(done=threading.Event(), cancel=threading.Event(),
                                       reference_pad=observation["pose"][0],
                                       initial_error=observation["error"], arrival_hits=0,
                                       deadline=now + min(6.0 if observation["pose"][0] != config["mission_pad"] else 4.0,
                                                          distance / dc.TAKEOFF_CLIMB_SPEED_CM_S + 1.5))
                        pending["thread"] = threading.Thread(
                            target=execute_move, args=(pending,), daemon=True)
                        pending["thread"].start()
                        report_event("MOVE_REQUESTED", f"pose={observation['pose']}; reference=m{pending['reference_pad']}; "
                                     f"own=m{config['mission_pad']}; target={observation['target']}")
                        status = "centering"
                    else:
                        status = "confirming_position"

                    # Never interleave RC with a go still awaiting its result.
                    if pending is None or (fault is not None and pending["done"].is_set()):
                        with command_locks[idx]:
                            if active(idx):
                                tello.send_rc_control(0, 0, 0, 0)
                    if active(idx):
                        dc.set_phase(idx, "wind_tunnel_" + status)
                    if now-last_report >= 1.0:
                        print(f"PAD HOLD: {config['name']} status={status} "
                              f"expected=m{config['mission_pad']} pose={observation['pose']} "
                              f"packet_age={observation['age']:.2f}s fault={fault}", flush=True)
                        last_report = now
                except Exception as exc:
                    if pending is not None:
                        pending["cancel"].set()
                    latch("control/telemetry error: " + str(exc))
                    if active(idx):
                        dc.set_phase(idx, "wind_tunnel_control_fault")
                stop_event.wait(0.1)
        finally:
            if pending is not None and not pending["done"].is_set():
                pending["cancel"].set()
                latch("controller stopped or landing requested during active recenter")
                # Keep the per-drone command lock owned by the executor until its
                # bounded SDK wait ends; no new go may race with normal landing.
                pending["thread"].join(timeout=8.0)

    threads = [threading.Thread(target=worker, args=(idx,), daemon=True)
               for idx in range(len(configs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def run(experiment_id):
    experiment = dc.load_experiment(experiment_id)
    if experiment.get("protocol") != "wind_tunnel":
        raise ValueError("This collector only runs records whose protocol is wind_tunnel.")
    configs = build_configs(experiment)
    front_tail = (
        configs[0]["formation"] == "front"
        and configs[0]["wind_direction"] == "tail wind"
    )
    dc.reset_runtime_state(configs)
    paths = dc.output_paths(experiment_id, configs)
    run_id, experiment_dir, coordination_path, battery_path, timeseries_path, drone_paths, battery_plot, temp_plot = paths
    dc.write_header(coordination_path, dc.COORDINATION_COLUMNS)
    dc.write_header(battery_path, dc.BATTERY_COLUMNS)
    dc.write_header(timeseries_path, dc.BATTERY_TIMESERIES_COLUMNS)
    for item in drone_paths.values():
        dc.write_header(item["coordination"], dc.COORDINATION_COLUMNS)
        dc.write_header(item["battery"], dc.BATTERY_COLUMNS)
    pad_event_path = coordination_path.with_name(coordination_path.stem + "_pad_control.csv")
    dc.write_header(pad_event_path, ["timestamp", "drone", "event", "detail"])
    pad_event_lock = threading.Lock()

    def save_pad_event(idx, kind, detail):
        with pad_event_lock:
            dc.append_row(pad_event_path, [datetime.now().isoformat(timespec="milliseconds"),
                                          configs[idx]["name"], kind, detail])

    print(f"Wind Tunnel experiment: {experiment_id}", flush=True)
    print(f"Formation={experiment['formation']}, distance={experiment['inter_drone_distance_cm']}cm, wind={experiment['wind_direction']} / {experiment['wind_speed']}", flush=True)
    if front_tail:
        print(
            "Front tail-wind frame: Pad 5 -> 6 -> 7 -> 8 -> 1 and printed "
            "arrows run left-to-right (+X); aircraft noses point up (+Y). "
            "The fan is behind the drones (-Y), blowing toward +Y.",
            flush=True,
        )
    elif str(experiment.get("formation", "")).strip().lower() == "front":
        print(
            "Front physical frame: Pad 5 -> 6 -> 7 -> 8 -> 1 runs bottom-to-top "
            "along global +Y; every pad rocket (+pad X) and aircraft nose points "
            "global +X (right).",
            flush=True,
        )
    elif str(experiment.get("wind_direction", "")).strip().lower() == "side wind":
        spacing = dc.experiment_inter_drone_distance_cm(experiment)
        side_layouts = {
            "vee": (
                "Vee: Pad 7 is the +X apex; 5-6-7 and 1-8-7 form the two arms; "
                f"adjacent centres are {spacing} cm and the included angle is 90 degrees."
            ),
            "diamond": (
                "Diamond: Pad 7 centre, Pad 8 top, Pad 6 bottom, Pad 5 left, "
                f"Pad 1 right; centre-to-outer-pad distance is {spacing} cm."
            ),
            "column": f"Column: Pad 5 -> 6 -> 7 -> 8 -> 1 runs left-to-right along global +X at {spacing} cm spacing.",
            "echalon": (
                "Echelon: Pad 1 is nearest the +Y fan and Pad 5 is farthest; "
                f"1-8-7-6-5 follows a 45-degree diagonal with {spacing} cm between adjacent centres."
            ),
            "echelon": (
                "Echelon: Pad 1 is nearest the +Y fan and Pad 5 is farthest; "
                f"1-8-7-6-5 follows a 45-degree diagonal with {spacing} cm between adjacent centres."
            ),
            "echolon": (
                "Echelon: Pad 1 is nearest the +Y fan and Pad 5 is farthest; "
                f"1-8-7-6-5 follows a 45-degree diagonal with {spacing} cm between adjacent centres."
            ),
        }
        description = side_layouts.get(str(experiment.get("formation", "")).strip().lower())
        if description:
            print(
                f"{spacing} cm side-wind layout: " + description + " "
                "Every pad rocket (+pad X) and aircraft nose points global +X (right).",
                flush=True,
            )
    print(
        "Physical wind direction: "
        + ("source at -Y; airflow -Y -> +Y (from behind the +Y-facing noses)"
           if front_tail else WIND_FLOW_DESCRIPTIONS.get(
            str(experiment.get("wind_direction", "")).strip().lower(),
            "unknown; verify fan placement before takeoff",
        )),
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
    control_inactive = [False] * 5
    command_locks = [threading.Lock() for _ in configs]
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
        if front_tail:
            print(
                "Place drones 1-5 on Pads 5,6,7,8,1 left-to-right along the "
                "printed arrows (+X). Keep all noses pointing up (+Y), away "
                "from the fan at -Y. Press Enter to take off all five drones...",
                flush=True,
            )
        elif str(experiment.get("formation", "")).strip().lower() == "front":
            print(
                "Place drone 1-5 above Mission Pads 5,6,7,8,1 bottom-to-top. "
                "Confirm every pad rocket (+pad X) and aircraft nose points right (+X). "
                "Press Enter to take off all five drones...",
                flush=True,
            )
        else:
            print(
                "Place drone 1-5 above Mission Pads 5,6,7,8,1 respectively. "
                "Press Enter to take off all five drones...",
                flush=True,
            )
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
        takeoffs = IndependentTakeoff(swarm, configs, abort_event, command_locks, save_pad_event)
        # Controllers wait on independent gates. A slow takeoff reply from one
        # aircraft must not prevent another confirmed aircraft from centering.
        controller_thread = threading.Thread(
            target=run_fixed_pad_hover_control,
            args=(swarm, configs),
            kwargs={"stop_event": abort_event, "landed": control_inactive,
                    "command_locks": command_locks, "event_sink": save_pad_event,
                    "ready_events": takeoffs.ready},
            daemon=True,
        )
        controller_thread.start()
        takeoffs.start()
        takeoffs.wait()  # Pad controllers are already running as each reply arrives.
        # Each aircraft enters hover only after its own measured alignment
        # check passes. Battery monitoring starts now, including acquisition
        # and re-centering; no fixed six-second success assumption.

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
                        control_inactive[idx] = True
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
                        control_inactive[idx] = True
                        print(
                            f"PROGRAM LAND COMMAND: {config['name']} confirmed battery <= "
                            f"{TARGET_BATTERY_PERCENT}% for {LOW_BATTERY_CONFIRMATION_HITS} "
                            f"consecutive readings (latest {battery}%). Landing requested; "
                            "waiting for any active pad command to finish.",
                            flush=True,
                        )
                        # Prevent a new correction after landing is requested.
                        # An outstanding SDK go must return before normal land.
                        with command_locks[idx]:
                            if abort_event.is_set():
                                return
                            tello.send_rc_control(0, 0, 0, 0)
                            tello.land()
                        landed[idx] = True
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
                    # The pad worker owns movement; battery read errors must
                    # not send RC zero in the middle of its go command.
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
                    with command_locks[idx]:
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
