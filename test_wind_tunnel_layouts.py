"""Offline geometry/control tests: never import the aircraft SDK or start a run."""

import ast
import itertools
import math
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parent
FORMATIONS = ("front", "vee", "column", "diamond", "echalon", "echelon", "echolon")
WINDS = ("head wind", "tail wind", "side wind")
PADS = (5, 6, 7, 8, 1)
REFERENCE_COMMIT = "3ff2f14c"


def load_offline(source=None):
    """Load isolated helpers, excluding SDK imports, file output and main."""
    dc_names = {
        "IP_PREFIX", "DRONE_NUMBER_TO_IP_SUFFIX", "ROW_SPACING_CM",
        "TAKEOFF_HEIGHT_CM", "int_field", "experiment_inter_drone_distance_cm",
        "TAKEOFF_CLIMB_SPEED_CM_S", "START_ALIGNMENT_TOLERANCE_CM",
        "start_pad_alignment_state",
        "clamp", "is_echalon_formation", "pad_origin_for_detection", "to_global",
    }
    tree = ast.parse((ROOT / "data_collector.py").read_text())
    nodes = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name in dc_names
    ) or (
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in dc_names for t in n.targets)
    )]
    dc_namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "data_collector.py", "exec"), dc_namespace)
    namespace = {"math": math, "threading": threading, "time": time,
                 "dc": SimpleNamespace(**dc_namespace)}
    tree = ast.parse(source if source is not None else (ROOT / "wind_tunnel_collector.py").read_text())
    pure_functions = {
        "build_configs", "_minimum_effective_control", "_signed_angle_degrees",
        "fixed_pad_hover_command",
        "run_fixed_pad_hover_control",
        "check_and_recenter_assigned_pad",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) or (
        isinstance(n, ast.ClassDef) and n.name in {"PadObservationGuard", "PadRecoveryGeometry", "IndependentTakeoff"}
    ) or (
        isinstance(n, ast.FunctionDef) and n.name in pure_functions
    )]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "wind_tunnel_collector.py", "exec"), namespace)
    return SimpleNamespace(**namespace)


def experiment(formation, wind, spacing, level="Level1"):
    return {
        "formation": formation, "wind_direction": wind, "wind_speed": level,
        "inter_drone_distance_cm": spacing,
        "drones": [
            {"drone_number": str(i), "takeoff_order": str(i),
             "mission_pad": str(pad), "battery_id": battery}
            for i, pad, battery in zip(range(1, 6), PADS, ("B11", "B10", "B13", "B14", "B12"))
        ],
    }


class WindTunnelLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wt = load_offline()

    def test_75_geometry_matches_each_confirmed_layout(self):
        u = 75 / math.sqrt(2)
        normal = {
            "front": [(0, 0), (0, 75), (0, 150), (0, 225), (0, 300)],
            "vee": [(0, 0), (u, u), (2*u, 2*u), (3*u, u), (4*u, 0)],
            "column": [(0, 300), (0, 225), (0, 150), (0, 75), (0, 0)],
            "diamond": [(75, 0), (0, 75), (75, 75), (150, 75), (75, 150)],
            "echalon": [(0, 4*u), (u, 3*u), (2*u, 2*u), (3*u, u), (4*u, 0)],
        }
        side = {
            "front": normal["front"],
            "vee": [(0, 0), (u, u), (2*u, 2*u), (u, 3*u), (0, 4*u)],
            "column": [(0, 0), (75, 0), (150, 0), (225, 0), (300, 0)],
            "diamond": [(0, 75), (75, 0), (75, 75), (75, 150), (150, 75)],
            "echalon": [(4*u, 0), (3*u, u), (2*u, 2*u), (u, 3*u), (0, 4*u)],
        }
        for formation, wind in itertools.product(FORMATIONS, WINDS):
            with self.subTest(formation=formation, wind=wind):
                configs = self.wt.build_configs(experiment(formation, wind, 75))
                key = "echalon" if self.wt.dc.is_echalon_formation(formation) else formation
                expected = (side if wind == "side wind" else normal)[key]
                if formation == "front" and wind == "tail wind":
                    expected = [(0, 0), (75, 0), (150, 0), (225, 0), (300, 0)]
                self.assertEqual([(c["start_x"], c["start_y"]) for c in configs],
                                 expected)

    def test_front_tail_corrects_in_actual_body_axes(self):
        for spacing in (50, 75):
            for c in self.wt.build_configs(experiment("front", "tail wind", spacing)):
                self.assertFalse(c["pad_x_aligned_with_body_forward"])
                for x, y, expected in ((20, 0, [-8, 0, 0, 0]),
                                       (-20, 0, [8, 0, 0, 0]),
                                       (0, 20, [0, -8, 0, 0]),
                                       (0, -20, [0, 8, 0, 0])):
                    state = {"mid": c["mission_pad"], "x": x, "y": y,
                             "z": 80, "mission_pad_yaw": 180}
                    self.assertEqual(self.wt.fixed_pad_hover_command(c, state), expected)

    def test_front_tail_pad_switch_preserves_global_position(self):
        for spacing in (50, 75):
            c = self.wt.build_configs(experiment("front", "tail wind", spacing))[0]
            on_five = {"mid": 5, "x": 20, "y": -25, "z": 80, "mission_pad_yaw": 180}
            on_six = dict(on_five, mid=6, x=20-spacing)
            self.assertEqual(self.wt.dc.to_global(c, on_five),
                             self.wt.dc.to_global(c, on_six))
            self.assertEqual(self.wt.fixed_pad_hover_command(c, on_five),
                             self.wt.fixed_pad_hover_command(c, on_six))

    def test_only_spacing_and_coordinates_differ(self):
        geometry = {"start_x", "start_y", "target_x", "target_y", "pad_origins_cm",
                    "inter_drone_distance_cm", "column_spacing_cm"}
        for formation, wind, level in itertools.product(FORMATIONS, WINDS, ("Level1", "Level2")):
            with self.subTest(formation=formation, wind=wind, level=level):
                small = self.wt.build_configs(experiment(formation, wind, 50, level))
                large = self.wt.build_configs(experiment(formation, wind, 75, level))
                for a, b in zip(small, large):
                    self.assertEqual({k:v for k,v in a.items() if k not in geometry},
                                     {k:v for k,v in b.items() if k not in geometry})
                    for key in ("start_x", "start_y", "target_x", "target_y"):
                        self.assertAlmostEqual(a[key], b[key] * 2/3)
                    for pad in PADS:
                        for axis in (0, 1):
                            self.assertAlmostEqual(a["pad_origins_cm"][pad][axis],
                                                   b["pad_origins_cm"][pad][axis] * 2/3)

    def test_selected_distance_matches_physical_layout(self):
        for formation, wind, spacing in itertools.product(FORMATIONS, WINDS, (50, 75)):
            with self.subTest(formation=formation, wind=wind, spacing=spacing):
                configs = self.wt.build_configs(experiment(formation, wind, spacing))
                positions = [(c["start_x"], c["start_y"]) for c in configs]
                pairs = [(2, i) for i in (0, 1, 3, 4)] if formation == "diamond" else list(zip(range(4), range(1, 5)))
                for a, b in pairs:
                    self.assertAlmostEqual(math.dist(positions[a], positions[b]), spacing)

    def test_same_local_error_produces_same_control(self):
        for formation, wind in itertools.product(FORMATIONS, WINDS):
            small = self.wt.build_configs(experiment(formation, wind, 50))
            large = self.wt.build_configs(experiment(formation, wind, 75))
            for a, b in zip(small, large):
                for x, y, yaw in itertools.product((-24, 0, 24), (-24, 0, 24), (150, 180, -170)):
                    state = {"mid": a["mission_pad"], "x": x, "y": y,
                             "z": 80, "mission_pad_yaw": yaw}
                    self.assertEqual(self.wt.fixed_pad_hover_command(a, state),
                                     self.wt.fixed_pad_hover_command(b, state))

    def test_neighbour_pad_correction_points_toward_assigned_pad(self):
        for formation, wind, spacing in itertools.product(FORMATIONS, WINDS, (50, 75)):
            for config in self.wt.build_configs(experiment(formation, wind, spacing)):
                for pad in PADS:
                    state = {"mid": pad, "x": 0, "y": 0, "z": 80, "mission_pad_yaw": 180}
                    lr, fb, ud, yaw = self.wt.fixed_pad_hover_command(config, state)
                    command_x, command_y = ((fb, -lr) if config["pad_x_aligned_with_body_forward"] else (lr, fb))
                    x, y = config["pad_origins_cm"][pad]
                    progress = ((config["target_x"] - x) * command_x
                                + (config["target_y"] - y) * command_y)
                    if pad == config["mission_pad"]:
                        self.assertEqual((lr, fb, ud, yaw), (0, 0, 0, 0))
                    else:
                        self.assertGreater(progress, 0, (formation, wind, spacing, pad))

    def test_missing_pad_and_heading_hold_are_shared(self):
        for formation, wind, spacing in itertools.product(FORMATIONS, WINDS, (50, 75)):
            for config in self.wt.build_configs(experiment(formation, wind, spacing)):
                state = {"mid": -1, "x": 40, "y": 40, "z": 80, "mission_pad_yaw": 180}
                self.assertEqual(self.wt.fixed_pad_hover_command(config, state), [0, 0, 0, 0])
                state["mid"] = config["mission_pad"]
                state["mission_pad_yaw"] = None
                self.assertEqual(self.wt.fixed_pad_hover_command(config, state), [0, 0, 0, 0])
                if config["mission_pad_heading_tolerance_deg"] is not None:
                    state["mission_pad_yaw"] = 100
                    self.assertEqual(self.wt.fixed_pad_hover_command(config, state), [0, 0, 0, 0])


class Published75RegressionTests(unittest.TestCase):
    """Read-only comparison against the last published 75 cm implementation.

    Geometry/body-frame adaptation is intentional; feedback gains, command
    generation and the takeoff/landing lifecycle must not silently change.
    These checks never connect to aircraft or execute the run function.
    """

    @classmethod
    def setUpClass(cls):
        cls.reference_source = subprocess.check_output(
            ["git", "show", f"{REFERENCE_COMMIT}:wind_tunnel_collector.py"],
            cwd=ROOT, text=True,
        )
        cls.current_source = (ROOT / "wind_tunnel_collector.py").read_text()

    def test_legacy_helpers_and_constants_unchanged_by_new_pad_controller(self):
        def control_nodes(source):
            return [ast.dump(n) for n in ast.parse(source).body if (
                isinstance(n, ast.Assign)
                or isinstance(n, ast.FunctionDef) and n.name not in {
                    "build_configs", "run", "run_fixed_pad_hover_control",
                    "check_and_recenter_assigned_pad"}
            )]
        self.assertEqual(control_nodes(self.current_source),
                         control_nodes(self.reference_source))

    def test_new_lifecycle_has_no_fixed_time_alignment_success(self):
        tree = ast.parse(self.current_source)
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        text = ast.unparse(run)
        self.assertNotIn('dc.set_phase_all("wind_tunnel_hover")', text.replace("'", '"'))
        self.assertNotIn("time.sleep(INITIAL_POSITION_CORRECTION_DURATION_SEC)", text)
        self.assertIn("command_locks", text)
        self.assertIn("control_inactive", text)
        self.assertNotIn("swarm.takeoff()", text)
        self.assertIn("ready_events", text)
        self.assertLess(text.index("controller_thread.start()"), text.index("takeoffs.start()"))
        self.assertLess(text.index("logger_thread.start()"), text.index("takeoffs.start()"))
        control = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "run_fixed_pad_hover_control")
        self.assertNotIn("fixed_pad_hover_command(", ast.unparse(control))

    def test_75_configs_keep_only_confirmed_front_tail_frame_exception(self):
        reference = load_offline(self.reference_source)
        current = load_offline(self.current_source)
        frame_fields = {"start_x", "start_y", "target_x", "target_y",
                        "pad_origins_cm", "pad_x_aligned_with_body_forward"}
        for formation, wind in itertools.product(FORMATIONS, WINDS):
            record = experiment(formation, wind, 75)
            for old, new in zip(reference.build_configs(record), current.build_configs(record)):
                if formation == "front" and wind == "tail wind":
                    old = {k: v for k, v in old.items() if k not in frame_fields}
                    new = {k: v for k, v in new.items() if k not in frame_fields}
                self.assertEqual(old, new, (formation, wind))


if __name__ == "__main__":
    unittest.main()
