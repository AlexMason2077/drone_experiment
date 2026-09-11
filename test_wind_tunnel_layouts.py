"""Historical Wind Tunnel layout checks using AST-isolated helpers only.

No data_collector/aircraft SDK module is imported; no flight entry point runs.
"""
import ast
import itertools
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent
REFERENCE_COMMIT = "3ff2f14c"
PADS = (5, 6, 7, 8, 1)
FORMATIONS = ("front", "vee", "column", "diamond", "echalon", "echelon", "echolon")
WINDS = ("head wind", "tail wind", "side wind")


def load_offline(source=None):
    dc_names = {
        "IP_PREFIX", "DRONE_NUMBER_TO_IP_SUFFIX", "COLUMN_SPACING_CM", "ROW_SPACING_CM",
        "TAKEOFF_HEIGHT_CM", "VEE_COLUMN_ORIGINS_CM", "VEE_75_COLUMN_ORIGINS_CM",
        "ECHALON_COLUMN_ORIGINS_CM", "int_field", "experiment_inter_drone_distance_cm",
        "clamp", "is_echalon_formation", "position_at_column_row",
        "pad_origin_for_detection", "to_global",
    }
    tree = ast.parse((ROOT / "data_collector.py").read_text())
    nodes = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name in dc_names
    ) or (
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in dc_names for t in n.targets)
    )]
    dc_namespace = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "isolated-dc", "exec"), dc_namespace)
    namespace = {"math": math, "dc": SimpleNamespace(**dc_namespace)}
    tree = ast.parse(source if source is not None else (ROOT / "wind_tunnel_collector.py").read_text())
    pure_functions = {
        "build_configs", "_minimum_effective_control", "_signed_angle_degrees",
        "fixed_pad_hover_command", "run_fixed_pad_hover_control",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) or (
        isinstance(n, ast.FunctionDef) and n.name in pure_functions
    )]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "isolated-wind-tunnel", "exec"), namespace)
    return SimpleNamespace(**namespace)


def experiment(formation, wind, spacing):
    return {
        "formation": formation, "wind_direction": wind, "wind_speed": "Level2",
        "inter_drone_distance_cm": spacing,
        "drones": [
            {"drone_number": str(i), "takeoff_order": str(i),
             "mission_pad": str(pad), "battery_id": battery}
            for i, pad, battery in zip(range(1, 6), PADS, ("B11", "B10", "B13", "B14", "B12"))
        ],
    }


class LegacyLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = load_offline()
        cls.reference = load_offline(subprocess.check_output(
            ["git", "show", REFERENCE_COMMIT + ":wind_tunnel_collector.py"],
            cwd=ROOT, text=True))

    def test_only_diamond_head_50_coordinates_differ_from_historical_source(self):
        geometry = {"start_x", "start_y", "target_x", "target_y", "pad_origins_cm"}
        for formation, wind, spacing in itertools.product(FORMATIONS, WINDS, (50, 75)):
            record = experiment(formation, wind, spacing)
            with self.subTest(formation=formation, wind=wind, spacing=spacing):
                current = self.w.build_configs(record)
                historical = self.reference.build_configs(record)
                if (formation, wind, spacing) == ("diamond", "head wind", 50):
                    self.assertNotEqual(current, historical)
                    for a, b in zip(current, historical):
                        self.assertEqual({k:v for k,v in a.items() if k not in geometry},
                                         {k:v for k,v in b.items() if k not in geometry})
                else:
                    self.assertEqual(current, historical)

    def test_diamond_head_50_centres_targets_and_pad_transforms(self):
        configs = self.w.build_configs(experiment("diamond", "head wind", 50))
        expected = {5:(50,0), 6:(0,50), 7:(50,50), 8:(100,50), 1:(50,100)}
        for config, pad in zip(configs, PADS):
            self.assertEqual(config["mission_pad"], pad)
            self.assertEqual(config["target_pad"], pad)
            self.assertEqual(config["pad_origins_cm"], expected)
            self.assertEqual((config["start_x"], config["start_y"]), expected[pad])
            self.assertEqual((config["target_x"], config["target_y"], config["target_z"]),
                             (*expected[pad],80))
            if pad != 7:
                self.assertEqual(math.dist(expected[7], expected[pad]), 50)
            for observed, origin in expected.items():
                raw = dict(mid=observed,x=3,y=-7,z=81)
                self.assertEqual(self.w.dc.to_global(config,raw),
                                 (origin[0]+3,origin[1]-7,81))
                self.assertEqual(raw,dict(mid=observed,x=3,y=-7,z=81))
        # The existing RC implementation consumes the corrected coordinates;
        # its gain, direction mapping and limits remain historical.
        centre = configs[2]
        for observed, command in ((5,[0,12,0,0]),(6,[12,0,0,0]),
                                  (7,[0,0,0,0]),(8,[-12,0,0,0]),(1,[0,-12,0,0])):
            raw = dict(mid=observed,x=0,y=0,z=80,mission_pad_yaw=180)
            self.assertEqual(self.w.fixed_pad_hover_command(centre,raw),command)

    def test_vee_head_50_and_75_have_actual_selected_arm_distance(self):
        for spacing in (50, 75):
            configs = self.w.build_configs(experiment("vee", "head wind", spacing))
            step = spacing / math.sqrt(2)
            expected = [(0,0),(step,step),(2*step,2*step),(3*step,step),(4*step,0)]
            for config, point, pad in zip(configs, expected, PADS):
                self.assertEqual(config["mission_pad"], pad)
                self.assertEqual(config["target_pad"], pad)
                self.assertEqual(set(config["pad_origins_cm"]), set(PADS))
                self.assertEqual(config["target_z"], 80)
                self.assertAlmostEqual(config["start_x"], point[0])
                self.assertAlmostEqual(config["start_y"], point[1])
                self.assertEqual((config["target_x"], config["target_y"]),
                                 (config["start_x"], config["start_y"]))
            positions = [(c["start_x"], c["start_y"]) for c in configs]
            for first, second in zip(positions, positions[1:]):
                self.assertAlmostEqual(math.dist(first, second), spacing)

    def test_current_dc_dependency_matches_historical_blob(self):
        historical = subprocess.check_output(
            ["git", "show", REFERENCE_COMMIT + ":data_collector.py"], cwd=ROOT)
        self.assertEqual((ROOT / "data_collector.py").read_bytes(), historical)


if __name__ == "__main__":
    unittest.main()
