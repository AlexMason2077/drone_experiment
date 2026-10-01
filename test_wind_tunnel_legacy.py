"""Offline restoration acceptance; AST plus RC doubles, never SDK or flight run."""
import ast
import contextlib
import io
import subprocess
from types import SimpleNamespace
import unittest

from test_wind_tunnel_layouts import ROOT, REFERENCE_COMMIT, load_offline, experiment


class LegacyRestorationTests(unittest.TestCase):
    def setUp(self):
        self.w = load_offline()
        self.config = self.w.build_configs(experiment("vee", "head wind", 50))[2]

    def test_entire_collector_matches_github_except_requested_layout_and_wind_text(self):
        original = subprocess.check_output(
            ["git", "show", "038b238265f5f312337964f3280622726365d37f:wind_tunnel_collector.py"],
            cwd=ROOT, text=True)
        column_anchor = '''        elif diamond_75_side:
'''
        column_branch = '''        elif formation == "column" and wind_direction in {"head wind", "tail wind", "side wind"} and spacing == 50:
            # 50 cm Column shares targets: Pad 5 at +Y, then 6, 7, 8, 1 toward -Y.
            fixed_positions.append((0.0, (4 - idx) * spacing))
'''
        self.assertEqual(original.count(column_anchor), 1)
        historical = '''        + ("source at +Y; airflow +Y -> -Y (against the +Y-facing noses)"
           if configs[0]["formation"] == "vee" and configs[0]["wind_direction"] == "head wind"'''
        correction = '''        + ("source at +X; airflow +X -> -X"
           if (dc.is_echalon_formation(configs[0]["formation"])
               or configs[0]["formation"] in {"vee", "column", "diamond"})
           and configs[0]["wind_direction"] == "side wind"
           and configs[0]["inter_drone_distance_cm"] == 50
           else "source at +Y; airflow +Y -> -Y (against the +Y-facing noses)"
           if (configs[0]["formation"] == "vee"
               or (configs[0]["formation"] in {"column", "diamond"}
                   and configs[0]["inter_drone_distance_cm"] == 50))
           and configs[0]["wind_direction"] == "head wind"
           else "source at -Y; airflow -Y -> +Y (from behind the +Y-facing noses)"
           if configs[0]["formation"] in {"column", "diamond"}
           and configs[0]["inter_drone_distance_cm"] == 50
           and configs[0]["wind_direction"] == "tail wind"'''
        self.assertEqual(original.count(historical), 1)
        current = (ROOT / "wind_tunnel_collector.py").read_text()
        expected = original.replace(column_anchor, column_branch + column_anchor)
        expected = expected.replace(
            'elif formation == "diamond" and wind_direction == "head wind" and spacing == 50:',
            'elif formation == "diamond" and wind_direction in {"head wind", "tail wind", "side wind"} and spacing == 50:',
        )
        expected = expected.replace(
            "WIND_FLOW_DESCRIPTIONS = {",
            "# Headwind defaults to Front's +X-facing frame; run() overrides Vee and 50 cm Column/Diamond.\n"
            "WIND_FLOW_DESCRIPTIONS = {",
        )
        expected = expected.replace(
            "# Physical frame shown on the Wind Tunnel floor plan:",
            "# Front physical frame shown on the Wind Tunnel floor plan:",
        )
        expected = expected.replace(
            "# Consequently +pad Y is global +Y, while aircraft body-right is global -Y.\n",
            "# Consequently +pad Y is global +Y, while aircraft body-right is global -Y.\n"
            "# At 50 cm, Column places Pad 5 at the +Y/front end, then 6, 7, 8, 1 toward -Y\n"
            "# for all wind directions; noses point +Y. Headwind runs +Y -> -Y,\n"
            "# tailwind -Y -> +Y, and sidewind +X -> -X.\n",
        )
        self.assertEqual(current, expected.replace(historical, correction))

    def test_column_and_diamond_preflight_text_changes_only_at_50_cm(self):
        tree = ast.parse((ROOT / "wind_tunnel_collector.py").read_text())
        run = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "run")
        message = next(node.args[0] for node in ast.walk(run)
                       if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == "print"
                       and node.args and any(isinstance(part, ast.Constant)
                                             and part.value == "Physical wind direction: "
                                             for part in ast.walk(node.args[0])))
        expression = compile(ast.Expression(message), "isolated-preflight-text", "eval")
        cases = (
            ("head wind", 50, "Physical wind direction: source at +Y; airflow +Y -> -Y (against the +Y-facing noses)"),
            ("head wind", 75, "Physical wind direction: source at +X; airflow +X -> -X (against the nose)"),
            ("tail wind", 50, "Physical wind direction: source at -Y; airflow -Y -> +Y (from behind the +Y-facing noses)"),
            ("tail wind", 75, "Physical wind direction: source at -X; airflow -X -> +X (from behind)"),
            ("side wind", 50, "Physical wind direction: source at +X; airflow +X -> -X"),
            ("side wind", 75, "Physical wind direction: source at +Y; airflow +Y -> -Y (down the pad line)"),
        )
        for formation in ("column", "diamond"):
            for wind, spacing, expected in cases:
                with self.subTest(formation=formation, wind=wind, spacing=spacing):
                    config = self.w.build_configs(experiment(formation, wind, spacing))[0]
                    actual = eval(expression, {"dc": self.w.dc, "configs": [config],
                                               "experiment": experiment(formation, wind, spacing),
                                               "WIND_FLOW_DESCRIPTIONS": self.w.WIND_FLOW_DESCRIPTIONS})
                    self.assertEqual(actual, expected)

    def test_own_pad_rc_points_toward_local_centre_and_target_height(self):
        for x,y,z,expected in ((20,0,80,[-8,0,0,0]),(0,20,80,[0,-8,0,0]),
                               (-20,0,80,[8,0,0,0]),(0,-20,80,[0,8,0,0]),
                               (0,0,100,[0,0,-6,0]),(0,0,60,[0,0,6,0])):
            state=dict(mid=7,x=x,y=y,z=z,mission_pad_yaw=180)
            self.assertEqual(self.w.fixed_pad_hover_command(self.config,state),expected)

    def test_neighbour_pad_uses_historical_global_recovery(self):
        state=dict(mid=8,x=0,y=0,z=80,mission_pad_yaw=180)
        self.assertEqual(self.w.fixed_pad_hover_command(self.config,state),[-12,12,0,0])
        self.assertNotEqual(self.w.dc.to_global(self.config,state),(None,None,None))

    def test_lost_or_unknown_pad_sends_zero_rc(self):
        for pad in (-1,2):
            self.assertEqual(self.w.fixed_pad_hover_command(
                self.config,dict(mid=pad,x=50,y=50,z=80,mission_pad_yaw=180)),[0,0,0,0])

    def test_legacy_loop_sends_repeated_rc_and_final_hold_without_go(self):
        class Clock:
            now = 0.0
            def monotonic(self):
                return self.now
            def sleep(self, seconds):
                self.now += max(seconds, .001)
        class RCOnlyDouble:
            def __init__(self):
                self.calls=[]
            def send_rc_control(self,*args):
                self.calls.append(args)
        clock=Clock();drone=RCOnlyDouble()
        self.w.run_fixed_pad_hover_control.__globals__["time"] = clock
        self.w.dc.get_state_safe=lambda _:dict(mid=7,x=20,y=0,z=80,mission_pad_yaw=180)
        with contextlib.redirect_stdout(io.StringIO()):
            self.w.run_fixed_pad_hover_control(SimpleNamespace(tellos=[drone]),[self.config],duration_sec=.35)
        self.assertGreaterEqual(drone.calls.count((-8,0,0,0)),3)
        self.assertEqual(drone.calls[-1],(0,0,0,0))

    def test_historical_thresholds_and_takeoff_order_are_retained(self):
        self.assertEqual(self.w.TARGET_BATTERY_PERCENT,20)
        self.assertEqual(self.w.LOW_BATTERY_CONFIRMATION_HITS,8)
        self.assertEqual(self.w.INITIAL_POSITION_CORRECTION_DURATION_SEC,6)
        self.assertEqual(self.w.FIXED_PAD_CONTROL_INTERVAL_SEC,.1)
        self.assertEqual(self.w.FIXED_PAD_XY_TOLERANCE_CM,8)
        tree=ast.parse((ROOT/"wind_tunnel_collector.py").read_text())
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any("go_xyz_speed_mid" in name for name in calls))
        self.assertFalse(any(isinstance(n,ast.ClassDef) and n.name in
                             {"PadObservationGuard","IndependentTakeoff"} for n in tree.body))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run")
        text=ast.unparse(run)
        self.assertLess(text.index("swarm.takeoff()"),text.index("controller_thread.start()"))
        self.assertLess(text.index("controller_thread.start()"),text.index("dc.wait_for_all_expected_start_pads"))
        self.assertIn("time.sleep(INITIAL_POSITION_CORRECTION_DURATION_SEC)",text)
        self.assertIn("low_battery_hits[idx] >= LOW_BATTERY_CONFIRMATION_HITS",text)
        self.assertIn("battery <= TARGET_BATTERY_PERCENT",text)


if __name__ == "__main__":
    unittest.main()
