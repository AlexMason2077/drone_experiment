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

    def test_entire_collector_has_only_authorized_description_and_layout_changes(self):
        original = subprocess.check_output(
            ["git", "show", REFERENCE_COMMIT + ":wind_tunnel_collector.py"], cwd=ROOT, text=True)
        historical = '''        + WIND_FLOW_DESCRIPTIONS.get(
            str(experiment.get("wind_direction", "")).strip().lower(),
            "unknown; verify fan placement before takeoff",
        ),'''
        correction = '''        + ("source at +Y; airflow +Y -> -Y (against the +Y-facing noses)"
           if configs[0]["formation"] == "vee" and configs[0]["wind_direction"] == "head wind"
           else WIND_FLOW_DESCRIPTIONS.get(
            str(experiment.get("wind_direction", "")).strip().lower(),
            "unknown; verify fan placement before takeoff",
        )),'''
        current = (ROOT / "wind_tunnel_collector.py").read_text()
        diamond_branch = '''        elif formation == "diamond" and wind_direction == "head wind" and spacing == 50:
            # Match the same diamond geometry at 50 cm centre-to-outer spacing.
            fixed_positions.append(tuple(value * (spacing / 75.0)
                                         for value in WIND_TUNNEL_DIAMOND_75_POSITIONS_CM[idx]))
'''
        self.assertEqual(current.count(diamond_branch), 1)
        current = current.replace(diamond_branch, "")
        self.assertEqual(original.count(historical), 1)
        self.assertEqual(current, original.replace(historical, correction))
        # Removing only the authorized layout branch and console correction
        # yields identical full AST, including RC, takeoff and landing logic.
        self.assertEqual(ast.dump(ast.parse(current.replace(correction, historical))),
                         ast.dump(ast.parse(original)))

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
