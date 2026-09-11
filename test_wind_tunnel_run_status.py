"""Exercise the runner's fault lifecycle without importing the app or SDK."""

import ast
import contextlib
from datetime import datetime
import io
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest


class WindTunnelRunStatusTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path(__file__).with_name("app.py").read_text())
        names = {"append_run_output", "reset_run_state", "monitor_experiment_process"}
        nodes = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name in names
        ) or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "RUN_STATE"
                    for target in node.targets)
        )]
        self.ns = {"RUN_LOCK": threading.Lock(), "datetime": datetime,
                   "FORMAL_TAKEOFF_PROMPT": "formal prompt",
                   "TAKEOFF_PROMPT": "takeoff prompt",
                   "DISCHARGE_PROMPT": "discharge prompt"}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), self.ns)
        self.state = self.ns["RUN_STATE"]
        self.state["status"] = "running"

    def append(self, line):
        with contextlib.redirect_stdout(io.StringIO()):
            self.ns["append_run_output"](line)

    def test_fault_survives_log_rotation_and_does_not_disable_running_state(self):
        line = "WIND TUNNEL INVALID: drone_3: localization_suspect"
        self.append(line)
        self.append(line)
        for _ in range(510):
            self.append("Live battery status: drone_3 battery_id=B13 battery=70%")
        self.assertNotIn(line, self.state["output"])
        self.assertEqual(self.state["wind_tunnel_faults"], ["drone_3: localization_suspect"])
        self.assertEqual(self.state["status"], "running")

    def test_zero_exit_does_not_turn_faulted_run_into_success(self):
        process = SimpleNamespace(
            stdout=io.StringIO("WIND TUNNEL INVALID: drone_1: no progress\n"),
            wait=lambda: 0,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.ns["monitor_experiment_process"](process)
        self.assertEqual(self.state["status"], "error")
        self.assertIn("invalid", self.state["message"])
        self.assertTrue(self.state["wind_tunnel_faults"])

    def test_manual_stop_remains_stopped_and_preserves_fault(self):
        self.append("WIND TUNNEL INVALID: drone_1: no progress")
        self.state["status"] = "stopped"
        self.ns["monitor_experiment_process"](SimpleNamespace(stdout=None, wait=lambda: 1))
        self.assertEqual(self.state["status"], "stopped")
        self.assertTrue(self.state["wind_tunnel_faults"])

    def test_normal_completion_is_unchanged(self):
        self.ns["monitor_experiment_process"](SimpleNamespace(stdout=None, wait=lambda: 0))
        self.assertEqual(self.state["status"], "finished")
        self.assertFalse(self.state["wind_tunnel_faults"])

    def test_reset_clears_old_faults(self):
        self.append("WIND TUNNEL INVALID: drone_1: no progress")
        self.ns["reset_run_state"]()
        self.assertFalse(self.state["wind_tunnel_faults"])
        self.assertEqual(self.state["status"], "idle")

    def test_both_launch_paths_clear_old_faults(self):
        tree = ast.parse(Path(__file__).with_name("app.py").read_text())
        for name in ("start_experiment_process", "start_baseline_process"):
            function = next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == name)
            resets = [keyword.value for node in ast.walk(function)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                      and node.func.id == "set_run_state"
                      for keyword in node.keywords if keyword.arg == "wind_tunnel_faults"]
            self.assertEqual(len(resets), 1, name)
            self.assertEqual(ast.literal_eval(resets[0]), [], name)


if __name__ == "__main__":
    unittest.main()
