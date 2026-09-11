"""Offline SDK doubles only: never import or connect to real aircraft."""
import threading
import time
import unittest
from test_wind_tunnel_layouts import load_offline, experiment


class FakeTello:
    def __init__(self, pad=5, x=30, y=-20, z=80, jitter=False):
        # A real Tello reports a new pose on every packet. A fake whose pose
        # never changes trips the guard's frozen/stale check instead of the
        # path under test, so tests that must survive >1.5 s set jitter=True.
        self.jitter = jitter
        self._jittered = False
        self.state = dict(mid=pad, x=x, y=y, z=z)
        self.retry_count = 3
        self.calls = []
        self.fail = False
        self.arrive = True
        self.in_go = False
        self.entered = threading.Event()
        self.release = None
        self.heartbeat = threading.Event()
        self.cached = False

    def get_current_state(self):
        if self.jitter:
            self.state["y"] += 1 if self._jittered else -1
            self._jittered = not self._jittered
        return self.state if self.cached else dict(self.state)

    def send_command_without_return(self, command):
        assert command == "stop"
        self.calls.append(("stop",))
        if self.release is not None:
            self.release.set()

    def send_rc_control(self, *args):
        assert not self.in_go, "RC must not interrupt go"
        assert args == (0, 0, 0, 0), "old RC micro-corrections must not run"
        self.calls.append(("rc", args))
        self.heartbeat.set()

    def go_xyz_speed_mid(self, *args):
        self.calls.append(("go", args, self.retry_count))
        self.in_go = True
        self.entered.set()
        try:
            if self.release is not None:
                assert self.release.wait(5), "test release timeout"
            if self.fail:
                raise RuntimeError("no valid marker")
            if self.arrive:
                self.state.update(mid=args[4], x=args[0], y=args[1], z=args[2])
        finally:
            self.in_go = False


class FakeTakeoffTello(FakeTello):
    def __init__(self, pad=5):
        super().__init__(pad=pad, x=16, y=0)
        self.takeoff_entered = threading.Event()
        self.takeoff_release = threading.Event()
        self.takeoff_failure = False
        self.in_takeoff = False

    def takeoff(self):
        self.in_takeoff = True
        self.calls.append(("takeoff", self.retry_count))
        self.takeoff_entered.set()
        try:
            if not self.takeoff_release.wait(8):
                raise TimeoutError("test takeoff timed out")
            if self.takeoff_failure:
                raise RuntimeError("takeoff reply missing")
        finally:
            self.in_takeoff = False

    def go_xyz_speed_mid(self, *args):
        assert not self.in_takeoff, "go must wait for this drone's takeoff reply"
        super().go_xyz_speed_mid(*args)

    def send_rc_control(self, *args):
        assert not self.in_takeoff, "RC must wait for this drone's takeoff reply"
        super().send_rc_control(*args)


class RetryTello(FakeTello):
    """First go outlives the watchdog; the retry after the brake arrives."""

    def __init__(self, brake_releases_go=True, **kwargs):
        super().__init__(jitter=True, **kwargs)
        self.arrive = False
        self.release = threading.Event()
        self.braked = threading.Event()
        self.second_go = threading.Event()
        # The brake is fire-and-forget. Whether the outstanding go actually
        # returns because of it is the variable under test, so the test drives
        # `release` itself when this is False.
        self.brake_releases_go = brake_releases_go
        self.gos = 0

    def send_command_without_return(self, command):
        assert command == "stop"
        self.calls.append(("stop",))
        self.braked.set()
        if self.brake_releases_go:
            self.release.set()

    def go_xyz_speed_mid(self, *args):
        self.gos += 1
        if self.gos > 1:
            self.arrive = True
            self.second_go.set()
        super().go_xyz_speed_mid(*args)


class PadCenteringTests(unittest.TestCase):
    def setUp(self):
        self.w = load_offline()
        self.w.dc.start_pad_alignment_state.__globals__["get_state_safe"] = lambda t: dict(t.state)
        self.phases = []
        self.w.dc.set_phase = lambda i, p: self.phases.append((i, p))
        self.configs = self.w.build_configs(experiment("front", "tail wind", 50))

    def test_original_goal_speed_and_single_attempt(self):
        t = FakeTello()
        status, _ = self.w.check_and_recenter_assigned_pad(t, self.configs[0])
        self.assertEqual(status, "aligned")
        self.assertEqual(t.calls, [("go", (0, 0, 80, 20, 5), 1)])
        self.assertEqual(t.retry_count, 3)

    def test_ack_without_arrival_is_not_success(self):
        t = FakeTello(); t.arrive = False
        status, _ = self.w.check_and_recenter_assigned_pad(t, self.configs[0])
        self.assertEqual(status, "not_centered")

    def test_original_15cm_xyz_tolerance(self):
        for axis in ("x", "y", "z"):
            t = FakeTello(x=0, y=0)
            t.state[axis] = (80 if axis == "z" else 0) + 15
            self.assertEqual(self.w.check_and_recenter_assigned_pad(t, self.configs[0])[0], "aligned")
            self.assertFalse(any(c[0] == "go" for c in t.calls))
            t.state[axis] += 1
            self.w.check_and_recenter_assigned_pad(t, self.configs[0])
            self.assertEqual(t.calls[-1][0], "go")

    def test_missing_or_unknown_pad_never_uses_old_coordinates(self):
        for pad in (-1, 2):
            t = FakeTello(pad=pad)
            self.assertEqual(self.w.check_and_recenter_assigned_pad(t, self.configs[0])[0], "waiting_pad")
            self.assertEqual(t.calls, [("rc", (0, 0, 0, 0))])

    def test_known_neighbour_uses_global_target_in_visible_pad_frame(self):
        for wind, expected in (("head wind", (0, 50, 80, 20, 7)),
                               ("tail wind", (50, 0, 80, 20, 7))):
            config = self.w.build_configs(experiment("front", wind, 50))[3]
            t = FakeTello(pad=7, x=31, y=8)
            status, _ = self.w.check_and_recenter_assigned_pad(t, config)
            self.assertEqual(t.calls[0], ("go", expected, 1))
            self.assertEqual(status, "not_centered")  # must detect own Pad8 to finish

    def test_changed_reference_before_dispatch_does_not_send_go(self):
        t = FakeTello(pad=7)
        status, _ = self.w.check_and_recenter_assigned_pad(t, self.configs[0], expected_pad=6)
        self.assertEqual(status, "waiting_pad")
        self.assertFalse(any(c[0] == "go" for c in t.calls))

    def test_all_layout_recovery_targets_map_back_to_assigned_origin(self):
        for formation in ("front", "vee", "column", "echalon", "diamond"):
            for wind in ("head wind", "tail wind", "side wind"):
                for spacing in (50, 75):
                    for config in self.w.build_configs(experiment(formation, wind, spacing)):
                        for pad in (5, 6, 7, 8, 1):
                            x, y, z = self.w.PadRecoveryGeometry.target(config, pad)
                            gx, gy, _ = self.w.dc.to_global(config, dict(mid=pad, x=x, y=y, z=z))
                            own = config["pad_origins_cm"][config["mission_pad"]]
                            self.assertAlmostEqual(gx, own[0])
                            self.assertAlmostEqual(gy, own[1])

    def test_rotated_common_pad_axes_recovery(self):
        config = dict(self.configs[3], mission_pad_axes_global=((0, 1), (-1, 0)))
        self.assertEqual(self.w.PadRecoveryGeometry.target(config, 7), (0, -50, 80))

    def test_guard_compares_global_positions_across_pad_switch(self):
        config = self.w.build_configs(experiment("front", "head wind", 50))[3]
        guard = self.w.PadObservationGuard(8, config)
        for at in (0, .2, .4):
            result = guard.observe(dict(mid=7, x=0, y=40, z=80), at)
        self.assertTrue(result["stable"])
        self.assertEqual(result["error"], 10)
        self.assertFalse(result["aligned"])
        result = guard.observe(dict(mid=8, x=0, y=-5, z=80), .5)
        self.assertFalse(result["jump"])
        self.assertTrue(result["aligned"])

    def test_cross_pad_recovery_finishes_only_after_own_pad_reacquired(self):
        config = self.w.build_configs(experiment("front", "head wind", 50))[3]
        t = FakeTello(pad=7, x=0, y=40); t.arrive = False; t.release = threading.Event()
        swarm = type("Swarm", (), {"tellos": [t]})()
        runner = threading.Thread(target=self.w.run_fixed_pad_hover_control,
                                  args=(swarm, [config]), kwargs=dict(duration_sec=1.5))
        runner.start()
        try:
            self.assertTrue(t.entered.wait(1))
            t.state.update(mid=7, x=0, y=45, z=80)
            time.sleep(.12)
            t.state.update(mid=8, x=0, y=-1, z=80)
            t.release.set()
        finally:
            t.release.set(); runner.join(3)
        self.assertFalse(runner.is_alive())
        self.assertEqual([c for c in t.calls if c[0] == "go"], [("go", (0, 50, 80, 20, 7), 1)])
        self.assertIn((0, "wind_tunnel_hover"), self.phases)
        self.assertFalse(any(c[0] == "stop" for c in t.calls))

    def test_no_pad_then_known_neighbour_can_start_recovery(self):
        t = FakeTello(pad=-1, x=0, y=0)
        swarm = type("Swarm", (), {"tellos": [t]})()
        runner = threading.Thread(target=self.w.run_fixed_pad_hover_control,
                                  args=(swarm, self.configs[:1]), kwargs=dict(duration_sec=1.0))
        runner.start()
        self.assertTrue(t.heartbeat.wait(.5))
        self.assertFalse(t.entered.is_set())
        t.state.update(mid=6, x=-40, y=0, z=80)
        self.assertTrue(t.entered.wait(.8))
        runner.join(3)
        self.assertEqual([c for c in t.calls if c[0] == "go"], [("go", (-50, 0, 80, 20, 6), 1)])

    def test_failure_restores_sdk_retry_setting(self):
        t = FakeTello(); t.fail = True
        with self.assertRaises(RuntimeError):
            self.w.check_and_recenter_assigned_pad(t, self.configs[0])
        self.assertEqual(t.retry_count, 3)
        self.assertEqual(len(t.calls), 1)

    def test_all_five_pad_ids_and_both_distances(self):
        for spacing in (50, 75):
            for c in self.w.build_configs(experiment("front", "tail wind", spacing)):
                t = FakeTello(pad=c["mission_pad"])
                self.w.check_and_recenter_assigned_pad(t, c)
                self.assertEqual(t.calls[0][1], (0, 0, 80, 20, c["mission_pad"]))

    def test_one_blocking_move_does_not_block_other_drone_or_send_rc(self):
        a, b = FakeTello(), FakeTello(pad=6, x=0, y=0)
        release = threading.Event(); a.release = release
        swarm = type("Swarm", (), {"tellos": [a, b]})()
        stop = threading.Event(); inactive = [False, False]
        thread = threading.Thread(target=self.w.run_fixed_pad_hover_control,
                                  args=(swarm, self.configs[:2]),
                                  kwargs=dict(stop_event=stop, landed=inactive))
        thread.start()
        try:
            self.assertTrue(a.entered.wait(1))
            self.assertTrue(b.heartbeat.wait(1))
            inactive[0] = True  # same gate used by a 20% landing request
            stop.set()
        finally:
            stop.set(); release.set(); thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(sum(c[0] == "go" for c in a.calls), 1)
        self.assertNotIn((0, "wind_tunnel_hover"), self.phases)

    def test_already_stopped_never_moves(self):
        t = FakeTello(); stop = threading.Event(); stop.set()
        swarm = type("Swarm", (), {"tellos": [t]})()
        self.w.run_fixed_pad_hover_control(swarm, self.configs[:1], stop_event=stop)
        self.assertEqual(t.calls, [])

    def test_fast_takeoff_enters_control_while_other_aircraft_still_taking_off(self):
        a, b = FakeTakeoffTello(), FakeTakeoffTello(pad=6)
        swarm = type("Swarm", (), {"tellos": [a, b]})()
        stop = threading.Event(); locks = [threading.Lock(), threading.Lock()]
        takeoffs = self.w.IndependentTakeoff(swarm, self.configs[:2], stop, locks)
        control = threading.Thread(target=self.w.run_fixed_pad_hover_control,
                                   args=(swarm, self.configs[:2]),
                                   kwargs=dict(stop_event=stop, command_locks=locks,
                                               ready_events=takeoffs.ready))
        control.start(); takeoffs.start()
        try:
            self.assertTrue(a.takeoff_entered.wait(1))
            self.assertTrue(b.takeoff_entered.wait(1))
            self.assertFalse(a.entered.is_set())
            self.assertFalse(b.heartbeat.is_set())
            a.takeoff_release.set()
            # Own control resumes at once; only the first go waits out the
            # post-takeoff settle window.
            self.assertTrue(a.heartbeat.wait(1))
            self.assertTrue(a.entered.wait(self.w.TAKEOFF_SETTLE_SEC + 1))
            self.assertTrue(b.in_takeoff)
            self.assertFalse(b.entered.is_set())
            self.assertFalse(b.heartbeat.is_set())
            b.takeoff_release.set()
            self.assertTrue(b.entered.wait(self.w.TAKEOFF_SETTLE_SEC + 1))
            takeoffs.wait()
            self.assertEqual(a.calls[0], ("takeoff", 1))
            self.assertEqual(b.calls[0], ("takeoff", 1))
        finally:
            stop.set(); a.takeoff_release.set(); b.takeoff_release.set()
            for thread in takeoffs.threads:
                thread.join(2)
            control.join(3)
        self.assertFalse(control.is_alive())
        self.assertEqual(a.retry_count, 3)
        self.assertEqual(b.retry_count, 3)

    def test_failed_takeoff_never_releases_control_gate(self):
        t = FakeTakeoffTello(); t.takeoff_failure = True; t.takeoff_release.set()
        swarm = type("Swarm", (), {"tellos": [t]})()
        stop = threading.Event(); locks = [threading.Lock()]; events = []
        takeoffs = self.w.IndependentTakeoff(swarm, self.configs[:1], stop, locks,
                                          lambda *row: events.append(row))
        control = threading.Thread(target=self.w.run_fixed_pad_hover_control,
                                   args=(swarm, self.configs[:1]),
                                   kwargs=dict(stop_event=stop, ready_events=takeoffs.ready,
                                               command_locks=locks))
        control.start(); takeoffs.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "takeoff reply missing"):
                takeoffs.wait()
            self.assertFalse(takeoffs.ready[0].is_set())
            self.assertFalse(t.entered.is_set())
            self.assertFalse(t.heartbeat.is_set())
            self.assertEqual(t.retry_count, 3)
            self.assertTrue(any(e[1] == "TAKEOFF_FAILED" for e in events))
        finally:
            stop.set(); control.join(2)

    def test_stop_before_dispatch_never_sends_takeoff(self):
        t = FakeTakeoffTello(); swarm = type("Swarm", (), {"tellos": [t]})()
        stop = threading.Event(); stop.set()
        takeoffs = self.w.IndependentTakeoff(swarm, self.configs[:1], stop, [threading.Lock()])
        takeoffs.start(); takeoffs.wait()
        self.assertEqual(t.calls, [])
        self.assertFalse(takeoffs.ready[0].is_set())

    def test_stop_during_takeoff_ignores_late_success_reply(self):
        t = FakeTakeoffTello(); swarm = type("Swarm", (), {"tellos": [t]})()
        stop = threading.Event()
        takeoffs = self.w.IndependentTakeoff(swarm, self.configs[:1], stop, [threading.Lock()])
        takeoffs.start()
        try:
            self.assertTrue(t.takeoff_entered.wait(1))
            stop.set(); t.takeoff_release.set(); takeoffs.wait()
            self.assertFalse(takeoffs.ready[0].is_set())
            self.assertEqual(t.retry_count, 3)
        finally:
            stop.set(); t.takeoff_release.set()
            for thread in takeoffs.threads:
                thread.join(2)

    def run_one(self, tello, duration=1.5):
        swarm = type("Swarm", (), {"tellos": [tello]})()
        events = []
        self.w.run_fixed_pad_hover_control(
            swarm, self.configs[:1], duration_sec=duration,
            event_sink=lambda *row: events.append(row))
        return [e for e in events if e[1] in ("SOFT_FAULT", "FAULT")]

    def tune(self, **constants):
        """Shrink control timing so behaviour, not wall-clock, is under test."""
        self.w.run_fixed_pad_hover_control.__globals__.update(constants)

    def test_cached_packet_cannot_arm_a_move(self):
        t = FakeTello(); t.cached = True
        self.run_one(t, 0.8)
        self.assertFalse(any(c[0] == "go" for c in t.calls))

    def test_error_latches_without_retry_even_after_pose_recovers(self):
        t = FakeTello(); t.fail = True
        self.run_one(t)
        self.assertEqual(sum(c[0] == "go" for c in t.calls), 1)
        self.assertEqual(sum(c[0] == "stop" for c in t.calls), 1)
        self.assertIn((0, "wind_tunnel_control_fault"), self.phases)

    def test_watchdog_brakes_pad_loss_during_blocking_go(self):
        t = FakeTello(); t.release = threading.Event()
        runner = threading.Thread(target=self.run_one, args=(t,))
        runner.start()
        self.assertTrue(t.entered.wait(1))
        t.state["mid"] = -1
        self.assertTrue(t.release.wait(0.5), "watchdog must run before SDK go returns")
        runner.join(3)
        self.assertFalse(runner.is_alive())
        self.assertEqual(sum(c[0] == "go" for c in t.calls), 1)
        self.assertEqual(sum(c[0] == "stop" for c in t.calls), 1)
        # Fake stop releases go, which reports arrival/ok. Fault must stay latched.
        self.assertNotIn((0, "wind_tunnel_hover"), self.phases)

    def test_timeout_or_frozen_pose_cannot_trigger_second_go(self):
        t = FakeTello(); t.arrive = False
        self.run_one(t, 2.0)
        self.assertEqual(sum(c[0] == "go" for c in t.calls), 1)
        self.assertEqual(sum(c[0] == "stop" for c in t.calls), 1)

    def test_guard_rejects_jumps_and_requires_multiple_packets(self):
        guard = self.w.PadObservationGuard(5)
        def sample(x, at):
            return guard.observe(dict(mid=5, x=x, y=0, z=80), at)
        self.assertFalse(sample(0, 0)["stable"])
        self.assertFalse(sample(16, 0.1)["stable"])
        self.assertFalse(sample(0, 0.2)["stable"])
        self.assertTrue(sample(50, 0.3)["jump"])
        self.assertFalse(sample(50, 0.4)["stable"])
        self.assertTrue(sample(50, 0.7)["stable"])
        self.assertTrue(sample(50, 2.0)["frozen"])

    def test_watchdog_brakes_divergence(self):
        t = FakeTello(x=16, y=0); t.release = threading.Event()
        runner = threading.Thread(target=self.run_one, args=(t,))
        runner.start()
        self.assertTrue(t.entered.wait(1))
        t.state["x"] = 34  # <20 cm jump, but >15 cm deterioration
        self.assertTrue(t.release.wait(0.5))
        runner.join(3)
        self.assertEqual(sum(c[0] == "go" for c in t.calls), 1)

    def test_fresh_aligned_packets_verify_success(self):
        t = FakeTello(x=16, y=0)
        self.run_one(t, 1.0)
        self.assertIn((0, "wind_tunnel_hover"), self.phases)
        self.assertFalse(any(c[0] == "stop" for c in t.calls))

    def test_single_threshold_crossing_does_not_trigger_move(self):
        guard = self.w.PadObservationGuard(5)
        for at in (0, 0.2, 0.4):
            guard.observe(dict(mid=5, x=14, y=0, z=80), at)
        self.assertFalse(guard.observe(dict(mid=5, x=16, y=0, z=80), 0.5)["stable"])
        guard.observe(dict(mid=5, x=14, y=0, z=80), 0.6)
        self.assertFalse(guard.observe(dict(mid=5, x=16, y=0, z=80), 0.7)["stable"])

    def test_watchdog_brakes_stale_telemetry_during_go(self):
        t = FakeTello(x=16, y=0); t.release = threading.Event()
        runner = threading.Thread(target=self.run_one, args=(t, 1.6))
        runner.start()
        self.assertTrue(t.entered.wait(1))
        t.cached = True
        self.assertTrue(t.release.wait(1.0))
        runner.join(3)
        self.assertEqual(sum(c[0] == "go" for c in t.calls), 1)
        self.assertEqual(sum(c[0] == "stop" for c in t.calls), 1)

    def test_watchdog_deadline_works_with_fresh_changing_telemetry(self):
        # Fresh, changing, non-diverging telemetry defeats every other guard,
        # so only the deadline can end this move. It is a recoverable failure:
        # brake and retry, never disable the aircraft for the whole run.
        self.tune(FIXED_PAD_MOVE_MAX_SEC=0.8)
        t = FakeTello(x=16, y=0); t.release = threading.Event()
        original_read = t.get_current_state
        def read():
            if t.in_go:
                t.state["x"] = 17 if t.state["x"] == 16 else 16
            return original_read()
        t.get_current_state = read
        events = self.run_one(t, 2.0)
        self.assertEqual(sum(c[0] == "stop" for c in t.calls), 1)
        self.assertEqual([e[1] for e in events], ["SOFT_FAULT"])
        self.assertIn("deadline exceeded", events[0][2])
        self.assertIn((0, "wind_tunnel_correction_retry"), self.phases)
        self.assertNotIn((0, "wind_tunnel_control_fault"), self.phases)

    def test_missing_startup_fields_wait_instead_of_arming(self):
        guard = self.w.PadObservationGuard(5)
        for at in (0, 0.2, 0.4, 1.0):
            observation = guard.observe({}, at)
            self.assertFalse(observation["valid"])
            self.assertFalse(observation["stable"])

    def test_fault_event_is_preserved_and_other_drone_continues(self):
        a, b = FakeTello(x=16, y=0), FakeTello(pad=6, x=0, y=0)
        a.fail = True
        events = []
        swarm = type("Swarm", (), {"tellos": [a, b]})()
        self.w.run_fixed_pad_hover_control(
            swarm, self.configs[:2], duration_sec=1.0,
            event_sink=lambda *row: events.append(row))
        faults = [e for e in events if e[1] == "FAULT"]
        self.assertEqual(len(faults), 1)
        self.assertEqual(faults[0][0], 0)
        self.assertIn("no valid marker", faults[0][2])
        self.assertIn((1, "wind_tunnel_hover"), self.phases)
        self.assertFalse(any(c[0] == "stop" for c in b.calls))

    def test_recenter_budget_covers_sdk_overhead_not_only_travel(self):
        """Field regression, 2026-09-10 column/side runs.

        drone_1 needed 3.42 s to complete a 36 cm recenter and drone_2 2.50 s
        for 18 cm.  The old distance/speed + 1.5 s budget allowed 3.34 s and
        2.40 s, so both were failed and disabled for the rest of the flight.
        """
        g = self.w.run_fixed_pad_hover_control.__globals__
        for distance, measured in ((36.8, 3.42), (18.0, 2.50), (13.0, 2.07)):
            budget = min(g["FIXED_PAD_MOVE_MAX_SEC"],
                         distance / self.w.dc.TAKEOFF_CLIMB_SPEED_CM_S
                         + g["FIXED_PAD_MOVE_OVERHEAD_SEC"])
            self.assertGreater(budget, measured, f"{distance} cm recenter")

    def test_larger_error_never_gets_a_smaller_budget(self):
        g = self.w.run_fixed_pad_hover_control.__globals__
        budgets = [min(g["FIXED_PAD_MOVE_MAX_SEC"],
                       d / self.w.dc.TAKEOFF_CLIMB_SPEED_CM_S + g["FIXED_PAD_MOVE_OVERHEAD_SEC"])
                   for d in range(0, 120, 5)]
        self.assertEqual(budgets, sorted(budgets))

    def test_only_an_explicit_error_reply_counts_as_motionless(self):
        prefix = "Command 'go 0 0 80 20 m5' was unsuccessful for 1 tries. Latest response:\t"
        self.assertTrue(self.w.go_was_rejected_without_motion(prefix + "'error Not joystick'"))
        self.assertTrue(self.w.go_was_rejected_without_motion(prefix + "'error Motor stop'"))
        # A timeout, an exhausted retry loop or a decode failure prove nothing
        # about whether the aircraft is moving: those must stay hard faults.
        for ambiguous in ("'Aborting command 'go 0 0 80 20 m5'. Did not receive a "
                          "response after 7 seconds'",
                          "'max retries exceeded'", "'response decode error'", "'ok'"):
            self.assertFalse(self.w.go_was_rejected_without_motion(prefix + ambiguous), ambiguous)
        self.assertFalse(self.w.go_was_rejected_without_motion("no valid marker"))

    def test_timed_out_recenter_brakes_then_retries_and_succeeds(self):
        self.tune(FIXED_PAD_MOVE_MAX_SEC=0.8, SOFT_FAULT_COOLDOWN_SEC=0.4)
        t = RetryTello(x=16, y=0)
        events = self.run_one(t, 3.5)
        self.assertEqual(sum(c[0] == "go" for c in t.calls), 2)
        self.assertEqual([e[1] for e in events], ["SOFT_FAULT"])
        self.assertIn((0, "wind_tunnel_correction_retry"), self.phases)
        self.assertIn((0, "wind_tunnel_hover"), self.phases)
        self.assertNotIn((0, "wind_tunnel_control_fault"), self.phases)

    def test_retry_waits_for_the_braked_go_to_return_before_moving_again(self):
        self.tune(FIXED_PAD_MOVE_MAX_SEC=0.8, SOFT_FAULT_COOLDOWN_SEC=0.4)
        t = RetryTello(x=16, y=0, brake_releases_go=False)
        runner = threading.Thread(target=self.run_one, args=(t, 4.0))
        runner.start()
        try:
            self.assertTrue(t.braked.wait(2), "watchdog must brake the stuck go")
            # The first go is still inside its SDK wait. Until it returns, no
            # second go and no RC heartbeat may be issued for this aircraft,
            # even though the cooldown itself has already elapsed.
            before = len(t.calls)
            time.sleep(0.8)
            self.assertEqual([c[0] for c in t.calls[before:]], [])
            self.assertFalse(t.second_go.is_set())
            t.release.set()  # the outstanding go finally reports back
            self.assertTrue(t.second_go.wait(2))
        finally:
            t.release.set(); runner.join(5)
        self.assertFalse(runner.is_alive())

    def test_braked_go_that_never_returns_escalates_to_a_hard_fault(self):
        self.tune(FIXED_PAD_MOVE_MAX_SEC=0.5, SOFT_FAULT_COOLDOWN_SEC=0.2,
                  SOFT_FAULT_BRAKE_GRACE_SEC=0.6)
        t = RetryTello(x=16, y=0, brake_releases_go=False)
        events = []
        runner = threading.Thread(target=lambda: events.extend(self.run_one(t, 2.0)))
        runner.start()
        try:
            self.assertTrue(t.braked.wait(2))
            time.sleep(1.0)  # longer than the brake grace
            self.assertIn((0, "wind_tunnel_control_fault"), self.phases)
        finally:
            t.release.set(); runner.join(5)
        self.assertFalse(runner.is_alive())
        self.assertEqual([e[1] for e in events], ["SOFT_FAULT", "FAULT"])
        self.assertIn("did not return", events[1][2])
        self.assertEqual(sum(c[0] == "go" for c in t.calls), 1)

    def test_repeated_soft_faults_still_latch_the_run(self):
        self.tune(FIXED_PAD_MOVE_MAX_SEC=0.5, SOFT_FAULT_COOLDOWN_SEC=0.2)
        t = FakeTello(x=16, y=0, jitter=True); t.arrive = False
        t.release = threading.Event()
        events = self.run_one(t, 6.0)
        maximum = self.w.run_fixed_pad_hover_control.__globals__["SOFT_FAULT_MAX_CONSECUTIVE"]
        self.assertEqual([e[1] for e in events],
                         ["SOFT_FAULT"] * (maximum - 1) + ["FAULT"])
        self.assertEqual(sum(c[0] == "go" for c in t.calls), maximum)
        self.assertIn((0, "wind_tunnel_control_fault"), self.phases)
        # Latched means latched: nothing after the hard fault may move again.
        last_go = max(i for i, c in enumerate(t.calls) if c[0] == "go")
        self.assertNotIn("go", [c[0] for c in t.calls[last_go + 1:]])

    def test_divergence_and_pad_loss_are_never_retried(self):
        for setup in ("diverge", "lose_pad"):
            with self.subTest(setup):
                self.setUp()
                self.tune(SOFT_FAULT_COOLDOWN_SEC=0.2)
                t = FakeTello(x=16, y=0); t.release = threading.Event()
                runner = threading.Thread(target=self.run_one, args=(t, 2.5))
                runner.start()
                try:
                    self.assertTrue(t.entered.wait(2))
                    if setup == "diverge":
                        t.state["x"] = 34  # <20 cm jump, but >15 cm deterioration
                    else:
                        t.state["mid"] = -1
                finally:
                    runner.join(5)
                self.assertEqual(sum(c[0] == "go" for c in t.calls), 1)
                self.assertIn((0, "wind_tunnel_control_fault"), self.phases)

    def test_takeoff_settle_window_is_configured_and_stays_fresh(self):
        settle = self.w.run_fixed_pad_hover_control.__globals__["TAKEOFF_SETTLE_SEC"]
        self.assertGreater(settle, 0.0, "takeoff drift needs time to bleed off")
        # PadObservationGuard calls a pose frozen after 1.5 s without a change,
        # so a longer settle would strand an aircraft in acquire_pad instead.
        self.assertLess(settle, 1.5)

    def test_first_recenter_waits_for_takeoff_drift_to_settle(self):
        self.tune(TAKEOFF_SETTLE_SEC=0.8)
        t = FakeTakeoffTello(); t.takeoff_release.set()
        swarm = type("Swarm", (), {"tellos": [t]})()
        stop = threading.Event(); locks = [threading.Lock()]
        takeoffs = self.w.IndependentTakeoff(swarm, self.configs[:1], stop, locks)
        control = threading.Thread(target=self.w.run_fixed_pad_hover_control,
                                   args=(swarm, self.configs[:1]),
                                   kwargs=dict(stop_event=stop, command_locks=locks,
                                               ready_events=takeoffs.ready))
        control.start(); takeoffs.start(); takeoffs.wait()
        try:
            released = time.monotonic()
            self.assertTrue(t.heartbeat.wait(1), "RC hold must start immediately")
            self.assertFalse(t.entered.wait(0.5), "first go must wait out the settle")
            self.assertTrue(t.entered.wait(2))
            self.assertGreaterEqual(time.monotonic() - released, 0.8)
        finally:
            stop.set(); control.join(3)
        self.assertFalse(control.is_alive())


if __name__ == "__main__":
    unittest.main()
