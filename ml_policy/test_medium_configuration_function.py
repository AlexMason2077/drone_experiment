"""Offline checks of the three-input function and its computed evidence."""
import csv
import hashlib
import inspect
import itertools
import json
import math
import unittest
from pathlib import Path

from ml_policy.medium_configuration_function import F, select_medium_configuration, explain_medium_configuration, RESULT_PATH
from output_py.build_medium_configuration_function import brute_force_makespan


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = RESULT_PATH.parent


def rows(name):
    with (OUTPUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


class MediumConfigurationFunctionTests(unittest.TestCase):
    def test_interface_has_exactly_three_condition_inputs(self):
        self.assertEqual(list(inspect.signature(select_medium_configuration).parameters),
                         ["charging_pad_availability", "wind_direction", "wind_level"])
        self.assertIs(F, select_medium_configuration)

    def test_real_example(self):
        result = F(2, "head", 1)
        self.assertEqual(result, dict(formation="front", inter_drone_spacing_cm=75,
                                     position_assignment={f"drone_{i}": i for i in range(1, 6)}))
        detail = explain_medium_configuration(2, "head", 1)
        self.assertAlmostEqual(detail["total_time_minutes"], 190.766, places=3)

    def test_all_30_inputs_return_safe_available_candidates(self):
        manifest = json.loads((OUTPUT / "manifest.json").read_text())
        unsafe = {tuple(r) for r in manifest["unsafe_excluded"]}
        observed = {(r["wind_direction"], int(r["wind_level"]), r["formation"], int(r["inter_drone_spacing_cm"]))
                    for r in rows("preprocessed_75pct_250cm.csv")}
        for pads, wind, level in itertools.product(range(1, 6), ("head", "tail", "side"), (1, 2)):
            c = F(pads, wind, level)
            key = (wind, level, c["formation"], c["inter_drone_spacing_cm"])
            self.assertIn(key, observed)
            self.assertNotIn(key, unsafe)
        self.assertEqual(len(observed), 55)

    def test_common_wind_aliases(self):
        self.assertEqual(F(2, "HEAD WIND", 1), F(2, "head", 1))
        self.assertEqual(F(3, "side_wind", 2), F(3, "side", 2))
        self.assertEqual(F(4, "tailwind", 2), F(4, "tail", 2))

    def test_unsupported_inputs_fail_instead_of_silently_clipping(self):
        for args in ((0, "head", 1), (6, "head", 1), (True, "head", 1),
                     (1.5, "head", 1), ("2", "head", 1), (2, "no wind", 1),
                     (2, None, 1), (2, "head", 3), (2, "head", False)):
            with self.assertRaises(ValueError):
                F(*args)

    def test_caller_cannot_mutate_future_decisions(self):
        original = F(2, "head", 1)
        altered = F(2, "head", 1)
        altered["formation"] = "invalid"
        altered["position_assignment"]["drone_1"] = 5
        self.assertEqual(original, F(2, "head", 1))

    def test_all_preprocessed_arrivals_remain_in_medium(self):
        preprocessed = rows("preprocessed_75pct_250cm.csv")
        for r in preprocessed:
            self.assertEqual(float(r["reference_start_soc_Bideal"]), 75)
            self.assertEqual(float(r["forward_duration_s"]), 25)
            self.assertEqual(int(r["distance_cm"]), 250)
            for i in range(1, 6):
                rate = float(r[f"position_{i}_discharge_rate_Bideal_pp_min"])
                arrival = float(r[f"position_{i}_arrival_soc_Bideal"])
                self.assertAlmostEqual(arrival, 75 - rate * 25 / 60)
                self.assertGreaterEqual(arrival, 52)
                self.assertLessEqual(arrival, 82)

    def test_every_candidate_charge_schedule_is_exact(self):
        candidates = rows("candidate_rankings.csv")
        self.assertEqual(len(candidates), 275)
        for row in candidates:
            jobs = json.loads(row["charging_seconds_by_position"])
            actual = float(row["charging_completion_seconds"])
            expected = brute_force_makespan(jobs, int(row["charging_pad_availability"]))
            self.assertAlmostEqual(actual, expected, places=7)
            self.assertAlmostEqual(float(row["total_time_seconds"]), expected + 25, places=7)

    def test_decisions_minimize_the_full_candidate_table(self):
        candidates = rows("candidate_rankings.csv")
        for pads, wind, level in itertools.product(range(1, 6), ("head", "tail", "side"), (1, 2)):
            matching = [r for r in candidates if (int(r["charging_pad_availability"]), r["wind_direction"], int(r["wind_level"])) == (pads, wind, level)]
            expected = min(float(r["total_time_seconds"]) for r in matching)
            result = explain_medium_configuration(pads, wind, level)
            self.assertAlmostEqual(result["total_time_seconds"], expected, places=7)
            self.assertEqual(result["candidate_count"], len(matching))

    def test_pad_limits_and_monotonicity(self):
        for wind, level in itertools.product(("head", "tail", "side"), (1, 2)):
            decisions = [explain_medium_configuration(k, wind, level) for k in range(1, 6)]
            self.assertAlmostEqual(decisions[0]["charging_completion_seconds"], sum(decisions[0]["charging_seconds_by_position"]))
            self.assertAlmostEqual(decisions[-1]["charging_completion_seconds"], max(decisions[-1]["charging_seconds_by_position"]))
            totals = [r["total_time_seconds"] for r in decisions]
            self.assertTrue(all(a >= b - 1e-7 for a, b in zip(totals, totals[1:])))

    def test_missing_real_condition_is_explicit(self):
        result = explain_medium_configuration(2, "side", 1)
        self.assertEqual(result["missing_safe_configurations"], [dict(formation="column", inter_drone_spacing_cm=50)])
        self.assertEqual(result["candidate_count"], 9)

    def test_build_is_traceable_and_did_not_change_inputs(self):
        audit = json.loads((OUTPUT / "manifest.json").read_text())
        self.assertTrue(all(audit["verification"].values()))
        for relative, expected in audit["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected, relative)
        for filename, expected in audit["output_sha256"].items():
            self.assertEqual(hashlib.sha256((OUTPUT / filename).read_bytes()).hexdigest(), expected, filename)


if __name__ == "__main__":
    unittest.main()
