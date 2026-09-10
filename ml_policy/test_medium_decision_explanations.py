"""Read-only regression and evidence checks for paper-facing explanations."""
import csv
import inspect
import itertools
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from ml_policy import medium_decision_explanations as explanations
from ml_policy.medium_configuration_function import explain_medium_configuration


class MediumExplanationTests(unittest.TestCase):
    def test_exactly_three_inputs_without_soc_preprocessing(self):
        self.assertEqual(list(inspect.signature(explanations.select_medium_with_explanation).parameters),
                         ["charging_pad_availability", "wind_direction", "wind_strength"])

    def test_all_30_decisions_match_frozen_result(self):
        for k, wind, strength in itertools.product(range(1, 6), ("head", "side", "tail"), ("low", "high")):
            with self.subTest(k=k, wind=wind, strength=strength):
                result = explanations.select_medium_with_explanation(k, wind, strength)
                original = explain_medium_configuration(k, wind, 1 if strength == "low" else 2)
                self.assertEqual(result["configuration"], original["configuration"])
                self.assertAlmostEqual(result["total_time_minutes"], original["total_time_minutes"], places=9)
                self.assertEqual(result["input"]["wind_strength"], strength)
                self.assertEqual(result["baseline_assumptions"]["flight_duration_seconds"], 25)
                self.assertIn("hypothesis", result["aerodynamic_evidence_status"].lower())
                self.assertTrue(result["data_explanation_en"])
                self.assertTrue(result["data_explanation_zh"])
                self.assertEqual(result["reference_ids"], [r["id"] for r in result["references"]])
                for reference in result["references"]:
                    self.assertIn(reference["url"], result["explanation_en"])
                    self.assertTrue(reference["boundary"])

    def test_aliases_and_validation(self):
        self.assertEqual(explanations.select_medium_with_explanation(2, "HEAD WIND", " LOW "),
                         explanations.select_medium_with_explanation(2, "head", "low"))
        for args in ((2, "head", "medium"), (2, "head", 1), (2, "head", True),
                     (0, "head", "low"), (True, "head", "high"), (2, "unknown", "low")):
            with self.assertRaises(ValueError):
                explanations.select_medium_with_explanation(*args)

    def test_copy_isolation(self):
        original = explanations.select_medium_with_explanation(2, "head", "low")
        changed = explanations.select_medium_with_explanation(2, "head", "low")
        changed["configuration"]["formation"] = "bad"
        changed["references"][0]["url"] = "bad"
        changed["baseline_assumptions"].clear()
        self.assertEqual(original, explanations.select_medium_with_explanation(2, "head", "low"))

    def test_modified_frozen_artifact_is_rejected(self):
        with patch.object(Path, "read_bytes", return_value=b"changed frozen artifact"):
            with self.assertRaisesRegex(ValueError, "Frozen decisions changed"):
                explanations._load_checked.__wrapped__(
                    explanations.EXPLANATIONS_PATH, None, explanations.baseline.RESULT_PATH, None)

    def test_all_five_small_margins_have_warning(self):
        artifact = json.loads(explanations.EXPLANATIONS_PATH.read_text())
        small = [v for v in artifact["decisions"].values() if v["gap_to_runner_up_seconds"] < 10]
        self.assertEqual(len(small), 5)
        self.assertTrue(all(any("small-gap flag" in c for c in r["caveats"]) for r in small))

    def test_explanations_focus_on_mechanisms_not_repeated_statistical_notes(self):
        artifact = json.loads(explanations.EXPLANATIONS_PATH.read_text())
        for result in artifact["decisions"].values():
            text = result["explanation_en"].lower()
            self.assertNotIn("unverified", text)
            self.assertNotIn("not a confidence probability", text)
            self.assertTrue(any(term in text for term in ("wake", "inflow")))
            self.assertIn("charging", text)

    def test_thesis_keeps_method_checks_in_separate_companion(self):
        folder = explanations.EXPLANATIONS_PATH.parent
        paper = (folder / "thesis_explanations.md").read_text()
        self.assertEqual(paper.count("### "), 30)
        self.assertNotIn("Evidence note:", paper)
        self.assertNotIn("unverified", paper.lower())
        self.assertNotIn("not a confidence probability", paper.lower())
        notes = (folder / "method_notes.md").read_text()
        self.assertEqual(notes.count("## "), 30)

    def test_rig_source_is_not_presented_as_tello_measurements(self):
        artifact = json.loads(explanations.EXPLANATIONS_PATH.read_text())
        source = next(r for r in artifact["reviewed_literature"] if r["id"] == "R10")
        self.assertIn("not five complete Tellos", source["boundary"])
        paper = (explanations.EXPLANATIONS_PATH.parent / "thesis_explanations.md").read_text()
        self.assertIn("rotor-rig experiments", paper)

    def test_uncertainty_counts_match_deleted_trial_audit(self):
        artifact = json.loads(explanations.EXPLANATIONS_PATH.read_text())
        with (explanations.EXPLANATIONS_PATH.parent / "leave_one_trial_out.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for key, result in artifact["decisions"].items():
            eligible = [r for r in rows if r["key"] == key and r["evaluable"] == "True"]
            retained = sum(r["retained"] == "True" for r in eligible)
            self.assertEqual(result["sensitivity"]["loo_evaluable"], len(eligible))
            self.assertEqual(result["sensitivity"]["loo_retained"], retained)


if __name__ == "__main__":
    unittest.main()
