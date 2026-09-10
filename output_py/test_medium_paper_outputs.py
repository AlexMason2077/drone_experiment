"""Independent checks of paper comparisons and the exported decision-map cells."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_results/medium_configuration_function_v1_20260909"
OUT = ROOT / "analysis_results/medium_decision_interpretability_20260909"


class MediumPaperOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = json.loads((BASE / "decision_function.json").read_text())
        cls.scores = pd.read_csv(BASE / "candidate_rankings.csv")
        cls.profiles = pd.read_csv(BASE / "preprocessed_75pct_250cm.csv")
        cls.comparisons = pd.read_csv(OUT / "cross_formation_best_available_spacing.csv")
        cls.matched = pd.read_csv(OUT / "cross_formation_matched_spacing.csv")

    def test_four_distinct_other_formations_for_each_state(self):
        self.assertEqual(len(self.comparisons), 120)
        self.assertEqual(self.comparisons.key.nunique(), 30)
        for key, group in self.comparisons.groupby("key"):
            c = self.model["decisions"][key]["configuration"]
            self.assertEqual(set(group.comparator_formation), {"front", "column", "vee", "echalon", "diamond"} - {c["formation"]})
            self.assertTrue((group.selected_formation == c["formation"]).all())
            self.assertTrue((group.selected_spacing_cm == c["inter_drone_spacing_cm"]).all())

    def test_other_formation_uses_its_best_available_spacing(self):
        for r in self.comparisons.to_dict("records"):
            candidates = self.scores[(self.scores.wind_direction == r["wind_direction"]) &
                                     (self.scores.wind_level == r["wind_level"]) &
                                     (self.scores.charging_pad_availability == r["charging_pads"]) &
                                     (self.scores.formation == r["comparator_formation"])]
            self.assertAlmostEqual(candidates.total_time_seconds.min(), r["comparator_total_time_s"], places=8)

    def test_matched_spacing_is_complete_without_filling_missing_profiles(self):
        self.assertEqual(len(self.matched), 99)
        for key, d in self.model["decisions"].items():
            k, wind, level = key.split("|")
            c = d["configuration"]
            candidates = self.scores[(self.scores.wind_direction == wind) &
                                     (self.scores.wind_level == int(level)) &
                                     (self.scores.charging_pad_availability == int(k)) &
                                     (self.scores.formation != c["formation"]) &
                                     (self.scores.inter_drone_spacing_cm == c["inter_drone_spacing_cm"])]
            self.assertEqual(set(self.matched[self.matched.key == key].comparator_formation), set(candidates.formation))
        self.assertTrue((self.matched.selected_spacing_cm == self.matched.comparator_spacing_cm).all())

    def test_vectors_drops_and_critical_jobs_recompute(self):
        for r in pd.concat([self.comparisons, self.matched]).to_dict("records"):
            for prefix in ("selected", "comparator"):
                p = self.profiles[(self.profiles.wind_direction == r["wind_direction"]) &
                                  (self.profiles.wind_level == r["wind_level"]) &
                                  (self.profiles.formation == r[prefix + "_formation"]) &
                                  (self.profiles.inter_drone_spacing_cm == r[prefix + "_spacing_cm"])].iloc[0]
                rates = np.array([p[f"position_{i}_discharge_rate_Bideal_pp_min"] for i in range(1, 6)])
                np.testing.assert_allclose(rates, [r[f"{prefix}_p{i}_rate"] for i in range(1, 6)], rtol=1e-12)
                self.assertAlmostEqual(rates.sum() * 25 / 60, r[prefix + "_swarm_drop_pp"], places=10)
                arrival = 75 - rates * 25 / 60
                jobs = 5400 / np.log(100) * np.log(100 - arrival)
                groups = json.loads(r[prefix + "_pad_groups"])
                loads = [sum(jobs[int(d.split("_")[-1]) - 1] for d in group) for group in groups]
                self.assertEqual(sorted(d for group in groups for d in group), [f"drone_{i}" for i in range(1, 6)])
                self.assertAlmostEqual(max(loads), r[prefix + "_critical_pad_seconds"], places=8)
                self.assertAlmostEqual(max(loads) + 25, r[prefix + "_total_time_s"], places=8)
            self.assertAlmostEqual(r["gap_s"], r["comparator_total_time_s"] - r["selected_total_time_s"], places=8)

    def test_time_saving_not_mislabeled_as_discharge_saving(self):
        greater = self.comparisons.selected_swarm_drop_pp > self.comparisons.comparator_swarm_drop_pp
        self.assertEqual(int(greater.sum()), 8)
        for text in self.comparisons[greater].explanation_en:
            self.assertIn("not an aggregate-discharge saving", text)
        self.assertTrue((self.comparisons.gap_s >= 0).all())

    def test_all_thirty_figure_cells_match_frozen_lookup(self):
        cells = pd.read_csv(ROOT / "output/figures/medium_configuration_decision_map_data.csv")
        self.assertEqual(len(cells), 30)
        for r in cells.to_dict("records"):
            key = f"{r['charging_pads']}|{r['wind_direction']}|{1 if r['wind_strength']=='Low' else 2}"
            d = self.model["decisions"][key]
            self.assertEqual(r["formation_key"], d["configuration"]["formation"])
            self.assertEqual(r["spacing_cm"], d["configuration"]["inter_drone_spacing_cm"])
            self.assertAlmostEqual(r["total_time_minutes"], d["total_time_minutes"], places=8)

    def test_paper_exports_exist_and_have_no_raster_pdf_image(self):
        from xml.etree import ElementTree
        figure = ROOT / "output/figures/medium_configuration_decision_map.svg"
        svg = ElementTree.parse(figure)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        self.assertEqual(len(svg.findall(".//svg:image", ns)), 0)
        self.assertGreaterEqual(len(svg.findall(".//svg:text", ns)), 60)
        pdf = (ROOT / "output/pdf/medium_configuration_decision_map.pdf").read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertNotIn(b"/Subtype /Image", pdf)
        self.assertIn(b"/FontFile2", pdf)

    def test_report_table_cells_are_scalar(self):
        a = json.loads((OUT / "artifact.json").read_text())
        for dataset in a["snapshot"]["datasets"].values():
            for row in dataset:
                self.assertTrue(all(not isinstance(v, (dict, list)) for v in row.values()))


if __name__ == "__main__":
    unittest.main()
