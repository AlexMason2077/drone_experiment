"""Offline checks for the Wind Tunnel GUI preview; no aircraft SDK is imported."""
import ast
from pathlib import Path
import unittest

from test_wind_tunnel_layouts import experiment, load_offline
from wind_tunnel_layouts import layout_description

ROOT = Path(__file__).resolve().parent


class LayoutPreviewTests(unittest.TestCase):
    def test_gui_targets_match_the_committed_collector_for_every_selection(self):
        collector = load_offline()
        for formation in ("front", "vee", "column", "diamond", "echalon"):
            for wind in ("head wind", "tail wind", "side wind"):
                for spacing in (50, 75):
                    with self.subTest(formation=formation, wind=wind, spacing=spacing):
                        layout = layout_description(formation, wind, spacing)
                        configs = collector.build_configs(experiment(formation, wind, spacing))
                        self.assertEqual(layout["pad_ids"], [c["mission_pad"] for c in configs])
                        self.assertEqual(layout["pad_axes"], [list(axis) for axis in configs[0]["mission_pad_axes_global"]])
                        self.assertEqual(layout["pad_x_aligned_with_body_forward"],
                                         configs[0]["pad_x_aligned_with_body_forward"])
                        self.assertEqual(layout["heading_tolerance_required"],
                                         configs[0]["mission_pad_heading_tolerance_deg"] is not None)
                        for point, config in zip(layout["positions"], configs):
                            self.assertAlmostEqual(point[0], config["target_x"])
                            self.assertAlmostEqual(point[1], config["target_y"])

    def test_side_wind_arrow_is_display_only(self):
        for formation in ("echalon", "vee", "column", "diamond"):
            with self.subTest(formation=formation):
                layout = layout_description(formation, "side wind", 50)
                self.assertEqual(layout["flow"]["text"], "+X -> -X")
                self.assertEqual(layout["flow"]["vector"], [-1, 0])
        self.assertEqual(layout_description("vee", "side wind", 75)["flow"]["text"], "+Y -> -Y")
        collector = (ROOT / "wind_tunnel_collector.py").read_text()
        self.assertNotIn("wind_tunnel_layouts", collector)
        self.assertIn("source at +X; airflow +X -> -X", collector)
        tree = ast.parse(collector)
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "build_configs"
                            for node in tree.body))

    def test_column_head_wind_preview_matches_the_floor_orientation(self):
        for spacing in (50, 75):
            with self.subTest(spacing=spacing):
                layout = layout_description("column", "head wind", spacing)
                self.assertEqual(layout["pad_ids"], [5, 6, 7, 8, 1])
                self.assertEqual(layout["positions"],
                                 [[0, (4 - i) * spacing] for i in range(5)])
                self.assertEqual(layout["nose"], "+Y")
                self.assertEqual(layout["flow"],
                                 {"source": "+Y", "vector": [0, -1],
                                  "text": "+Y -> -Y"})
                if spacing == 50:
                    self.assertIn("Pad 5 is at the +Y/front end", layout["layout_note"])
                else:
                    self.assertEqual(layout["layout_note"], "")

    def test_column_tail_and_side_50_share_headwind_layout_except_airflow(self):
        head = layout_description("column", "head wind", 50)
        for wind, source, vector in (("tail wind", "-Y", [0, 1]),
                                     ("side wind", "+X", [-1, 0])):
            with self.subTest(wind=wind):
                layout = layout_description("column", wind, 50)
                self.assertEqual(layout["positions"], [[0, 200], [0, 150], [0, 100], [0, 50], [0, 0]])
                self.assertEqual(layout["nose"], "+Y")
                self.assertEqual(layout["flow"]["source"], source)
                self.assertEqual(layout["flow"]["vector"], vector)
                self.assertEqual({k: v for k, v in head.items() if k != "flow"},
                                 {k: v for k, v in layout.items() if k != "flow"})


    def test_diamond_50_wind_variants_share_the_accepted_headwind_layout(self):
        head = layout_description("diamond", "head wind", 50)
        expected = [[50, 0], [0, 50], [50, 50], [100, 50], [50, 100]]
        for wind, source, vector, text in (
            ("head wind", "+Y", [0, -1], "+Y -> -Y"),
            ("tail wind", "-Y", [0, 1], "-Y -> +Y"),
            ("side wind", "+X", [-1, 0], "+X -> -X"),
        ):
            with self.subTest(wind=wind):
                layout = layout_description("diamond", wind, 50)
                self.assertEqual(layout["positions"], expected)
                self.assertEqual(layout["pad_ids"], [5, 6, 7, 8, 1])
                self.assertEqual(layout["nose"], "+Y")
                self.assertEqual(layout["flow"], {"source": source, "vector": vector, "text": text})
                self.assertEqual({k: v for k, v in head.items() if k != "flow"},
                                 {k: v for k, v in layout.items() if k != "flow"})


if __name__ == "__main__":
    unittest.main()
