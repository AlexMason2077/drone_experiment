import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from simulation_viewer import (
    STAGES, SimulationDataError, contained_file, create_simulation_blueprint,
    current_catalog, read_run, signature,
)


class SimulationViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "simulation_data" / "active"
        (self.root / "runs").mkdir(parents=True)
        self.pointer = self.root.parent / "CURRENT.json"
        self.pointer.write_text(json.dumps({"path": "active"}))
        self.run_id = "SIM_front_50_side_lv1_r01_s1"
        self.relative = f"runs/{self.run_id}_all_coordination.csv.gz"
        self.raw = self.root / self.relative
        self.records = []
        for i in range(1, 6):
            for time, soc in ((0, 100), (2, 99), (100 + i, 20), (110, 20)):
                self.records.append(dict(experiment_id=self.run_id, is_simulated="True",
                    drone_name=f"drone_{i}", battery_id=f"B{i}", elapsed_time=time,
                    battery=soc, target_x=(i - 1) * 50, target_y=0, target_z=70,
                    mission_pad=i, x="", yaw="", templ=""))
        self.write_raw()
        manifest = [dict(experiment_id=self.run_id, is_simulated="True", file=self.relative,
            rows=len(self.records), duration_s=105, condition_id="front_50_side_lv1", formation="front",
            spacing_cm=50, wind_direction="side", wind_level=1, seed=1)]
        self.write_csv(self.root / "run_manifest.csv", manifest)
        rates = [dict(experiment_id=self.run_id, is_simulated="True", position=i, stage=s,
                 battery_id=f"B{i}", soc_start=99, soc_end=80, duration_s=50,
                 raw_rate_pp_min=20, bideal_rate_pp_min=21) for i in range(1, 6) for s in STAGES]
        self.write_csv(self.root / "simulated_stage_rates_long.csv", rates)
        (self.root.parent / "duplicate_export").mkdir()
        self.write_csv(self.root.parent / "duplicate_export" / "run_manifest.csv", manifest)
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.add_url_rule("/", "index", lambda: "Real experiments")
        self.app.register_blueprint(create_simulation_blueprint(self.base))
        self.client = self.app.test_client()
        self.detail = "/simulations/" + self.run_id

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def write_raw(self):
        with gzip.open(self.raw, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)

    def test_listing_uses_only_current_manifest_without_loading_telemetry(self):
        with patch("simulation_viewer.gzip.open", side_effect=AssertionError("not for listing")):
            response = self.client.get("/simulations")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"1 matching simulations", response.data)
        self.assertNotIn(b"duplicate_export", response.data)
        self.assertEqual(len(current_catalog(self.root.parent)[1]), 1)

    def test_filters_search_and_bad_page(self):
        for query, expected in (("?spacing_cm=75", b"0 matching simulations"),
                                ("?q=front_50_side&page=oops", b"1 matching simulations")):
            response = self.client.get("/simulations" + query)
            self.assertEqual(response.status_code, 200)
            self.assertIn(expected, response.data)

    def test_detail_preview_preserves_blanks_and_has_no_flight_buttons(self):
        response = self.client.get(self.detail)
        self.assertEqual(response.status_code, 200)
        for content in (b"Generate plots", b"Bideal", b"not modelled", b"SIMULATED"):
            self.assertIn(content, response.data)
        self.assertNotIn(b"Prepare experiment", response.data)
        self.assertNotIn(b"/start", response.data)
        self.assertEqual(self.client.post(self.detail + "/start").status_code, 404)

    def test_curves_are_steps_and_end_at_individual_threshold(self):
        data = read_run(self.raw, signature(self.raw), self.run_id)
        self.assertEqual(data["count"], 20)
        for i, drone in enumerate(data["drones"], 1):
            self.assertEqual(drone["times"], [0, 2, 100 + i])
            self.assertEqual(drone["soc"], [100, 99, 20])
            self.assertEqual(drone["end_s"], 100 + i)

    def test_plot_cache_and_download_do_not_change_source_files(self):
        original_hash = hashlib.sha256(self.raw.read_bytes()).hexdigest()
        self.assertEqual(self.client.get(self.detail + "/plots/battery.png").status_code, 404)
        images = {key: b"PNG fixture" for key in ("battery", "duration", "stage_rates")}
        with patch("simulation_viewer.render_plots", return_value=images) as renderer:
            self.assertEqual(self.client.post(self.detail + "/plots").status_code, 302)
            self.assertEqual(self.client.post(self.detail + "/plots").status_code, 302)
            self.assertEqual(renderer.call_count, 1)
        image = self.client.get(self.detail + "/plots/battery.png")
        self.assertEqual(image.status_code, 200)
        image.close()
        download = self.client.get(self.detail + "/download")
        self.assertEqual(download.data, self.raw.read_bytes())
        download.close()
        self.assertEqual(original_hash, hashlib.sha256(self.raw.read_bytes()).hexdigest())
        self.assertFalse((self.base / "database").exists())
        self.assertFalse((self.root / "plots").exists())

    def test_unknown_run_and_plot_types_are_not_served(self):
        for path in ("/simulations/SIM_REMOVED", self.detail + "/plots/temperature.png",
                     "/simulations/../../app.py", self.detail + "/plots/../../app.py"):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_paths_cannot_escape_export(self):
        for value in ("../secret", "/tmp/secret"):
            with self.assertRaises(SimulationDataError):
                contained_file(self.root, value)
        self.pointer.write_text('{"path":"../../secret"}')
        self.assertEqual(self.client.get("/simulations").status_code, 503)

    def test_missing_pointer_is_empty_but_corrupt_export_reports_error(self):
        self.pointer.unlink()
        response = self.client.get("/simulations")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No active export", response.data)
        self.pointer.write_text('{"path":"missing"}')
        self.assertEqual(self.client.get("/simulations").status_code, 503)

    def test_provenance_mismatch_is_rejected(self):
        self.records[0]["is_simulated"] = "False"
        self.write_raw()
        self.assertEqual(self.client.get(self.detail).status_code, 503)


if __name__ == "__main__":
    unittest.main()
