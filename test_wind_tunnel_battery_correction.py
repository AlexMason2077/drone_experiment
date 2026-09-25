"""Offline tests for the side-wind wind-tunnel battery metadata correction."""

import unittest

import pandas as pd

from wind_tunnel_battery_correction import (
    correct_battery_dataframe,
    correct_battery_row,
    corrected_battery_id,
)


class WindTunnelBatteryCorrectionTests(unittest.TestCase):
    def test_only_side_wind_tunnel_drone_five_is_corrected(self):
        original = {
            "soc_mode": "wind_tunnel",
            "wind_direction": "side wind",
            "drone_name": "drone_5",
            "battery_id": "B06",
        }
        self.assertEqual(corrected_battery_id(original), "B12")
        corrected = correct_battery_row(original)
        self.assertEqual(corrected["battery_id"], "B12")
        self.assertEqual(original["battery_id"], "B06")

        for changed_field, value in (
            ("soc_mode", "medium"),
            ("wind_direction", "head wind"),
            ("drone_name", "drone_4"),
        ):
            row = {**original, changed_field: value}
            self.assertEqual(corrected_battery_id(row), "B06")

    def test_registry_protocol_and_dataframe_boundaries(self):
        self.assertEqual(
            corrected_battery_id({
                "protocol": "wind_tunnel", "wind_direction": "side wind",
                "drone_number": "5", "battery_id": "B06",
            }),
            "B12",
        )
        original = pd.DataFrame([
            {"soc_mode": "wind_tunnel", "wind_direction": "side wind", "drone_name": "drone_5", "battery_id": "B06"},
            {"soc_mode": "wind_tunnel", "wind_direction": "head wind", "drone_name": "drone_5", "battery_id": "B06"},
            {"soc_mode": "medium", "wind_direction": "side wind", "drone_name": "drone_5", "battery_id": "B06"},
            {"soc_mode": "wind_tunnel", "wind_direction": "side wind", "drone_name": "drone_4", "battery_id": "B06"},
        ])
        effective = correct_battery_dataframe(original)
        self.assertEqual(effective["battery_id"].tolist(), ["B12", "B06", "B06", "B06"])
        self.assertEqual(original["battery_id"].tolist(), ["B06"] * 4)


if __name__ == "__main__":
    unittest.main()
