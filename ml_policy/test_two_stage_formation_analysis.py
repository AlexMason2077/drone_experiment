"""Deterministic checks for the meeting-compliant two-stage calculation."""

from __future__ import annotations

import unittest

import numpy as np

from two_stage_formation_analysis import (
    ENERGY_NUMERIC_FEATURES,
    TIME_NUMERIC_FEATURES,
    model_candidates,
    stage2_components,
)


class TwoStageFormationAnalysisTests(unittest.TestCase):
    def test_pad_availability_is_not_a_regression_feature(self) -> None:
        self.assertNotIn("charging_pad_availability", ENERGY_NUMERIC_FEATURES)
        self.assertNotIn("charging_pad_availability", TIME_NUMERIC_FEATURES)

    def test_polynomial_regression_is_a_candidate(self) -> None:
        names = model_candidates()
        self.assertIn("Polynomial Ridge degree=2 alpha=0.01", names)

    def test_stage2_separates_service_and_queue_time(self) -> None:
        service, queue, total = stage2_components(
            np.asarray([4.5]),
            np.asarray([30.0]),
            pads=2,
            charge_rate=4.5,
        )
        # Five equal jobs on two pads require three waves: one service wave and
        # two queue waves.
        self.assertAlmostEqual(float(service[0]), 60.0)
        self.assertAlmostEqual(float(queue[0]), 120.0)
        self.assertAlmostEqual(float(total[0]), 210.0)

    def test_five_pads_remove_queue_wait(self) -> None:
        service, queue, total = stage2_components(
            np.asarray([4.5]),
            np.asarray([30.0]),
            pads=5,
            charge_rate=4.5,
        )
        self.assertAlmostEqual(float(service[0]), 60.0)
        self.assertAlmostEqual(float(queue[0]), 0.0)
        self.assertAlmostEqual(float(total[0]), 90.0)


if __name__ == "__main__":
    unittest.main()
