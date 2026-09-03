"""Regression tests for the adaptive neural-shortlist controller."""

from __future__ import annotations

import unittest

from ml_policy.adaptive_topn_policy import (
    DEFAULT_TOP_N_BY_K,
    adaptive_top_n,
    predict_adaptive_configuration,
)
from ml_policy.oracle_optimizer import OracleState, solve_oracle


class AdaptiveTopNPolicyTests(unittest.TestCase):
    def test_shortlist_sizes(self) -> None:
        self.assertEqual(
            {value: adaptive_top_n(value) for value in range(1, 6)},
            DEFAULT_TOP_N_BY_K,
        )
        with self.assertRaises(ValueError):
            adaptive_top_n(0)

    def test_reference_state_matches_full_oracle(self) -> None:
        state = OracleState(
            wind_direction="head",
            wind_level=1,
            charging_pad_count=3,
            current_soc=(82.0, 76.0, 91.0, 85.0, 73.0),
            remaining_distance_m=10.0,
            minimum_arrival_soc=30.0,
        )
        adaptive = predict_adaptive_configuration(state)
        oracle = solve_oracle(state)
        selected = adaptive["selected_configuration"]
        self.assertEqual(
            adaptive["policy"]["exactly_evaluated_configuration_count"], 36
        )
        self.assertAlmostEqual(
            selected["total_completion_minutes"],
            oracle.selected.total_completion_minutes,
            places=9,
        )
        self.assertEqual(selected["position"], oracle.selected.position_mapping(state.drone_ids))


if __name__ == "__main__":
    unittest.main()
