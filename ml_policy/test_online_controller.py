"""Deterministic checks for the fixed-interval controller contract."""

from __future__ import annotations

import unittest

from algorithm import (
    Configuration,
    ExponentialChargingModel,
    OnlineConfigurationController,
    WindCondition,
)


class _ZeroEnergyModel:
    def predict_battery_drop(
        self,
        condition,
        configuration,
        drone_ids,
        distance_m,
    ):
        return (0.0,) * len(drone_ids)


class _FixedCrossingTimeModel:
    def predict_crossing_seconds(self, condition, configuration, distance_m):
        return distance_m * (2.0 if configuration.formation == "column" else 1.0)


class OnlineIntervalTests(unittest.TestCase):
    def test_generator_defaults_match_25_second_interval(self) -> None:
        from ml_policy.generate_training_states import _build_parser

        args = _build_parser().parse_args([])
        self.assertEqual(args.history_step_min_m, 2.5)
        self.assertEqual(args.history_step_max_m, 2.5)
        self.assertIn('interval25s', str(args.output_csv))

    def _controller(self) -> OnlineConfigurationController:
        return OnlineConfigurationController(
            charging_pad_availability=2,
            energy_model=_ZeroEnergyModel(),
            crossing_time_model=_FixedCrossingTimeModel(),
            charging_model=ExponentialChargingModel(),
            candidate_configurations=(
                Configuration("column", ("D1", "D2", "D3", "D4", "D5"), 50),
            ),
            evaluation_distance_m=10.0,
        )

    def test_every_epoch_refreshes_k_and_observations(self) -> None:
        controller = self._controller()
        initial = controller.start(WindCondition("head", 1), timestamp_seconds=0.0)
        self.assertEqual(initial.charging_pad_availability, 2)
        self.assertEqual(initial.projected_flight_seconds, 20.0)
        self.assertEqual(controller.decision_interval_seconds, 25.0)
        self.assertEqual(controller.next_decision_timestamp, 25.0)
        self.assertFalse(controller.decision_due(24.9))
        self.assertTrue(controller.decision_due(25.0))

        updated = controller.on_decision_interval(
            WindCondition("side", 2),
            measured_battery=(98, 97, 98, 99, 98),
            timestamp_seconds=25.0,
            charging_pad_availability=4,
            remaining_distance_m=7.0,
        )
        self.assertEqual(updated.charging_pad_availability, 4)
        self.assertEqual(updated.observed_condition, WindCondition("side", 2))
        self.assertEqual(updated.evaluation_distance_m, 7.0)
        self.assertEqual(controller.next_decision_timestamp, 50.0)
        self.assertEqual(controller.current_battery, (98, 97, 98, 99, 98))

        controller.on_decision_interval(
            WindCondition("tail", 1),
            measured_battery=(96, 95, 96, 97, 96),
            timestamp_seconds=50.0,
            charging_pad_availability=1,
            remaining_distance_m=4.0,
        )
        self.assertEqual(controller.next_decision_timestamp, 75.0)

    def test_update_before_next_epoch_is_rejected(self) -> None:
        controller = self._controller()
        controller.start(WindCondition("head", 1), timestamp_seconds=0.0)
        with self.assertRaisesRegex(ValueError, "Next decision is due"):
            controller.on_decision_interval(
                WindCondition("head", 2),
                measured_battery=(99, 99, 99, 99, 99),
                timestamp_seconds=24.9,
                charging_pad_availability=3,
                remaining_distance_m=8.0,
            )

    def test_formation_dependent_crossing_time_enters_score(self) -> None:
        candidates = (
            Configuration("column", ("D1", "D2", "D3", "D4", "D5"), 50),
            Configuration("vee", ("D1", "D2", "D3", "D4", "D5"), 50),
        )
        controller = OnlineConfigurationController(
            charging_pad_availability=5,
            energy_model=_ZeroEnergyModel(),
            crossing_time_model=_FixedCrossingTimeModel(),
            charging_model=ExponentialChargingModel(),
            candidate_configurations=candidates,
            evaluation_distance_m=10.0,
        )
        decision = controller.start(WindCondition("head", 1))
        self.assertEqual(decision.selected_configuration.formation, "vee")
        self.assertEqual(decision.projected_flight_seconds, 10.0)
        self.assertEqual(
            decision.online_score_seconds,
            decision.projected_flight_seconds + decision.projected_charging_seconds,
        )


if __name__ == "__main__":
    unittest.main()
