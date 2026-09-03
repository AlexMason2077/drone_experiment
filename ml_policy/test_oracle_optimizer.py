"""Deterministic validation checks for the offline Oracle."""

from __future__ import annotations

import itertools
import random
import unittest

from ml_policy.charging_model import (
    charging_time_constant_minutes,
    exponential_charging_minutes,
)
from ml_policy.oracle_optimizer import (
    EmpiricalRateTable,
    OracleState,
    SafetyTier,
    _optimal_parallel_charging_schedule,
    solve_oracle,
)


class ChargingScheduleTests(unittest.TestCase):
    def test_known_five_job_optima_for_every_k(self) -> None:
        jobs = (2.28, 2.81, 3.17, 4.02, 4.38)
        expected = {1: 16.66, 2: 8.40, 3: 6.30, 4: 5.09, 5: 4.38}
        for pad_count, expected_makespan in expected.items():
            with self.subTest(pad_count=pad_count):
                schedule = _optimal_parallel_charging_schedule(jobs, pad_count)
                self.assertAlmostEqual(
                    schedule.makespan_minutes,
                    expected_makespan,
                    places=9,
                )

    def test_exact_scheduler_matches_independent_brute_force(self) -> None:
        rng = random.Random(20260817)
        for case_index in range(50):
            jobs = tuple(rng.uniform(0.0, 30.0) for _ in range(5))
            for pad_count in range(1, 6):
                with self.subTest(case_index=case_index, pad_count=pad_count):
                    effective_pads = min(pad_count, len(jobs))
                    brute_force_best = min(
                        max(
                            sum(
                                jobs[job_index]
                                for job_index, assigned_pad in enumerate(assignment)
                                if assigned_pad == pad_index
                            )
                            for pad_index in range(effective_pads)
                        )
                        for assignment in itertools.product(
                            range(effective_pads),
                            repeat=len(jobs),
                        )
                    )
                    schedule = _optimal_parallel_charging_schedule(jobs, pad_count)
                    self.assertAlmostEqual(
                        schedule.makespan_minutes,
                        brute_force_best,
                        places=9,
                    )


class ExponentialChargingModelTests(unittest.TestCase):
    def test_zero_to_fully_charged_is_exactly_90_minutes(self) -> None:
        self.assertAlmostEqual(exponential_charging_minutes(0.0), 90.0, places=12)

    def test_ninety_three_to_fully_charged_is_about_38_minutes(self) -> None:
        self.assertAlmostEqual(
            exponential_charging_minutes(93.0),
            38.0294118006,
            places=6,
        )

    def test_ninety_nine_or_higher_requires_no_more_charging(self) -> None:
        self.assertEqual(exponential_charging_minutes(99.0), 0.0)
        self.assertEqual(exponential_charging_minutes(100.0), 0.0)

    def test_time_constant_is_calibrated_not_assumed_as_point_three(self) -> None:
        self.assertAlmostEqual(charging_time_constant_minutes(), 19.5432516856, places=9)

    def test_charging_time_decreases_monotonically_with_soc(self) -> None:
        times = [exponential_charging_minutes(soc) for soc in range(0, 100)]
        self.assertTrue(all(left >= right for left, right in zip(times, times[1:])))


class EmpiricalOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rate_table = EmpiricalRateTable.from_csv()

    def test_rate_table_applies_explicit_safety_mask(self) -> None:
        head_level_2 = {
            cell.structure.label
            for cell in self.rate_table.structures_for("head", 2)
        }
        side_level_2 = {
            cell.structure.label
            for cell in self.rate_table.structures_for("side", 2)
        }
        tail_level_2 = {
            cell.structure.label
            for cell in self.rate_table.structures_for("tail", 2)
        }
        self.assertNotIn("column_50", head_level_2)
        self.assertNotIn("column_50", side_level_2)
        self.assertNotIn("diamond_50", side_level_2)
        self.assertNotIn("diamond_50", tail_level_2)

    def test_head_level_1_selects_column_50_from_full_battery(self) -> None:
        for pad_count in (1, 2, 5):
            with self.subTest(pad_count=pad_count):
                result = solve_oracle(
                    OracleState(
                        wind_direction="head",
                        wind_level=1,
                        charging_pad_count=pad_count,
                        current_soc=(100, 100, 100, 100, 100),
                        remaining_distance_m=2.5,
                    ),
                    self.rate_table,
                )
                self.assertEqual(result.selected.structure.label, "column_50")

    def test_head_level_1_column_50_uses_exponential_k2_charging_time(self) -> None:
        result = solve_oracle(
            OracleState(
                wind_direction="head",
                wind_level=1,
                charging_pad_count=2,
                current_soc=(100, 100, 100, 100, 100),
                remaining_distance_m=2.5,
            ),
            self.rate_table,
        )
        self.assertEqual(result.selected.structure.label, "column_50")
        self.assertAlmostEqual(
            result.selected.charging_schedule.makespan_minutes,
            58.810955106059,
            places=5,
        )
        self.assertAlmostEqual(
            result.selected.remaining_flight_minutes,
            25.0 / 60.0,
            places=12,
        )

    def test_position_reacts_to_current_soc_when_five_pads_are_available(self) -> None:
        result = solve_oracle(
            OracleState(
                wind_direction="head",
                wind_level=1,
                charging_pad_count=5,
                current_soc=(100, 100, 100, 100, 90),
                remaining_distance_m=2.5,
            ),
            self.rate_table,
        )
        selected_rates = next(
            cell
            for cell in self.rate_table.structures_for("head", 1)
            if cell.structure == result.selected.structure
        )
        rate_by_slot = {
            slot.slot_id: slot.rate_pp_per_min
            for slot in selected_rates.slots
        }
        d5_slot = result.selected.position_mapping(result.state.drone_ids)["D5"]
        self.assertEqual(rate_by_slot[d5_slot], min(rate_by_slot.values()))

    def test_one_run_cell_is_backup_only(self) -> None:
        tail_level_2 = self.rate_table.structures_for("tail", 2)
        by_label = {cell.structure.label: cell for cell in tail_level_2}
        self.assertEqual(
            by_label["column_50"].safety_tier,
            SafetyTier.BACKUP_ONLY,
        )
        self.assertEqual(
            by_label["echelon_75"].safety_tier,
            SafetyTier.BACKUP_ONLY,
        )
        self.assertEqual(
            by_label["column_75"].safety_tier,
            SafetyTier.SAFE,
        )

    def test_tail_level_2_prefers_safe_candidate_over_faster_backup(self) -> None:
        result = solve_oracle(
            OracleState(
                wind_direction="tail",
                wind_level=2,
                charging_pad_count=2,
                current_soc=(100, 100, 100, 100, 100),
                remaining_distance_m=2.5,
            ),
            self.rate_table,
        )
        self.assertEqual(result.selected.structure.label, "column_75")
        self.assertEqual(result.selected.safety_tier, SafetyTier.SAFE)

    def test_side_level_2_two_run_echelon_50_remains_safe(self) -> None:
        result = solve_oracle(
            OracleState(
                wind_direction="side",
                wind_level=2,
                charging_pad_count=2,
                current_soc=(100, 100, 100, 100, 100),
                remaining_distance_m=2.5,
            ),
            self.rate_table,
        )
        self.assertEqual(result.selected.structure.label, "echelon_50")
        self.assertEqual(result.selected.safety_tier, SafetyTier.SAFE)


if __name__ == "__main__":
    unittest.main()
