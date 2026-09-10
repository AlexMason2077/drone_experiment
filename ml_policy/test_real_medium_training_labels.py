"""Independent checks for battery-adjusted supervision and exact scheduling."""

import itertools
import math
import random
import unittest

from ml_policy.build_real_medium_training_labels import (
    FEATURES,
    candidate_row,
    cost_values,
    exact_schedule,
    mission_values,
)


def brute_schedule(jobs, k):
    return min(
        max(sum(q for q, pad in zip(jobs, a) if pad == p) for p in range(k))
        for a in itertools.product(range(k), repeat=5)
    )


class LabelTests(unittest.TestCase):
    def test_partition_solver_matches_all_labelled_assignments(self):
        rng = random.Random(7401)
        cases = [[0] * 5, [20] * 5, [0, 2, 3, 70, 4]]
        cases += [[rng.uniform(0, 90) for _ in range(5)] for _ in range(8)]
        for jobs in cases:
            for k in range(1, 6):
                cost, assignment = exact_schedule(jobs, k)
                self.assertAlmostEqual(cost, brute_schedule(jobs, k), places=10)
                self.assertAlmostEqual(cost, max(sum(q for q, a in zip(jobs, assignment) if a == p) for p in range(k)), places=10)

    def test_position_changes_rate_but_not_battery_identity(self):
        soc = [70] * 5
        rates = [12, 9, 8, 7, 6]
        scales = [1.2, 1, 1, 1, 0.9]
        assigned, physical, arrival = mission_values(soc, rates, scales, (4, 1, 2, 3, 0))
        self.assertEqual(assigned, [6, 9, 8, 7, 12])
        self.assertAlmostEqual(physical[0], 5)
        self.assertAlmostEqual(physical[4], 12 / 0.9)
        self.assertAlmostEqual(arrival[0], 70 - 5 * 25 / 60)
        with self.assertRaises(ValueError):
            mission_values(soc, rates, [0, 1, 1, 1, 1], (0, 1, 2, 3, 4))
        with self.assertRaises(ValueError):
            mission_values(soc, rates, scales, (0, 0, 2, 3, 4))

    def test_residual_reconstruction_and_analytic_cases(self):
        arrival = [56, 57, 58, 54, 55]
        for k in range(1, 6):
            jobs, _, v = cost_values(arrival, k)
            # Independent expression of the anchored exponential charging rule.
            expected_jobs = [90 / math.log(100) * math.log((100 - s) / 1) for s in arrival]
            for q, expected in zip(jobs, expected_jobs):
                self.assertAlmostEqual(q, expected, places=10)
            self.assertAlmostEqual(v['total_required_time_min'], brute_schedule(expected_jobs, k) + 25 / 60, places=10)
            self.assertAlmostEqual(v['cost_lower_bound_min'] + math.expm1(v['target_log1p_residual']), v['total_required_time_min'], places=10)
            if k in (1, 5):
                self.assertAlmostEqual(v['target_log1p_residual'], 0, places=10)
        with self.assertRaises(ValueError):
            cost_values([39.9, 60, 60, 60, 60], 2)

    def test_invalid_sources_are_retained_without_targets(self):
        b = {
            'soc': [70] * 5, 'rates': [6] * 5, 'scales': [1] * 5,
            'flags': [], 'unsafe': False, 'formation': 'front', 'wind': 'tail',
            'spacing': 50, 'level': 2, 'exp': 'fixture', 'run': 'fixture',
            'source_group': 'fixture::fixture', 'observed_positions': (0, 1, 2, 3, 4),
            'metadata_warning': '', 'rows': [{'battery_id': f'B{i}'} for i in range(5)],
        }
        good = candidate_row(b, (0, 1, 2, 3, 4), 2)
        self.assertTrue(good['training_eligible'])
        self.assertEqual(len(FEATURES), 26)
        self.assertTrue(all(isinstance(good[f], (int, float)) for f in FEATURES))
        self.assertFalse(any('charging_time' in f or 'arrival' in f or 'target' in f for f in FEATURES))
        for patch in ({'flags': ['flat_forward_SOC_curve']}, {'soc': [40] * 5}, {'unsafe': True}):
            bad = candidate_row(dict(b, **patch), (0, 1, 2, 3, 4), 2)
            self.assertFalse(bad['training_eligible'])
            self.assertEqual(bad['total_required_time_min'], '')
            self.assertEqual(bad['target_log1p_residual'], '')


if __name__ == '__main__':
    unittest.main()
