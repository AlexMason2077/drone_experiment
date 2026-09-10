import unittest
from unittest.mock import patch

import numpy as np

from ml_policy.medium_gap_policy import candidates_for_state, shortlist


class FakeTensor:
    def __init__(self, x):
        self.x = x

    def numpy(self):
        return self.x


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.state = {'wind': 'head', 'level': 1, 'k': 2, 'soc': [70]*5, 'scales': [1, 2, 1, 1, 1]}
        self.reference = [{'wind': 'head', 'level': 1, 'formation': 'front', 'spacing': 75, 'rates': [1, 2, 3, 4, 5]}]

    def test_feature_layout_and_no_oracle_during_construction(self):
        with patch('ml_policy.medium_gap_policy.cost_values', side_effect=AssertionError('No cost allowed')):
            cs, x = candidates_for_state(self.state, self.reference)
        self.assertEqual(x.shape, (120, 26))
        np.testing.assert_allclose(x[0], [1, 0, 0, 1, .4, .7, .7, .7, .7, .7,
                                        0, 0, 0, 1, 0, 1, 1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
        self.assertAlmostEqual(cs[0]['arrival_soc'][1], 70-1*25/60)

    def test_only_shortlisted_candidates_are_exactly_scored(self):
        policy = {'reference': self.reference, 'scaler': {'mean': [0]*26, 'scale': [1]*26},
                  'model': lambda x, training: FakeTensor(np.arange(len(x), dtype=float)[:, None])}
        with patch('ml_policy.medium_gap_policy.cost_values', return_value=([], [], {'total_required_time_min': 200})) as cost:
            result = shortlist(policy, self.state, 3)
        self.assertEqual(cost.call_count, 3)
        self.assertEqual(result['exact_evaluations'], 3)
        self.assertEqual(result['candidate_count'], 120)

    def test_domain_and_safety_constraints(self):
        low = dict(self.state, soc=[40]*5)
        cs, x = candidates_for_state(low, self.reference)
        self.assertEqual(len(cs), 0)
        self.assertEqual(x.shape, (0, 26))
        unsafe = [dict(self.reference[0], level=2, formation='column', spacing=50)]
        self.assertEqual(len(candidates_for_state(dict(self.state, level=2), unsafe)[0]), 0)
        with self.assertRaises(ValueError):
            candidates_for_state(dict(self.state, k=0), self.reference)


if __name__ == '__main__':
    unittest.main()
