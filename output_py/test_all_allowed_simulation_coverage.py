"""Offline scope tests; no flight SDK or generation invoked."""
import unittest
from complete_wind_tunnel_simulation_coverage import planned_coverage, unsafe_condition_ids


class AllAllowedCoverageTests(unittest.TestCase):
    def test_exact_legacy_safety_exclusions(self):
        self.assertEqual(unsafe_condition_ids(), {'column_50_head_lv2', 'column_50_side_lv2',
                                               'diamond_50_side_lv2', 'diamond_50_tail_lv2'})

    def test_every_other_condition_has_three_requested_runs(self):
        cov, _ = planned_coverage()
        self.assertEqual(len(cov), 60)
        self.assertEqual(int(cov.safety_excluded.sum()), 4)
        self.assertTrue(cov.loc[~cov.safety_excluded, 'requested_simulation_replicates'].eq(3).all())
        self.assertTrue(cov.loc[cov.safety_excluded, 'requested_simulation_replicates'].eq(0).all())
        self.assertEqual(int(cov.requested_simulation_replicates.sum()), 168)

    def test_data_availability_is_not_a_scope_filter(self):
        cov, _ = planned_coverage()
        existing = cov[cov.coverage.eq('locally_supported_all_15_cells')]
        self.assertEqual(len(existing), 7)
        self.assertTrue(existing.requested_simulation_replicates.eq(3).all())
        missing_not_unsafe = cov[cov.condition_id.eq('column_50_side_lv1')].iloc[0]
        self.assertFalse(missing_not_unsafe.safety_excluded)
        self.assertEqual(missing_not_unsafe.requested_simulation_replicates, 3)


if __name__ == '__main__':
    unittest.main()
