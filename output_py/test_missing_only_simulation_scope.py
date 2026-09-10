"""Current user scope: observed condition => no simulation, regardless of stages."""
import unittest
from generate_wind_tunnel_simulation import coverage, missing_allowed_conditions


class MissingOnlyScopeTests(unittest.TestCase):
    def test_only_missing_conditions_remain(self):
        cov,_=coverage();selected=missing_allowed_conditions(cov)
        self.assertEqual(len(selected),32)
        self.assertTrue(selected.usable_flights.eq(0).all())

    def test_partial_and_full_real_coverage_both_excluded(self):
        cov,_=coverage();selected=set(missing_allowed_conditions(cov).condition_id)
        self.assertFalse(selected & set(cov.loc[cov.usable_flights.gt(0),'condition_id']))
        self.assertNotIn('front_50_head_lv2',selected)
        self.assertNotIn('front_75_head_lv2',selected)
        self.assertNotIn('diamond_75_side_lv1',selected)

    def test_missing_is_distinct_from_safety_exclusion(self):
        cov,_=coverage();selected=set(missing_allowed_conditions(cov).condition_id)
        self.assertIn('column_50_side_lv1',selected)
        for cid in ('column_50_head_lv2','column_50_side_lv2','diamond_50_side_lv2','diamond_50_tail_lv2'):
            self.assertNotIn(cid,selected)


if __name__=='__main__':unittest.main()
