"""Offline unit tests for SOC boundary selection and mean aggregation."""
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from output_py.recompute_medium_rates_v3 import medium_intervals, aggregate, CELL, RATE
from battery_normalization import DischargeCurve


class MediumRateTests(unittest.TestCase):
    def selected(self, battery, moving=None, upper=82, lower=52):
        soc=np.array(battery,dtype=float)
        times=np.arange(len(soc),dtype=float)
        active=np.ones(len(soc),dtype=bool)
        return medium_intervals(times,soc,active if moving is None else np.array(moving),active,upper,lower)[1]

    def test_own_middle_and_boundary_plateau(self):
        self.assertEqual(self.selected([83,82,82,81,52,52,51]).tolist(),
                         [False,True,True,True,False,False])

    def test_crossing_not_interpolated(self):
        self.assertEqual(self.selected([54,53,51]).tolist(),[True,False])

    def test_hover_time_is_not_reintroduced(self):
        self.assertEqual(self.selected([75,74,73],[False,True,True]).tolist(),[False,True])

    def test_outside_range_and_missing(self):
        self.assertFalse(self.selected([90,85,np.nan,40,39]).any())

    def test_upward_step_remains_identifiable(self):
        self.assertEqual(self.selected([70,71,70]).tolist(),[True,True])

    def test_own_middle_scale_equals_curve_integral(self):
        own=DischargeCurve((100,85,56,20),(12,7,11))
        bideal_rate=7.9648164842029185
        self.assertAlmostEqual(own.equivalent_seconds(73,69)*bideal_rate/60,
                               (73-69)*bideal_rate/7)

    def test_equal_trial_mean_missing_is_not_zero(self):
        cell=dict(zip(CELL,['head',1,'front',50]))
        rows=[dict(**cell,position=1,status='included',qc_flags='',**{RATE:2.0}),
              dict(**cell,position=1,status='included',qc_flags='',**{RATE:10.0}),
              dict(**cell,position=5,status='missing_battery_calibration',qc_flags='',**{RATE:np.nan})]
        rates,counts,coverage=aggregate(pd.DataFrame(rows),pd.DataFrame([cell]))
        self.assertEqual(rates.iloc[0][f'position_1_{RATE}'],6)
        self.assertEqual(counts.iloc[0].position_1_run_count,2)
        self.assertIsNone(rates.iloc[0][f'position_5_{RATE}'])
        self.assertEqual(counts.iloc[0].position_5_run_count,0)
        self.assertEqual(coverage[coverage.position.eq(5)].iloc[0].missing_calibration_count,1)

    def test_b15_does_not_change_reference_or_other_batteries(self):
        root=Path(__file__).resolve().parent
        old=json.loads((root/'analysis_results/battery_normalization_extended_v3_20260909/model.json').read_text())
        new=json.loads((root/'analysis_results/battery_normalization_v3_with_b15_20260909/model.json').read_text())
        self.assertEqual(old['reference'],new['reference'])
        self.assertEqual(old['fitting']['battery_weights'],new['fitting']['battery_weights'])
        self.assertNotIn('B15',new['fitting']['battery_weights'])
        for b in old['batteries']:
            self.assertEqual(old['batteries'][b],new['batteries'][b])
        self.assertEqual(new['batteries']['B15']['run_id'],'20260513_150444')
        self.assertFalse(new['batteries']['B15']['included_in_Bideal'])


if __name__=='__main__':
    unittest.main()
