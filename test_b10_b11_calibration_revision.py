"""Checks for the pooled calibration; no changes to any source or flight code."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from battery_normalization import BatteryNormalizer, DischargeCurve
from output_py.revise_b10_b11_calibration import review_sources, pooled_curve, model_error_on_trace
from output_py.build_battery_normalization_candidate import mean_rate_curve, fit_reference

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis_results/battery_normalization_pooled_v2_20260909'


class PooledCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=json.loads((OUT/'model.json').read_text())
        cls.old=json.loads((ROOT/'analysis_results/battery_normalization_candidate_20260908/model.json').read_text())
        cls.traces,cls.profiles,_=review_sources()

    def test_exact_primary_source_membership(self):
        expected={'B10':{'20260513_180556','20260906_173013'},'B11':{'20260513_143201','20260906_164725'}}
        for b,runs in expected.items():
            self.assertEqual({r['run_id'] for r in self.model['batteries'][b]['sources']},runs)
            self.assertEqual(self.model['batteries'][b]['run_weights'],[.5,.5])

    def test_mean_at_same_soc_not_average_boundary_numbers(self):
        for b,drone,knots in [('B10','drone_2',(95,85,56,20)),('B11','drone_1',(95,82,51,20))]:
            traces=[t for t in self.traces if t['battery_id']==b and t['drone']==drone]
            curve,_,_,levels,times=pooled_curve(traces)
            self.assertEqual(curve.boundaries,knots)
            for s,time in zip(levels,times):
                self.assertAlmostEqual(time,sum(t['crossing'][int(s)]-t['crossing'][95] for t in traces)/2)
            self.assertAlmostEqual(curve.equivalent_seconds(95,20),times[-1])
            self.assertEqual(list(curve.rates_pp_min),self.model['batteries'][b]['rates_pp_min'])

    def test_other_batteries_unchanged_and_reference_rebuilt(self):
        for b in ['B12','B13','B14']:self.assertEqual(self.model['batteries'][b],self.old['batteries'][b])
        curves=[DischargeCurve(tuple(m['boundaries_soc']),tuple(m['rates_pp_min'])) for m in self.model['batteries'].values()]
        edges,_,times=mean_rate_curve(curves)
        curve,_,_=fit_reference(edges,times)
        self.assertEqual(curve.boundaries,(95,82,52,20))
        self.assertEqual(list(curve.rates_pp_min),self.model['reference']['rates_pp_min'])

    def test_model_load_and_cross_band_inverse(self):
        n=BatteryNormalizer.load(OUT/'model.json')
        for b,curve in n.curves.items():
            for start,end in [(95,20),(90,49),(82,51),(70,40)]:
                self.assertAlmostEqual(curve.advance(start,curve.equivalent_seconds(start,end)),end)

    def test_out_of_range_predictions_not_clamped_in_scoring(self):
        trace=next(t for t in self.traces if t['run_id']=='20260906_164725')
        item=self.model['batteries']['B11']
        curve=DischargeCurve(tuple(item['boundaries_soc']),tuple(item['rates_pp_min']))
        r=model_error_on_trace(curve,trace)
        self.assertLess(r['coverage_fraction'],1)
        self.assertGreater(r['in_domain_soc_rmse_pp'],6)
        self.assertLess(r['evaluated_samples'],r['observed_samples'])

    def test_confirmed_conditions_and_no_holdout_claim(self):
        selected=[p for p in self.profiles if p['use']=='primary_pooled_calibration']
        self.assertEqual(len(selected),4)
        self.assertTrue(all(p['conditions']=='user_confirmed_no_wind_same_set_height' for p in selected))
        self.assertTrue(self.model['revision']['not_held_out_validation'])
        self.assertFalse(self.model['revision']['upper_80_constraint_applied'])


if __name__=='__main__':unittest.main(verbosity=2)
