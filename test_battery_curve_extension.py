"""Offline tests for explicit, provenance-labeled 100..95 linear extension."""
import copy
import csv
import hashlib
import json
import unittest

from battery_normalization import BatteryNormalizer, DischargeCurve
from output_py.extend_battery_curves_to_100 import SOURCE, DEFAULT_OUT, extend_model


class CurveExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=json.loads(SOURCE.read_text())
        cls.source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        cls.model=extend_model(cls.original,cls.source_hash)
        cls.old=BatteryNormalizer(cls.original)
        cls.new=BatteryNormalizer(cls.model)

    def pairs(self):
        return [('Bideal',self.old.reference,self.new.reference),
                *[(b,self.old.curves[b],self.new.curves[b]) for b in self.old.curves]]

    def test_three_segments_and_identical_slopes(self):
        for name,old,new in self.pairs():
            with self.subTest(curve=name):
                self.assertEqual(new.boundaries,(100,*old.boundaries[1:]))
                self.assertEqual(new.rates_pp_min,old.rates_pp_min)
                self.assertEqual(len(new.rates_pp_min),3)

    def test_original_input_model_is_not_mutated(self):
        before=copy.deepcopy(self.original)
        extend_model(self.original,self.source_hash)
        self.assertEqual(self.original,before)
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),self.source_hash)

    def test_continuity_and_entire_old_curve_preserved(self):
        for name,old,new in self.pairs():
            with self.subTest(curve=name):
                added=300/old.rates_pp_min[0]
                self.assertAlmostEqual(new.advance(100,added),95)
                self.assertAlmostEqual(new.advance(100,added/2),97.5)
                for end in range(95,19,-1):
                    self.assertAlmostEqual(new.equivalent_seconds(100,end),added+old.equivalent_seconds(95,end))
                    self.assertAlmostEqual(new.advance(100,added+old.equivalent_seconds(95,end)),end)

    def test_no_implicit_extension_of_old_model_or_lower_floor(self):
        with self.assertRaises(ValueError):self.old.curves['B10'].rate_at(100)
        for soc in [101,19]:
            with self.assertRaises(ValueError):self.new.curves['B10'].rate_at(soc)
        with self.assertRaises(ValueError):extend_model(self.model,self.source_hash)

    def test_calibration_range_does_not_expand(self):
        self.assertEqual(self.model['calibrated_soc_range'],[20,95])
        self.assertEqual(self.model['supported_soc_range'],[20,100])
        self.assertEqual(self.new.calibrated_soc_range,(20,95))
        malformed=copy.deepcopy(self.model);malformed['supported_soc_range']=[20,99]
        with self.assertRaises(ValueError):BatteryNormalizer(malformed)

    def test_battery_and_reference_extrapolation_are_distinguished(self):
        for start,end,ref,expected in [(100,98,70,(True,False)),(90,88,100,(False,True)),
                                       (96,94,100,(True,True)),(95,93,95,(False,False))]:
            with self.subTest(start=start,reference=ref):
                r=self.new.normalize_interval('B10','drone_2',start,end,10,ref)
                self.assertEqual(r['bn_battery_uses_extrapolation'],expected[0])
                self.assertEqual(r['bn_reference_uses_extrapolation'],expected[1])
                self.assertEqual(r['bn_uses_extrapolation'],any(expected))

    def test_original_domain_normalization_numbers_unchanged(self):
        for battery in self.old.curves:
            args=(battery,self.old.pairs[battery],80,77,25,70)
            before=self.old.normalize_interval(*args);after=self.new.normalize_interval(*args)
            for key in ['bn_equivalent_hover_seconds','bn_relative_hover_factor','bn_reference_soc_end','bn_Bideal_drop_pp']:
                self.assertAlmostEqual(before[key],after[key])
            self.assertFalse(after['bn_uses_extrapolation'])

    def test_written_samples_are_explicitly_not_observations(self):
        if not (DEFAULT_OUT/'model.json').exists():self.skipTest('Build extension artifacts first')
        self.assertEqual(json.loads((DEFAULT_OUT/'model.json').read_text()),self.model)
        with (DEFAULT_OUT/'model_extrapolated_100_to_95.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),36)
        self.assertTrue(all(row['is_observed']=='False' for row in rows))
        self.assertEqual(sum(row['is_extrapolated']=='True' for row in rows),30)
        self.assertTrue(all(row['source_model_sha256']==self.source_hash for row in rows))


if __name__=='__main__':unittest.main(verbosity=2)
