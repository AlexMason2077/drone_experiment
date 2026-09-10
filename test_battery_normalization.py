"""Offline tests only: no drone connection, no flight control or app imports."""
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from battery_normalization import BatteryNormalizer, DischargeCurve, finite, normalize_csv


def fixture():
    return dict(schema_version=1,kind='individual_battery_normalization_candidate',version='test_only',
        reference=dict(boundaries_soc=[95,80,50,20],rates_pp_min=[12,6,15]),
        batteries={'B10':dict(drone_id='drone_2',boundaries_soc=[95,70,40,20],rates_pp_min=[10,5,20])})


class CurveTests(unittest.TestCase):
    def setUp(self):
        self.curve=DischargeCurve((95,70,40,20),(10,5,20))

    def test_domain_and_bad_curve_validation(self):
        for bounds,rates in [((95,70,20),(10,5)),((95,40,70,20),(10,5,20)),
                             ((101,70,40,20),(10,5,20)),((95,70,40,-1),(10,5,20)),
                             ((95,70,40,20),(10,0,20)),((95,70,40,20),(10,math.inf,20))]:
            with self.subTest(bounds=bounds,rates=rates),self.assertRaises(ValueError):
                DischargeCurve(bounds,rates)

    def test_numeric_validation(self):
        for value in [None,True,'bad',math.nan,math.inf,-math.inf]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                finite(value,'test')
        self.assertEqual(finite('25','test'),25)

    def test_boundary_rate_is_lower_segment(self):
        for soc,expected in [(95,10),(70.01,10),(70,5),(40.01,5),(40,20),(20,20)]:
            self.assertEqual(self.curve.rate_at(soc),expected)

    def test_no_extrapolation(self):
        for soc in [100,19.99]:
            with self.assertRaises(ValueError):self.curve.rate_at(soc)
            with self.assertRaises(ValueError):self.curve.equivalent_seconds(95,soc)
            with self.assertRaises(ValueError):self.curve.advance(soc,0)

    def test_closed_form_integral_crosses_all_three_segments(self):
        # 90->70: 120s; 70->40: 360s; 40->30: 30s.
        self.assertAlmostEqual(self.curve.equivalent_seconds(90,30),510)
        self.assertAlmostEqual(self.curve.advance(90,510),30)

    def test_additivity_and_inverse(self):
        for start,middle,end in [(95,70,20),(90,60,25),(70,40,20),(55,50,45)]:
            whole=self.curve.equivalent_seconds(start,end)
            parts=self.curve.equivalent_seconds(start,middle)+self.curve.equivalent_seconds(middle,end)
            self.assertAlmostEqual(whole,parts)
            self.assertAlmostEqual(self.curve.advance(start,whole),end)

    def test_continuous_at_knots(self):
        for soc in [70,40]:
            t=self.curve.equivalent_seconds(95,soc)
            self.assertAlmostEqual(self.curve.advance(95,t),soc)
            self.assertLess(abs(self.curve.advance(95,t-1e-6)-soc),1e-5)
            self.assertLess(abs(self.curve.advance(95,t+1e-6)-soc),1e-5)

    def test_zero_drop_and_exact_floor(self):
        self.assertEqual(self.curve.equivalent_seconds(70,70),0)
        self.assertEqual(self.curve.advance(70,0),70)
        self.assertEqual(self.curve.advance(20,0),20)
        self.assertEqual(self.curve.advance(95,570),20)

    def test_rebound_and_exhausted_reference_raise(self):
        with self.assertRaises(ValueError):self.curve.equivalent_seconds(70,71)
        with self.assertRaises(ValueError):self.curve.advance(95,571)
        with self.assertRaises(ValueError):self.curve.advance(95,-1)


class NormalizerTests(unittest.TestCase):
    def setUp(self):self.normalizer=BatteryNormalizer(fixture())

    def test_cross_individual_and_reference_boundaries(self):
        # Own 75->65 takes 30+60=90 equivalent seconds. Reference 85->80
        # takes 25s, then 65s at 6pp/min -> 73.5%.
        r=self.normalizer.normalize_interval('B10','drone_2',75,65,30,85)
        self.assertAlmostEqual(r['bn_equivalent_hover_seconds'],90)
        self.assertAlmostEqual(r['bn_relative_hover_factor'],3)
        self.assertAlmostEqual(r['bn_reference_soc_end'],73.5)
        self.assertAlmostEqual(r['bn_Bideal_average_rate_pp_min'],23)
        self.assertAlmostEqual(r['bn_reference_rate_at_start_pp_min'],36)
        self.assertIn('not_measured_energy',r['bn_kind'])

    def test_identity_mapping(self):
        model=fixture(); model['reference']={k:v for k,v in model['batteries']['B10'].items() if k!='drone_id'}
        n=BatteryNormalizer(model)
        r=n.normalize_interval('B10','drone_2',90,35,120,90)
        self.assertAlmostEqual(r['bn_reference_soc_end'],35)
        self.assertAlmostEqual(r['bn_Bideal_drop_pp'],55)

    def test_continuous_reference_chain(self):
        n=self.normalizer
        a=n.normalize_interval('B10','drone_2',90,75,25,95)
        b=n.normalize_interval('B10','drone_2',75,65,25,a['bn_reference_soc_end'])
        full=n.normalize_interval('B10','drone_2',90,65,50,95)
        self.assertAlmostEqual(b['bn_reference_soc_end'],full['bn_reference_soc_end'])

    def test_pair_and_duration_rejection(self):
        for battery,drone in [('B99','drone_2'),('B10','drone_1')]:
            with self.assertRaises(ValueError):self.normalizer.relative_drain(battery,drone,70,65,25)
        for duration in [0,-1,None,True,math.inf]:
            with self.assertRaises(ValueError):self.normalizer.relative_drain('B10','drone_2',70,65,duration)

    def test_reference_start_must_be_explicit_and_supported(self):
        with self.assertRaises(TypeError):self.normalizer.normalize_interval('B10','drone_2',70,65,25)
        with self.assertRaises(ValueError):self.normalizer.normalize_interval('B10','drone_2',70,65,25,21)

    def test_model_schema_and_domain_mismatch(self):
        model=fixture();model['schema_version']=2
        with self.assertRaises(ValueError):BatteryNormalizer(model)
        model=fixture();model['batteries']['B10']['boundaries_soc'][0]=94
        with self.assertRaises(ValueError):BatteryNormalizer(model)


class CsvTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.model=self.folder/'model.json'
        self.model.write_text(json.dumps(fixture()),encoding='utf-8')
        self.input=self.folder/'input.csv';self.output=self.folder/'output.csv'
        self.header='battery_id,drone_id,soc_start,soc_end,duration_s,note\n'
        self.row='B10,drone_2,75,65,30,original\n'
        self.input.write_text(self.header+self.row,encoding='utf-8')

    def run_csv(self):return normalize_csv(self.model,self.input,self.output,85)

    def test_preserves_raw_fields_and_input_bytes(self):
        before=self.input.read_bytes();result=self.run_csv()
        self.assertEqual(self.input.read_bytes(),before)
        self.assertEqual(result['rows'],1)
        with self.output.open(newline='') as stream:row=next(csv.DictReader(stream))
        self.assertEqual(row['note'],'original');self.assertEqual(row['soc_start'],'75')
        self.assertEqual(row['bn_model_sha256'],hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.assertEqual(float(row['bn_reference_soc_end']),73.5)

    def test_refuses_existing_output_and_same_input(self):
        self.run_csv();before=self.output.read_bytes()
        with self.assertRaises(FileExistsError):self.run_csv()
        self.assertEqual(before,self.output.read_bytes())
        with self.assertRaises(FileExistsError):normalize_csv(self.model,self.input,self.input,85)

    def test_all_rows_validated_before_output(self):
        self.input.write_text(self.header+self.row+'B10,drone_2,75,76,30,bad rebound\n')
        with self.assertRaisesRegex(ValueError,'CSV row 3'):self.run_csv()
        self.assertFalse(self.output.exists())

    def test_empty_and_malformed_inputs(self):
        for body in ['',self.header,self.header+self.row.rstrip()+',extra\n',
                     'battery_id,battery_id\nB10,B10\n',self.header+'B10,drone_2,75,,30,bad\n']:
            with self.subTest(body=body):
                self.input.write_text(body)
                with self.assertRaises(ValueError):self.run_csv()
                self.assertFalse(self.output.exists())

    def test_refuses_double_normalization(self):
        self.run_csv()
        with self.assertRaisesRegex(ValueError,'double normalization'):
            normalize_csv(self.model,self.output,self.folder/'twice.csv',85)


class ReferenceConstructionTests(unittest.TestCase):
    def test_same_soc_average_uses_union_not_stage_index(self):
        from output_py.build_battery_normalization_candidate import mean_rate_curve
        curves=[DischargeCurve((95,80,50,20),(10,5,20)),
                DischargeCurve((95,70,40,20),(12,6,18))]
        edges,rates,times=mean_rate_curve(curves)
        self.assertEqual(edges.tolist(),[95,80,70,50,40,20])
        self.assertEqual(rates.tolist(),[11,8.5,5.5,13,19])
        self.assertAlmostEqual(times[-1],15*60/11+10*60/8.5+20*60/5.5+10*60/13+20*60/19)

    def test_exact_integral_has_known_answer(self):
        import numpy as np
        from output_py.build_battery_normalization_candidate import exact_line_rmse
        # Same endpoints with a triangular difference peak of 1 -> RMSE sqrt(1/3).
        rmse,maximum=exact_line_rmse(np.array([0.,1.,2.]),np.array([4.,3.,0.]),
                                     np.array([0.,2.]),np.array([4.,0.]))
        self.assertAlmostEqual(rmse,math.sqrt(1/3));self.assertEqual(maximum,1)

    def test_reference_of_identical_three_line_curves_is_exact(self):
        from output_py.build_battery_normalization_candidate import mean_rate_curve, fit_reference
        curve=DischargeCurve((95,80,50,20),(10,5,20))
        edges,_,times=mean_rate_curve([curve]*5)
        result,_,grid=fit_reference(edges,times)
        self.assertEqual(result.boundaries,curve.boundaries)
        self.assertAlmostEqual(float(grid.iloc[0].rmse_pp),0)


class CandidateArtifactTests(unittest.TestCase):
    def test_current_candidate_and_source_provenance(self):
        root=Path(__file__).resolve().parent
        folder=root/'analysis_results/battery_normalization_candidate_20260908'
        if not (folder/'model.json').exists():self.skipTest('Build candidate artifacts first')
        n=BatteryNormalizer.load(folder/'model.json')
        self.assertEqual(set(n.curves),{'B10','B11','B12','B13','B14'})
        self.assertEqual(n.reference.boundaries,(95,83,51,20))
        sources=json.loads((folder/'sources.json').read_text())
        self.assertEqual(len(sources['calibration']),5)
        self.assertNotIn('20260906_165812',{s['run_id'] for s in sources['calibration']})
        for source in sources['calibration']+sources['historical']:
            self.assertEqual(hashlib.sha256((root/source['source']).read_bytes()).hexdigest(),source['sha256'])
        model=json.loads((folder/'model.json').read_text())
        for battery,record in model['batteries'].items():
            duration=record['anchor_elapsed_times_s'][-1]-record['anchor_elapsed_times_s'][0]
            self.assertAlmostEqual(n.relative_drain(battery,record['drone_id'],95,20,duration),1)


if __name__=='__main__':unittest.main(verbosity=2)
