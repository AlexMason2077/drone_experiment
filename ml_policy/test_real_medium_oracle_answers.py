"""Offline regression checks; no network, telemetry, or flight commands."""
import math
import unittest

from ml_policy.build_real_medium_oracle_answers import (
    FEATURES, answer_row, evaluate_run, make_rate_bank,
)


def fixture(name='a', formation='front', rates=None, flags=None):
    return dict(soc=[70, 69, 68, 67, 66], rates=rates or [5, 6, 7, 8, 9],
                scales=[1, 1.1, 0.9, 1.2, 0.95], flags=flags or [], unsafe=False,
                wind='head', level=1, formation=formation, spacing=75,
                source_group=name, exp=name, run='1', metadata_warning='',
                observed_positions=(0, 1, 2, 3, 4), rows=[{'battery_id': f'B{i}'} for i in range(5)])


class OracleAnswerTests(unittest.TestCase):
    def test_safety_only_keeps_qc_sources_but_not_unsafe_or_out_of_domain(self):
        a = fixture()
        flagged = fixture('flagged', rates=[0, 6, 7, 8, 9], flags=['flat_forward_SOC_curve'])
        unsafe = dict(fixture('unsafe', rates=[90]*5), unsafe=True)
        bank = make_rate_bank([a, flagged, unsafe], include_flagged=True)
        cell = bank[('head', 1, 'front', 75)]
        self.assertEqual(cell['rates'][0], 2.5)
        self.assertEqual(cell['source_run_count'], 2)
        self.assertEqual(cell['qc_flagged_source_count'], 1)
        self.assertEqual(cell['zero_fitted_rate_source_count'], 1)
        for k, candidates in evaluate_run(flagged, bank).items():
            r = answer_row(flagged, bank, k, candidates, include_flagged=True)
            self.assertTrue(r['training_eligible'])
            self.assertTrue(r['source_trial_time_available'])
            self.assertEqual(r['source_qc_flags'], ['flat_forward_SOC_curve'])
        r = answer_row(unsafe, bank, 2, evaluate_run(unsafe, bank)[2], include_flagged=True)
        self.assertFalse(r['training_eligible'])
        low = dict(flagged, soc=[40]*5)
        r = answer_row(low, bank, 2, evaluate_run(low, bank)[2], include_flagged=True)
        self.assertFalse(r['training_eligible'])

    def test_pooled_mean_excludes_flagged_and_unsafe(self):
        a, b = fixture(), fixture('b', rates=[7, 8, 9, 10, 11])
        bad = fixture('bad', rates=[100]*5, flags=['flat_forward_SOC_curve'])
        unsafe = dict(fixture('unsafe'), unsafe=True)
        bank = make_rate_bank([a, b, bad, unsafe])
        cell = bank[('head', 1, 'front', 75)]
        self.assertEqual(cell['rates'], [6, 7, 8, 9, 10])
        self.assertEqual(cell['source_groups'], ['a', 'b'])

    def test_current_and_best_use_same_pool_and_no_extra_training_rows(self):
        a, b = fixture(), fixture('b', rates=[7, 8, 9, 10, 11])
        c = fixture('c', formation='column', rates=[2, 3, 4, 5, 6])
        bank = make_rate_bank([a, b, c])
        evaluated = evaluate_run(a, bank)
        self.assertEqual(len(evaluated), 5)
        for k, candidates in evaluated.items():
            self.assertEqual(len(candidates), 240)
            r = answer_row(a, bank, k, candidates)
            self.assertTrue(r['training_eligible'])
            self.assertAlmostEqual(r['best_total_time_min'], min(c['total_time_min'] for c in candidates))
            self.assertAlmostEqual(r['worst_total_time_min'], max(c['total_time_min'] for c in candidates))
            self.assertAlmostEqual(r['time_gap_min'], r['current_total_time_min']-r['best_total_time_min'])
            self.assertAlmostEqual(math.expm1(r['target_log1p_time_gap']), r['time_gap_min'])
            self.assertNotEqual(r['source_trial_total_time_min'], r['current_total_time_min'])
            self.assertEqual(r['assigned_Bideal_rate_d1'], 6)
            self.assertEqual(r['source_Bideal_rate_d1'], 5)
            self.assertEqual(len(FEATURES), 26)
            self.assertTrue(all(isinstance(r[f], (int, float)) for f in FEATURES))
            self.assertFalse(r['covers_all_safe_structures'])

    def test_all_ties_retained(self):
        a = dict(fixture(rates=[5]*5), soc=[70]*5, scales=[1]*5)
        bank = make_rate_bank([a])
        for k, candidates in evaluate_run(a, bank).items():
            r = answer_row(a, bank, k, candidates)
            self.assertEqual(r['best_tie_count'], 120)
            self.assertEqual(r['worst_tie_count'], 120)
            self.assertAlmostEqual(r['time_gap_min'], 0)

    def test_out_of_domain_and_flagged_sources_not_promoted(self):
        a, bad = fixture(), fixture('bad', flags=['flat_forward_SOC_curve'])
        bank = make_rate_bank([a, bad])
        for k, candidates in evaluate_run(bad, bank).items():
            r = answer_row(bad, bank, k, candidates)
            self.assertTrue(r['oracle_answer_available'])
            self.assertFalse(r['training_eligible'])
            self.assertFalse(r['source_trial_time_available'])
        low = dict(a, soc=[40]*5)
        for k, candidates in evaluate_run(low, bank).items():
            r = answer_row(low, bank, k, candidates)
            self.assertFalse(r['oracle_answer_available'])
            self.assertFalse(r['training_eligible'])
            self.assertEqual(r['time_gap_min'], '')
        missing = fixture(formation='vee')
        r = answer_row(missing, bank, 2, evaluate_run(missing, bank)[2])
        self.assertTrue(r['oracle_answer_available'])
        self.assertFalse(r['training_eligible'])
        self.assertEqual(r['current_total_time_min'], '')


if __name__ == '__main__':
    unittest.main()
