"""Non-flight checks for immutable-by-convention rate snapshots."""
import tempfile
import unittest
from pathlib import Path

from ml_policy.freeze_discharge_rates import freeze, read_rows, verify, write_csv


class FrozenRateTests(unittest.TestCase):
    def test_csv_round_trip_accepts_bom_and_plain_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.csv'
            expected = [{'experiment_directory': 'trial', 'run_id': '001'}]
            write_csv(path, expected)
            self.assertEqual(read_rows(path), expected)
            path.write_text('experiment_directory,run_id\ntrial,001\n', encoding='utf-8')
            self.assertEqual(read_rows(path), expected)

    def test_current_reference_freeze_and_tamper_detection(self):
        root = Path(__file__).resolve().parents[1]
        reference = root / 'analysis_outputs/ml_policy/real_medium_oracle_answers_safety_only_20260907'
        sources = root / 'analysis_outputs/forward_discharge_rate_modeling'
        if not (reference / 'manifest.json').exists():
            self.skipTest('Local processed reference is not available')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'test_snapshot'
            result = freeze(reference, sources, target)
            self.assertEqual(result['position_rates'], 275)
            self.assertEqual(result['source_runs'], 134)
            self.assertEqual(verify(target), result)
            self.assertEqual(
                (reference / 'rate_reference.json').read_bytes(),
                (target / 'rate_reference.json').read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                freeze(reference, sources, target)
            # Only damage the disposable test snapshot, never the frozen release.
            (target / 'rate_reference.json').write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
                verify(target)


if __name__ == '__main__':
    unittest.main()
