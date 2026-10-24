import unittest
from embodiment.paired import bootstrap, category


def rows(values):
    return [dict(scenario_id=f'synthetic-fixture-{index}', success=value, controller='vla_only',
                 evidence_source='unit_fixture', fallback_triggers=[]) for index, value in enumerate(values)]


class Paired(unittest.TestCase):
    def test_known_outcomes_and_reordering(self):
        left, right = rows([False]*50), rows([True]*50)
        result = bootstrap(left, right)
        self.assertEqual(result['difference'], 1.)
        self.assertEqual(result['paired_bootstrap_95'], [1., 1.])
        self.assertEqual(result['wins'], 50)
        self.assertEqual(result, bootstrap(left[::-1], right))
        negative = bootstrap(right, left)
        self.assertEqual(negative['paired_bootstrap_95'], [-1., -1.])

    def test_null_and_discordance(self):
        result = bootstrap(rows([True, False]*25), rows([False, True]*25))
        self.assertEqual((result['wins'], result['losses'], result['ties']), (25, 25, 0))
        self.assertLess(result['paired_bootstrap_95'][0], 0)
        self.assertGreater(result['paired_bootstrap_95'][1], 0)
        self.assertEqual(bootstrap(rows([True]*3), rows([True]*3))['paired_bootstrap_95'], [0., 0.])

    def test_missing_duplicates_and_interventions(self):
        with self.assertRaises(ValueError): bootstrap(rows([True]), rows([]))
        with self.assertRaises(ValueError): bootstrap(rows([True])*2, rows([True]))
        row = rows([True])[0]
        self.assertEqual(category({**row, 'rejected_policy_actions': ['workspace']}), 'safety_rejection_or_stop')
        self.assertEqual(category({**row, 'controller': 'fallback_assisted', 'fallback_triggers': ['stall']}), 'fallback_assisted_completion')
        self.assertEqual(category({**row, 'success': False, 'infrastructure_error': 'process died'}), 'infrastructure_failure')
        with self.assertRaises(ValueError): category({**row, 'policy_errors': ['TimeoutError']})


class PhysicalEvidenceGate(unittest.TestCase):
    def test_report_without_raw_trial_hashes_cannot_be_published(self):
        from embodiment.tests.test_benchmark_report import fixture
        from embodiment.paired import verify_evidence
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'missing or corrupted'):
                verify_evidence(Path(temporary), fixture())
