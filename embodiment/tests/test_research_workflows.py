from pathlib import Path
import tempfile
import unittest
from embodiment.pilot import prepare
from embodiment.qualification_pair import qualify


class ResearchWorkflows(unittest.TestCase):
    def test_pilot_is_deterministic_and_never_a_training_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = prepare(root)
            self.assertEqual(first, prepare(root))
            self.assertFalse(first['simulation_executed'])
            self.assertFalse(first['training_ready'])
            seeds = [row['seed'] for cases in first['scenarios'].values() for row in cases]
            self.assertEqual(len(seeds), len(set(seeds)))
            self.assertEqual(len(first['commands']), 2)
            with self.assertRaises(ValueError): prepare(root, 20)
            with self.assertRaises(ValueError): prepare(root, 100)

    def test_common_qualification_dryrun_cannot_reuse_legacy_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = qualify(Path(temporary))
            self.assertEqual(result['status'], 'NOT_EXECUTED')
            self.assertFalse(result['simulation_executed'])
            self.assertEqual(len(result['commands']), 2)
            self.assertEqual(result, qualify(Path(temporary)))
