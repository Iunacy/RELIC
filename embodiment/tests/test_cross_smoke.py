from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from embodiment.cross_smoke import workflow


class CrossSmoke(unittest.TestCase):
    def test_dry_run_records_missing_inputs_without_compute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('embodiment.cross_smoke.verify_source'), patch('embodiment.cross_smoke.smoke', side_effect=AssertionError('compute forbidden')):
                result = workflow(root, root/'a', root/'b', root/'base', root/'run')
            self.assertEqual(result['status'], 'BLOCKED')
            self.assertFalse(result['training_executed'])
            self.assertFalse(result['closed_loop_executed'])
            self.assertTrue((root/'run/result.json').is_file())

    def test_completed_preflight_is_idempotent_and_change_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('embodiment.cross_smoke.verify_source'), \
                 patch('embodiment.cross_smoke.validate_export', side_effect=lambda p: {'embodiment': 'openmanipulator_x' if p.name == 'a' else 'franka_panda', 'split': 'train'}), \
                 patch('embodiment.cross_smoke.validate_with_upstream', return_value={'reader': 'unit_fixture'}), \
                 patch('embodiment.cross_smoke.dataset_digest', return_value='fixture'), \
                 patch('embodiment.cross_smoke.validate_checkpoint', return_value={'weight_paths': []}), \
                 patch('embodiment.cross_smoke.file_digest', return_value='fixture'), \
                 patch('embodiment.cross_smoke.smoke', side_effect=AssertionError('compute forbidden')):
                for _ in range(2):
                    result = workflow(root, root/'a', root/'b', root/'base', root/'run')
                    self.assertEqual(result['status'], 'PREFLIGHT_ONLY')
                changed = workflow(root, root/'a', root/'b', root/'different', root/'run')
                self.assertEqual(changed['status'], 'BLOCKED')
                self.assertIn('changed', changed['errors'][0])
