from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from embodiment.cloud_workflow import commands, run


class CloudWorkflow(unittest.TestCase):
    def test_default_is_zero_compute_and_roles_use_separate_interpreters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('embodiment.cloud_workflow.execute', side_effect=AssertionError('compute forbidden')):
                result = run(root, '/sim/python', '/train/python', '/isaac', '/groot', '/weights')
            self.assertEqual(result['status'], 'NOT_EXECUTED')
            self.assertFalse(result['provisions_hardware'])
            planned = result['commands']
            self.assertEqual(planned['qualification'][0], '/sim/python')
            self.assertEqual(planned['pilot'][0], '/train/python')
            self.assertNotIn('--execute', planned['g3_preflight'])
            self.assertIn('--execute', planned['g3'])
            self.assertEqual(run(root, '/sim/python', '/train/python', '/isaac', '/groot', '/weights'), result)

    def test_training_stage_requires_physical_qualification_before_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch('embodiment.cloud_workflow.source_provenance', return_value={'relic_worktree_dirty': False, 'source_sha256': {'test':'fixture'}}), \
                 patch('embodiment.cloud_workflow.execute', side_effect=AssertionError('compute forbidden')):
                with self.assertRaises(FileNotFoundError):
                    run(Path(temporary), '/sim/python', '/train/python', '/isaac', '/groot', '/weights', 'g3', True)
