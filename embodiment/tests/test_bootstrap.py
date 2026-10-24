from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from embodiment.bootstrap import plan, setup
from embodiment.cloud_run import workflow


class IsolatedSetup(unittest.TestCase):
    def test_plans_are_isolated_and_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('embodiment.bootstrap.subprocess.run', side_effect=AssertionError('installation forbidden')):
                for role in ('training', 'simulation'):
                    result = setup(root, role)
                    self.assertEqual(result['status'], 'NOT_EXECUTED')
                    self.assertFalse(result['downloads_models'])
                    self.assertFalse(result['provisions_hardware'])
            training = plan(root, 'training')
            self.assertIn('--frozen', training['commands'][0])
            self.assertNotEqual(training['python'], plan(root, 'simulation')['python'])
            self.assertTrue((root/'setup/training.plan.json').is_file())

    def test_offline_setup_cannot_clone_missing_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch('embodiment.bootstrap.shutil.disk_usage', return_value=type('Usage', (), {'free': 100*1024**3})()), \
                 patch('embodiment.bootstrap.shutil.which', return_value='/uv'), \
                 patch('embodiment.bootstrap.subprocess.check_output', return_value='uv 0.13.4 (fixture)'), \
                 patch('embodiment.bootstrap.subprocess.run', side_effect=AssertionError('network forbidden')):
                with self.assertRaisesRegex(RuntimeError, 'pinned source checkout'):
                    setup(Path(temporary), 'training', True, True)

    def test_entrypoint_derives_paths_without_starting_compute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = workflow(root/'cache', root/'run', root/'model')
            self.assertEqual(result['status'], 'NOT_EXECUTED')
            self.assertEqual(result['commands']['qualification'][0], str(root/'cache/simulation/bin/python'))
            self.assertEqual(result['commands']['g3'][0], str(root/'cache/sources/GR00T/.venv/bin/python'))
