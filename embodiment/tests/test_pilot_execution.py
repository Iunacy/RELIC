from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from embodiment.pilot import prepare, run


class ExecutablePilot(unittest.TestCase):
    def test_dry_run_has_separate_runtime_commands_and_no_compute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('embodiment.pilot.run_process', side_effect=AssertionError('compute forbidden')):
                result = run(root, python='/simulation/python', data_python='/data/python')
            self.assertEqual(result['status'], 'NOT_EXECUTED')
            self.assertFalse(result['training_ready'])
            command = result['commands']['franka_panda']['train']
            self.assertEqual(command[0], '/data/python')
            self.assertEqual(command[command.index('--python')+1], '/simulation/python')
            self.assertEqual(command[command.index('--target')+1], '12')
            self.assertEqual(len(result['commands']['franka_panda']), 3)

    def test_underpowered_host_is_blocked_before_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch('embodiment.pilot.subprocess.run', return_value=SimpleNamespace(stdout='{"ready": false, "blockers": ["VRAM"]}', stderr='')), \
                 patch('embodiment.pilot.run_process', side_effect=AssertionError('compute forbidden')):
                result = run(Path(temporary), execute=True, groot_source=Path('/source'))
            self.assertEqual(result['status'], 'BLOCKED')
            self.assertFalse(result['simulation_executed'])
