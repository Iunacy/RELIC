from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from embodiment.collection import atomic_json, collect, execute, locked
from embodiment.scenarios import plan


class CollectionRecovery(unittest.TestCase):
    def fake_episode(self, path):
        metadata = json.loads((path/'metadata.json').read_text())
        return SimpleNamespace(metadata=metadata, outcome={'success': metadata['seed'] % 2 == 0}, rows=[{}, {}])

    def fake_runner(self, command, job, timeout):
        request = json.loads((job/'request.json').read_text())
        for case in request['cases']:
            path = job/'run'/f"trial_{case['seed']:02d}"
            path.mkdir(parents=True)
            atomic_json(path/'metadata.json', {**case, 'embodiment': request['robot'], 'purpose': 'collection'})
        atomic_json(job/'process.json', {'wall_seconds': 10., 'returncode': 0, 'timed_out': False})

    def mocks(self):
        stack = ExitStack()
        stack.enter_context(patch('embodiment.collection.load_episode', side_effect=self.fake_episode))
        stack.enter_context(patch('embodiment.collection.validate_collection'))
        stack.enter_context(patch('builtins.print'))
        return stack

    def test_resume_and_expand_without_duplicate_cases(self):
        with tempfile.TemporaryDirectory() as temp, self.mocks():
            root = Path(temp)
            first = collect(root, 'openmanipulator_x', 'train', 3, batch_size=2, runner=self.fake_runner)
            self.assertEqual(first['status'], 'COMPLETE')
            before = sorted((root/'jobs').glob('job-*'))
            with patch('embodiment.collection.execute', side_effect=AssertionError('unexpected execution')):
                resumed = collect(root, 'openmanipulator_x', 'train', 3, runner=lambda *args: self.fail('reran complete collection'))
            self.assertEqual(first, resumed)
            expanded = collect(root, 'openmanipulator_x', 'train', 5, batch_size=2, runner=self.fake_runner)
            self.assertEqual(expanded['successful_episodes'], 5)
            self.assertTrue(all(path.exists() for path in before))
            seeds = [case['seed'] for job in (root/'jobs').glob('job-*')
                     for case in json.loads((job/'request.json').read_text())['cases']]
            self.assertEqual(len(seeds), len(set(seeds)))
            self.assertGreater(expanded['successful_episodes_per_hour'], 0)

    def test_interrupted_job_recovers_completed_episode_and_preserves_partial(self):
        with tempfile.TemporaryDirectory() as temp, self.mocks():
            root = Path(temp)
            def interrupted(command, job, timeout):
                self.fake_runner(command, job, timeout)
                (job/'process.json').unlink()
                case = json.loads((job/'request.json').read_text())['cases'][-1]
                (job/'run'/f"trial_{case['seed']:02d}"/'metadata.json').unlink()
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                collect(root, 'franka_panda', 'validation', 2, batch_size=2, runner=interrupted)
            result = collect(root, 'franka_panda', 'validation', 2, batch_size=2, runner=self.fake_runner)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertGreaterEqual(result['failed_or_incomplete_attempts'], 1)
            self.assertFalse(result['timing_complete'])
            self.assertIsNone(result['successful_episodes_per_hour'])

    def test_corruption_and_configuration_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp, self.mocks():
            root = Path(temp)
            result = collect(root, 'openmanipulator_x', 'train', 1, runner=self.fake_runner)
            with self.assertRaisesRegex(ValueError, 'source, embodiment or split changed'):
                collect(root, 'franka_panda', 'train', 1, runner=self.fake_runner)
            (root/result['episodes'][0]/'metadata.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                collect(root, 'openmanipulator_x', 'train', 1, runner=self.fake_runner)

    def test_exclusive_lock_and_bounded_failures(self):
        with tempfile.TemporaryDirectory() as temp, self.mocks():
            root = Path(temp)
            with locked(root), self.assertRaisesRegex(RuntimeError, 'another collector'):
                collect(root, 'openmanipulator_x', 'train', 1)
            result = collect(root, 'openmanipulator_x', 'train', 1, runner=lambda *args: None)
            self.assertEqual(result['jobs'], 3)
            self.assertEqual(result['successful_episodes'], 0)
            self.assertIsNone(result['successful_episodes_per_hour'])

    def test_timeout_records_actual_process_failure(self):
        import sys
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            result = execute([sys.executable, '-c', 'import time; time.sleep(30)'], job, .1)
            self.assertTrue(result['timed_out'])
            self.assertNotEqual(result['returncode'], 0)

    def test_scenario_offset_preserves_prefix_and_rejects_invalid_ranges(self):
        self.assertEqual(plan('train', 3, 2), plan('train', 5)[2:])
        self.assertEqual([asdict(case) for case in plan('evaluation', 2, 7)],
                         [asdict(case) for case in plan('evaluation', 9)[7:]])
        for start in (-1, True, 100000):
            with self.assertRaises(ValueError):
                plan('train', 1, start)
