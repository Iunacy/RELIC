from dataclasses import asdict, replace
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodiment.benchmark import Action, COLORS, CONTROL_DT, Observation
from embodiment.dataset import DOF
from embodiment.experiment import aggregate, campaign, cases, run_trial
from embodiment.perturbations import KINDS, Perturbation, plan_robustness
from embodiment.tests.test_replay import episode


class Robustness(unittest.TestCase):
    def test_seeded_conditions_are_matched_and_disjoint_from_training(self):
        plan = plan_robustness(3)
        self.assertEqual(plan, plan_robustness(3))
        self.assertEqual(len({case['scenario_id'] for case in plan}), 15)
        self.assertTrue(all(case['seed'] >> 48 == 4 and case['split'] == 'robustness' for case in plan))
        for case in plan:
            perturbation = Perturbation(**case['perturbation'])
            self.assertEqual(set(perturbation.positions(case['seed'])), set(COLORS))
            self.assertEqual(perturbation.offsets(3), perturbation.offsets(3))
        self.assertNotEqual(Perturbation('observation_noise', 1, .001).offsets(1), Perturbation('observation_noise', 1, .001).offsets(2))
        self.assertEqual(Perturbation('observation_bias', 1, .001).offsets(1), Perturbation('observation_bias', 1, .001).offsets(2))
        self.assertEqual(len(cases('heldout', 20)), 20)
        with self.assertRaises(ValueError): cases('heldout', 2)
        for arguments in (('position_shift', 0, .004), ('observation_noise', 0, float('nan')), ('control_latency', 0, .5), ('unknown', 0, 0.)):
            with self.assertRaises(ValueError): Perturbation(*arguments)

    def test_incomplete_and_corrupt_campaign_results_are_never_complete(self):
        planned = plan_robustness(1, ['baseline'])
        self.assertFalse(aggregate([], planned)['completed'])
        trials = []
        for robot in DOF:
            trials.append({**planned[0], 'robot': robot, 'controller': 'classical_only',
                           'simulation_executed': True, 'success': True, 'duration': 20.,
                           'safety_events': [], 'fallback_triggers': [], 'failure_reason': None,
                           'pick_result': True, 'placement_result': True, 'mean_tracking_error_m': .001,
                           'peak_tracking_error_m': .003, 'placement_error_xy_m': .002})
        result = aggregate(trials, planned)
        self.assertTrue(result['completed'])
        self.assertEqual(result['physical_trials'], 2)
        self.assertLess(result['by_embodiment']['franka_panda']['baseline']['wilson_95'][0], 1.)
        for changed in ([trials[0], trials[0]], [{**trials[0], 'duration': float('nan')}],
                        [{**trials[0], 'target': 'blue'}], [{**trials[0], 'controller': 'vla_only'}]):
            with self.assertRaises(ValueError): aggregate(changed, planned)

    def test_missing_runtime_preserves_failure_and_restart_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            def missing(command, job, timeout):
                calls.append(command)
                (job/'process.log').write_text('unit fixture: missing runtime')
            result = campaign(Path(temporary), count=1, conditions=['baseline'], runner=missing)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertEqual(result['physical_trials'], 0)
            self.assertEqual(len(result['errors']), 2)
            campaign(Path(temporary), count=1, conditions=['baseline'], runner=lambda *args: self.fail('overwrote failed run'))
            self.assertEqual(len(calls), 2)

    def test_changed_report_metrics_cannot_override_recorded_outcome(self):
        planned = plan_robustness(1, ['baseline'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = {}
            def worker(command, job, timeout):
                request = json.loads((job/'request.json').read_text())
                robot = request['robot']
                result = {**planned[0], 'robot': robot, 'controller': 'classical_only',
                    'simulation_executed': True, 'success': True, 'duration': 20.,
                    'safety_events': [], 'fallback_triggers': [], 'failure_reason': None,
                    'pick_result': True, 'placement_result': True, 'mean_tracking_error_m': .001,
                    'peak_tracking_error_m': .003, 'placement_error_xy_m': .002}
                folder = job/'run/trial_0000'
                folder.mkdir(parents=True)
                recorded = episode()
                recorded.metadata.update(embodiment=robot, source_sha256=request['source_sha256'], scenario_id=planned[0]['scenario_id'])
                recorded.outcome = dict(result)
                episodes[folder] = recorded
                result.update(mean_tracking_error_m=0., evidence_sha256={'fixture': 'test'})
                (job/'run/report.json').write_text(json.dumps({'source_sha256': request['source_sha256'],
                    'source_changed_during_run': [], 'error': None, 'trials': [result]}))
            with patch('embodiment.replay.load_episode', side_effect=lambda folder: episodes[folder]), patch('embodiment.experiment.hashes', return_value={'fixture': 'test'}):
                result = campaign(root, count=1, conditions=['baseline'], runner=worker)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertEqual(result['physical_trials'], 0)
            self.assertEqual(len(result['errors']), 2)
            self.assertTrue((root/'summary.json').is_file())

    def test_safety_rejections_and_partial_ticks_are_recorded_as_physical_failures(self):
        record = episode()
        class Backend:
            embodiment = 'openmanipulator_x'
            arm_names, finger_names = record.metadata['joint_names'][:4], record.metadata['joint_names'][4:]
            camera = None
            def __init__(self, elapsed, unsafe): self.elapsed, self.unsafe = elapsed, unsafe
            def reset(self, seed, cube_positions):
                row, physical = record.rows[0], record.telemetry[0]
                self.initial = Observation(0., tuple(row['ee_position']), tuple(row['joint_position']),
                    tuple(row['joint_velocity']), physical['cube_poses'], {c: 0. for c in COLORS},
                    contacts=physical['contacts'], gripper_open=1., contact_forces=physical['contact_forces'])
                return self.initial
            def frame(self): return None
            def step(self, action):
                return replace(self.initial, time=self.elapsed, unsafe=self.unsafe,
                    safety_events=('unit_guard_rejection',) if self.unsafe else ())
            def safe_stop(self): pass
        case = plan_robustness(1, ['baseline'])[0]
        from embodiment.replay import load_episode
        for elapsed in (0., CONTROL_DT/2):
            with self.subTest(elapsed=elapsed), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/'episode'
                result = run_trial(Backend(elapsed, True), case, path, {'source_sha256': {'fixture': 'not physical evidence'}})
                self.assertFalse(result['success'])
                self.assertEqual(result['failure_reason'], 'unsafe_movement')
                self.assertEqual(result['safety_events'], ['unit_guard_rejection'])
                self.assertEqual(result['recovery_behavior'], 'safe_stop_no_retry')
                self.assertEqual(aggregate([result], [case])['physical_trials'], 1)
                replay = load_episode(path)
                self.assertEqual(replay.duration, elapsed)
                self.assertEqual(replay.state(elapsed)['status'], 'failure')
                self.assertEqual(replay.state(elapsed)['frame_index'], len(replay.rows)-1)
                if elapsed == 0.:
                    self.assertIsNone(result['mean_tracking_error_m'])
                    rejected = json.loads((path/'execution.jsonl').read_text())
                    self.assertIsNone(rejected['applied_action'])
                    self.assertEqual(replay.rows[-1]['action'][:3], [0., 0., 0.])
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(RuntimeError, 'stalled'):
            run_trial(Backend(0., False), case, Path(temporary)/'episode', {'source_sha256': {'fixture': 'not physical evidence'}})

    def test_delay_records_applied_actions_and_true_observations(self):
        record = episode()
        class Backend:
            embodiment = 'openmanipulator_x'
            arm_names, finger_names = record.metadata['joint_names'][:4], record.metadata['joint_names'][4:]
            camera = None
            def observation(self):
                row, physical = record.rows[self.index], record.telemetry[self.index]
                return Observation(row['timestamp'], tuple(row['ee_position']), tuple(row['joint_position']),
                    tuple(row['joint_velocity']), physical['cube_poses'], {c: 0. for c in COLORS},
                    contacts=physical['contacts'], gripper_open=1., contact_forces=physical['contact_forces'])
            def reset(self, seed, cube_positions): self.index = 0;return self.observation()
            def frame(self): return None
            def step(self, action): self.index += 1;return self.observation()
            def safe_stop(self): pass
        class ExpertFixture:
            failure = None
            lifted = released = True
            phase = 'approach'
            def __init__(self, *args): self.tick = 0
            def update(self, observation):
                self.phase = 'done' if self.tick == 2 else 'approach';self.tick += 1
                return Action((.001, 0., 0.), 1.)
        case = plan_robustness(1, ['control_latency'])[0]
        with tempfile.TemporaryDirectory() as temporary, patch('embodiment.experiment.Expert', ExpertFixture):
            result = run_trial(Backend(), case, Path(temporary)/'episode', {'source_sha256': {'fixture': 'test'}})
            rows = [json.loads(line) for line in (Path(temporary)/'episode/samples.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['action'][:3], [0., 0., 0.])
            self.assertEqual(rows[1]['action'][:3], [.001, 0., 0.])
            self.assertEqual(result['controller'], 'classical_only')
            self.assertEqual(result['split'], 'robustness')
