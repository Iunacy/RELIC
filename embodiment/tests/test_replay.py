from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from embodiment.assets import ARM, FINGERS, MANIFEST
from embodiment.preflight import SIM, LAB_COMMIT
from embodiment.benchmark import BIN, COLORS, CONTROL_DT, SCENE_REVISION
from embodiment.dataset import DOF
from embodiment.panda import ARM as PANDA_ARM, FINGERS as PANDA_FINGERS
from embodiment.replay import Episode, Playback, load_episode, load_qualified_pair, validate_pair
from embodiment.dual_media import frame_schedule


def episode(robot='openmanipulator_x'):
    count = DOF[robot]+2
    names = ARM+FINGERS if count == 6 else PANDA_ARM+PANDA_FINGERS
    metadata = {'embodiment': robot, 'controller': 'classical_only', 'seed': 0, 'target': 'red',
                'purpose': 'qualification', 'split': 'qualification', 'scene_revision': SCENE_REVISION,
                'runtime_pin': {'isaacsim': SIM, 'isaaclab_commit': LAB_COMMIT, 'robot_commit': MANIFEST['commit']},
                'source_sha256': {'unit_fixture': 'not physical evidence'}, 'physics_engine': 'physx',
                'action_frame': 'world', 'action_alignment': 'observation_to_next_control_tick',
                'joint_names': list(names), 'simulation_executed': True, 'evidence_source': 'isaac_lab_physx'}
    rows, telemetry = [], []
    for i in range(3):
        row = {'timestamp': i*CONTROL_DT, 'frame': None, 'joint_position': [0.]*count,
               'joint_velocity': [0.]*count, 'ee_position': [.17, 0., .085], 'action': [.001, 0., 0., 1.],
               'instruction': 'Pick up the red cube and place it in the bin.', 'phase': 'done' if i == 2 else 'lift', 'success': i == 2}
        cubes = {'red': [*BIN[:2], .08 if i < 2 else .0165, 1., 0., 0., 0.],
                 'green': [.17, .05, .0125, 1., 0., 0., 0.], 'blue': [.19, -.02, .0125, 1., 0., 0., 0.]}
        telemetry.append({'timestamp': row['timestamp'], 'cube_poses': cubes, 'contacts': {c: i < 2 for c in COLORS},
                          'contact_forces': {c: [.1, .1] for c in COLORS}, 'unsafe': False, 'safety_events': []})
        rows.append(row)
    outcome = {'seed': 0, 'target': 'red', 'controller': 'classical_only', 'simulation_executed': True,
               'duration': rows[-1]['timestamp'], 'success': True, 'pick_result': True, 'placement_result': True,
               'failure_reason': None, 'final_object_pose': telemetry[-1]['cube_poses']['red']}
    return Episode(Path('/unit-fixture'), metadata, outcome, rows, telemetry, {'test': 'unit-fixture'})


class ReplayValidation(unittest.TestCase):
    def test_matched_embodiments_and_real_state_dimensions(self):
        pair = {robot: episode(robot) for robot in DOF}
        self.assertAlmostEqual(validate_pair(pair), 2/30)
        self.assertEqual(len(pair['openmanipulator_x'].state(0.)['joint_position']), 6)
        self.assertEqual(len(pair['franka_panda'].state(0.)['joint_position']), 9)
        pair['franka_panda'].metadata['target'] = 'blue'
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            validate_pair(pair)

    def test_tracking_error_uses_next_observation_and_terminal_is_unavailable(self):
        record = episode()
        record.rows[1]['ee_position'][0] += .0005
        self.assertAlmostEqual(record.state(0.)['tracking_error_m'], .0005)
        self.assertAlmostEqual(record.state(0.)['requested_step_m'], .001)
        self.assertIsNone(record.state(record.duration)['tracking_error_m'])
        self.assertIsNone(record.state(0.)['checkpoint_sha256'])
        self.assertFalse(record.state(0.)['learned_evidence'])

    def test_shared_clock_holds_shorter_trial_without_rescaling(self):
        first = episode()
        for t, expected in ((0., 0), (.02, 0), (1/30, 1), (10., 2)):
            self.assertEqual(first.index(t), expected)
        self.assertTrue(first.state(10.)['holding_final_state'])
        self.assertEqual(first.state(10.)['recorded_time'], 2/30)
        for value in (-1., float('nan'), True):
            with self.assertRaises(ValueError):
                first.index(value)

    def test_pause_resume_reset_and_completion_use_monotonic_clock(self):
        now = [10.]
        playback = Playback(5., lambda: now[0])
        playback.command('start');now[0] += 2.
        self.assertEqual(playback.command('pause')['elapsed'], 2.)
        now[0] += 100.
        self.assertEqual(playback.snapshot()['elapsed'], 2.)
        playback.command('resume');now[0] += 1.
        self.assertEqual(playback.snapshot()['elapsed'], 3.)
        self.assertEqual(playback.command('reset')['elapsed'], 0.)
        playback.command('seek', 4.);playback.command('resume');now[0] += 3.
        self.assertEqual(playback.snapshot()['elapsed'], 5.)
        self.assertFalse(playback.snapshot()['playing'])
        self.assertEqual(playback.command('start')['elapsed'], 0.)
        for value in (-1., 6., float('nan'), True):
            with self.assertRaises(ValueError):
                playback.command('seek', value)

    def test_corrupt_or_misaligned_trajectory_is_rejected(self):
        for defect in ('dimension', 'timestamp', 'controller', 'lift', 'hash', 'placement', 'outcome', 'safety'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                record = episode()
                path = Path(temporary)
                if defect == 'dimension': record.rows[0]['joint_position'].append(0.)
                if defect == 'timestamp': record.telemetry[1]['timestamp'] += .01
                if defect == 'controller': record.outcome['controller'] = 'vla_only'
                if defect == 'lift': record.telemetry[0]['contact_forces']['red'] = [0., 0.];record.telemetry[1]['contact_forces']['red'] = [0., 0.]
                if defect == 'placement': record.outcome['final_object_pose'] = [0.]*7
                if defect == 'outcome': record.outcome['duration'] = 9.
                if defect == 'safety': record.telemetry[0]['safety_events'] = ['unsafe']
                for name, value in (('metadata.json', record.metadata), ('outcome.json', record.outcome)):
                    (path/name).write_text(json.dumps(value))
                for name, value in (('samples.jsonl', record.rows), ('telemetry.jsonl', record.telemetry)):
                    (path/name).write_text(''.join(json.dumps(row)+'\n' for row in value))
                with self.assertRaises(ValueError):
                    load_episode(path, {'bad': 'hash'} if defect == 'hash' else None)

    def test_export_frame_schedule_is_explicit_and_time_preserving(self):
        times = frame_schedule(26.1, 15, 1.)
        self.assertEqual(times[0], 0.)
        self.assertEqual(times[-1], 26.1)
        self.assertAlmostEqual(times[1]-times[0], 1/15)
        self.assertAlmostEqual(frame_schedule(26.1, 15, 4.)[1], 4/15)
        for args in ((1., 0, 1.), (1., 15, 0.), (-1., 15, 1.), (1., 15, float('nan'))):
            with self.assertRaises(ValueError): frame_schedule(*args)
