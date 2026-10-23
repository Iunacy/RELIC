import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import asdict
from embodiment.benchmark import Action, Observation, SCENE_REVISION, layout
from embodiment.dataset import export, load_episode, read_json, validate_collection, validate_export
from embodiment.recording import Recorder
from embodiment.scenarios import plan
from embodiment.preflight import LAB_COMMIT, SIM
from embodiment.assets import MANIFEST

HAS_IMAGES = importlib.util.find_spec('PIL') is not None
HAS_EXPORT = HAS_IMAGES and all(importlib.util.find_spec(name) is not None for name in ('numpy','pyarrow','av'))


def fixture(root: Path, case=None) -> Path:
    import numpy as np
    case = case or plan('train', 1)[0]
    instruction = f'Pick up the {case.target} cube and place it in the bin.'
    metadata = {**asdict(case), 'embodiment': 'openmanipulator_x', 'joint_names': [f'joint{i}' for i in range(6)],
                'control_dt': 1/30, 'simulation_executed': False, 'evidence_source': 'unit_fixture',
                'purpose': 'collection','scene_revision':SCENE_REVISION, 'physics_engine': 'none', 'controller': 'fixture'}
    recorder = Recorder(root, instruction, metadata)
    for i in range(3):
        observation = Observation(i/30, (.17, 0., .085), (0.,)*6, (0.,)*6,
                                  {color: (*position, 1, 0, 0, 0) for color, position in layout(case.seed).items()}, {})
        recorder.append(observation, Action((0.,0.,0.),1.), 'done' if i == 2 else 'approach', i == 2,
                        np.full((8, 8, 3), i*40, dtype=np.uint8))
    recorder.close({'success': True})
    return root


@unittest.skipUnless(HAS_IMAGES and importlib.util.find_spec('numpy') is not None, 'image validation dependencies unavailable')
class EpisodeValidation(unittest.TestCase):
    def test_fixture_is_rejected_as_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(Path(temp)/'episode')
            self.assertEqual(len(load_episode(path, require_physics=False).rows), 3)
            with self.assertRaisesRegex(ValueError, 'physical'):
                load_episode(path)

    def test_corrupt_image_and_time_alignment(self):
        for defect in ('image', 'time', 'language', 'boundary', 'dimension'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                path = fixture(Path(temp)/'episode')
                rows = [json.loads(line) for line in (path/'samples.jsonl').read_text().splitlines()]
                if defect == 'image':
                    (path/'camera/000001.png').write_bytes(b'not an image')
                elif defect == 'time':
                    rows[1]['timestamp'] = .04
                elif defect == 'language':
                    rows[0]['instruction'] = 'Pick up the blue cube and place it in the bin.'
                elif defect == 'boundary':
                    rows[0]['success'] = True
                else:
                    rows[0]['joint_position'].append(0.)
                    rows[0]['joint_velocity'].append(0.)
                (path/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with self.assertRaises((ValueError, OSError)):
                    load_episode(path, require_physics=False)

    def test_success_labels_do_not_replace_contact_evidence(self):
        from embodiment.assets import ARM,FINGERS
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(Path(temp)/'episode')
            metadata = read_json(path/'metadata.json')
            metadata.update(simulation_executed=True,physics_engine='physx',controller='classical_only',
                            evidence_source='isaac_lab_physx',joint_names=list(ARM+FINGERS),
                            runtime_pin={'isaacsim':SIM,'isaaclab_commit':LAB_COMMIT,'robot_commit':MANIFEST['commit']},
                            source_sha256={'fixture':'test-only'},relic_commit='unit-fixture')
            (path/'metadata.json').write_text(json.dumps(metadata))
            outcome = {'success':True,'robot':metadata['embodiment'],'seed':metadata['seed'],
                       'target':metadata['target'],'simulation_executed':True,'controller':'classical_only',
                       'pick_result':True,'placement_result':True}
            (path/'outcome.json').write_text(json.dumps(outcome))
            with self.assertRaisesRegex(ValueError,'invalid contact labels|contact/lift'):
                load_episode(path)

    def test_bilateral_forces_and_safety_events_are_required(self):
        from embodiment.assets import ARM, FINGERS
        from embodiment.benchmark import BIN, COLORS
        for defect in (None, 'one_finger', 'safety', 'nan'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                path = fixture(Path(temp)/'episode')
                metadata = read_json(path/'metadata.json')
                metadata.update(simulation_executed=True, physics_engine='physx', controller='classical_only',
                                evidence_source='isaac_lab_physx', joint_names=list(ARM+FINGERS),
                                runtime_pin={'isaacsim':SIM,'isaaclab_commit':LAB_COMMIT,'robot_commit':MANIFEST['commit']},
                                source_sha256={'fixture':'test-only'},relic_commit='unit-fixture')
                (path/'metadata.json').write_text(json.dumps(metadata))
                target = metadata['target']
                poses = {color: [*pose, 1, 0, 0, 0] for color, pose in layout(metadata['seed']).items()}
                telemetry = []
                for index in range(3):
                    current = {color: list(pose) for color, pose in poses.items()}
                    current[target] = [*BIN[:2], .085 if index < 2 else .0165, 1, 0, 0, 0]
                    forces = {color: [.1, .1] for color in COLORS}
                    if defect == 'one_finger':
                        forces[target] = [.1, 0.]
                    if defect == 'nan':
                        current[next(color for color in COLORS if color != target)][2] = float('nan')
                    telemetry.append({'timestamp': index/30, 'cube_poses': current,
                                      'contacts': {color: True for color in COLORS}, 'contact_forces': forces,
                                      'unsafe': False, 'safety_events': ['tracking_error'] if defect == 'safety' else []})
                (path/'telemetry.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in telemetry))
                outcome = {'success': True, 'robot': metadata['embodiment'], 'seed': metadata['seed'],
                           'target': target, 'simulation_executed': True, 'controller': 'classical_only',
                           'pick_result': True, 'placement_result': True,
                           'final_object_pose': telemetry[-1]['cube_poses'][target]}
                (path/'outcome.json').write_text(json.dumps(outcome))
                if defect is None:
                    self.assertTrue(load_episode(path).outcome['success'])
                else:
                    with self.assertRaises(ValueError):
                        load_episode(path)

    def test_duplicate_and_qualification_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(Path(temp)/'episode')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                validate_collection([path,path], require_physics=False)
            meta = read_json(path/'metadata.json')
            meta['purpose'] = 'qualification'
            (path/'metadata.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'qualification'):
                validate_collection([path], require_physics=False)


@unittest.skipUnless(HAS_EXPORT, 'Parquet and video dependencies unavailable')
class LeRobotSerialization(unittest.TestCase):
    def test_actual_parquet_and_video_roundtrip(self):
        import av
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(Path(temp)/'episode')
            output = Path(temp)/'lerobot'
            info = export([path], output, require_physics=False)
            report = validate_export(output,require_physics=False)
            self.assertEqual(report['frames'],3)
            with self.assertRaisesRegex(ValueError,'fixture'):
                validate_export(output)
            self.assertEqual(info['total_frames'], 3)
            self.assertFalse(info['relic']['physical_evidence_validated'])
            self.assertEqual(info['features']['observation.state']['shape'], [16])
            table = pq.read_table(output/'data/chunk-000/episode_000000.parquet')
            self.assertEqual(table['next.done'].to_pylist(), [False,False,True])
            self.assertEqual(table['action'].type.value_type.bit_width, 32)
            with av.open(str(output/'videos/chunk-000/observation.images.overhead/episode_000000.mp4')) as container:
                frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 3)
            self.assertEqual([float(frame.time) for frame in frames], [0.,1/30,2/30])
            with self.assertRaises(FileExistsError):
                export([path], output, require_physics=False)
            stats = read_json(output/'meta/stats.json')
            stats['action']['mean'][0] = 1.
            (output/'meta/stats.json').write_text(json.dumps(stats))
            with self.assertRaisesRegex(ValueError,'statistics'):
                validate_export(output,require_physics=False)

    def test_export_failure_does_not_publish_partial_dataset(self):
        from unittest.mock import patch
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = fixture(root/'episode')
            output = root/'lerobot'
            with patch('embodiment.dataset.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ffmpeg')):
                with self.assertRaises(subprocess.CalledProcessError):
                    export([path], output, require_physics=False)
            self.assertFalse(output.exists())
            self.assertEqual(sorted(item.name for item in root.iterdir()), ['episode'])
            export([path], output, require_physics=False)
            self.assertEqual(validate_export(output, require_physics=False)['episodes'], 1)


if __name__ == '__main__':
    unittest.main()
