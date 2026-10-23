import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import asdict
from embodiment.benchmark import Action, Observation, layout
from embodiment.dataset import export, load_episode, read_json, validate_collection, validate_export
from embodiment.recording import Recorder
from embodiment.scenarios import plan

HAS_IMAGES = importlib.util.find_spec('PIL') is not None
HAS_EXPORT = HAS_IMAGES and all(importlib.util.find_spec(name) is not None for name in ('numpy','pyarrow','av'))


def fixture(root: Path, case=None) -> Path:
    import numpy as np
    case = case or plan('train', 1)[0]
    instruction = f'Pick up the {case.target} cube and place it in the bin.'
    metadata = {**asdict(case), 'embodiment': 'openmanipulator_x', 'joint_names': [f'joint{i}' for i in range(6)],
                'control_dt': 1/30, 'simulation_executed': False, 'evidence_source': 'unit_fixture',
                'purpose': 'collection', 'physics_engine': 'none', 'controller': 'fixture'}
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


if __name__ == '__main__':
    unittest.main()
