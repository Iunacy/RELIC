from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import unittest

from embodiment.benchmark import Action, Observation, SCENE_REVISION, layout
from embodiment.dataset import DOF, export
from embodiment.recording import Recorder
from embodiment.scenarios import plan
from embodiment.upstream_validation import validate_with_upstream


@unittest.skipUnless(os.environ.get('RELIC_GROOT_SOURCE'), 'set RELIC_GROOT_SOURCE for pinned upstream integration')
class UpstreamCompatibility(unittest.TestCase):
    def test_both_embodiments_decode_chunk_and_normalize_with_real_upstream(self):
        import numpy as np
        source = Path(os.environ['RELIC_GROOT_SOURCE'])
        for robot, dof in DOF.items():
            with self.subTest(robot=robot), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                case = plan('train', 1)[0]
                joints = dof + 2
                metadata = {**asdict(case), 'embodiment': robot,
                            'joint_names': [f'fixture_joint_{i}' for i in range(joints)],
                            'control_dt': 1/30, 'simulation_executed': False,
                            'evidence_source': 'unit_fixture', 'purpose': 'collection',
                            'scene_revision': SCENE_REVISION}
                recorder = Recorder(root/'episode', f'Pick up the {case.target} cube and place it in the bin.', metadata)
                for index in range(24):
                    observation = Observation(index/30, (.17, 0., .085), (.001*index,)*joints,
                        (.01,)*joints, {color: (*pose, 1., 0., 0., 0.) for color, pose in layout(case.seed).items()}, {})
                    recorder.append(observation, Action((.0001*(index % 3), 0., 0.), float(index % 2)),
                                    'done' if index == 23 else 'fixture', index == 23,
                                    np.full((16, 16, 3), index*8, dtype=np.uint8))
                recorder.close({'success': True})
                library_path = os.environ.pop('LD_LIBRARY_PATH', None)
                try:
                    export([root/'episode'], root/'lerobot', require_physics=False)
                finally:
                    if library_path is not None:
                        os.environ['LD_LIBRARY_PATH'] = library_path
                result = validate_with_upstream(source, root/'lerobot', decode_video=True)
                self.assertEqual(result['frames'], 24)
                self.assertEqual(result['action_chunk_samples_verified'], 3)
                self.assertTrue(result['upstream_normalization_verified'])
                self.assertTrue(result['world_action_semantics_verified'])
                self.assertFalse(result['training_executed'])
