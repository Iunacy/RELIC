import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from embodiment.dual_media import produce
from embodiment.replay import load_qualified_pair
from embodiment.replay_render import Renderer


@unittest.skipUnless(os.environ.get('RELIC_PHYSICAL_REPLAY_TESTS') == '1', 'set RELIC_PHYSICAL_REPLAY_TESTS=1 for real asset and media checks')
class RecordedPhysicalMedia(unittest.TestCase):
    def test_real_assets_preserve_recorded_kinematics_and_grippers(self):
        import numpy as np
        for robot, episode in load_qualified_pair(0).items():
            renderer = Renderer(robot)
            try:
                for elapsed in (0., 12., episode.duration):
                    image = renderer.render(episode, elapsed)
                    self.assertEqual(image.shape, (480, 640, 3))
                    self.assertEqual(image.dtype, np.uint8)
                    self.assertGreater(int(image.max())-int(image.min()), 100)
                self.assertLess(renderer.provenance['maximum_fk_error_m'], .0002)
                self.assertFalse(renderer.provenance['dynamics_reexecuted'])
                if robot == 'franka_panda':
                    self.assertGreater(renderer.provenance['visual_parts'], 40)
                    fingers = [index for index in range(renderer.model.ngeom)
                               if 'panda_rightfinger' in (renderer.mj.mj_id2name(renderer.model, renderer.mj.mjtObj.mjOBJ_GEOM, index) or '')
                               and renderer.model.geom_group[index] == 2]
                    self.assertTrue(fingers)
                    for index in fingers:
                        mesh = renderer.model.geom_dataid[index]
                        left = next(candidate for candidate in range(renderer.model.ngeom)
                                    if renderer.model.geom_dataid[candidate] == mesh and renderer.model.geom_group[candidate] == 2
                                    and 'panda_leftfinger' in (renderer.mj.mj_id2name(renderer.model, renderer.mj.mjtObj.mjOBJ_GEOM, candidate) or ''))
                        inverse = renderer.model.geom_quat[left].copy()
                        inverse[1:] *= -1
                        relative = np.zeros(4)
                        renderer.mj.mju_mulQuat(relative, renderer.model.geom_quat[index], inverse)
                        self.assertAlmostEqual(abs(relative[-1]), 1., places=6)
            finally:renderer.close()

    def test_real_replay_media_encodes_decodes_and_preserves_source_indices(self):
        import av
        with tempfile.TemporaryDirectory() as temporary:
            result = produce(load_qualified_pair(0), Path(temporary)/'media', fps=1, speed=8., gif=True)
            self.assertEqual(result['playback_speed'], 8.)
            self.assertFalse(result['learned_transfer_demonstrated'])
            self.assertEqual(result['source_frame_indices'][0]['franka_panda'], 0)
            for variant in ('clean', 'comparison'):
                with av.open(str(Path(temporary)/'media'/(variant+'.mp4'))) as container:
                    frames = list(container.decode(video=0))
                self.assertEqual(len(frames), result['frames'])
            self.assertIn('simulation.gif', result['outputs'])
