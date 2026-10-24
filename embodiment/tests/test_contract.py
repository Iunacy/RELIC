import math
import unittest
from embodiment.contract import ADAPTERS, TaskFrame, manifest


class Contract(unittest.TestCase):
    def test_native_state_and_gripper_roundtrip(self):
        for robot, adapter in ADAPTERS.items():
            self.assertEqual(adapter.state_dimension, 16 if adapter.arm_dof == 4 else 22)
            self.assertEqual(len(manifest(robot)['joint_names']), adapter.arm_dof+2)
            for opening in (0., .1, .5, 1.):
                joints = (0.,)*adapter.arm_dof+adapter.fingers(opening)
                self.assertAlmostEqual(adapter.aperture(joints), opening)
            with self.assertRaises(ValueError): adapter.fingers(1.1)
            with self.assertRaises(ValueError): adapter.aperture((0.,)*3)
            with self.assertRaises(ValueError): adapter.aperture((0.,)*adapter.arm_dof+(1., 1.))

    def test_rigid_frames_roundtrip_and_displacement(self):
        frame = TaskFrame(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)), (.4, -.3, .1))
        for value in ((0., 0., 0.), (.1, -.2, .05)):
            for delta in (True, False):
                self.assertLess(math.dist(frame.task(frame.world(value, delta), delta), value), 1e-12)
        self.assertEqual(frame.world((.1, 0., 0.), True), (0., .1, 0.))
        with self.assertRaises(ValueError): TaskFrame(((1., 0., 0.), (0., 1., 0.), (0., 0., -1.)))
        with self.assertRaises(ValueError): TaskFrame(((2., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
