import math
import unittest
from embodiment.panda import ARM, FINGERS, bounded_reference, downward_quaternion


class PandaMapping(unittest.TestCase):
    def test_downward_radial_frame(self):
        for position in ((.17, 0, .085), (.18, -.08, .02), (.16, .115, .065)):
            w, x, y, z = downward_quaternion(position)
            self.assertAlmostEqual(w*w+x*x+y*y+z*z, 1)
            self.assertAlmostEqual(1-2*(x*x+y*y), -1)
            yaw = math.atan2(position[1], position[0]-.012)
            self.assertAlmostEqual(1-2*(y*y+z*z), math.cos(yaw))
            self.assertAlmostEqual(2*(x*y+w*z), math.sin(yaw))
        self.assertEqual(len(set(ARM+FINGERS)), 9)

    def test_joint_projection_is_bounded(self):
        limits = ((-2., 2.),) * 7
        result = bounded_reference((0.,)*7, (1.,)*7, limits, 1/30)
        self.assertTrue(all(abs(v-.5/30) < 1e-12 for v in result))
        for current, goal, bounds, dt in (((0.,)*7, (3.,)*7, limits, .03),
                ((0.,)*7, (1.,)*7, limits, 0), ((0.,)*6, (1.,)*7, limits, .03),
                ((0.,)*7, (float('nan'),)*7, limits, .03)):
            with self.assertRaises(ValueError):
                bounded_reference(current, goal, bounds, dt)


if __name__ == '__main__':
    unittest.main()
