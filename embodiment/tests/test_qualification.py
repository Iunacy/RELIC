import math
import unittest
from dataclasses import replace
from embodiment.benchmark import BIN, COLORS, Observation, layout
from embodiment.evaluation import metrics, wilson
from embodiment.expert import Expert
from embodiment.openmanipulator import bounded_reference,ik


def observed(time, **changes):
    base = Observation(time, (.17, 0, .085), (0, 0, 0, 0, .02, .02), (0,) * 6,
                       {c: (*p, 1, 0, 0, 0) for c, p in layout(0).items()},
                       {c: 0 for c in COLORS})
    return replace(base, **changes)


class PhysicalEvidence(unittest.TestCase):
    def test_aperture_alone_does_not_prove_grasp(self):
        expert = Expert('red')
        expert.phase = 'verify'
        obs = observed(1, ee=layout(0)['red'], joints=(0, 0, 0, 0, .002, .002))
        expert.update(obs)
        self.assertEqual(expert.failure, 'failed_grasp')

    def test_bilateral_contact_and_lift_are_required(self):
        for contact, z, expected in ((False, .085, 'failed_grasp'), (True, .015, 'failed_grasp'), (True, .085, None)):
            expert = Expert('red')
            expert.phase = 'lift'
            expert.pick = layout(0)['red']
            x, y, _ = expert.pick
            obs = observed(1, ee=(x, y, .085), contacts={'red': contact})
            obs.cubes['red'] = (x, y, z, 1, 0, 0, 0)
            expert.update(obs)
            self.assertEqual(expert.failure, expected)
            self.assertEqual(expert.lifted, expected is None)

    def test_transport_lowers_before_releasing(self):
        expert = Expert('red')
        expert.phase = 'transport'
        expert.lifted = True
        obs = observed(1, ee=(*BIN[:2], .085), contacts={'red': True})
        obs.cubes['red'] = (*BIN[:2], .07, 1, 0, 0, 0)
        action = expert.update(obs)
        self.assertEqual(expert.phase, 'lower')
        self.assertEqual(action.gripper, 0.)
        obs.time = 2
        obs.ee = (*BIN[:2], .065)
        action = expert.update(obs)
        self.assertEqual(expert.phase, 'release')
        self.assertEqual(action.gripper, 0.)

    def test_bounded_ik_reference_preserves_downward_pitch(self):
        current = ik((.17,0.,.085))
        for goal in ((.19,-.05,.0285),(*BIN[:2],.065),(.15,.04,.085)):
            desired = ik(goal)
            projected = bounded_reference(current,desired,1/30)
            self.assertAlmostEqual(sum(projected[1:]),math.pi/2)
            self.assertTrue(all(abs(a-b) <= .5/30+1e-12 for a,b in zip(projected,current)))

    def test_transport_budget_remains_bounded(self):
        expert = Expert('red')
        expert.phase = 'transport'
        expert.lifted = True
        obs = observed(11.,contacts={'red':True})
        obs.cubes['red'] = (.17,0.,.07,1.,0.,0.,0.)
        expert.update(obs)
        self.assertIsNone(expert.failure)
        obs.time = 14.1
        expert.update(obs)
        self.assertEqual(expert.failure,'timeout')

    def test_wilson_reference_counts(self):
        low, high = wilson(18, 20)
        self.assertAlmostEqual(low, .6989663547715127)
        self.assertAlmostEqual(high, .9721335187862318)
        self.assertIsNone(wilson(0, 0))
        self.assertEqual(wilson(0, 20)[0], 0.)
        self.assertAlmostEqual(wilson(20, 20)[1], 1.)
        for args in ((21, 20), (-1, 20), (True, 2), (1, -1)):
            with self.assertRaises(ValueError):
                wilson(*args)
        self.assertIsNone(metrics([])['success_rate'])


if __name__ == '__main__':
    unittest.main()
