import unittest
from dataclasses import replace
from embodiment.benchmark import Action, Observation
from embodiment.safety import Guard, SafetyConfig, SafetyViolation


CONFIG = SafetyConfig(((-2., 2.),) * 4 + ((-.011, .02),) * 2, 4)


def observation(time=0., **changes):
    obs = Observation(time, (.17, 0., .085), (0.,)*6, (0.,)*6, {}, {})
    return replace(obs, **changes)


class SafetyGuards(unittest.TestCase):
    def test_state_rejection_categories(self):
        cases = (({'unsafe': True}, 'unsafe_state'), ({'ee': (.4, 0., .08)}, 'workspace_state'),
                 ({'joints': (3., 0., 0., 0., 0., 0.)}, 'joint_position'),
                 ({'velocities': (1.3, 0., 0., 0., 0., 0.)}, 'joint_velocity'),
                 ({'velocities': (0., 0., 0., 0., .16, 0.)}, 'joint_velocity'),
                 ({'joints': (float('nan'),)*6}, 'nonfinite_or_malformed_state'))
        for changes, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(SafetyViolation, reason):
                Guard(CONFIG).state(observation(**changes))

    def test_time_must_advance(self):
        guard = Guard(CONFIG)
        guard.state(observation())
        with self.assertRaisesRegex(SafetyViolation, 'nonmonotonic_time'):
            guard.state(observation())

    def test_action_boundary(self):
        guard = Guard(CONFIG)
        for raw in ((0, 0, 0), (0, 0, 0, 1, 4), (float('nan'), 0, 0, 1), (0, 0, 0, 2), None):
            with self.assertRaises(SafetyViolation):
                guard.command(observation(), raw)
        self.assertEqual(guard.command(observation(), [0., 0., .003, 1.]).delta, (0., 0., .003))
        with self.assertRaisesRegex(SafetyViolation, 'workspace_command'):
            guard.command(observation(ee=(.27, .07, .19)), (0., 0., .001, 1.))

    def test_joint_command_velocity_and_position(self):
        guard = Guard(CONFIG)
        guard.joint_command((0.,)*6, (.02, 0., 0., 0., .001, 0.))
        for command, reason in (((.05, 0., 0., 0., 0., 0.), 'joint_command_velocity'),
                                ((0., 0., 0., 0., .025, 0.), 'joint_command_limit')):
            with self.assertRaisesRegex(SafetyViolation, reason):
                guard.joint_command((0.,)*6, command)

    def test_tracking_dwell_and_recovery(self):
        guard = Guard(CONFIG)
        reference = (.3, 0., 0., 0., 0., 0.)
        self.assertIsNone(guard.tracking(observation(), reference))
        self.assertIsNone(guard.tracking(observation(.4), reference))
        self.assertEqual(guard.tracking(observation(.5), reference), 'tracking_error')
        self.assertIsNone(guard.tracking(observation(.6), (0.,)*6))
        self.assertIsNone(guard.tracking(observation(.7), reference))

    def test_stall_does_not_include_grasp_dwell(self):
        guard = Guard(CONFIG)
        action = Action((.001, 0., 0.), 1.)
        self.assertIsNone(guard.progress(observation(), action))
        self.assertEqual(guard.progress(observation(2.), action), 'lack_of_progress')
        self.assertIsNone(guard.progress(observation(2.1), Action((0., 0., 0.), 0.)))
        self.assertIsNone(guard.progress(observation(8.), action))
        self.assertIsNone(guard.progress(observation(9., ee=(.173, 0., .085)), action))
        self.assertIsNone(guard.progress(observation(10.), action))

    def test_invalid_configuration(self):
        for kwargs in ({'arm_dof': 5}, {'joint_limits': ((0., 0.),)*6}, {'stall_timeout': 0.}):
            with self.assertRaises(ValueError):
                replace(CONFIG, **kwargs)


if __name__ == '__main__':
    unittest.main()
