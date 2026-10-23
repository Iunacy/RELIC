import importlib.util
from pathlib import Path
import tempfile
import unittest

from embodiment.assets import ARM, FINGERS
from embodiment.benchmark import Action, COLORS, Observation, layout
from embodiment.dataset import load_episode
from embodiment.evaluate import run_trial
from embodiment.execution import ExecutionConfig
from embodiment.safety import Guard, SafetyConfig
from embodiment.scenarios import plan


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'), 'image dependencies unavailable')
class EvaluationRecording(unittest.TestCase):
    def test_failed_learned_trial_records_real_commands_and_remains_out_of_expert_data(self):
        import numpy as np
        case = plan('evaluation', 1)[0]
        class Backend:
            embodiment = 'openmanipulator_x'
            arm_names, finger_names = ARM, FINGERS
            safety_config = SafetyConfig(((-3., 3.),)*4 + ((-.011, .02),)*2, 4)
            def __init__(self):
                self.guard = Guard(self.safety_config)
                self.commands = []
                self.stopped = False
            def reset(self, seed):
                self.observation = Observation(0., (.17, 0., .085), (0.,)*6, (0.,)*6,
                    {color: (*pose, 1., 0., 0., 0.) for color, pose in layout(seed).items()},
                    {color: 0. for color in COLORS}, contacts={color: False for color in COLORS},
                    contact_forces={color: (0., 0.) for color in COLORS}, gripper_open=1.)
                return self.observation
            def step(self, action):
                self.commands.append(action)
                from dataclasses import replace
                self.observation = replace(self.observation, time=1/30, unsafe=True)
                return self.observation
            def frame(self):
                return np.zeros((8, 8, 3), dtype=np.uint8)
            def safe_stop(self):
                self.stopped = True
        class Policy:
            metadata = {'checkpoint_sha256': 'unit-fixture-no-model-inference'}
            def reset(self):
                self.inputs = []
            def actions(self, observation):
                self.inputs.append(observation)
                return (Action((0., 0., 0.), 1.),)
        backend, policy = Backend(), Policy()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'trial'
            result = run_trial(backend, policy, case, output,
                               {'source_sha256': {'test': 'unit-fixture'}, 'relic_commit': 'unit-fixture'},
                               ExecutionConfig(fallback_enabled=False), lambda: True)
            self.assertFalse(result['success'])
            self.assertEqual(result['controller'], 'vla_only')
            self.assertEqual(result['failure_reason'], 'unsafe_movement')
            self.assertEqual(len(backend.commands), 1)
            self.assertTrue(backend.stopped)
            self.assertFalse(hasattr(policy.inputs[0], 'cubes'))
            episode = load_episode(output, expert_only=False)
            self.assertEqual(len(episode.rows), 2)
            self.assertEqual(episode.rows[0]['action'], [0., 0., 0., 1.])
            with self.assertRaisesRegex(ValueError, 'scripted demonstration'):
                load_episode(output)
