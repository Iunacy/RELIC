from dataclasses import replace
import unittest
import json
from pathlib import Path
import tempfile
from embodiment.benchmark import Action
from embodiment.perception import DisturbedPolicy, condition, verify_trace
from embodiment.perturbations import Perturbation
from embodiment.policy import PolicyInput
from embodiment.paired import compare


class PerceptionSchedule(unittest.TestCase):
    def test_noise_is_indexed_by_physical_time_and_true_state_is_preserved(self):
        class Policy:
            def reset(self): self.inputs = []
            def actions(self, observation):
                self.inputs.append(observation)
                return (Action((0.,0.,0.),1.),)
        original = PolicyInput('openmanipulator_x', 'reach', (0.,)*6, (0.,)*6, (.17,0.,.085), 1., None, .5)
        first = DisturbedPolicy(Policy(), Perturbation('observation_noise', 4, .001))
        second = DisturbedPolicy(Policy(), Perturbation('observation_noise', 4, .001))
        first.reset(); second.reset()
        first.actions(original)
        second.actions(replace(original, timestamp=.1))
        second.actions(original)
        self.assertEqual(first.trace[-1], second.trace[-1])
        self.assertEqual(original.ee_position, (.17,0.,.085))
        self.assertNotEqual(first.policy.inputs[-1].ee_position, original.ee_position)
        self.assertEqual(first.policy.inputs[-1].joint_position, original.joint_position)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'perception.jsonl'
            path.write_text(json.dumps(first.trace[0])+'\n')
            metadata = {'seed': 4, 'observation_condition':condition('observation_noise')}
            samples = [{'timestamp':original.timestamp, 'ee_position':list(original.ee_position)}]
            verify_trace(path, metadata, samples)
            corrupted = dict(first.trace[0], control_tick=0)
            path.write_text(json.dumps(corrupted)+'\n')
            with self.assertRaises(ValueError): verify_trace(path, metadata, samples)
        with self.assertRaises(ValueError): first.actions(replace(original, timestamp=None))

    def test_comparisons_reject_mismatched_disturbances_before_aggregation(self):
        with self.assertRaisesRegex(ValueError, 'disturbances differ'):
            compare({'observation_condition':condition()}, {'observation_condition':condition('observation_noise')})
        with self.assertRaises(ValueError): condition('observation_noise', .1)
