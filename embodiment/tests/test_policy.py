import importlib.util
import unittest
from dataclasses import replace
from embodiment.benchmark import COLORS, Observation, layout
from embodiment.policy import GrootPolicy, PolicyInput

HAS_NUMPY = importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_NUMPY,'NumPy unavailable')
class PolicyBoundary(unittest.TestCase):
    def input(self):
        import numpy as np
        obs = Observation(0.,(.17,0.,.085),(0.,)*6,(0.,)*6,
                          {c:(*p,1,0,0,0) for c,p in layout(0).items()}, {c:0 for c in COLORS})
        return PolicyInput.from_observation(obs,np.zeros((8,8,3),dtype=np.uint8),'openmanipulator_x',
                                            'Pick up the red cube and place it in the bin.')

    def test_privileged_state_is_excluded(self):
        observation = self.input()
        self.assertFalse(hasattr(observation,'cubes'))
        self.assertFalse(hasattr(observation,'target_pose'))
        packed = GrootPolicy(None).pack(observation)
        self.assertEqual(set(packed),{'video','state','language'})
        self.assertEqual(packed['state']['joint_position'].shape,(1,1,6))
        self.assertEqual(packed['video']['overhead'].shape,(1,1,8,8,3))
        self.assertAlmostEqual(packed['state']['gripper'][0,0,0],.011/.015)

    def test_actual_public_interface_and_chunk_mapping(self):
        import numpy as np
        class MockPublicPolicy:
            def get_action(self, observation):
                self.observation = observation
                return {'delta_position':np.zeros((1,16,3),dtype=np.float32),
                        'gripper':np.ones((1,16,1),dtype=np.float32)}, {'mode':'mock'}
            def reset(self):
                self.was_reset = True
        model = MockPublicPolicy()
        policy = GrootPolicy(model)
        policy.reset()
        self.assertTrue(model.was_reset)
        self.assertEqual(len(policy.actions(self.input())),16)
        self.assertEqual(policy.actions(self.input())[0].gripper,1.)

    def test_invalid_camera_and_state(self):
        import numpy as np
        for changes in ({'image':None},{'image':np.zeros((8,8,3))}, {'joint_position':(0.,)*9}, {'instruction':''}):
            with self.assertRaises(ValueError):
                GrootPolicy(None).pack(replace(self.input(),**changes))

    def test_entire_chunk_is_checked_before_execution(self):
        import numpy as np
        for defect in ('nan','oversized','length','gripper','keys'):
            delta = np.zeros((1,16,3),dtype=np.float32)
            gripper = np.ones((1,16,1),dtype=np.float32)
            if defect == 'nan': delta[0,15,0] = float('nan')
            if defect == 'oversized': delta[0,15,0] = .1
            if defect == 'length': delta = np.zeros((1,17,3))
            if defect == 'gripper': gripper = np.ones((1,16,2))
            output = {'delta_position':delta,'gripper':gripper}
            if defect == 'keys': output['unknown'] = 0
            class MockPublicPolicy:
                def get_action(self, observation): return output, {'mode':'mock'}
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                GrootPolicy(MockPublicPolicy()).actions(self.input())


if __name__ == '__main__':
    unittest.main()
