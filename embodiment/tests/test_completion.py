import json
import unittest
from embodiment.benchmark import BIN, COLORS
from embodiment.completion import CONTRACT, validate_completion
from embodiment.contract import manifest


class PhysicalCompletion(unittest.TestCase):
    def fixture(self):
        metadata = {'episode_contract': CONTRACT, 'embodiment': 'openmanipulator_x', 'target': 'red',
                    'robot_contract': json.loads(json.dumps(manifest('openmanipulator_x')))}
        rows = [{'timestamp': i/30, 'joint_position': [0.]*4+[.004]*2, 'ee_position': [.17, 0., .085]} for i in range(34)]
        telemetry = [{'timestamp': row['timestamp'], 'cube_poses': {'red': [*BIN[:2], .0165, 1., 0., 0., 0.]},
                      'cube_speeds': {color: 0. for color in COLORS}, 'contacts': {'red': False}, 'unsafe': False, 'safety_events': []} for row in rows]
        return metadata, {'success': True}, rows, telemetry

    def test_stable_release_required_and_legacy_not_promoted(self):
        result = validate_completion(*self.fixture())
        self.assertTrue(result['terminal_dwell_verified'])
        self.assertFalse(validate_completion({}, {}, [], [])['terminal_dwell_verified'])
        for defect in ('speed', 'contact', 'time', 'contract', 'gripper', 'retreat'):
            with self.subTest(defect=defect):
                metadata, outcome, rows, physical = self.fixture()
                if defect == 'speed': physical[-1]['cube_speeds']['red'] = .1
                if defect == 'contact': physical[-1]['contacts']['red'] = True
                if defect == 'time': rows, physical = rows[-10:], physical[-10:]
                if defect == 'contract': metadata['robot_contract']['state_dimension'] = 12
                if defect == 'gripper': rows[-1]['joint_position'][-2] = -.011
                if defect == 'retreat': rows[-1]['ee_position'][2] = .05
                with self.assertRaises(ValueError): validate_completion(metadata, outcome, rows, physical)
