import copy
import json
from pathlib import Path
import tempfile
import unittest

from embodiment.checkpoint_compatibility import validate_processor
from embodiment.dataset import DOF


def fixture(robot):
    count = DOF[robot]+2
    groups = {'state': {'joint_position': count, 'joint_velocity': count, 'ee_position': 3, 'gripper': 1},
              'action': {'delta_position': 3, 'gripper': 1}}
    modalities = {name: {'modality_keys': list(keys), 'delta_indices': list(range(16)) if name == 'action' else [0]}
                  for name, keys in groups.items()}
    modalities['action']['action_configs'] = [{'rep': 'ABSOLUTE', 'type': 'NON_EEF', 'format': 'DEFAULT'}]*2
    modalities['video'] = {'modality_keys': ['overhead'], 'delta_indices': [0]}
    modalities['language'] = {'modality_keys': ['annotation.human.task_description'], 'delta_indices': [0]}
    statistics = {name: {key: {field: ([1.]*size if field in ('std', 'max', 'q99') else [0.]*size)
                    for field in ('min', 'max', 'mean', 'std', 'q01', 'q99')} for key, size in keys.items()}
                  for name, keys in groups.items()}
    return {'processor_kwargs': {'modality_configs': {'new_embodiment': modalities}}}, {'new_embodiment': statistics}


class CheckpointCompatibility(unittest.TestCase):
    def test_real_dimensions_distinguish_the_two_embodiments(self):
        for robot in DOF:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, stats = fixture(robot)
                (root/'processor_config.json').write_text(json.dumps(config));(root/'statistics.json').write_text(json.dumps(stats))
                self.assertEqual(validate_processor(root, robot)['embodiment'], robot)
                other = next(name for name in DOF if name != robot)
                with self.assertRaisesRegex(ValueError, 'another embodiment'):
                    validate_processor(root, other)

    def test_semantic_mismatches_fail_before_weights_are_loaded(self):
        for defect in ('relative', 'horizon', 'dimension', 'nan', 'std', 'order', 'camera'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary);config, stats = fixture('openmanipulator_x')
                modalities = config['processor_kwargs']['modality_configs']['new_embodiment']
                if defect == 'relative': modalities['action']['action_configs'][0] = {'rep': 'RELATIVE', 'type': 'EEF', 'format': 'DEFAULT'}
                if defect == 'horizon': modalities['action']['delta_indices'] = [0]
                if defect == 'dimension': stats['new_embodiment']['state']['joint_velocity']['mean'].append(0.)
                if defect == 'nan': stats['new_embodiment']['state']['joint_position']['mean'][0] = float('nan')
                if defect == 'std': stats['new_embodiment']['state']['joint_position']['std'][0] = -1.
                if defect == 'order': modalities['state']['modality_keys'].reverse()
                if defect == 'camera': modalities['video']['modality_keys'] = ['wrist']
                (root/'processor_config.json').write_text(json.dumps(config));(root/'statistics.json').write_text(json.dumps(stats))
                with self.assertRaises(ValueError): validate_processor(root)
