import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from embodiment.tests.test_checkpoint_compatibility import fixture
from embodiment.transfer import prepare, validate_architecture


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'), 'CPU torch and safetensors required')
class Transfer(unittest.TestCase):
    def checkpoint(self, root, robot, fill):
        import torch
        from safetensors.torch import save_file
        root.mkdir()
        config = {'model_type': 'gr00t_n1d7', 'max_state_dim': 29, 'max_action_dim': 4,
                  'state_history_length': 1, 'max_num_embodiments': 11}
        (root/'config.json').write_text(json.dumps(config))
        tensors = {'backbone.weight': torch.full((2, 2), fill), 'action_head.model.weight': torch.full((2, 2), fill)}
        for module in ('state_encoder', 'action_encoder', 'action_decoder'):
            layers = ('W1', 'W2', 'W3') if module == 'action_encoder' else ('layer1', 'layer2')
            for layer in layers:
                size = 29 if module == 'state_encoder' and layer == 'layer1' else 4
                tensors[f'action_head.{module}.{layer}.W'] = torch.full((11, size, 4), fill)
                tensors[f'action_head.{module}.{layer}.b'] = torch.full((11, 4), fill)
        save_file(tensors, root/'model.safetensors')
        cfg, stats = fixture(robot)
        cfg['processor_kwargs'].update(max_state_dim=29, max_action_dim=4)
        for name, value in (('processor_config.json', cfg), ('statistics.json', stats), ('embodiment_id.json', {'new_embodiment': 10})):
            (root/name).write_text(json.dumps(value))
        return tensors

    def test_matched_projectors_and_source_shared_weights(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.checkpoint(root/'a', 'openmanipulator_x', 3.)
            self.checkpoint(root/'base', 'openmanipulator_x', 1.)
            cfg, stats = fixture('franka_panda')
            cfg['processor_kwargs'].update(max_state_dim=29, max_action_dim=4)
            with patch('embodiment.transfer.target_processor', return_value=(cfg, stats, {'new_embodiment': 10})), \
                 patch('embodiment.transfer.effective_configuration', side_effect=lambda config, upstream: config):
                for condition in ('E2', 'E3'):
                    result = prepare(root/'a', root/'base', root/'b', root, root/condition, condition)
                    self.assertFalse(result['upstream_training_load_verified'])
            baseline, transfer = [load_file(root/name/'model.safetensors') for name in ('E2', 'E3')]
            for key in baseline:
                if 'encoder.' in key or 'decoder.' in key:
                    self.assertTrue(baseline[key][10].equal(transfer[key][10]))
                    self.assertTrue((transfer[key][10] == 1).all())
                else:
                    self.assertTrue((baseline[key] == 1).all())
                    self.assertTrue((transfer[key] == 3).all())
            self.assertEqual(json.loads((root/'a/statistics.json').read_text())['new_embodiment']['state']['joint_position']['mean'], [0.]*6)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                prepare(root/'a', root/'base', root/'b', root, root/'E3')

    def test_architecture_and_tensor_mismatch_fail_closed(self):
        from safetensors.torch import load_file, save_file
        for defect in ('config', 'missing', 'shape', 'unexpected'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.checkpoint(root/'a', 'openmanipulator_x', 3.)
                self.checkpoint(root/'base', 'openmanipulator_x', 1.)
                if defect == 'config':
                    path = root/'a/config.json'; value = json.loads(path.read_text()); value['max_state_dim'] = 30; path.write_text(json.dumps(value))
                else:
                    path = root/'a/model.safetensors'; tensors = load_file(path)
                    if defect == 'missing': del tensors['backbone.weight']
                    if defect == 'shape': tensors['backbone.weight'] = tensors['backbone.weight'][:1]
                    if defect == 'unexpected': tensors['unknown'] = tensors['backbone.weight'].clone()
                    save_file(tensors, path)
                with self.assertRaises(ValueError): validate_architecture(root/'a', root/'base')

    def test_mixed_training_precision_is_allowed_but_integer_reinterpretation_is_rejected(self):
        from safetensors.torch import load_file, save_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.checkpoint(root/'a', 'openmanipulator_x', 3.)
            self.checkpoint(root/'base', 'openmanipulator_x', 1.)
            path = root/'base/model.safetensors'
            tensors = load_file(path)
            save_file({name: tensor.bfloat16() for name, tensor in tensors.items()}, path)
            self.assertTrue(validate_architecture(root/'a', root/'base'))
            save_file({name: tensor.int() for name, tensor in tensors.items()}, path)
            with self.assertRaises(ValueError): validate_architecture(root/'a', root/'base')
