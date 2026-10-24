import importlib.util
import unittest
from embodiment.training_worker import parameter_inventory


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU torch required')
class TrainingInventory(unittest.TestCase):
    def test_records_actual_trainable_names_and_excludes_frozen_parameters(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 1))
        model[0].requires_grad_(False)
        result = parameter_inventory(model)
        self.assertEqual(set(result), {'1.weight', '1.bias'})
        self.assertEqual(result['1.weight'], {'shape': [1, 4], 'elements': 4})
