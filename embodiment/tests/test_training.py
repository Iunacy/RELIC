import copy
from dataclasses import asdict
from pathlib import Path
import unittest
from embodiment.training import BASE_MODEL,Budget,training_arguments,validate_matched


class TrainingProtocol(unittest.TestCase):
    def runs(self):
        base = {'condition':'b_only','base_model':BASE_MODEL,'embodiment':'franka_panda',
                'dataset_digest':'verified-dataset-digest','budget':asdict(Budget()),
                'evaluation_scenarios':[{'seed':42,'target':'red'}],
                'success_definition':'relic.cube_bin.v2','action_horizon':16,'execution_horizon':16}
        transfer = {**copy.deepcopy(base),'condition':'a_to_b','base_model':'/external/a/checkpoint-2000'}
        return base,transfer

    def test_matching_data_and_budget_are_required(self):
        base,transfer = self.runs()
        self.assertEqual(len(validate_matched(base,transfer)),64)
        for field,value in (('dataset_digest','different'),('budget',asdict(Budget(steps=1000))),
                            ('evaluation_scenarios',[{'seed':43}]),('execution_horizon',8),
                            ('success_definition','different')):
            changed = {**copy.deepcopy(transfer),field:value}
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_matched(base,changed)
        with self.assertRaises(ValueError):
            validate_matched({**base,'dataset_digest':None},transfer)

    def test_original_baseline_checkpoint_is_required(self):
        base,transfer = self.runs()
        with self.assertRaises(ValueError):
            validate_matched({**base,'base_model':transfer['base_model']},transfer)
        with self.assertRaises(ValueError):
            validate_matched(base,{**transfer,'base_model':BASE_MODEL})

    def test_budget_validation(self):
        for arguments in ({'steps':0},{'batch_size':True},{'learning_rate':float('nan')},{'save_steps':3000},{'seed':7}):
            with self.assertRaises(ValueError):
                Budget(**arguments)
        self.assertEqual(Budget().steps*Budget().batch_size,24000)

    def test_explicit_dataset_and_control_mapping(self):
        arguments = training_arguments(Path('/data/panda'),BASE_MODEL,Path('/runs/b'),Budget())
        self.assertIn('NEW_EMBODIMENT',arguments)
        self.assertIn('--no-tune-llm',arguments)
        self.assertIn('--no-tune-visual',arguments)
        self.assertEqual(arguments[arguments.index('--global-batch-size')+1],'12')


if __name__ == '__main__':
    unittest.main()


class MatchedTargetInitialization(unittest.TestCase):
    def test_new_protocol_compares_target_recipe_and_preserves_source_cost(self):
        from unittest.mock import patch
        from embodiment.training import prepare
        with patch('embodiment.training.verify_source'), \
             patch('embodiment.training.validate_export', side_effect=[
                 {'embodiment': 'openmanipulator_x', 'split': 'train', 'episodes': 100},
                 {'embodiment': 'franka_panda', 'split': 'train', 'episodes': 100}]), \
             patch('embodiment.training.dataset_digest', return_value='unit-fixture-digest'):
            result = prepare(Path('/source'), Path('/a'), Path('/b'), Path('/runs'), pretrained=Path('/weights'))
        baseline, transfer = result['runs'][1:]
        self.assertEqual(len(transfer['evaluation_scenarios']), 50)
        self.assertEqual(transfer['demonstration_budget'], 100)
        self.assertTrue(validate_matched(baseline, transfer))
        self.assertIsNone(baseline['additional_source_budget'])
        self.assertEqual(transfer['additional_source_budget']['steps'], 2000)
        for key, value in (('trainable_selection', ['different']), ('normalizer_source', 'other'),
                           ('target_initialization_donor', '/different'), ('optimizer', 'sgd'),
                           ('inference_config', {'execution_horizon': 1}), ('safety_config', 'changed')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_matched(baseline, {**transfer, key: value})
