import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodiment.policy import checkpoint_digest, validate_checkpoint
from embodiment.training import Budget, prepare, training_arguments, validate_upstream_arguments
from embodiment.training_smoke import smoke


class TrainingSmoke(unittest.TestCase):
    def checkpoint(self, root):
        (root/'config.json').write_text(json.dumps({'model_type': 'gr00t_n1d7'}))
        (root/'model.safetensors').write_bytes(b'unit-fixture-not-real-weights')
        (root/'processor').mkdir()
        (root/'processor/processor_config.json').write_text('{}')
        (root/'processor/statistics.json').write_text('{}')
        (root/'processor/embodiment_id.json').write_text('{}')

    def test_checkpoint_requires_complete_shards_and_processor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.checkpoint(root)
            self.assertEqual(validate_checkpoint(root)['weight_files'], 1)
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'tensor': 'missing.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                validate_checkpoint(root)
            (root/'model.safetensors.index.json').unlink()
            (root/'processor/statistics.json').unlink()
            with self.assertRaisesRegex(ValueError, 'statistics'):
                validate_checkpoint(root)

    def test_checkpoint_identity_excludes_training_logs_and_nested_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.checkpoint(root)
            before = checkpoint_digest(root)
            (root/'trainer_state.json').write_text('{"global_step": 2}')
            (root/'checkpoint-1').mkdir()
            (root/'checkpoint-1/model.safetensors').write_bytes(b'unit-fixture')
            self.assertEqual(before, checkpoint_digest(root))
            (root/'processor/statistics.json').write_text('{"changed": true}')
            self.assertNotEqual(before, checkpoint_digest(root))

    def test_preflight_cannot_claim_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root/'base'
            base.mkdir()
            self.checkpoint(base)
            with patch('embodiment.training_smoke.verify_source'), \
                 patch('embodiment.training_smoke.validate_export', return_value={'split': 'train'}), \
                 patch('embodiment.training_smoke.dataset_digest', return_value='unit-fixture'), \
                 patch('embodiment.training_smoke.validate_with_upstream', return_value={}), \
                 patch('embodiment.training_smoke.validate_upstream_arguments', return_value={}), \
                 patch('embodiment.collection.execute', side_effect=AssertionError('must not execute')):
                result = smoke(root, root/'data', base, root/'smoke')
            self.assertEqual(result['status'], 'PREFLIGHT_ONLY')
            self.assertFalse(result['training_executed'])
            self.assertFalse(result['inference_executed'])
            self.assertTrue((root/'smoke/result.json').exists())

    def test_unavailable_dataset_is_a_recorded_blocker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch('embodiment.training_smoke.verify_source'):
                result = smoke(root, root/'missing', root/'base', root/'smoke')
            self.assertEqual(result['status'], 'BLOCKED')
            self.assertFalse(result['training_executed'])
            self.assertTrue(result['errors'])

    def test_final_output_is_transfer_source_even_without_final_periodic_save(self):
        with patch('embodiment.training.verify_source'), \
             patch('embodiment.training.validate_export', side_effect=OSError('no dataset')):
            result = prepare(Path('/source'), Path('/data/a'), Path('/data/b'), Path('/runs'),
                             Budget(steps=11, save_steps=5))
        self.assertEqual(result['runs'][2]['base_model'], '/runs/a')
        self.assertEqual(result['runs'][0]['checkpoint'], '/runs/a')


if __name__ == '__main__':
    unittest.main()
