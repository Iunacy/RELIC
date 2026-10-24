import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodiment.policy import checkpoint_digest, prepare_training_checkpoint, validate_checkpoint
from embodiment.training import Budget, prepare, training_arguments, validate_upstream_arguments
from embodiment.training_smoke import smoke, validate_resume


class TrainingSmoke(unittest.TestCase):
    def checkpoint(self, root):
        if importlib.util.find_spec('safetensors') is None:
            self.skipTest('safetensors required for checkpoint validation')
        import numpy as np
        from safetensors.numpy import save_file
        (root/'config.json').write_text(json.dumps({'model_type': 'gr00t_n1d7'}))
        save_file({'unit_fixture': np.zeros(1, dtype=np.float32)}, root/'model.safetensors')
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

    def test_invalid_tensor_contents_and_incorrect_index_are_rejected(self):
        for defect in ('contents', 'index'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.checkpoint(root)
                if defect == 'contents':
                    (root/'model.safetensors').write_bytes(b'unit-fixture-not-real-weights')
                else:
                    (root/'model.safetensors.index.json').write_text(json.dumps({
                        'weight_map': {'nonexistent_tensor': 'model.safetensors'}}))
                with self.assertRaisesRegex(ValueError, 'invalid safetensors'):
                    validate_checkpoint(root)

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
        self.assertEqual(result['runs'][2]['base_model'], '/runs/initialization/a_to_b')
        self.assertEqual(result['runs'][1]['base_model'], '/runs/initialization/b_only')
        self.assertIn('/runs/a', result['runs'][2]['initialization_commands'][1])
        self.assertEqual(result['runs'][0]['checkpoint'], '/runs/a')
        self.assertEqual(result['runs'][2]['initialization_commands'][0][-1], '/runs/a')

    def test_transfer_preparation_copies_only_saved_processor_and_is_restartable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.checkpoint(root)
            weights = (root/'model.safetensors').read_bytes()
            result = prepare_training_checkpoint(root)
            self.assertEqual(result['processor'], str(root))
            self.assertEqual(len(result['prepared_processor_files']), 3)
            self.assertEqual(prepare_training_checkpoint(root)['prepared_processor_files'], [])
            self.assertEqual((root/'model.safetensors').read_bytes(), weights)
            (root/'processor_config.json').unlink()
            (root/'statistics.json').unlink()
            self.assertEqual(len(prepare_training_checkpoint(root)['prepared_processor_files']), 2)

    def test_transfer_preparation_rejects_conflicting_processor_before_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.checkpoint(root)
            (root/'statistics.json').write_text('{"conflict": true}')
            with self.assertRaisesRegex(ValueError, 'disagree'):
                prepare_training_checkpoint(root)
            self.assertFalse((root/'processor_config.json').exists())


if __name__ == '__main__':
    unittest.main()


class OptimizerRecovery(unittest.TestCase):
    def test_resume_requires_complete_integrity_checked_optimizer_state(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            TrainingSmoke().checkpoint(root)
            (root/'trainer_state.json').write_text('{"global_step": 1}')
            with self.assertRaisesRegex(ValueError, 'complete upstream'):
                validate_resume(root)
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
                with zipfile.ZipFile(root/name, 'w') as archive:
                    archive.writestr('unit_fixture', 'synthetic optimizer archive; never training evidence')
            result = validate_resume(root)
            self.assertEqual(result['step'], 1)
            self.assertEqual(len(result['optimizer_state_sha256']), 3)
            (root/'optimizer.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): validate_resume(root)
