import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from embodiment.weights import inspect_weights


@unittest.skipUnless(importlib.util.find_spec('safetensors') and importlib.util.find_spec('numpy'), 'safe tensor dependencies required')
class WeightIntegrity(unittest.TestCase):
    def test_exact_shard_catalog_and_metadata_are_hashed_without_model_loading(self):
        import numpy as np
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'config.json').write_text('{"model_type":"unit_fixture"}')
            save_file({'tensor': np.zeros(3, dtype=np.float32)}, root/'model.safetensors')
            first = inspect_weights(root)
            self.assertFalse(first['weights_loaded'])
            self.assertEqual(first['tensors'], 1)
            save_file({'tensor': np.ones(3, dtype=np.float32)}, root/'model.safetensors')
            self.assertNotEqual(first['sha256'], inspect_weights(root)['sha256'])
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'wrong': 'model.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'index'): inspect_weights(root)
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'wrong': '../model.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'unsafe'): inspect_weights(root)
