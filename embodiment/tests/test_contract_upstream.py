import os
import sys
import unittest
from embodiment.tests.test_checkpoint_compatibility import fixture


@unittest.skipUnless(os.environ.get('RELIC_GROOT_SOURCE'), 'set RELIC_GROOT_SOURCE for actual upstream contract tests')
class UpstreamContract(unittest.TestCase):
    def test_public_checkpoint_defaults_and_actual_forty_to_sixteen_decoder(self):
        from pathlib import Path
        from types import SimpleNamespace
        import numpy as np
        from embodiment.dataset import read_json, file_digest
        from embodiment.models import SOURCES
        from embodiment.transfer import effective_configuration
        path = Path(__file__).resolve().parents[1]/'assets/groot_model_config.json'
        self.assertEqual(file_digest(path), SOURCES['groot']['config_sha256'])
        config = effective_configuration(read_json(path), Path(os.environ['RELIC_GROOT_SOURCE']))
        self.assertEqual(config['state_history_length'], 1)
        self.assertEqual(config['action_horizon'], 40)
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor, validate_action_horizons
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.state_action.state_action_processor import StateActionProcessor
        from embodiment.groot_modalities import RELIC_CONFIG
        modalities = {'new_embodiment': RELIC_CONFIG}
        validate_action_horizons(modalities, config['action_horizon'])
        _, statistics = fixture('franka_panda')
        normalizer = StateActionProcessor(modalities, statistics, use_percentiles=True, use_relative_action=True)
        normalizer.eval()
        context = SimpleNamespace(modality_configs=modalities, state_action_processor=normalizer)
        decoded = Gr00tN1d7Processor.decode_action(context, np.zeros((1,40,132), dtype=np.float32), EmbodimentTag.NEW_EMBODIMENT)
        self.assertEqual(decoded['delta_position'].shape, (1,16,3))
        self.assertEqual(decoded['gripper'].shape, (1,16,1))

    def test_real_normalization_roundtrips_and_clips_both_native_states(self):
        sys.path.insert(0, os.environ['RELIC_GROOT_SOURCE'])
        import numpy as np
        from gr00t.data.state_action.state_action_processor import StateActionProcessor
        from embodiment.groot_modalities import RELIC_CONFIG
        for robot in ('openmanipulator_x', 'franka_panda'):
            _, statistics = fixture(robot)
            processor = StateActionProcessor({'new_embodiment': RELIC_CONFIG}, statistics,
                                             use_percentiles=True, use_relative_action=True)
            processor.eval()
            state = {name: np.full((1, len(stats['mean'])), .5, dtype=np.float32)
                     for name, stats in statistics['new_embodiment']['state'].items()}
            action = {'delta_position': np.full((16, 3), .25, dtype=np.float32),
                      'gripper': np.full((16, 1), .75, dtype=np.float32)}
            normalized_state, normalized_action = processor.apply(state, action, 'new_embodiment')
            restored_state = processor.unapply_state(normalized_state, 'new_embodiment')
            restored_action = processor.unapply_action(normalized_action, 'new_embodiment', state=state)
            for original, restored in ((state, restored_state), (action, restored_action)):
                for name in original: np.testing.assert_allclose(restored[name], original[name], atol=1e-6)
            action['gripper'][:, 0] = 2.
            normalized = processor.apply_action(action, 'new_embodiment', state=state)
            self.assertTrue((normalized['gripper'] <= 1.).all())

    def test_real_projector_uses_fixed_capacity_without_reinterpreting_joints(self):
        sys.path.insert(0, os.environ['RELIC_GROOT_SOURCE'])
        import torch
        from pathlib import Path
        import importlib.util
        from embodiment.upstream_validation import verify_source
        source = Path(os.environ['RELIC_GROOT_SOURCE'])
        verify_source(source)
        spec = importlib.util.spec_from_file_location('verified_upstream_mlp', source/'gr00t/model/modules/embodiment_conditioned_mlp.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        CategorySpecificMLP = module.CategorySpecificMLP
        projector = CategorySpecificMLP(11, 29, 4, 4)
        self.assertEqual(tuple(projector.layer1.W.shape), (11, 29, 4))
        for native_dim in (16, 22):
            normalized = torch.ones((1, 1, native_dim))
            model_input = torch.nn.functional.pad(normalized, (0, 29-native_dim))
            self.assertEqual(projector(model_input, torch.tensor([10])).shape, (1, 1, 4))
            self.assertEqual(model_input[..., native_dim:].count_nonzero().item(), 0)
