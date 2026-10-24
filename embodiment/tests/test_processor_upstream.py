import os
from pathlib import Path
import json
import sys
import tempfile
import unittest

from embodiment.checkpoint_compatibility import validate_processor
from embodiment.tests.test_checkpoint_compatibility import fixture


@unittest.skipUnless(os.environ.get('RELIC_GROOT_SOURCE'), 'set RELIC_GROOT_SOURCE for real serialization checks')
class UpstreamProcessorSerialization(unittest.TestCase):
    def test_actual_launcher_parses_resume_as_boolean(self):
        from embodiment.training import Budget, training_arguments, validate_upstream_arguments
        arguments = training_arguments(Path('/data'), '/weights', Path('/run'), Budget(steps=2, batch_size=1, save_steps=2))
        result = validate_upstream_arguments(Path(os.environ['RELIC_GROOT_SOURCE']), arguments+['--resume-from-checkpoint'])
        self.assertIs(result['resume_from_checkpoint'], True)
        self.assertIs(result['save_only_model'], False)

    def test_actual_enum_serialization_matches_checkpoint_gate(self):
        sys.path.insert(0, os.environ['RELIC_GROOT_SOURCE'])
        from gr00t.data.utils import to_json_serializable
        from embodiment.groot_modalities import RELIC_CONFIG
        config, statistics = fixture('franka_panda')
        config['processor_kwargs']['modality_configs']['new_embodiment'] = to_json_serializable(RELIC_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'processor_config.json').write_text(json.dumps(config))
            (root/'statistics.json').write_text(json.dumps(statistics))
            self.assertEqual(validate_processor(root)['joint_count'], 9)
