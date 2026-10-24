from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from embodiment.collection import atomic_json
from embodiment.tests import test_training_smoke as fixtures
from embodiment.training import Budget
from embodiment.training_campaign import run, restore_final_snapshot


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('numpy', 'safetensors')), 'safe checkpoint dependencies required')
class TrainingCampaign(unittest.TestCase):
    def test_completed_checkpoint_recovers_missing_final_snapshot_without_an_extra_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/'checkpoint-2'
            checkpoint.mkdir()
            fixtures.TrainingSmoke().checkpoint(checkpoint)
            result = restore_final_snapshot(checkpoint, root)
            self.assertEqual(result['weight_files'], 1)
            self.assertTrue((root/'processor_config.json').is_file())
            self.assertEqual(restore_final_snapshot(checkpoint, root), result)
            (root/'config.json').unlink()
            (root/'config.json').write_text('{"conflict":true}')
            with self.assertRaisesRegex(ValueError, 'disagree'):
                restore_final_snapshot(checkpoint, root)

    def test_plan_is_zero_compute_and_real_smoke_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root/'weights'
            weights.mkdir()
            fixtures.TrainingSmoke().checkpoint(weights)
            fixture = {'datasets': {'a': {'digest': 'a'}, 'b': {'digest': 'b'}}, 'runs': []}
            provenance = {'relic_worktree_dirty': False, 'source_sha256': {'fixture': 'unit'}, 'relic_commit': 'fixture'}
            args = (root/'source', root/'a', root/'b', weights, root/'run', '/sim/python', root/'g3')
            with patch('embodiment.training_campaign.prepare', return_value=fixture), \
                 patch('embodiment.training_campaign.source_provenance', return_value=provenance), \
                 patch('embodiment.training_campaign.shutil.disk_usage', return_value=type('Usage', (), {'free': 100*1024**3})()), \
                 patch('embodiment.training_campaign.execute', side_effect=AssertionError('compute forbidden')):
                self.assertEqual(run(*args)['status'], 'NOT_EXECUTED')
                (root/'g3').mkdir()
                atomic_json(root/'g3/result.json', {'status': 'PREFLIGHT_ONLY'})
                atomic_json(root/'g3/configuration.json', {})
                with self.assertRaisesRegex(ValueError, 'real G3'):
                    run(*args, 'train_a', True)
                with self.assertRaisesRegex(ValueError, 'inputs changed'):
                    run(*args, budget=Budget(steps=1000))
