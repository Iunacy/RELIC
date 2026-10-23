import json
from pathlib import Path
import tempfile
import unittest

from embodiment.collection import collect
from embodiment.inventory import inventory
from embodiment.tests import test_collection


class DatasetInventory(unittest.TestCase):
    def test_counts_only_verified_successes_and_preserves_failed_attempts(self):
        helper = test_collection.CollectionRecovery()
        with tempfile.TemporaryDirectory() as temporary, helper.mocks():
            root = Path(temporary)/'collection'
            summary = collect(root, 'openmanipulator_x', 'train', 3, batch_size=2, runner=helper.fake_runner)
            result = inventory([root])
            self.assertEqual(result['status'], 'VALIDATED')
            self.assertEqual(result['counts']['openmanipulator_x']['train'], summary['successful_episodes'])
            self.assertEqual(result['counts']['franka_panda']['train'], 0)
            record = result['collections'][0]
            self.assertEqual(len(record['failures']), summary['failed_or_incomplete_attempts'])
            self.assertTrue(record['seed_range'])
            self.assertFalse(record['initial_target_complete'])
            self.assertTrue(all(item['sha256'] for item in record['episodes']))

    def test_duplicate_roots_and_corrupt_manifests_block_inventory(self):
        helper = test_collection.CollectionRecovery()
        with tempfile.TemporaryDirectory() as temporary, helper.mocks():
            root = Path(temporary)/'collection'
            collect(root, 'openmanipulator_x', 'train', 1, runner=helper.fake_runner)
            duplicate = inventory([root, root])
            self.assertEqual(duplicate['status'], 'BLOCKED')
            self.assertEqual(duplicate['counts']['openmanipulator_x']['train'], 1)
            configuration = json.loads((root/'collection.json').read_text())
            configuration['schema'] = 'invalid'
            (root/'collection.json').write_text(json.dumps(configuration))
            self.assertEqual(inventory([root])['status'], 'BLOCKED')
