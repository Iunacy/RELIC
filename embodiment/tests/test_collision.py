import importlib.util
import unittest
from embodiment.collision import pairs, apply_usd


class ManufacturerCollisionMatrix(unittest.TestCase):
    def test_pinned_matrix_covers_the_observed_self_contact_pairs(self):
        disabled = {(first, second) for first, second, reason in pairs()}
        self.assertIn(('link1', 'link2'), disabled)
        self.assertIn(('link1', 'link3'), disabled)
        self.assertEqual(len(disabled), 10)
        self.assertTrue(all('World' not in name and 'cube' not in name for pair in disabled for name in pair))

    @unittest.skipUnless(importlib.util.find_spec('pxr'), 'OpenUSD authoring unavailable')
    def test_real_usd_api_authors_only_declared_self_pair_relationships(self):
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim('/World/Robot')
        for name in {name for first, second, reason in pairs() for name in (first, second)}:
            stage.DefinePrim('/World/Robot/'+name)
        result = apply_usd(stage)
        self.assertEqual(len(result['pairs']), 10)
        for row in result['pairs']:
            relation = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(row['first'])).GetFilteredPairsRel()
            self.assertIn(row['second'], [str(path) for path in relation.GetTargets()])
