import unittest
from embodiment.scene import Scene, KINDS, variation
from embodiment.manipulation import cases


class PhysicalSceneConfiguration(unittest.TestCase):
    def test_variations_are_bounded_deterministic_and_sealed_into_scenarios(self):
        for kind in KINDS:
            scene = variation(kind, 7)
            self.assertEqual(scene, variation(kind, 7))
            self.assertEqual(Scene(**scene.manifest()).manifest(), scene.manifest())
            planned = cases(physical_variation=kind, variation_seed=7)
            self.assertTrue(all(case['physical_scene'] == scene.manifest() for case in planned))
        self.assertNotEqual(cases()[0]['scenario_id'], cases(physical_variation='geometry')[0]['scenario_id'])
        self.assertEqual(variation('geometry').geometry, 'cylinder')
        for parameters in ({'geometry':'unknown'}, {'mass_kg':1.}, {'static_friction':.2}, {'camera_eye':(0.,0.,0.)}):
            with self.assertRaises(ValueError): Scene(**parameters)
