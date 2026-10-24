import importlib.util
import math
import os
from pathlib import Path
import unittest
from embodiment.kinematics import Kinematics, audit, audit_dynamics
from embodiment.openmanipulator import fk, ik


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required')
class UrdfKinematics(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('VIPER_SIMULATOR_AUDIT'), 'set VIPER_SIMULATOR_AUDIT for real CPU dynamics')
    def test_manufacturer_matrix_repairs_omx_self_contacts_without_removing_floor_probe(self):
        raw = audit_dynamics('openmanipulator_x', manufacturer_filters=False)
        filtered = audit_dynamics('openmanipulator_x')
        self.assertFalse(raw['passed'])
        self.assertTrue(raw['robot_contact_pairs'])
        self.assertTrue(filtered['passed'])
        self.assertLess(filtered['endpoint_joint_error_rad'], .003)
        self.assertGreater(filtered['maximum_cube_floor_normal_force_n'], .05)
        self.assertTrue(audit_dynamics('franka_panda')['passed'])

    def test_omx_chain_matches_existing_analytic_kinematics(self):
        chain = Kinematics('openmanipulator_x')
        for position in ((.17, 0., .085), (.18, -.04, .06), (.14, .145, .085)):
            q = ik(position)
            self.assertLess(math.dist(chain.forward(q)[:3, 3], fk(q)), 1e-10)
        with self.assertRaises(ValueError): chain.forward((0.,)*7)
        with self.assertRaises(ValueError): chain.forward((9.,)*4)

    @unittest.skipUnless(os.environ.get('VIPER_SIMULATOR_AUDIT'), 'explicit CPU simulator audit with pinned assets required')
    def test_both_official_assets_against_actual_mujoco(self):
        for robot in ('openmanipulator_x', 'franka_panda'):
            result = audit(robot)
            self.assertTrue(result['passed'])
            self.assertFalse(result['dynamics_executed'])
            self.assertFalse(result['manipulation_qualified'])
            self.assertGreater(result['collision_geometries'], 0)
