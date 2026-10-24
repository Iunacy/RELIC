from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from embodiment.benchmark import Action
from embodiment.manipulation import TaskController, cases, campaign, summarize
from embodiment.tests.test_execution import observation


class ManipulationProtocol(unittest.TestCase):
    def test_frozen_tasks_starts_and_perturbations(self):
        planned = cases(2, starts=('home', 'left', 'right'), perturbed=True)
        self.assertEqual(len(planned), 30)
        self.assertEqual(planned, cases(2, starts=('home', 'left', 'right'), perturbed=True))
        self.assertEqual(len({case['scenario_id'] for case in planned}), len(planned))
        for args in ({'count': 100}, {'tasks': ('unknown',)}, {'starts': ('x',)}):
            with self.assertRaises(ValueError): cases(**args)
        with tempfile.TemporaryDirectory() as temporary:
            result = campaign(Path(temporary))
            self.assertEqual(result['status'], 'NOT_EXECUTED')
            self.assertFalse(result['simulation_executed'])

    def test_reach_requires_arrival_velocity_and_dwell(self):
        controller = TaskController('reach', 'openmanipulator_x', 'red')
        first = replace(observation(), ee=(.17, 0., .12))
        controller.update(first)
        self.assertFalse(controller.success)
        controller.update(replace(first, time=.6))
        self.assertTrue(controller.success)

    def test_recovery_is_bounded_and_explicitly_classical(self):
        controller = TaskController('recover_grasp', 'openmanipulator_x', 'red')
        controller.expert.phase = 'verify'
        controller.expert.entered = 0.
        obs = observation(.5)
        controller.update(obs)
        self.assertTrue(controller.recovering)
        self.assertEqual(controller.interventions[0]['reason'], 'classical_grasp_retry')
        controller.update(replace(obs, time=.6, ee=(obs.ee[0], obs.ee[1], .085)))
        controller.update(replace(obs, time=1.2, ee=(obs.ee[0], obs.ee[1], .085)))
        self.assertFalse(controller.recovering)
        controller.expert.phase = 'verify'
        controller.expert.entered = 1.2
        controller.update(replace(obs, time=1.8))
        self.assertEqual(controller.failure, 'failed_grasp')
        self.assertEqual(len(controller.attempts), 2)

    def test_unsafe_state_never_recovers(self):
        controller = TaskController('recover_grasp', 'franka_panda', 'red')
        controller.update(replace(observation(), unsafe=True))
        self.assertEqual(controller.failure, 'unsafe_movement')
        self.assertEqual(controller.interventions, [])

    def test_missing_infrastructure_results_stay_in_denominator(self):
        assigned = cases(tasks=('reach',))
        rows = [{**assigned[0], 'robot': robot, 'success': False, 'failure_reason': 'infrastructure_failure',
                 'infrastructure_error': 'unit_fixture'} for robot in ('openmanipulator_x', 'franka_panda')]
        result = summarize(rows, assigned)
        self.assertTrue(result['completed'])
        self.assertEqual(result['by_robot']['franka_panda']['trials'], 1)
        self.assertEqual(result['by_robot']['franka_panda']['success_rate'], 0)
        with self.assertRaises(ValueError): summarize(rows+rows, assigned)
