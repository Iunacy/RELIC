import unittest
from embodiment.paired import bootstrap
from embodiment.experiment import cases


class ReferenceComparison(unittest.TestCase):
    def test_classical_success_is_reference_success_without_counting_as_policy(self):
        reference = [{'scenario_id': str(i), 'controller': 'classical_only', 'success': True} for i in range(4)]
        policy = [{'scenario_id': str(i), 'controller': 'vla_only', 'success': False} for i in range(4)]
        result = bootstrap(reference, policy, baseline_metric='classical_controller_completion')
        self.assertEqual(result['difference'], -1.)
        self.assertEqual(result['paired_bootstrap_95'], [-1., -1.])
        self.assertEqual(bootstrap(reference, policy)['difference'], 0.)

    def test_classical_primary_schedule_matches_fifty_learned_scenes(self):
        from embodiment.scenarios import plan
        self.assertEqual([case['scenario_id'] for case in cases('heldout', 50)], [case.scenario_id for case in plan('evaluation', 50)])
