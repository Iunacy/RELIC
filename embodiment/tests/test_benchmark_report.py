import copy
from dataclasses import asdict
import unittest
from embodiment.benchmark import BIN, SCENE_REVISION
from embodiment.benchmark_report import validate_run
from embodiment.scenarios import plan


def fixture():
    return {'schema':'relic.evaluation.v1','scene_revision':SCENE_REVISION,
            'evidence_source':'isaac_lab_physx','simulation_executed':True,'completed':True,
            'source_changed_during_run':[],'embodiment':'franka_panda','condition':'classical',
            'runtime_pin':{'fixture':True},'source_sha256':{'fixture':'test'},'fallback_protocol_sha256':'fixture',
            'trials':[{**asdict(case),'success':True,'simulation_executed':True,'duration':20.,
                       'controller':'classical_only','fallback_triggers':[],'safety_events':[],
                       'failure_reason':None,'pick_result':True,'placement_result':True,
                       'final_object_pose':[BIN[0],BIN[1],BIN[2]+.0125,1,0,0,0]}
                      for case in plan('evaluation',20)]}


class BenchmarkReport(unittest.TestCase):
    def test_counts_and_intervals(self):
        result = validate_run(fixture())
        self.assertEqual(result['successes'],20)
        self.assertEqual(result['successes_by_controller']['classical_only'],20)
        self.assertLess(result['wilson_95'][0],1)
        self.assertEqual(result['wrong_color_rate'], 0.)
        self.assertEqual(result['mean_success_placement_error_xy_m'], 0.)

    def test_rejects_incomplete_duplicate_unsafe_and_false_success(self):
        for change in ('partial','duplicate','unsafe','outside','no_pick','not_physical'):
            run = fixture()
            if change == 'partial': run['completed'] = False
            if change == 'duplicate': run['trials'][1] = copy.deepcopy(run['trials'][0])
            if change == 'unsafe': run['trials'][0]['safety_events'] = ['joint_limit']
            if change == 'outside': run['trials'][0]['final_object_pose'][0] = .3
            if change == 'no_pick': run['trials'][0]['pick_result'] = False
            if change == 'not_physical': run['evidence_source'] = 'fixture'
            with self.subTest(change=change), self.assertRaises(ValueError): validate_run(run)

    def test_fallback_cannot_count_as_vla_only(self):
        run = fixture()
        run['condition'] = 'b_only'
        for row in run['trials']: row['controller'] = 'vla_only'
        run['trials'][0].update(controller='fallback_assisted',fallback_triggers=['progress_stall'])
        result = validate_run(run)
        self.assertEqual(result['successes_by_controller']['vla_only'],19)
        self.assertEqual(result['successes_by_controller']['fallback_assisted'],1)
        self.assertEqual(result['success_rates_by_controller']['vla_only'], .95)
        self.assertEqual(result['fallback_trigger_counts'], {'progress_stall': 1})
        self.assertLess(result['wilson_95_by_controller']['vla_only'][0], .95)
        run['trials'][0]['controller'] = 'vla_only'
        with self.assertRaises(ValueError): validate_run(run)
