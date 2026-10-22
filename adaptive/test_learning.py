import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import learning as l
import pipeline as p

EXE, EXAMPLES, PROBE = (Path(v).resolve() for v in sys.argv[1:4])
sys.argv = sys.argv[:1]


def model_text(bounds, embodiment=0):
    mag, rate = l.bound_values(bounds)
    arrays = [mag, rate, [0.]*16, [1.]*16, [.01*math.sin(i) for i in range(32*16)], [.01]*32,
              [.01*math.cos(i) for i in range(32*32)], [-.01]*32, [.02]*32, [.05]]
    return f'RELIC_MLP_V1 relic.local.v1 16 32 32 {embodiment}\n'+'\n'.join(' '.join(format(v, '.17g') for v in a) for a in arrays)+'\n', arrays


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.bounds = self.out/'bounds.txt'
        self.bounds.write_text('RELIC_BOUNDS_V1 4\n'+'.1 .001\n'*4)
        cfg = json.loads((p.ROOT/'adaptive/campaign.json').read_text())
        self.spec = p.spec(40, cfg)
        self.spec.update(joint_descriptors=[0., 0., 1., -2.8, 2.8, 0., 0., .75]*3)

    def tearDown(self):
        self.temp.cleanup()

    def test_native_python_inference_parity_and_loading(self):
        model = self.out/'model.txt'
        text, arrays = model_text(self.bounds)
        model.write_text(text)
        for seed in range(6):
            x = [math.sin(seed+i) for i in range(16)]
            a = [math.tanh(arrays[5][i]+sum(arrays[4][16*i+j]*x[j] for j in range(16))) for i in range(32)]
            b = [math.tanh(arrays[7][i]+sum(arrays[6][32*i+j]*a[j] for j in range(32))) for i in range(32)]
            expected = math.tanh(arrays[9][0]+sum(arrays[8][i]*b[i] for i in range(32)))*.1
            value = subprocess.check_output([str(PROBE), str(model), '0']+[repr(v) for v in x], text=True)
            self.assertAlmostEqual(float(value), expected, places=14)
        for broken in [text.replace('RELIC_MLP_V1', 'RELIC_MLP_V2'), text.replace(' 16 32 32 ', ' 17 32 32 '), text+'garbage', text.replace('.01', 'nan', 1), text[:100]]:
            model.write_text(broken)
            self.assertNotEqual(subprocess.run([str(PROBE), str(model), '0']+['0']*16, capture_output=True).returncode, 0)

    def test_label_acceptance_rejection(self):
        base = dict(outcome='Success', completed_ticks=10, planned_ticks=10, fallbacks=0,
                    metrics=dict(contact_observations=0, joint_rmse_rad=.1, final_cartesian_m=.002))
        good = copy.deepcopy(base)
        good['metrics']['joint_rmse_rad'] = .05
        self.assertEqual(l.acceptance(good, base), (True, 'accepted'))
        self.assertEqual(l.acceptance(base, base), (False, 'no_rmse_improvement'))
        bad = copy.deepcopy(good)
        bad['metrics']['final_cartesian_m'] = .01
        self.assertEqual(l.acceptance(bad, base), (False, 'final_error_regressed'))
        bad = copy.deepcopy(good)
        bad['fallbacks'] = 1
        self.assertEqual(l.acceptance(bad, base), (False, 'unsafe_or_incomplete'))
        bad = copy.deepcopy(good)
        bad['completed_ticks'] = 9
        self.assertEqual(l.acceptance(bad, base), (False, 'unsafe_or_incomplete'))

    def test_causality(self):
        row = dict(joint='1', q_des_rad='.2', v_des_analytic_rad_s='.1', a_des_analytic_rad_s2='.3',
                   q_pre_rad='.15', v_pre_rad_s='.05', previous_error_rad='.04', q_actual_rad='.16', error_rad='.04')
        original = l.local_features(self.spec, row, .03, .02)
        poisoned = copy.deepcopy(row)
        poisoned.update(q_actual_rad='100000', v_actual_rad_s='-999', error_rad='nan', actuator_force_Nm='12345')
        hidden = copy.deepcopy(self.spec)
        hidden.update(physics={'mass_scale': 999, 'gain_scale': -1}, goal_q_rad=[999]*3, seed=-1)
        self.assertEqual(original, l.local_features(hidden, poisoned, .03, .02))
        self.assertEqual(len(original), 16)

        folder = self.out/'base'
        l.run_native(EXE, EXAMPLES, folder, self.spec, 'conventional', self.bounds)
        before = list(l.examples_from_trace(folder))
        rows = l.read_rows(folder)
        cutoff = 100
        for row in rows:
            if int(row['tick']) > cutoff:
                row['q_pre_rad'] = '123'
                row['previous_error_rad'] = '456'
        reader = l.read_rows
        try:
            l.read_rows = lambda _: rows
            after = list(l.examples_from_trace(folder))
        finally:
            l.read_rows = reader
        prefix = cutoff//4*self.spec['dof']
        self.assertEqual(before[:prefix], after[:prefix])
        self.assertNotEqual(before[prefix:], after[prefix:])

    def test_fallback_and_closed_loop_repeatability(self):
        bad = self.out/'invalid.txt'
        bad.write_text('wrong-version')
        base = l.run_native(EXE, EXAMPLES, self.out/'baseline', self.spec, 'conventional', self.bounds)
        fallback = l.run_native(EXE, EXAMPLES, self.out/'fallback', self.spec, 'mlp', self.bounds, bad)
        self.assertGreater(fallback['fallbacks'], 0)
        self.assertEqual(fallback['last_fallback_reason'], 'model_version_or_dimension')
        self.assertEqual(base['metrics'], fallback['metrics'])
        analytical = l.run_native(EXE, EXAMPLES, self.out/'analytical', self.spec, 'analytical', self.bounds)
        repeat = l.run_native(EXE, EXAMPLES, self.out/'repeat', self.spec, 'analytical', self.bounds)
        self.assertEqual(analytical['trace_sha256'], repeat['trace_sha256'])
        self.assertLess(analytical['metrics']['joint_rmse_rad'], base['metrics']['joint_rmse_rad'])
        model = self.out/'per3.txt'
        model.write_text(model_text(self.bounds, embodiment=3)[0])
        unsupported = l.run_native(EXE, EXAMPLES, self.out/'unsupported', self.spec, 'mlp', self.bounds, model)
        self.assertEqual(unsupported['last_fallback_reason'], 'unsupported_embodiment')

if __name__ == '__main__':
    unittest.main()
