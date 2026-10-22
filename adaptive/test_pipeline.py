import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pipeline as p

EXE, EXAMPLES = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.argv = sys.argv[:1]

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.config = json.loads((p.ROOT / 'adaptive/campaign.json').read_text())
        self.identity = dict(config=self.config, executable_sha256=p.digest(EXE), source_tree_sha256=p.source_hash(), source_dirty=True)
        p.atomic_json(self.path/'campaign.json', self.identity)

    def tearDown(self):
        self.temp.cleanup()

    def generate(self, index):
        spec = p.spec(index, self.config)
        p.generate_one(EXE, EXAMPLES, self.path, spec, self.identity)
        return self.path/spec['episode']

    def test_generation_resume_and_repeatability(self):
        folder = self.generate(40)
        m = p.validate_episode(folder)
        self.assertEqual(m['completed_ticks'], m['planned_ticks'])
        initial = p.digest(folder/'telemetry.csv.gz')
        self.assertFalse(p.generate_one(EXE, EXAMPLES, self.path, p.spec(40, self.config), self.identity)[0])
        import shutil
        shutil.rmtree(folder)
        self.generate(40)
        self.assertEqual(initial, p.digest(folder/'telemetry.csv.gz'))

        (self.path/'.partial-interrupted').mkdir()
        (self.path/'.partial-interrupted'/'telemetry.csv').write_text('partial')
        self.assertEqual(p.validate_dataset(self.path)['episodes'], 1)
        identity = copy.deepcopy(self.identity)
        identity['executable_sha256'] = 'wrong'
        with self.assertRaises(AssertionError):
            p.generate_one(EXE, EXAMPLES, self.path, p.spec(40, self.config), identity)

    def test_faults_and_corruption(self):
        nominal = p.validate_episode(self.generate(40))
        faulty_folder = self.generate(41)
        faulty = p.validate_episode(faulty_folder)
        self.assertLess(faulty['physics']['actuator_kp'][0], nominal['physics']['actuator_kp'][0])
        trace = faulty_folder/'telemetry.csv.gz'
        content = gzip.decompress(trace.read_bytes())
        trace.write_bytes(gzip.compress(content.replace(b',yaw,', b',wrong,', 1)))
        with self.assertRaises(AssertionError):
            p.validate_episode(faulty_folder, check_hash=False)

    def test_missing_samples_and_metadata(self):
        folder = self.generate(42)
        trace = folder/'telemetry.csv.gz'
        lines = gzip.decompress(trace.read_bytes()).splitlines(keepends=True)
        trace.write_bytes(gzip.compress(b''.join(lines[:4]+lines[5:])))
        with self.assertRaises(AssertionError):
            p.validate_episode(folder, check_hash=False)

    def test_canonical_fixture_import(self):
        meta = p.validate_episode(p.ROOT/'adaptive/fixtures/episode')
        self.assertEqual(meta['controller'], 'conventional')
        self.assertTrue(meta['task_success'])

    def test_splits(self):
        assignments = {}
        for i in range(10000):
            s = p.spec(i, self.config)
            self.assertEqual(assignments.setdefault(s['family'], s['partition']), s['partition'])
            if s['partition'] == 'severity_test':
                self.assertGreaterEqual(s['severity'], .85)
            elif s['partition'] == 'train':
                self.assertLess(s['severity'], .85)
        folder = self.generate(40)
        m = json.loads((folder/'metadata.json').read_text())
        m['partition'] = 'trajectory_test'
        p.atomic_json(folder/'metadata.json', m)
        with self.assertRaises(AssertionError):
            p.validate_dataset(self.path, full=False)

if __name__ == '__main__':
    unittest.main()
