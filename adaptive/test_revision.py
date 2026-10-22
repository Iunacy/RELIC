#!/usr/bin/env python3
import json
from pathlib import Path
import unittest
import sys
import tempfile
import shutil
import hashlib
import revision as r
import learning as l
import revision_geometry as geometry

EXE=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else None
sys.argv=sys.argv[:1]

class RevisionTests(unittest.TestCase):
    def test_split_independent_of_embodiment(self):
        value=r.protocol()
        families={}
        for m in value['specs']:
            entry=families.setdefault(m['family'],dict(partition=m['partition'],dofs=set(),categories=set()))
            self.assertEqual(entry['partition'],m['partition'])
            entry['dofs'].add(m['dof']);entry['categories'].add(m['category'])
        self.assertTrue(all(v['dofs']=={3,4} and v['categories']==set(r.CATEGORIES) for v in families.values()))
        self.assertEqual(sum(v['partition'].endswith('_test') for v in families.values()),40)
        self.assertEqual(sum(m['partition']=='train' for m in value['specs']),1024)
        self.assertEqual(value,r.protocol())
        old=json.loads((r.ROOT/'adaptive/fixtures/episode/metadata.json').read_text()) if (r.ROOT/'adaptive/fixtures/episode/metadata.json').exists() else None
        if old:self.assertNotIn(old['family'],families)

    def test_geometry_leakage_audit(self):
        audit=geometry.audit(r.protocol())
        self.assertEqual(audit['embodied_families'],224)
        self.assertGreater(audit['minimum_endpoint_max_norm_rad'],.05)
        value=r.protocol()
        duplicate=dict(value['specs'][0],partition='severity_test',dof=3)
        value['specs'].append(duplicate)
        with self.assertRaises(AssertionError):geometry.audit(value)

    @unittest.skipIf(EXE is None,'native executable argument required')
    def test_interrupted_teacher_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shutil.copy2(EXE,root/'generator-pinned')
            directory=root/'resume';directory.mkdir()
            spec=r.protocol()['specs'][0]
            first=r.teacher_task((directory,spec,1,0))
            folder=directory/'teachers'/spec['episode']
            identities={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.rglob('*.gz')}
            (folder/'result.json').rename(folder/'result-before-interruption.json')
            broken=folder/'attempt_1';broken.mkdir();(broken/'telemetry.csv').write_text('interrupted')
            second=r.teacher_task((directory,spec,2,0))
            self.assertEqual(first['attempts'][0],second['attempts'][0])
            self.assertEqual(len(second['attempts']),2)
            self.assertEqual(len(list(folder.glob('attempt_1-interrupted-*'))),1)
            for path,digest in identities.items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),digest)

    def test_causality(self):
        m=dict(dof=3,joint_descriptors=list(range(24)))
        row=dict(joint='1',q_des_rad='.2',v_des_analytic_rad_s='.1',a_des_analytic_rad_s2='.3',q_pre_rad='.1',v_pre_rad_s='.2',previous_error_rad='.01',error_rad='999',q_actual_rad='999',correction_rad='999',future_q='999',gain_scale='999')
        before=l.local_features(m,row,.02,.03)
        for key in ['error_rad','q_actual_rad','correction_rad','future_q','gain_scale']:row[key]='-999'
        self.assertEqual(before,l.local_features(m,row,.02,.03))
        row['q_pre_rad']='.4'
        self.assertNotEqual(before,l.local_features(m,row,.02,.03))

    def test_acceptance_is_not_training_loss(self):
        best=dict(metrics=dict(joint_rmse_rad=.02,final_cartesian_m=.001))
        trial=dict(outcome='Success',completed_ticks=100,planned_ticks=100,fallbacks=0,metrics=dict(contact_observations=0,joint_rmse_rad=.01,final_cartesian_m=.001))
        self.assertTrue(l.acceptance(trial,best)[0])
        trial['fallbacks']=1
        self.assertFalse(l.acceptance(trial,best)[0])

if __name__=='__main__':unittest.main()
