"""Offline checks only. Synthetic observations here never qualify the robot."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from embodiment.assets import MANIFEST, ROOT, URDF, fetch, validate_urdf
from embodiment.benchmark import Action, BIN, COLORS, CUBE, GRASP_Z, Observation, inside_bin, layout
from embodiment.expert import Expert
from embodiment.openmanipulator import bounded_reference, fk, ik
from embodiment.qualify import aggregate
from embodiment.recording import Recorder, lerobot_rows, validate_sample


class Assets(unittest.TestCase):
    def test_official_source(self):
        key = 'open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf'
        self.assertEqual(hashlib.sha256(URDF.read_bytes()).hexdigest(),MANIFEST['files'][key])
        self.assertEqual(len(validate_urdf()),8)
        self.assertIn('Apache License',(ROOT/'assets/LICENSE').read_text())

    def test_invalid_mapping_limits_mimic_mass(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.urdf'
            for xpath,attribute,value in [('joint','name','unknown'),
                    ("joint[@name='joint1']/limit",'lower','nan'),
                    ("joint[@name='joint1']/limit",'upper','-8'),
                    ("joint[@name='gripper_right_joint']/mimic",'multiplier','-1'),
                    ("joint[@name='world_fixed']",'type','revolute'),
                    ("link[@name='link1']/inertial/mass",'value','-1'),
                    ("link[@name='link1']/inertial/mass",'value','nan'),
                    ("link[@name='link1']/inertial/inertia",'ixy','1')]:
                tree = ET.parse(URDF)
                tree.find(xpath).set(attribute,value)
                tree.write(path)
                with self.subTest(xpath=xpath,attribute=attribute), self.assertRaises(ValueError):
                    validate_urdf(path)

    def test_cache_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'LICENSE').write_text('changed')
            with self.assertRaisesRegex(ValueError,'cached asset changed'):
                fetch(temp)


class Geometry(unittest.TestCase):
    def test_seeded_workspace_and_tool_paths(self):
        for seed in range(2000):
            scene = layout(seed)
            self.assertEqual(scene,layout(seed))
            self.assertEqual(set(scene),set(COLORS))
            poses = list(scene.values())
            self.assertTrue(all(math.dist(poses[i],poses[j])>=.055 for i in range(3) for j in range(i)))
            for p in poses:
                for z in (GRASP_Z,.085):
                    q = ik((*p[:2],z))
                    self.assertLess(math.dist(fk(q),(*p[:2],z)),1e-9)
                    self.assertAlmostEqual(sum(q[1:]),math.pi/2)
                # Cartesian transport interpolates along a safe-height segment.
                for i in range(11):
                    t = i/10
                    point = (p[0]*(1-t)+BIN[0]*t,p[1]*(1-t)+BIN[1]*t,.085)
                    self.assertLess(math.dist(fk(ik(point)),point),1e-9)
        for z in (.065,.085):
            self.assertLess(math.dist(fk(ik((*BIN[:2],z))),(*BIN[:2],z)),1e-9)

    def test_fk_against_independent_urdf_chain(self):
        joints = validate_urdf()
        def multiply(a,b):
            return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
        def identity():
            return [[float(i==j) for j in range(4)] for i in range(4)]
        for pose in ((.17,0,GRASP_Z),(.17,-.06,.085),(*BIN[:2],.065)):
            values=dict(zip(('joint1','joint2','joint3','joint4'),ik(pose)))
            result=identity()
            for name in ('world_fixed','joint1','joint2','joint3','joint4','end_effector_joint'):
                joint=joints[name]
                origin=joint.find('origin')
                self.assertEqual([float(v) for v in origin.attrib['rpy'].split()],[0,0,0])
                translation=identity()
                for i,v in enumerate(map(float,origin.attrib['xyz'].split())): translation[i][3]=v
                result=multiply(result,translation)
                if name in values:
                    x,y,z=map(float,joint.find('axis').attrib['xyz'].split())
                    angle=values[name]
                    c,t,s=math.cos(angle),1-math.cos(angle),math.sin(angle)
                    rotation=[[t*x*x+c,t*x*y-s*z,t*x*z+s*y,0],
                              [t*x*y+s*z,t*y*y+c,t*y*z-s*x,0],
                              [t*x*z-s*y,t*y*z+s*x,t*z*z+c,0],[0,0,0,1]]
                    result=multiply(result,rotation)
            self.assertLess(math.dist([result[i][3] for i in range(3)],pose),1e-9)
            self.assertAlmostEqual(result[2][0],-1)
        # From the official palm mesh: origin .0817 + tip .065 - tool .126.
        self.assertGreaterEqual(GRASP_Z-(.0817+.065-.126),.003)

    def test_invalid_geometry_actions(self):
        for p in ((float('nan'),0,0),(.8,0,.1),(.25,0,.19),(.1,)):
            with self.subTest(p=p),self.assertRaises(ValueError):
                ik(p)
        for delta,grip in (((.004,0,0),1),((0,0,0),2),((0,0,float('nan')),0),((0,0),0),((True,0,0),0)):
            with self.subTest(delta=delta),self.assertRaises(ValueError):
                Action(delta,grip)
        for seed in (-1,True,'2'):
            with self.assertRaises(ValueError):
                layout(seed)

    def test_velocity_reference(self):
        current = ik((.17,0,.085))
        goal = ik((.19,-.05,.02))
        cmd = bounded_reference(current,goal,1/30)
        self.assertTrue(all(abs(a-b)<=.5/30+1e-12 for a,b in zip(cmd,current)))
        with self.assertRaises(ValueError):
            bounded_reference(current,(8,0,0,0),.03)
        with self.assertRaises(ValueError):
            bounded_reference(current,goal,0)

    def test_bin_requires_complete_cube(self):
        self.assertTrue(inside_bin((*BIN[:2],BIN[2]+CUBE/2,1,0,0,0)))
        for pose in ((BIN[0]+.025,BIN[1],.02,1,0,0,0),(*BIN[:2],.15,1,0,0,0),
                     (*BIN[:2],.006,1,0,0,0), (BIN[0]+.014,BIN[1],.02,math.cos(math.pi/8),0,0,math.sin(math.pi/8))):
            self.assertFalse(inside_bin(pose))
        with self.assertRaises(ValueError):
            inside_bin((*BIN[:2],.02,2,0,0,0))


def observation(time=0,**kwargs):
    cubes = {c:(*p,1,0,0,0) for c,p in layout(0).items()}
    return Observation(time=time,ee=(.17,0,.085),joints=(0,0,0,0,0,0),
                       velocities=(0,)*6,cubes=cubes,cube_speeds={c:0 for c in COLORS},**kwargs)


class Controller(unittest.TestCase):
    def test_distinct_failures(self):
        cases = [('unsafe_movement','approach',dict(unsafe=True)),('timeout','approach',{}),
                 ('wrong_color','approach',{}),('failed_grasp','verify',{}),('dropped_object','transport',{})]
        for reason,phase,kw in cases:
            expert = Expert('red')
            expert.phase = phase
            expert.lifted = reason=='dropped_object'
            obs = observation(46 if reason=='timeout' else 1,**kw)
            if reason=='wrong_color': obs.cubes['blue']=(.17,0,.1,1,0,0,0)
            if reason=='failed_grasp': obs.joints=(0,0,0,0,-.011,-.011)
            if reason=='dropped_object': obs.cubes['red']=(.2,0,.01,1,0,0,0)
            expert.update(obs)
            self.assertEqual(expert.failure,reason)

    def test_no_success_before_release_dwell_and_retreat(self):
        expert = Expert('red')
        expert.phase='verify_placement'
        obs = observation(0)
        obs.ee = (*BIN[:2],.065)
        obs.cubes['red']=(*BIN[:2],BIN[2]+CUBE/2,1,0,0,0)
        expert.update(obs)
        self.assertEqual(expert.phase,'verify_placement')
        obs.time=1.1
        expert.update(obs)
        self.assertEqual(expert.phase,'retreat')
        obs.time=1.2
        obs.ee=(*BIN[:2],.085)
        expert.update(obs)
        self.assertEqual(expert.phase,'retreat')
        obs.time=2.3
        expert.update(obs)
        self.assertEqual(expert.phase,'done')

    def test_placement_dwell_resets_on_motion(self):
        expert = Expert('red')
        expert.phase='verify_placement'
        obs=observation(0)
        obs.cubes['red']=(*BIN[:2],BIN[2]+CUBE/2,1,0,0,0)
        expert.update(obs)
        obs.time=.9
        obs.cube_speeds['red']=.05
        expert.update(obs)
        obs.time=1.1
        obs.cube_speeds['red']=0
        expert.update(obs)
        self.assertEqual(expert.phase,'verify_placement')

    def test_lift_requires_actual_object_motion(self):
        expert=Expert('red')
        expert.phase='lift'
        obs=observation(1)
        expert.pick=obs.cubes['red'][:3]
        obs.ee=(*expert.pick[:2],.085)
        expert.update(obs)
        self.assertEqual(expert.failure,'failed_grasp')
        self.assertFalse(expert.lifted)

    def test_release_is_latched_and_waits_for_open_finger(self):
        expert=Expert('red')
        expert.phase='release'
        obs=observation(1.2)
        obs.ee=(*BIN[:2],.065)
        action=expert.update(obs)
        self.assertEqual(action.gripper,1)
        self.assertEqual(expert.phase,'release')
        obs.time=1.3
        obs.ee=(*BIN[:2],.070)
        self.assertEqual(expert.update(obs).gripper,1)
        obs.time=1.4
        obs.ee=(*BIN[:2],.065)
        obs.joints=(0,0,0,0,.02,.02)
        expert.update(obs)
        self.assertEqual(expert.phase,'verify_placement')

    def test_feedback_time_and_target(self):
        with self.assertRaises(ValueError): Expert('yellow')
        expert=Expert('red')
        expert.update(observation())
        with self.assertRaises(ValueError): expert.update(observation())


class DataAndReports(unittest.TestCase):
    def test_record_schema_and_no_fake_images(self):
        with tempfile.TemporaryDirectory() as temp:
            rec=Recorder(Path(temp)/'ep','Pick up the red cube and place it in the bin.')
            rec.append(observation(),Action((0,0,0),1),'approach')
            rec.close({'success':False})
            row=json.loads((rec.path/'samples.jsonl').read_text())
            self.assertIsNone(row['frame'])
            self.assertEqual(validate_sample(row),0)
            for field,value in [('timestamp',float('nan')),('action',[0,0,0,8]),('frame','../secret'),('success','true'),('joint_velocity',[0])]:
                with self.subTest(field=field),self.assertRaises(ValueError):
                    validate_sample({**row,field:value})
            with self.assertRaisesRegex(ValueError,'camera'):
                list(lerobot_rows(rec.path))
            with self.assertRaises(FileExistsError):
                Recorder(rec.path,'x')

    def test_gate_never_accepts_offline_or_partial_runs(self):
        trials=[dict(seed=i,success=i<18,failure_reason=None if i<18 else 'timeout') for i in range(20)]
        self.assertEqual(aggregate(trials,True,True)['status'],'PASS')
        trials[17]['success']=False
        trials[17]['failure_reason']='failed_grasp'
        self.assertEqual(aggregate(trials,True,True)['status'],'FAIL')
        for rows,executed,completed,error in [(trials,False,True,None),(trials[:19],True,True,None),
             (trials,True,False,None),(trials,True,True,'runtime error'),(trials[::-1],True,True,None)]:
            self.assertEqual(aggregate(rows,executed,completed,error)['status'],'BLOCKED')

    def test_offline_cli_is_blocked_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'gate'
            args=[sys.executable,'-m','embodiment.qualify','--preflight-only','--output',str(output)]
            result=subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'],'BLOCKED')
            self.assertFalse(report['simulation_executed'])
            self.assertFalse(report['threshold_achieved'])
            before=(output/'report.json').read_bytes()
            self.assertNotEqual(subprocess.run(args,capture_output=True).returncode,0)
            self.assertEqual(before,(output/'report.json').read_bytes())


if __name__=='__main__': unittest.main()
