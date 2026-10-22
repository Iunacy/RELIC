import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .actions import ActionError, Client, validate_action
from .arm import Arm, HOME, ROOT
from .engine import comparison, synthetic_run, synthetic_scene
from .app import Application


class FakeAPI(BaseHTTPRequestHandler):
    mode = 'valid'
    def log_message(self, *args):
        pass
    def do_GET(self):
        if self.server.mode == 'auth':
            self.send_response(401);self.end_headers();return
        if self.server.mode == 'timeout':
            time.sleep(.15)
        body = json.dumps(dict(data=[dict(id='nvidia/test')])).encode()
        try:
            self.send_response(200);self.end_headers();self.wfile.write(body)
        except BrokenPipeError:
            pass
    def do_POST(self):
        value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        instruction = value['messages'][1]['content']
        target = 'target_right' if 'right' in instruction else 'target_left'
        content = '{bad' if self.server.mode == 'malformed' else json.dumps(dict(action='reach_target',target_id=target))
        body = json.dumps(dict(id='test-request',choices=[dict(finish_reason='stop',message=dict(content=content))],
                              usage=dict(prompt_tokens=10,completion_tokens=10,total_tokens=20))).encode()
        self.send_response(200);self.end_headers();self.wfile.write(body)


class Actions(unittest.TestCase):
    def test_strict_actions(self):
        targets = {'target_left': [0,0,1]}
        self.assertEqual(validate_action('{"action":"reach_target","target_id":"target_left"}',targets)['target_id'],'target_left')
        for value in ['{}','[]','null','not json','{"action":"reach_target","target_id":"unknown"}',
                      '{"action":"reach_target","target_id":[]}',
                      '{"action":"reach_target","target_id":"target_left","q":[0]}',
                      '{"action":"reach_target","action":"reach_target","target_id":"target_left"}',
                      '{"action":"reject_task","target_id":null}',
                      '{"action":"execute_code","target_id":"target_left"}']:
            with self.subTest(value=value),self.assertRaises(ActionError):
                validate_action(value,targets)

    def test_transport_and_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            server=ThreadingHTTPServer(('127.0.0.1',0),FakeAPI)
            server.mode='valid'
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                client=Client(f'http://127.0.0.1:{server.server_port}/v1/',model='nvidia/test',key='secret-not-for-logs')
                targets={'target_left':[0,0,1],'target_right':[0,1,1]}
                for target in targets:
                    path=Path(temp)/(target+'.json')
                    self.assertEqual(client.select('reach '+target,targets,path)['target_id'],target)
                    log=json.loads(path.read_text());self.assertEqual(log['mode'],'mock')
                    self.assertEqual(log['token_usage']['total_tokens'],20)
                    self.assertNotIn('secret-not-for-logs',path.read_text())
                app=Application(Path(temp)/'app',client)
                app.busy=True
                request=dict(robot='synthetic',mode='live',target='target_left',instruction='reach right',degraded=True)
                with patch('interactive.app.comparison',side_effect=lambda robot,target,degraded,out,update: comparison(robot,target,degraded,out,capture=False)):
                    app.work(request,'mock-integrated')
                self.assertEqual(app.current['status'],'complete')
                self.assertEqual(app.current['report']['target_id'],'target_right')
                self.assertEqual(app.current['report']['selection_mode'],'mock')
                self.assertTrue(app.current['report']['controllers'][0]['metrics']['simulation_s']>1)
                server.mode='malformed'
                with patch('interactive.app.comparison') as blocked:
                    app.work(request,'mock-rejected')
                    blocked.assert_not_called()
                self.assertEqual(app.current['status'],'failed')
                self.assertFalse((app.output/'mock-rejected'/'execution').exists())
                for mode,reason in [('auth','api_http_401'),('malformed','malformed_action_json'),('timeout','api_timeout_or_connection_failure')]:
                    server.mode=mode;client.timeout=.03 if mode=='timeout' else 1
                    path=Path(temp)/(mode+'.json')
                    with self.assertRaisesRegex(ActionError,reason):client.select('left',targets,path)
                    self.assertEqual(json.loads(path.read_text())['validated_output'],None)
            finally:
                server.shutdown();server.server_close();thread.join()

    def test_missing_key_and_endpoint(self):
        with self.assertRaises(ActionError):Client('http://example.com/')
        with self.assertRaises(ActionError):Client('https://user:pass@example.com/')
        client=Client(key='x');client.key=None
        with tempfile.TemporaryDirectory() as temp,self.assertRaisesRegex(ActionError,'missing_nebius_api_key'):
            client.select('left',{'target_left':[0,0,1]},Path(temp)/'log.json')


class ImportedArm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm=Arm()

    def test_assets(self):
        manifest=json.loads((ROOT/'vendor/manifest.json').read_text())
        for name,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'vendor/low_cost_robot_arm'/name).read_bytes()).hexdigest(),digest)

    def test_limits_mapping_targets(self):
        arm=self.arm
        self.assertEqual(arm.model.nq,6);self.assertEqual(arm.model.nu,6)
        for target in arm.targets.values():
            q=arm.solve(target)
            self.assertTrue(np.array_equal(q[4:],HOME[4:]))
            self.assertLess(np.linalg.norm(arm.fk(q)-target),.0001)
        for q in [np.zeros(4),np.full(6,np.nan),np.full(6,10)]:
            with self.assertRaises(ValueError):arm.check_q(q)
        with self.assertRaises(ValueError):arm.solve([5,0,0])
        with self.assertRaises(ValueError):arm.solve([0,0,.49])
        with self.assertRaises(ValueError):arm.run('unknown')
        with self.assertRaisesRegex(ValueError,'no_mlp_transfer'):arm.run('target_left','mlp')

    def test_contact_path_and_isolation(self):
        arm=self.arm;arm.reset();before=arm.data.qpos.copy()
        bad=np.array([1.1944483717040004,-.6151889233281287,1.0086218013359198,.7815773950750811,0,-.65])
        with self.assertRaisesRegex(ValueError,'sampled_contact_path'):arm.safe_path(HOME,bad)
        self.assertTrue(np.array_equal(arm.data.qpos,before))
        self.assertNotEqual(Arm(True).model.actuator_gainprm[0,0],arm.model.actuator_gainprm[0,0])
        arm.data.qpos[0]=arm.upper[0]+.01
        with self.assertRaisesRegex(ValueError,'joint_or_control_limit'):arm.health()
        arm.reset()
        bad_arm=Arm()
        with patch.object(bad_arm,'health',side_effect=[None,ValueError('injected_observed_health_failure')]):
            stopped=bad_arm.run('target_left')
        self.assertEqual(stopped['status'],'Stopped')
        self.assertEqual(stopped['reason'],'injected_observed_health_failure')
        self.assertIsNotNone(stopped['failure_observation'])
        self.assertFalse(stopped['success'])

    def test_repeatability_withdrawal(self):
        a=Arm(True).run('target_left','integral',inject=300)
        b=Arm(True).run('target_left','integral',inject=300)
        self.assertEqual(a['trace'],b['trace'])
        correction=np.array([r['correction'] for r in a['trace']])
        self.assertGreater(np.max(np.abs(correction[299])),.001)
        self.assertGreater(a['metrics']['fallbacks'],0)
        self.assertLessEqual(np.max(np.abs(np.diff(correction,axis=0))),.0005+1.e-12)
        self.assertTrue(np.array_equal(correction[-1],np.zeros(6)))
        command=np.array([r['command'] for r in a['trace']])
        self.assertLessEqual(np.max(np.abs(np.diff(command,axis=0))),.005+1.e-12)
        self.assertEqual(a['status'],'Success')


class NativeBridge(unittest.TestCase):
    def test_equivalent_and_repeatable(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);targets=synthetic_scene(out/'scene');rows=[]
            for method in ('conventional','integral','mlp'):
                row=synthetic_run('target_left',targets,method,True,out/method)
                self.assertTrue(row['trace']);self.assertEqual(row['initial_q'],[0,-.5,.5,0])
                rows.append(row)
            for row in rows[1:]:
                self.assertEqual(row['goal_q'],rows[0]['goal_q']);self.assertEqual(row['physics'],rows[0]['physics'])
            repeated=synthetic_run('target_left',targets,'mlp',True,out/'repeat')
            self.assertEqual(repeated['trace'],rows[-1]['trace'])
            injected=synthetic_run('target_left',targets,'mlp',True,out/'injected',inject=250)
            c=np.array([r['correction'] for r in injected['trace']])
            self.assertGreater(np.max(np.abs(c[248])),.001)
            self.assertGreater(injected['metrics']['fallbacks'],0)
            self.assertEqual(injected['status'],rows[0]['status'])
            self.assertEqual(injected['status'],'NonConvergence')
            self.assertLessEqual(np.max(np.abs(np.diff(c,axis=0))),.000515)
            self.assertTrue(np.array_equal(c[-1],np.zeros(4)))
            missing=synthetic_run('target_left',targets,'mlp',True,out/'missing',model=out/'absent')
            self.assertGreater(missing['metrics']['fallbacks'],0)
            self.assertTrue(all(r['correction']==[0,0,0,0] for r in missing['trace']))
            native_bad=synthetic_run('unreachable',{'unreachable':[5,0,0]},'conventional',False,out/'bad')
            self.assertFalse(native_bad['success']);self.assertFalse(native_bad['trace'])


if __name__=='__main__':
    unittest.main(verbosity=2)
