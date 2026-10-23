import importlib.util
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
import unittest
import urllib.error
import urllib.request

from embodiment.dataset import DOF
from embodiment.tests.test_replay import episode
from interactive.research import Handler, Research


def loader(seed, roots):
    if type(seed) is not int or not 0 <= seed < 20:
        raise ValueError('invalid seed')
    return {robot: episode(robot) for robot in DOF}


class ResearchViewer(unittest.TestCase):
    def test_commands_and_dispatch_are_validated_without_simulator(self):
        app = Research(loader=loader)
        try:
            self.assertEqual(app.state()['mode'], 'prerecorded')
            self.assertFalse(app.state()['cloud_required'])
            for command in ({'action': 'run_code'}, {'action': 'start', 'extra': True}, {'action': 'seek', 'position': -1.}, {'action': 'select', 'seed': -1}):
                with self.assertRaises(ValueError): app.command(command)
            app.command({'action': 'start'})
            self.assertTrue(app.state()['playing'])
            app.command({'action': 'pause'})
            self.assertFalse(app.state()['playing'])
            app.command({'action': 'reset'})
            self.assertEqual(app.state()['elapsed'], 0.)
            app.command({'action': 'select', 'seed': 2})
            self.assertEqual(app.state()['seed'], 2)
        finally: app.close()

    def test_actual_http_state_and_same_origin_commands(self):
        app = Research(loader=loader)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler);server.app = app
        thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base+'/api/state') as response:
                self.assertEqual(set(json.load(response)['robots']), set(DOF))
            data = json.dumps({'action': 'start'}).encode()
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(urllib.request.Request(base+'/api/control', data=data))
            request = urllib.request.Request(base+'/api/control', data=data, headers={'Origin': base})
            with urllib.request.urlopen(request) as response:
                self.assertTrue(json.load(response)['playing'])
            with urllib.request.urlopen(base) as response:
                self.assertIn(b'PRERECORDED PHYSX', response.read())
        finally:server.shutdown();server.server_close();thread.join();app.close()

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'), 'image dependencies unavailable')
    def test_renderer_failure_preserves_telemetry_and_reports_blocker(self):
        def unavailable(robot): raise ImportError('unit-fixture missing renderer')
        app = Research(loader=loader, renderer_factory=unavailable)
        try:
            with self.assertRaisesRegex(ValueError, 'Renderer unavailable'):
                app.image('franka_panda', 0., 0)
            self.assertIn('Renderer unavailable', app.state()['error'])
            self.assertEqual(app.state()['robots']['franka_panda']['arm_dof'], 7)
        finally:app.close()
