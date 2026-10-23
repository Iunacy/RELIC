import base64
import importlib.util
from http.server import HTTPServer
import json
import threading
import unittest
from unittest.mock import patch

from embodiment.benchmark import Action
from embodiment.dataset import GROOT_COMMIT
from embodiment.policy import PolicyInput
from embodiment.policy_service import RemotePolicy, decode_observation, encode_observation, handler


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy unavailable')
class PolicyService(unittest.TestCase):
    def observation(self):
        import numpy as np
        return PolicyInput('openmanipulator_x', 'Pick up the red cube and place it in the bin.',
                           (0.,)*6, (0.,)*6, (.17, 0., .085), 1., np.zeros((8, 8, 3), dtype=np.uint8))

    def test_wire_schema_excludes_privileged_state_and_validates_images(self):
        payload = encode_observation(self.observation())
        decoded = decode_observation(payload)
        self.assertEqual(decoded.instruction, self.observation().instruction)
        self.assertNotIn('cube_poses', payload)
        for changed in ({**payload, 'cube_poses': {}}, {**payload, 'image_shape': [2, 2, 3]},
                        {**payload, 'image_base64': 'invalid###'}):
            with self.assertRaises(ValueError):
                decode_observation(changed)

    def test_actual_local_http_roundtrip_and_reset(self):
        observation = self.observation()
        class Fixture:
            resets = 0
            def reset(self):
                self.resets += 1
            def actions(self, received):
                return (Action((0., 0., 0.), 1.),)
        fixture = Fixture()
        metadata = {'schema': 'relic.policy_service.v1', 'groot_commit': GROOT_COMMIT,
                    'checkpoint_sha256': 'unit-fixture-only'}
        server = HTTPServer(('127.0.0.1', 0), handler(fixture, metadata))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            import contextlib
            with patch.dict('sys.modules', {'torch': type('Torch', (), {'inference_mode': staticmethod(contextlib.nullcontext)})}):
                remote = RemotePolicy(server.server_port)
                remote.reset()
                self.assertEqual(remote.actions(observation), (Action((0., 0., 0.), 1.),))
                self.assertEqual(fixture.resets, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_entire_remote_chunk_is_validated_before_execution(self):
        with patch.object(RemotePolicy, 'request', return_value={'schema': 'relic.policy_service.v1',
                          'groot_commit': GROOT_COMMIT, 'checkpoint_sha256': 'unit-fixture'}):
            remote = RemotePolicy()
        for payload in ({'actions': [[0., 0., 0., 1.], [.1, 0., 0., 1.]]},
                        {'actions': []}, {'actions': [[0., 0., 0.]]}, {'actions': [[0., 0., float('nan'), 1.]]}):
            with patch.object(remote, 'request', return_value={**payload, 'checkpoint_sha256': 'unit-fixture'}), self.assertRaises(ValueError):
                remote.actions(self.observation())

    def test_checkpoint_change_is_rejected(self):
        metadata = {'schema': 'relic.policy_service.v1', 'groot_commit': GROOT_COMMIT,
                    'checkpoint_sha256': 'unit-fixture'}
        with patch.object(RemotePolicy, 'request', return_value=metadata):
            remote = RemotePolicy()
        with patch.object(remote, 'request', return_value={'actions': [[0., 0., 0., 1.]], 'checkpoint_sha256': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                remote.actions(self.observation())
        with patch.object(remote, 'request', return_value={**metadata, 'checkpoint_sha256': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'service changed'):
                remote.reset()
