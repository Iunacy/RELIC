import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import urllib.error
import urllib.request

from .benchmark import Action
from .dataset import GROOT_COMMIT
from .policy import GrootPolicy, PolicyInput
from .qualify import source_provenance
from .training import dataset_digest

MAX_MESSAGE = 8 * 1024 * 1024


def encode_observation(observation):
    import numpy as np
    if not isinstance(observation.image, np.ndarray) or observation.image.dtype != np.uint8:
        raise ValueError('uint8 camera observation required')
    return {'embodiment': observation.embodiment, 'instruction': observation.instruction,
            'joint_position': observation.joint_position, 'joint_velocity': observation.joint_velocity,
            'ee_position': observation.ee_position, 'gripper_open': observation.gripper_open,
            'image_shape': list(observation.image.shape),
            'image_base64': base64.b64encode(observation.image.tobytes()).decode('ascii')}


def decode_observation(payload):
    import numpy as np
    expected = {'embodiment', 'instruction', 'joint_position', 'joint_velocity', 'ee_position',
                'gripper_open', 'image_shape', 'image_base64'}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError('policy observation fields differ from the authorized schema')
    shape = payload['image_shape']
    if (not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3
            or any(type(value) is not int or not 1 <= value <= 4096 for value in shape)):
        raise ValueError('invalid camera dimensions')
    raw = base64.b64decode(payload['image_base64'], validate=True)
    if len(raw) != shape[0]*shape[1]*shape[2] or len(raw) > MAX_MESSAGE:
        raise ValueError('invalid camera byte count')
    image = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    return PolicyInput(payload['embodiment'], payload['instruction'], tuple(payload['joint_position']),
                       tuple(payload['joint_velocity']), tuple(payload['ee_position']), payload['gripper_open'], image)


class RemotePolicy:
    def __init__(self, port=8766, timeout=30):
        if type(port) is not int or not 1 <= port <= 65535 or not 0 < timeout <= 120:
            raise ValueError('invalid local policy connection')
        self.url = f'http://127.0.0.1:{port}'
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.metadata = self.request('/metadata')
        if (self.metadata.get('schema') != 'relic.policy_service.v1'
                or self.metadata.get('groot_commit') != GROOT_COMMIT
                or not self.metadata.get('checkpoint_sha256')):
            raise ValueError('unverified policy service provenance')

    def request(self, endpoint, data=None):
        body = None if data is None else json.dumps(data, allow_nan=False).encode()
        if body is not None and len(body) > MAX_MESSAGE:
            raise ValueError('policy request exceeds message limit')
        request = urllib.request.Request(self.url + endpoint, data=body,
                                         headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = response.read(MAX_MESSAGE + 1)
            if len(payload) > MAX_MESSAGE:
                raise ValueError('policy response exceeds message limit')
            return json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite policy response')))
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError('local policy service unavailable or rejected the request') from error

    def reset(self):
        if self.request('/metadata') != self.metadata:
            raise ValueError('policy service changed during evaluation')
        self.request('/reset', {})

    def actions(self, observation):
        response = self.request('/actions', encode_observation(observation))
        if not isinstance(response, dict) or set(response) != {'actions', 'checkpoint_sha256'}:
            raise ValueError('invalid policy response')
        if response['checkpoint_sha256'] != self.metadata['checkpoint_sha256']:
            raise ValueError('policy checkpoint changed during evaluation')
        rows = response['actions']
        if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
            raise ValueError('invalid policy action horizon')
        if any(not isinstance(row, list) or len(row) != 4 for row in rows):
            raise ValueError('invalid policy action shape')
        return tuple(Action(tuple(row[:3]), row[3]) for row in rows)


def handler(policy, metadata):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.send(200, metadata) if self.path == '/metadata' else self.send(404, {'error': 'unknown endpoint'})

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_MESSAGE:
                    raise ValueError('invalid request length')
                payload = json.loads(self.rfile.read(length))
                if self.path == '/reset':
                    if payload != {}:
                        raise ValueError('reset payload must be empty')
                    policy.reset()
                    self.send(200, {'reset': True})
                elif self.path == '/actions':
                    import torch
                    with torch.inference_mode():
                        actions = policy.actions(decode_observation(payload))
                    self.send(200, {'actions': [[*action.delta, action.gripper] for action in actions],
                                    'checkpoint_sha256': metadata['checkpoint_sha256']})
                else:
                    self.send(404, {'error': 'unknown endpoint'})
            except Exception as error:
                self.send(422, {'error': type(error).__name__})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('invalid port')
    policy = GrootPolicy.from_checkpoint(args.source, args.checkpoint, args.device)
    metadata = {'schema': 'relic.policy_service.v1', 'groot_commit': GROOT_COMMIT,
                'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': dataset_digest(args.checkpoint),
                'source_sha256': source_provenance()['source_sha256']}
    with HTTPServer(('127.0.0.1', args.port), handler(policy, metadata)) as server:
        print(json.dumps(metadata), flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
