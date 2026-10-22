import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .actions import Client, load_env
from .arm import ROOT
from .engine import Capture, comparison, scene


class Application:
    def __init__(self, output, client):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.lock = threading.Lock()
        self.busy = False
        self.current = dict(status='ready', mode=None, proposal=None, error=None, report=None, image=None)
        self.latest = None
        self.reports = {}
        self.targets = {robot:scene(robot, self.output/'scenes'/robot) for robot in ('menagerie', 'synthetic')}

    def publish(self, **value):
        with self.lock:
            self.current.update(value)

    def image(self, path):
        with self.lock:
            self.latest = Path(path)
            self.current['image'] = '/frame?t='+str(time.time_ns())

    def start(self, request):
        if not isinstance(request, dict) or set(request) != {'robot','mode','target','instruction','degraded'}:
            raise ValueError('invalid_request_fields')
        robot, mode = request['robot'], request['mode']
        if robot not in self.targets or mode not in ('live', 'manual') or type(request['degraded']) is not bool:
            raise ValueError('invalid_request')
        if mode == 'manual' and request['target'] not in self.targets[robot]:
            raise ValueError('unknown_target')
        if not isinstance(request['instruction'], str) or len(request['instruction']) > 1500:
            raise ValueError('invalid_instruction')
        with self.lock:
            if self.busy:
                raise ValueError('execution_busy')
            self.busy = True
            self.current = dict(status='selecting' if mode == 'live' else 'validating', mode=mode,
                                proposal=None, error=None, report=None, image=self.current['image'])
        run_id = uuid.uuid4().hex
        threading.Thread(target=self.work, args=(request,run_id), daemon=True).start()
        return run_id

    def work(self, request, run_id):
        out = self.output/run_id
        out.mkdir()
        try:
            robot = request['robot']
            if request['mode'] == 'live':
                action = self.client.select(request['instruction'], self.targets[robot], out/'selection.json')
            else:
                action = dict(action='reach_target', target_id=request['target'])
                (out/'selection.json').write_text(json.dumps(dict(mode='manual', validated_output=action,
                    note='direct scene selection; no NVIDIA inference')))
            self.publish(proposal=action, status='executing')
            report = comparison(robot, action['target_id'], request['degraded'], out/'execution', self.image)
            report['selection_mode'] = 'mock' if request['mode'] == 'live' and self.client.mock else request['mode']
            report['run_id'] = run_id
            with self.lock:
                self.reports[run_id] = report
            self.publish(status='complete', report=report)
        except Exception as e:
            error = str(e) if isinstance(e, ValueError) else type(e).__name__
            (out/'failure.json').write_text(json.dumps(dict(error=error, status='failed')))
            self.publish(status='failed', error=error)
        finally:
            with self.lock:
                self.busy = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, value, content_type='application/json'):
        raw = json.dumps(value).encode() if content_type == 'application/json' else value
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self'")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        app = self.server.app
        url = urlparse(self.path)
        if url.path == '/':
            return self.send(200, (ROOT/'index.html').read_bytes(), 'text/html; charset=utf-8')
        if url.path == '/state':
            with app.lock:
                state = dict(app.current)
            state['targets'] = app.targets
            state['model'] = app.client.model
            state['live_available'] = bool(app.client.key)
            return self.send(200, state)
        if url.path == '/frame':
            with app.lock:
                path = app.latest
            if path and path.is_file():
                return self.send(200, path.read_bytes(), 'image/jpeg')
        if url.path == '/trace':
            try:
                params = parse_qs(url.query)
                run = params['run'][0]
                method = params['method'][0]
                report = app.reports[run]
                if not any(x['controller'] == method for x in report['controllers']):
                    raise ValueError
                value = json.loads((app.output/run/'execution'/method/'execution.json').read_text())
                return self.send(200, value['trace'])
            except (KeyError, ValueError):
                return self.send(404, dict(error='not_found'))
        if url.path == '/replay':
            try:
                params = parse_qs(url.query)
                run = params['run'][0]
                method = params['method'][0]
                index = int(params['index'][0])
                report = app.reports[run]
                control = next(x for x in report['controllers'] if x['controller'] == method)
                if index < 0 or index >= len(control['frames']):
                    raise ValueError
                path = app.output/run/'execution'/control['frames'][index]
                return self.send(200, path.read_bytes(), 'image/jpeg')
            except (KeyError, IndexError, ValueError, StopIteration):
                pass
        return self.send(404, dict(error='not_found'))

    def do_POST(self):
        if self.path != '/execute':
            return self.send(404, dict(error='not_found'))
        origin = self.headers.get('Origin')
        if origin != f'http://{self.headers.get("Host")}':
            return self.send(403, dict(error='same_origin_required'))
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size < 1 or size > 8192:
                raise ValueError('invalid_request_size')
            request = json.loads(self.rfile.read(size))
            run_id = self.server.app.start(request)
            return self.send(202, dict(run_id=run_id))
        except (ValueError, TypeError) as e:
            return self.send(400, dict(error=str(e)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--output', type=Path, default=ROOT/'runs/app')
    p.add_argument('--env-file', type=Path)
    p.add_argument('--base-url')
    p.add_argument('--model')
    args = p.parse_args()
    load_env(args.env_file)
    app = Application(args.output, Client(args.base_url, args.model))
    def preview():
        recorder = Capture('menagerie', args.output/'preview', app.targets['menagerie'], 'target_left', app.image)
        recorder.close()
    worker = threading.Thread(target=preview)
    worker.start()
    worker.join()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.app = app
    print(f'RELIC http://127.0.0.1:{args.port} — live key {"available" if app.client.key else "missing"}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    main()
