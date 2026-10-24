import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

from embodiment.dataset import DOF
from embodiment.replay import Playback, load_qualified_pair, validate_pair
from .http import LocalHandler

ROOT = Path(__file__).resolve().parent


class Research:
    def __init__(self, roots=None, seed=0, loader=load_qualified_pair, renderer_factory=None, catalog=None):
        self.roots, self.loader = roots, loader
        self.catalog = catalog or [{'id': seed, 'label': f'Seed {seed} / '+('red', 'green', 'blue')[seed % 3]} for seed in range(20)]
        self.renderer_factory = renderer_factory
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.RLock()
        self.renderers, self.cache, self.error = {}, {}, None
        self.select(seed)

    def select(self, seed):
        episodes = self.loader(seed, self.roots)
        duration = validate_pair(episodes)
        with self.lock:
            self.seed, self.episodes = seed, episodes
            self.playback = Playback(duration)
            self.cache.clear()
            self.error = None

    def state(self):
        with self.lock:
            playback = self.playback.snapshot()
            return {**playback, 'seed': self.seed, 'target': next(iter(self.episodes.values())).metadata['target'],
                    'source': 'validated PhysX trajectories; MuJoCo state rendering',
                    'purpose': next(iter(self.episodes.values())).metadata['purpose'], 'scenarios': self.catalog,
                    'robots': {robot: episode.state(playback['elapsed']) for robot, episode in self.episodes.items()},
                    'evidence': {'learned_policy': 'available' if any(episode.metadata['controller'] != 'classical_only' for episode in self.episodes.values()) else 'unavailable',
                                 'transfer_comparison': 'unavailable; matched E2/E3 evaluation not loaded',
                                 'common_revision_qualification': 'not established by legacy replay',
                                 'scene': next(iter(self.episodes.values())).metadata.get('scenario_id', f'qualification-seed-{self.seed}')},
                    'error': self.error, 'cloud_required': False}

    def command(self, request):
        if not isinstance(request, dict) or request.get('action') not in ('start', 'pause', 'resume', 'reset', 'seek', 'select'):
            raise ValueError('invalid viewer command')
        allowed = {'action', 'position'} if request['action'] == 'seek' else {'action', 'seed'} if request['action'] == 'select' else {'action'}
        if set(request) != allowed:
            raise ValueError('unexpected viewer command fields')
        with self.lock:
            if request['action'] == 'select':
                self.select(request['seed'])
            else:
                self.playback.command(request['action'], request.get('position'))
            return self.state()

    def image(self, robot, elapsed, seed):
        if robot not in DOF or type(seed) is not int or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid frame identity')
        with self.lock:
            if seed != self.seed or elapsed > self.playback.duration + 1e-6:
                raise ValueError('stale or out-of-range frame request')
            episode = self.episodes[robot]
            key = seed, robot, episode.index(elapsed)
            if key in self.cache:
                return self.cache[key]
        def render():
            if robot not in self.renderers:
                if self.renderer_factory is None:
                    from embodiment.replay_render import Renderer
                    factory = Renderer
                else:
                    factory = self.renderer_factory
                self.renderers[robot] = factory(robot)
            from PIL import Image
            image = self.renderers[robot].render(episode, elapsed)
            output = io.BytesIO()
            Image.fromarray(image).save(output, format='JPEG', quality=90)
            return output.getvalue()
        try:
            value = self.worker.submit(render).result(timeout=30)
        except Exception as error:
            with self.lock:
                self.error = f'Renderer unavailable: {type(error).__name__}. Telemetry remains available.'
            raise ValueError(self.error) from error
        with self.lock:
            if len(self.cache) >= 48:
                self.cache.pop(next(iter(self.cache)))
            self.error = None
            self.cache[key] = value
        return value

    def close(self):
        def close_renderers():
            for renderer in self.renderers.values():
                renderer.close()
        self.worker.submit(close_renderers).result(timeout=30)
        self.worker.shutdown()


class Handler(LocalHandler):
    def do_GET(self):
        route = urlparse(self.path)
        try:
            if route.path == '/':
                return self.send(200, (ROOT/'research.html').read_bytes(), 'text/html; charset=utf-8')
            if route.path == '/api/state':
                return self.send(200, self.server.app.state())
            if route.path == '/api/frame':
                query = parse_qs(route.query, strict_parsing=True)
                if set(query) != {'robot', 't', 'seed'} or any(len(value) != 1 for value in query.values()):
                    raise ValueError('invalid frame query')
                return self.send(200, self.server.app.image(query['robot'][0], float(query['t'][0]), int(query['seed'][0])), 'image/jpeg')
            self.send(404, {'error': 'not_found'})
        except (ValueError, KeyError, OSError) as error:
            self.send(400, {'error': str(error)})

    def do_POST(self):
        if self.path != '/api/control':
            return self.send(404, {'error': 'not_found'})
        try:
            return self.send(200, self.server.app.command(self.read_request()))
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.send(400, {'error': str(error)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--omx-campaign', type=Path)
    parser.add_argument('--panda-campaign', type=Path)
    parser.add_argument('--experiment', type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('invalid port')
    roots = {robot: value for robot, value in [('openmanipulator_x', args.omx_campaign), ('franka_panda', args.panda_campaign)] if value is not None}
    try:
        if args.experiment:
            from embodiment.replay import load_experiment_pair
            configuration = json.loads((args.experiment/'experiment.json').read_text())
            catalog = [{'id': index, 'label': case['perturbation']['kind']+' / '+case['target']+' / '+str(case['seed'])}
                       for index, case in enumerate(configuration['cases'])]
            app = Research(seed=args.seed, loader=lambda index, _: load_experiment_pair(args.experiment, index), catalog=catalog)
        else:
            app = Research(roots, args.seed)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Cannot load physical replay: {error}. Supply verified campaign roots.\n')
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        server.app = app
        print(f'VIPER research viewer http://127.0.0.1:{args.port} | prerecorded physical trajectories', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            app.close()


if __name__ == '__main__':
    main()
