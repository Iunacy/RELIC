import json
import hashlib
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .arm import Arm, ROOT, camera, markers

REPO = ROOT.parent
BOUNDS = REPO/'adaptive/revision/bounds.txt'
MODEL = REPO/'adaptive/revision/models/shared_full.txt'
CONFIG = REPO/'examples/synthetic_4dof.cfg'


def executable():
    value = Path(os.getenv('RELIC_INTERACTIVE_EXE', str(REPO/'build-interactive/relic_interactive'))).resolve()
    if not value.is_file():
        raise ValueError('native_bridge_missing_run_bash_interactive_setup_sh')
    return value


def native(args):
    try:
        completed = subprocess.run([str(executable()), *map(str, args)], capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise ValueError('native_execution_timeout') from None
    if completed.returncode not in (0, 2):
        raise ValueError('native_execution_error')
    return completed.returncode


def synthetic_scene(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    native(['--scene', CONFIG, out])
    return json.loads((out/'scene.json').read_text())


def scene(robot, out):
    if robot == 'menagerie':
        return Arm().targets
    if robot == 'synthetic':
        return synthetic_scene(out)
    raise ValueError('unknown_robot')


def synthetic_run(target_id, targets, method, degraded, out, inject=-1, model=MODEL):
    if target_id not in targets or method not in ('conventional', 'integral', 'mlp'):
        raise ValueError('invalid_native_task')
    native([CONFIG, out, method, BOUNDS, model, *targets[target_id], int(degraded), inject])
    result = json.loads((out/'result.json').read_text())
    result.update(schema='relic.interactive.v1', robot='synthetic', target_id=target_id,
                  controller=method, physics=dict(gain_scale=.4 if degraded else 1.,
                  damping_scale=2. if degraded else 1., friction_Nm=.15 if degraded else 0.))
    trace = out/'trace.jsonl'
    result['trace'] = [json.loads(row) for row in trace.read_text().splitlines()] if trace.exists() else []
    result['execution_binary_sha256'] = hashlib.sha256(executable().read_bytes()).hexdigest()
    result['integration_source_sha256'] = hashlib.sha256((ROOT/'bridge.cpp').read_bytes()).hexdigest()
    result['source_tree_dirty_at_run'] = bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO))
    result['assets'] = {str(path.relative_to(REPO)):hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (CONFIG,BOUNDS,MODEL)}
    failure=out/'failure_observation.json'
    result['failure_observation'] = json.loads(failure.read_text()) if failure.exists() else None
    result['trajectory'] = 'production plan_cartesian_motion linear reference; zero acceleration feature except undefined endpoint discontinuities'
    result['learned_scope'] = 'synthetic-only experimental residual on production-planned references; no imported-arm transfer'
    return result


def visual_model(robot, out):
    if robot == 'menagerie':
        arm = Arm()
        return arm.model, arm.data
    synthetic_scene(out)
    model = mujoco.MjModel.from_xml_path(str(Path(out)/'model.xml'))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    model.geom_rgba[:] = [.62, .72, .85, 1.]
    model.geom('ground').rgba[:] = [.13, .19, .26, 1.]
    data = mujoco.MjData(model)
    data.qpos[:] = [0, -.5, .5, 0]
    mujoco.mj_forward(model, data)
    return model, data


def view_camera(robot, model):
    if robot == 'menagerie':
        return camera(model)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [.22, 0, .28]
    cam.distance = 1.45
    cam.azimuth = 130
    cam.elevation = -22
    return cam


class Capture:
    def __init__(self, robot, out, targets, selected, update=None):
        self.robot, self.out, self.targets, self.selected = robot, Path(out), targets, selected
        self.out.mkdir(parents=True, exist_ok=True)
        self.model, self.data = visual_model(robot, self.out/'scene')
        self.renderer = mujoco.Renderer(self.model, height=540, width=810)
        self.camera = view_camera(robot, self.model)
        self.frames = []
        self.update = update
        self.image()

    def image(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        markers(self.renderer.scene, self.targets, self.selected)
        path = self.out/f'frame-{len(self.frames):05d}.jpg'
        Image.fromarray(self.renderer.render()).save(path, quality=88)
        self.frames.append(str(path))
        if self.update:
            self.update(path)

    def observed(self, arm, row):
        self.data.qpos[:] = row['observed']
        self.data.qvel[:] = row['velocity']
        self.data.time = row['t']
        mujoco.mj_forward(self.model, self.data)
        self.image()

    def trace(self, rows):
        for i, row in enumerate(rows):
            if i % 16 == 0 or i == len(rows)-1:
                self.observed(None, row)

    def close(self):
        self.renderer.close()


def comparison(robot, target_id, degraded, out, update=None, capture=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    targets = scene(robot, out/'scene')
    if target_id not in targets:
        raise ValueError('unknown_target')
    results = []
    for method in (('conventional', 'integral') if robot == 'menagerie' else ('conventional', 'integral', 'mlp')):
        folder = out/method
        folder.mkdir()
        recorder = Capture(robot, folder/'frames', targets, target_id, update) if capture else None
        try:
            if robot == 'menagerie':
                result = Arm(degraded).run(target_id, method, recorder.observed if recorder else None)
            else:
                result = synthetic_run(target_id, targets, method, degraded, folder)
                if recorder:
                    recorder.trace(result['trace'])
            result['degraded'] = bool(degraded)
            result['visualization'] = 'sampled actual observed states replayed without interpolation'
            if recorder:
                result['frames'] = [str(Path(x).relative_to(out)) for x in recorder.frames]
            (folder/'execution.json').write_text(json.dumps(result)+'\n')
            results.append(result)
        finally:
            if recorder:
                recorder.close()
    first = results[0]
    for value in results[1:]:
        if value.get('initial_q') != first.get('initial_q') or value.get('goal_q') != first.get('goal_q') or value['physics'] != first['physics']:
            raise ValueError('controller_comparison_conditions_differ')
    report = dict(schema='relic.comparison.v1', robot=robot, target_id=target_id, degraded=degraded,
                  targets=targets, controllers=[{k:v for k,v in x.items() if k!='trace'} for x in results])
    (out/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
