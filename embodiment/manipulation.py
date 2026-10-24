"""Bounded classical manipulation suite; physical execution is explicitly opt-in."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .benchmark import Action, COLORS, CONTROL_DT, GRASP_Z, TIMEOUT, inside_bin
from .collection import atomic_json, execute, hashes, locked
from .dataset import DOF, read_json
from .evaluation import metrics
from .expert import Expert
from .perturbations import Perturbation
from .qualify import source_provenance
from .scenarios import plan

TASKS = ('reach', 'approach', 'grasp', 'pick_place', 'recover_grasp')
STARTS = ('home', 'left', 'right')


def cases(count=1, tasks=TASKS, starts=('home',), perturbed=False):
    if type(count) is not int or not 1 <= count <= 20 or not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError('invalid bounded benchmark task schedule')
    if not starts or len(set(starts)) != len(starts) or any(start not in STARTS for start in starts):
        raise ValueError('unknown initial configuration')
    result = []
    for scene in plan('evaluation', count):
        for task in tasks:
            for start in starts:
                item = {**asdict(scene), 'task': task, 'initial_configuration': start,
                        'perturbation': asdict(Perturbation('position_shift', scene.seed % 65536, .003)) if perturbed else asdict(Perturbation())}
                item['scene_id'] = item.pop('scenario_id')
                item['scenario_id'] = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                result.append(item)
    return result


class TaskController:
    def __init__(self, task, robot, target):
        if task not in TASKS or robot not in DOF or target not in COLORS: raise ValueError('unknown task or robot')
        self.task, self.target = task, target
        self.grasp_z = GRASP_Z if robot == 'openmanipulator_x' else .0135
        self.expert = Expert(target, self.grasp_z)
        self.success, self.failure = False, None
        self.attempts, self.interventions = [], []
        self.recovering, self.stable_since = False, None
        self.phase = 'approach'
        self.goal = None
        self.last_time = -1.

    def update(self, obs):
        if self.success or self.failure: raise RuntimeError('task is terminal')
        if obs.time <= self.last_time: raise ValueError('nonmonotonic benchmark observation')
        self.last_time = obs.time
        neutral = Action((0., 0., 0.), obs.gripper_open if obs.gripper_open is not None else 1.)
        if obs.unsafe or obs.safety_events:
            self.failure = 'unsafe_movement'
            return neutral
        if obs.time >= TIMEOUT:
            self.failure = 'timeout'
            return neutral
        if self.task in ('reach', 'approach') or self.recovering:
            cube = obs.cubes[self.target]
            self.goal = (.17, 0., .12) if self.task == 'reach' else (cube[0], cube[1], .085)
            if self.recovering: self.goal = (obs.ee[0], obs.ee[1], .085)
            arrived = math.dist(obs.ee, self.goal) < .004 and max(map(abs, obs.velocities[:obs.arm_dof])) < .12
            if self.recovering: arrived &= obs.gripper_open is not None and obs.gripper_open >= .84
            self.stable_since = (obs.time if self.stable_since is None else self.stable_since) if arrived else None
            if self.stable_since is not None and obs.time-self.stable_since >= .5:
                if self.recovering:
                    self.expert = Expert(self.target, self.grasp_z)
                    self.expert.entered = obs.time
                    self.recovering, self.stable_since = False, None
                else: self.success = True
            delta = tuple(b-a for a, b in zip(obs.ee, self.goal, strict=True))
            scale = min(1., .0015/max(math.dist(delta, (0., 0., 0.)), 1e-12))
            return Action(tuple(value*scale for value in delta), 1.)
        action = self.expert.update(obs)
        self.phase = self.expert.phase
        self.goal = obs.cubes[self.target][:3]
        if self.expert.failure:
            self.attempts.append({'timestamp': obs.time, 'failure': self.expert.failure})
            if self.task == 'recover_grasp' and self.expert.failure == 'failed_grasp' and not self.expert.lifted and len(self.attempts) == 1:
                self.interventions.append({'timestamp': obs.time, 'reason': 'classical_grasp_retry'})
                self.recovering, self.stable_since = True, None
                self.phase = 'reopen_and_retreat'
                return Action((0., 0., .0015), 1.)
            self.failure = self.expert.failure
        elif self.task == 'grasp' and self.expert.lifted:
            self.success = bool(obs.contacts.get(self.target) and min(obs.contact_forces.get(self.target, (0., 0.))) > .005
                                and obs.cubes[self.target][2] >= .055)
        elif self.expert.phase == 'done':
            self.success = bool(self.expert.lifted and self.expert.released and inside_bin(obs.cubes[self.target]))
        return action


def initial_joints(backend, start):
    if start == 'home': return None
    if start not in STARTS: raise ValueError('unknown start')
    initial = (0.,)*4 if backend.embodiment == 'openmanipulator_x' else (0., -.5, 0., -2., 0., 1.5, .785)
    return backend.kinematics.inverse((.17, -.025 if start == 'left' else .025, .11), initial)


def run_trial(backend, case, output, running=lambda: True):
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    obs = backend.reset(case['seed'], cube_positions=Perturbation(**case['perturbation']).positions(case['seed']),
                        initial_joints=initial_joints(backend, case['initial_configuration']))
    physical_check = backend.kinematic_check()
    controller = TaskController(case['task'], backend.embodiment, case['target'])
    errors, collisions, events = [], 0, []
    with (output/'trace.jsonl').open('x') as trace:
        for tick in range(math.ceil(TIMEOUT/CONTROL_DT)+1):
            if not running(): raise RuntimeError('simulator stopped during task')
            action = controller.update(obs)
            if controller.success or controller.failure: break
            previous = obs
            obs = backend.step(action)
            target = tuple(a+b for a, b in zip(previous.ee, action.delta, strict=True))
            error = math.dist(obs.ee, target)
            errors.append(error)
            events.extend(obs.safety_events)
            physical_collisions = backend.collision_state()
            collisions += len(physical_collisions['forbidden'])
            trace.write(json.dumps({'observation': asdict(previous), 'requested_action': asdict(action),
                'response': asdict(obs), 'tracking_error_m': error, 'phase': controller.phase,
                'collisions': physical_collisions}, allow_nan=False)+'\n')
            trace.flush()
            if physical_collisions['forbidden']:
                controller.failure = 'forbidden_collision'
                break
            if obs.time <= previous.time:
                controller.failure = 'unsafe_movement'
                break
        else: controller.failure = 'timeout'
    backend.safe_stop()
    result = {**case, 'robot': backend.embodiment, 'controller': 'classical_only',
        'success': controller.success, 'failure_reason': controller.failure if not controller.success else None,
        'simulation_executed': True, 'evidence_source': 'isaac_lab_physx', 'kinematics': physical_check,
        'duration': obs.time, 'wall_seconds': time.monotonic()-started,
        'endpoint_error_m': math.dist(obs.ee, controller.goal) if controller.goal is not None else None,
        'mean_tracking_error_m': sum(errors)/len(errors) if errors else None,
        'collision_measurement': 'net-body force event samples; intentional finger-cube contact allowed; not unique contact-pair count',
        'collisions': collisions, 'safety_events': events, 'interventions': controller.interventions,
        'attempt_failures': controller.attempts, 'recovery_assisted': bool(controller.interventions),
        'capability_status': 'executed; see outcome', 'learned_transfer_evidence': False}
    atomic_json(output/'outcome.json', result)
    result['evidence_sha256'] = hashes(output)
    return result


def summarize(trials, assigned):
    expected = {(robot, case['scenario_id']) for robot in DOF for case in assigned}
    seen = set()
    by_case = {case['scenario_id']: case for case in assigned}
    for row in trials:
        key = row.get('robot'), row.get('scenario_id')
        if key not in expected or key in seen: raise ValueError('duplicate or unassigned benchmark trial')
        if any(row.get(name) != value for name, value in by_case[key[1]].items()): raise ValueError('trial scenario changed')
        if type(row.get('success')) is not bool: raise ValueError('explicit task result required')
        if row['success'] and (row.get('simulation_executed') is not True or row.get('evidence_source') != 'isaac_lab_physx' or not row.get('evidence_sha256')):
            raise ValueError('successful task lacks physical evidence')
        if row['success'] and (row.get('safety_events') or row.get('collisions') or row.get('infrastructure_error')):
            raise ValueError('unsafe or infrastructure result cannot be successful')
        seen.add(key)
    groups = {}
    for robot in DOF:
        for task in sorted({case['task'] for case in assigned}):
            rows = [row for row in trials if row['robot'] == robot and row['task'] == task]
            measured = [row for row in rows if row.get('endpoint_error_m') is not None]
            groups[robot+'/'+task] = {**metrics(rows), 'intervention_frequency': sum(bool(row.get('interventions')) for row in rows)/len(rows) if rows else None,
                'mean_endpoint_error_m': sum(row['endpoint_error_m'] for row in measured)/len(measured) if measured else None,
                'endpoint_measured_trials': len(measured), 'forbidden_contact_samples': sum(row.get('collisions', 0) for row in rows)}
    physical_complete = seen == expected and all(row.get('simulation_executed') is True and not row.get('infrastructure_error') for row in trials)
    return {'by_task': groups, 'schema': 'viper.manipulation_summary.v1', 'completed': seen == expected,
        'physical_execution_complete': physical_complete, 'status': 'COMPLETE' if physical_complete else 'BLOCKED',
        'missing_trials': sorted(expected-seen), 'trials': trials,
        'by_robot': {robot: {**metrics([row for row in trials if row['robot'] == robot]),
            'intervention_trials': sum(bool(row.get('interventions')) for row in trials if row['robot'] == robot),
            'infrastructure_failures': sum(bool(row.get('infrastructure_error')) for row in trials if row['robot'] == robot)} for robot in DOF},
        'claim': 'classical controller benchmark only; no learned policy or transfer claim'}


def worker(request, output):
    from .host import inspect_host
    requirements = inspect_host('simulation')
    if not requirements['ready']: raise RuntimeError('; '.join(requirements['blockers']))
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(['--headless', '--device', 'cuda:0'])
    app = AppLauncher(args).app
    backend = None
    rows = []
    provenance = source_provenance()
    try:
        from .isaac_backend import IsaacBackend
        backend = IsaacBackend('cuda:0', False, request['robot'], collision_audit=True)
        for index, case in enumerate(request['cases']):
            folder = output/f'trial_{index:04d}'
            try:
                row = run_trial(backend, case, folder, app.is_running)
            except Exception as error:
                row = {**case, 'robot': request['robot'], 'success': False, 'failure_reason': 'infrastructure_failure',
                       'infrastructure_error': f'{type(error).__name__}: {error}', 'simulation_executed': False}
            rows.append(row)
            atomic_json(output/'report.json', {'source_sha256': provenance['source_sha256'], 'trials': rows,
                'source_changed_during_run': source_provenance()['source_sha256'] != provenance['source_sha256']})
    finally:
        if backend is not None: backend.safe_stop()
        app.close(wait_for_replicator=False)
    return rows


def campaign(root, count=1, tasks=TASKS, starts=('home',), perturbed=False, run=False, python=sys.executable):
    root = Path(root).resolve()
    assigned = cases(count, tasks, starts, perturbed)
    configuration = {'schema': 'viper.manipulation.v1', 'cases': assigned, 'source': source_provenance()}
    with locked(root):
        path = root/'benchmark.json'
        if path.exists() and read_json(path) != configuration: raise ValueError('frozen benchmark changed')
        atomic_json(path, configuration)
        commands = {robot: [str(python), '-m', 'embodiment.manipulation', '--worker', '--request', str(root/robot/'request.json'),
                    '--root', str(root/robot/'run')] for robot in DOF}
        if not run: return {'status': 'NOT_EXECUTED', 'simulation_executed': False, 'commands': commands}
        if configuration['source']['relic_worktree_dirty']: raise ValueError('physical benchmarks require a clean committed revision')
        import subprocess
        probe = subprocess.run([str(python), '-m', 'embodiment.host', '--stage', 'simulation', '--volume', str(root)], text=True, capture_output=True, timeout=30)
        try: host = json.loads(probe.stdout)
        except ValueError: host = {'ready': False, 'blockers': [probe.stderr or 'simulation interpreter unavailable']}
        if not host['ready']: return {'status': 'BLOCKED', 'simulation_executed': False, 'blockers': host['blockers']}
        trials = []
        for robot, command in commands.items():
            job = root/robot
            job.mkdir(exist_ok=True)
            atomic_json(job/'request.json', {'robot': robot, 'cases': assigned})
            try:
                if not (job/'run/report.json').exists() and not (job/'process.log').exists(): execute(command, job, 2400)
                report = read_json(job/'run/report.json')
                if report['source_sha256'] != configuration['source']['source_sha256'] or report.get('source_changed_during_run') is not False:
                    raise ValueError('benchmark worker source changed')
                if len(report['trials']) != len(assigned): raise ValueError('worker omitted assigned benchmark scenes')
                verified = []
                for index, row in enumerate(report['trials']):
                    folder = job/'run'/f'trial_{index:04d}'
                    if row.get('evidence_sha256') and (hashes(folder) != row['evidence_sha256'] or
                            read_json(folder/'outcome.json') != {key: value for key, value in row.items() if key != 'evidence_sha256'}):
                        raise ValueError('benchmark episode corruption')
                    verified.append(row)
                trials.extend(verified)
            except (OSError, ValueError, KeyError, TypeError) as error:
                trials.extend({**case, 'robot': robot, 'success': False, 'failure_reason': 'infrastructure_failure',
                    'infrastructure_error': f'{type(error).__name__}: {error}', 'simulation_executed': False} for case in assigned)
        result = summarize(trials, assigned)
        atomic_json(root/'summary.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--tasks', choices=TASKS, nargs='+', default=list(TASKS))
    parser.add_argument('--starts', choices=STARTS, nargs='+', default=['home'])
    parser.add_argument('--perturbed', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if args.worker:
        args.root.mkdir(parents=True, exist_ok=False)
        results = worker(read_json(args.request), args.root)
        raise SystemExit(2 if any(row.get('infrastructure_error') for row in results) else 0)
    else:
        result = campaign(args.root, args.episodes, args.tasks, args.starts, args.perturbed, args.execute, args.python)
        print(json.dumps(result, indent=2))
        raise SystemExit(2 if result['status'] == 'BLOCKED' else 0)
