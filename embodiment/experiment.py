import argparse
from collections import Counter, deque
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

from .assets import MANIFEST
from .benchmark import Action, BIN, CONTROL_DT, GRASP_Z, SCENE_REVISION, inside_bin
from .collection import atomic_json, execute, hashes
from .dataset import DOF, read_json
from .evaluation import metrics
from .expert import Expert
from .perturbations import KINDS, Perturbation, plan_robustness
from .preflight import LAB_COMMIT, SIM
from .qualify import source_provenance
from .recording import Recorder
from .scenarios import plan


def cases(suite, count, conditions=KINDS):
    if suite == 'robustness':
        return plan_robustness(count, conditions)
    if suite != 'heldout' or count != 20:
        raise ValueError('held-out reference campaigns require the frozen twenty cases')
    return [{**asdict(case), 'namespace': 'relic.heldout.v1', 'perturbation': asdict(Perturbation())}
            for case in plan('evaluation', 20)]


def run_trial(backend, case, output, provenance, running=lambda: True):
    perturbation = Perturbation(**case['perturbation'])
    positions = perturbation.positions(case['seed'])
    observation = backend.reset(case['seed'], cube_positions=positions)
    robot = backend.embodiment
    from .panda import GRASP_Z as PANDA_GRASP_Z
    expert = Expert(case['target'], PANDA_GRASP_Z if robot == 'franka_panda' else GRASP_Z)
    metadata = {**case, **provenance, 'embodiment': robot, 'scene_revision': SCENE_REVISION,
                'joint_names': [*backend.arm_names, *backend.finger_names], 'control_dt': CONTROL_DT,
                'controller': 'classical_only', 'purpose': 'robustness' if case['split'] == 'robustness' else 'evaluation',
                'evidence_source': 'isaac_lab_physx', 'physics_engine': 'physx', 'simulation_executed': True,
                'runtime_pin': {'isaacsim': SIM, 'isaaclab_commit': LAB_COMMIT, 'robot_commit': MANIFEST['commit']},
                'initial_object_poses': observation.cubes, 'camera_enabled': backend.camera is not None}
    recorder = Recorder(output, f"Pick up the {case['target']} cube and place it in the bin.", metadata)
    queue = deque()
    errors, events, tick = [], list(observation.safety_events), 0
    latency = int(perturbation.magnitude) if perturbation.kind == 'control_latency' else 0
    try:
        with (output/'execution.jsonl').open('x') as execution:
            while running():
                requested = expert.update(perturbation.observe(observation, tick))
                terminal = expert.failure is not None or expert.phase == 'done'
                success = bool(expert.phase == 'done' and expert.lifted and expert.released and expert.failure is None and inside_bin(observation.cubes[case['target']]))
                queue.append(requested)
                action = queue.popleft() if len(queue) > latency else Action((0., 0., 0.), observation.gripper_open)
                if terminal:
                    recorder.append(observation, Action((0., 0., 0.), observation.gripper_open), expert.phase, success, backend.frame())
                    backend.safe_stop()
                    break
                previous, frame = observation, backend.frame()
                observation = backend.step(action)
                events.extend(observation.safety_events)
                if observation.time <= previous.time:
                    if observation.time == previous.time and observation.unsafe:
                        expert.fail('unsafe_movement')
                        success = False
                        recorder.append(observation, Action((0., 0., 0.), observation.gripper_open), expert.phase, False, frame)
                        execution.write(json.dumps({'timestamp': previous.time, 'requested_action': [*requested.delta, requested.gripper],
                            'rejected_action': [*action.delta, action.gripper], 'applied_action': None,
                            'safety_events': list(observation.safety_events)}, allow_nan=False)+'\n')
                        backend.safe_stop()
                        break
                    raise RuntimeError('simulator stalled or returned nonmonotonic time')
                recorder.append(previous, action, expert.phase, False, frame)
                target = tuple(value+delta for value, delta in zip(previous.ee, action.delta, strict=True))
                error = math.dist(observation.ee, target)
                errors.append(error)
                execution.write(json.dumps({'timestamp': previous.time, 'requested_action': [*requested.delta, requested.gripper],
                    'applied_action': [*action.delta, action.gripper], 'desired_ee': target,
                    'observed_next_ee': observation.ee, 'tracking_error_m': error}, allow_nan=False)+'\n')
                execution.flush()
                tick += 1
            else:
                raise RuntimeError('simulation stopped before trial completion')
        result = {**case, 'robot': robot, 'controller': 'classical_only', 'success': success,
                  'duration': observation.time, 'failure_reason': None if success else expert.failure or 'timeout',
                  'pick_result': expert.lifted, 'placement_result': success,
                  'final_object_pose': list(observation.cubes[case['target']]),
                  'safety_events': sorted(set(events)), 'fallback_triggers': [],
                  'simulation_executed': True, 'recovery_behavior': 'not_needed' if success else 'safe_stop_no_retry',
                  'tracking_samples': len(errors),
                  'mean_tracking_error_m': sum(errors)/len(errors) if errors else None,
                  'peak_tracking_error_m': max(errors) if errors else None,
                  'placement_error_xy_m': math.dist(observation.cubes[case['target']][:2], BIN[:2])}
        recorder.close(result)
        from .replay import load_episode
        load_episode(output)
        result['evidence_sha256'] = hashes(output)
        return result
    finally:
        recorder.abort()
        backend.safe_stop()


def aggregate(trials, planned):
    expected = {(robot, case['scenario_id']) for robot in DOF for case in planned}
    seen = set()
    assigned = {case['scenario_id']: case for case in planned}
    for trial in trials:
        key = trial.get('robot'), trial.get('scenario_id')
        if key not in expected or key in seen:
            raise ValueError('unexpected or duplicate experiment result')
        seen.add(key)
        if any(trial.get(field) != value for field, value in assigned[key[1]].items()):
            raise ValueError('experiment result differs from assigned condition')
        if not isinstance(trial.get('safety_events'), list) or not isinstance(trial.get('fallback_triggers'), list):
            raise ValueError('missing safety or fallback accounting')
        for field in ('duration', 'mean_tracking_error_m', 'peak_tracking_error_m', 'placement_error_xy_m'):
            value = trial.get(field)
            if field in ('mean_tracking_error_m', 'peak_tracking_error_m') and value is None and type(trial.get('tracking_samples')) is int and trial['tracking_samples'] == 0 and trial.get('duration') == 0 and trial.get('failure_reason') == 'unsafe_movement' and trial.get('success') is False:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid measured experiment metric')
        if any(type(trial.get(field)) is not bool for field in ('pick_result', 'placement_result')):
            raise ValueError('invalid grasp or placement outcome')
        if trial.get('success') is False and (not isinstance(trial.get('failure_reason'), str) or not trial['failure_reason']):
            raise ValueError('failed physical result lacks a reason')
        if trial.get('controller') != 'classical_only':
            raise ValueError('unsupported experiment controller')
        if trial['success'] and (trial.get('pick_result') is not True or trial.get('placement_result') is not True or trial.get('failure_reason') is not None or trial['safety_events']):
            raise ValueError('inconsistent physical success')
        if trial.get('simulation_executed') is not True or type(trial.get('success')) is not bool:
            raise ValueError('experiment requires explicit physical outcomes')
    summaries = {}
    for robot in DOF:
        summaries[robot] = {}
        for condition in sorted({case['perturbation']['kind'] for case in planned}):
            rows = [row for row in trials if row['robot'] == robot and row['perturbation']['kind'] == condition]
            measured = metrics(rows)
            tracked = [row for row in rows if row['mean_tracking_error_m'] is not None]
            measured.update(safety_event_trials=sum(bool(row['safety_events']) for row in rows),
                fallback_trials=sum(bool(row['fallback_triggers']) for row in rows),
                grasp_failures=sum(row['failure_reason'] == 'failed_grasp' for row in rows),
                placement_failures=sum(row['pick_result'] and not row['placement_result'] for row in rows),
                mean_tracking_error_m=sum(row['mean_tracking_error_m'] for row in tracked)/len(tracked) if tracked else None,
                tracking_measured_trials=len(tracked), recovery_policy='safe_stop_no_retry',
                mean_task_duration=sum(row['duration'] for row in rows)/len(rows) if rows else None)
            summaries[robot][condition] = measured
    return {'schema': 'relic.experiment_summary.v1', 'completed': seen == expected,
            'planned_trials': len(expected), 'physical_trials': len(seen), 'missing_results': sorted(expected-seen),
            'by_embodiment': summaries, 'claim': 'classical robustness/reference simulation; no learned transfer'}


def worker(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    request = read_json(args.request)
    report = {'schema': 'relic.experiment_worker.v1', 'trials': [], 'completed': False,
              'simulation_executed': False, 'source_changed_during_run': [], 'error': None}
    provenance = source_provenance()
    report.update(provenance)
    app = backend = None
    try:
        from packaging.version import Version
        if sys.version_info[:2] != (3, 11) or Version(importlib.metadata.version('isaacsim')) != Version(SIM):
            raise RuntimeError('worker requires pinned Isaac Sim Python 3.11 runtime')
        import isaaclab
        lab = Path(isaaclab.__file__).resolve().parents[3]
        if subprocess.check_output(['git', '-C', str(lab), 'rev-parse', 'HEAD'], text=True).strip() != LAB_COMMIT:
            raise ValueError('Isaac Lab differs from pin')
        if request['source_sha256'] != provenance['source_sha256']:
            raise ValueError('worker source differs from frozen experiment')
        app = AppLauncher(args).app
        from .isaac_backend import IsaacBackend
        backend = IsaacBackend(args.device, args.enable_cameras, request['robot'])
        report['simulation_executed'] = True
        for index, case in enumerate(request['cases']):
            report['trials'].append(run_trial(backend, case, args.output/f'trial_{index:04d}', provenance, app.is_running))
            atomic_json(args.output/'report.json', report)
        report['completed'] = True
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = f'{type(error).__name__}: {error}'
        (args.output/'error.txt').write_text(traceback.format_exc())
    finally:
        report['source_changed_during_run'] = [] if source_provenance()['source_sha256'] == provenance['source_sha256'] else ['runtime_source']
        report.update(provenance)
        atomic_json(args.output/'report.json', report)
        if backend is not None: backend.safe_stop()
        if app is not None: app.close(wait_for_replicator=False)
    return 0 if report['completed'] and not report['source_changed_during_run'] else 2


def campaign(root, suite='robustness', count=3, conditions=KINDS, python=sys.executable, timeout=1800, runner=execute):
    root = Path(root).resolve()
    planned = cases(suite, count, conditions)
    configuration = {'schema': 'relic.experiment.v1', 'suite': suite, 'cases': planned,
                     'source_sha256': source_provenance()['source_sha256']}
    from .collection import locked
    with locked(root):
        manifest = root/'experiment.json'
        if manifest.exists():
            if read_json(manifest) != configuration:
                raise ValueError('experiment configuration or runtime changed')
        else:
            atomic_json(manifest, configuration)
        trials, errors = [], []
        for robot in DOF:
            job = root/robot
            job.mkdir(exist_ok=True)
            request = {'robot': robot, 'cases': planned, 'source_sha256': configuration['source_sha256']}
            if not (job/'request.json').exists(): atomic_json(job/'request.json', request)
            elif read_json(job/'request.json') != request: raise ValueError('experiment request corrupted')
            if not (job/'run/report.json').exists() and not (job/'process.log').exists():
                command = [str(python), '-m', 'embodiment.experiment', '--worker', '--request', str(job/'request.json'),
                           '--output', str(job/'run'), '--headless', '--device', 'cuda:0']
                atomic_json(job/'command.json', command)
                runner(command, job, timeout)
            try:
                report = read_json(job/'run/report.json')
                if report.get('source_sha256') != configuration['source_sha256'] or report.get('source_changed_during_run') != []:
                    raise ValueError('worker source provenance mismatch')
                if report.get('error'): errors.append({'robot': robot, 'error': report['error']})
                for index, result in enumerate(report['trials']):
                    folder = job/'run'/f'trial_{index:04d}'
                    from .replay import load_episode
                    episode = load_episode(folder)
                    if (result.get('robot') != robot or episode.metadata['embodiment'] != robot
                            or episode.metadata['source_sha256'] != configuration['source_sha256']
                            or hashes(folder) != result['evidence_sha256'] or episode.metadata['scenario_id'] != planned[index]['scenario_id']
                            or episode.outcome != {key: value for key, value in result.items() if key != 'evidence_sha256'}):
                        raise ValueError('experiment episode identity or checksum mismatch')
                    aggregate([result], planned)
                    trials.append(result)
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                errors.append({'robot': robot, 'error': f'{type(error).__name__}: {error}'})
        summary = aggregate(trials, planned)
        summary.update(errors=errors, trials=trials, suite=suite,
                       status='COMPLETE' if summary['completed'] and not errors else 'INCOMPLETE')
        atomic_json(root/'summary.json', summary)
        return summary


def main():
    if '--worker' in sys.argv:
        return worker([arg for arg in sys.argv[1:] if arg != '--worker'])
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--suite', choices=['robustness', 'heldout'], default='robustness')
    parser.add_argument('--scenarios', type=int, default=3)
    parser.add_argument('--conditions', choices=KINDS, nargs='+', default=list(KINDS))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    result = campaign(args.root, args.suite, args.scenarios, args.conditions, args.python, args.timeout)
    print(json.dumps({key: result[key] for key in ('status', 'physical_trials', 'planned_trials', 'errors')}, indent=2))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
