import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .assets import MANIFEST
from .benchmark import Action, CONTROL_DT, SCENE_REVISION
from .benchmark_report import validate_run
from .collection import atomic_json, hashes
from .dataset import load_episode, read_json
from .execution import ClosedLoop, ExecutionConfig
from .policy_service import RemotePolicy
from .preflight import LAB_COMMIT, SIM
from .qualify import _save, source_provenance
from .recording import Recorder
from .scenarios import plan


class RecordedBackend:
    def __init__(self, backend, recorder):
        self.backend, self.recorder = backend, recorder
        self.current = self.image = self.loop = None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @property
    def guard(self):
        return self.backend.guard

    @guard.setter
    def guard(self, value):
        self.backend.guard = value

    def step(self, action):
        phase = 'fallback' if self.loop.expert is not None else 'policy'
        self.recorder.append(self.current, action, phase, False, self.image)
        return self.backend.step(action)


def run_trial(backend, policy, case, output, provenance, config, running):
    observation = backend.reset(case.seed)
    metadata = {**asdict(case), **provenance, 'embodiment': backend.embodiment,
                'joint_names': [*backend.arm_names, *backend.finger_names], 'control_dt': CONTROL_DT,
                'simulation_executed': True, 'physics_engine': 'physx', 'evidence_source': 'isaac_lab_physx',
                'controller': 'vla_only', 'purpose': 'evaluation', 'scene_revision': SCENE_REVISION,
                'runtime_pin': {'isaacsim': SIM, 'isaaclab_commit': LAB_COMMIT,
                                'robot_commit': MANIFEST['commit']},
                'policy_service': policy.metadata}
    recorder = Recorder(output, f'Pick up the {case.target} cube and place it in the bin.', metadata)
    wrapper = RecordedBackend(backend, recorder)
    try:
        loop = ClosedLoop(wrapper, policy, case.target, config)
        wrapper.loop = loop
        while running():
            image = backend.frame()
            wrapper.current, wrapper.image = observation, image
            next_observation = loop.tick(observation, image)
            if next_observation is None:
                neutral = Action((0., 0., 0.), observation.gripper_open if observation.gripper_open is not None else 1.)
                recorder.append(observation, neutral, 'done' if loop.success else 'failed', loop.success, image)
                result = {**asdict(case), **loop.result(), 'duration': observation.time,
                          'robot': backend.embodiment, 'simulation_executed': True,
                          'final_object_pose': list(observation.cubes[case.target])}
                recorder.close(result)
                metadata['controller'] = result['controller']
                atomic_json(output/'metadata.json', {**read_json(output/'metadata.json'), **metadata})
                load_episode(output, expert_only=False)
                with (output/'execution.jsonl').open('x') as stream:
                    for trace in loop.action_trace:
                        stream.write(json.dumps(trace, allow_nan=False)+'\n')
                result['evidence_sha256'] = hashes(output)
                return result
            observation = next_observation
        raise RuntimeError('simulation application stopped before trial completed')
    finally:
        recorder.abort()
        backend.safe_stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--robot', choices=['openmanipulator_x', 'franka_panda'], required=True)
    parser.add_argument('--condition', choices=['a', 'b_only', 'a_to_b'], required=True)
    parser.add_argument('--training-plan', type=Path, required=True)
    parser.add_argument('--policy-port', type=int, default=8766)
    parser.add_argument('--no-fallback', action='store_true')
    known, _ = parser.parse_known_args()
    known.output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance()
    report = {'schema': 'relic.evaluation.v1', 'status': 'BLOCKED', 'mode': 'evaluation',
              'scene_revision': SCENE_REVISION, 'evidence_source': 'isaac_lab_physx',
              'embodiment': known.robot, 'condition': known.condition, 'trials': [],
              'simulation_executed': False, 'completed': False, 'error': None}
    app = backend = None
    try:
        training = read_json(known.training_plan)
        run = next(item for item in training['runs'] if item['condition'] == known.condition)
        if not run.get('dataset_digest'):
            raise ValueError('evaluation requires the experiment plan with a validated training dataset')
        if run['embodiment'] != known.robot:
            raise ValueError('training condition and evaluation embodiment differ')
        policy = RemotePolicy(known.policy_port)
        if policy.metadata.get('source_sha256') != provenance['source_sha256']:
            raise ValueError('policy service and simulator source versions differ')
        from .policy import checkpoint_digest
        checkpoint = Path(run['checkpoint'])
        if not checkpoint.is_dir() or checkpoint_digest(checkpoint) != policy.metadata['checkpoint_sha256']:
            raise ValueError('served checkpoint differs from the planned experiment checkpoint')
        report['training_run'] = run
        report['policy_service'] = policy.metadata
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError('Isaac Sim requires the isolated Python 3.11 environment')
        import importlib.metadata
        from packaging.version import Version
        if Version(importlib.metadata.version('isaacsim')) != Version(SIM):
            raise RuntimeError('Isaac Sim differs from pin')
        import isaaclab
        import subprocess
        lab_root = Path(isaaclab.__file__).resolve().parents[3]
        if subprocess.check_output(['git', '-C', str(lab_root), 'rev-parse', 'HEAD'], text=True).strip() != LAB_COMMIT:
            raise RuntimeError('Isaac Lab differs from pin')
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        if not args.enable_cameras:
            raise ValueError('policy evaluation requires --enable_cameras')
        app = AppLauncher(args).app
        from .isaac_backend import IsaacBackend
        backend = IsaacBackend(args.device, True, args.robot)
        report['simulation_executed'] = True
        atomic_json(args.output/'articulation.json', backend.qualify_articulation())
        config = ExecutionConfig(fallback_enabled=not args.no_fallback)
        assigned = run['evaluation_scenarios']
        cases = plan('evaluation', len(assigned))
        if len(cases) not in (20, 50) or assigned != [asdict(case) for case in cases]:
            raise ValueError('evaluation scenes differ from frozen training plan')
        for case in cases:
            result = run_trial(backend, policy, case, args.output/f'trial_{case.seed}', provenance, config, app.is_running)
            report['trials'].append(result)
            digest = result['fallback_protocol_sha256']
            if 'fallback_protocol_sha256' in report and report['fallback_protocol_sha256'] != digest:
                raise ValueError('fallback protocol changed during evaluation')
            report['fallback_protocol_sha256'] = digest
            _save(args.output, report, provenance)
            if report['source_changed_during_run']:
                raise RuntimeError('evaluation source changed')
        report['completed'] = True
        report['statistics'] = validate_run(report)
        report['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='BLOCKED', completed=False, error=f'{type(error).__name__}: {error}')
    finally:
        if backend is not None:
            backend.safe_stop()
        if app is not None:
            app.close(wait_for_replicator=False)
        _save(known.output, report, provenance)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    sys.exit(main())
