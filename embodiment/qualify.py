"""20 physical trials; offline validation and missing runtime can never earn PASS."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback
from .assets import ARM, FINGERS, MANIFEST
from .benchmark import CAMERA_EYE, CAMERA_LOOK, COLORS, CONTROL_DT, inside_bin
from .expert import Expert, FAILURES
from .evaluation import metrics
from .preflight import LAB_COMMIT, SIM
from .recording import Recorder


def aggregate(trials, executed, completed, error=None):
    seeds = [t['seed'] for t in trials]
    successes = sum(t['success'] for t in trials)
    valid = executed and completed and len(trials)==20 and seeds==list(range(20)) and error is None
    return dict(status=('PASS' if successes>=18 else 'FAIL') if valid else 'BLOCKED',
                simulation_executed=executed,completed=completed,trials_executed=len(trials),
                successes=successes,required_successes=18,required_trials=20,
                threshold_achieved=bool(valid and successes>=18),
                failures=dict(Counter(t['failure_reason'] for t in trials if not t['success'])),
                error=error,trials=trials,statistics=metrics(trials))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    known,_ = parser.parse_known_args()
    # Refuse to overwrite historical evidence, including failed or partial attempts.
    known.output.mkdir(parents=True,exist_ok=False)
    trials,executed,completed = [],False,False
    error = None
    app = None
    try:
        if known.preflight_only:
            raise RuntimeError('preflight only; real Isaac Lab simulation was not executed')
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        if sys.version_info[:2] != (3,11):
            raise RuntimeError('selected Isaac Sim 5.1.0 runtime requires Python 3.11')
        import importlib.metadata
        from packaging.version import Version
        if Version(importlib.metadata.version('isaacsim')) != Version(SIM):
            raise RuntimeError('Isaac Sim version differs from pin')
        import isaaclab
        lab_root = Path(isaaclab.__file__).resolve().parents[3]
        commit = subprocess.check_output(['git','-C',str(lab_root),'rev-parse','HEAD'],text=True).strip()
        if commit != LAB_COMMIT:
            raise RuntimeError('Isaac Lab source commit differs from pin')
        launcher = AppLauncher(args)
        app = launcher.app
        from .isaac_backend import IsaacBackend
        backend = IsaacBackend(args.device,args.enable_cameras)
        executed = True
        checks = backend.qualify_articulation()
        (args.output/'articulation.json').write_text(json.dumps(checks,indent=2)+'\n')
        for seed in range(20):
            target = COLORS[seed%3]
            obs = backend.reset(seed)
            executed = True
            initial_positions = {color:list(pose) for color,pose in obs.cubes.items()}
            safety_events = []
            expert = Expert(target)
            recorder = Recorder(args.output/f'trial_{seed:02d}',f'Pick up the {target} cube and place it in the bin.',
                metadata={'embodiment':'openmanipulator_x','joint_names':[*ARM,*FINGERS],
                          'joint_units':['radian']*4+['meter']*2,'control_dt':CONTROL_DT,
                          'camera_eye':CAMERA_EYE,'camera_look':CAMERA_LOOK,
                          'camera_enabled':args.enable_cameras,'seed':seed,'target':target,
                          'initial_object_poses':initial_positions,'physics_engine':'physx',
                          'simulation_executed':True,'controller':'classical_only',
                          'purpose':'qualification','split':'evaluation','scene_revision':2,
                          'cube_yaw':'radial_about_openmanipulator_base_joint',
                          'light_intensity_range':[1800,2800],
                          'grasp_evidence':'bilateral_finger_contact_and_lift'})
            try:
                while app.is_running():
                    action = expert.update(obs)
                    terminal = expert.failure is not None or expert.phase=='done'
                    success = (expert.phase=='done' and expert.lifted and expert.released and expert.failure is None and inside_bin(obs.cubes[target]))
                    recorder.append(obs,action,expert.phase,success,backend.frame())
                    if terminal:
                        break
                    new_obs = backend.step(action)
                    safety_events.extend(new_obs.safety_events)
                    if new_obs.time <= obs.time:
                        expert.failure = 'unsafe_movement'
                        obs = new_obs
                        break
                    obs = new_obs
                success = bool(expert.phase=='done' and expert.lifted and expert.released and expert.failure is None and inside_bin(obs.cubes[target]))
                reason = None if success else (expert.failure or 'timeout')
                if reason is not None and reason not in FAILURES:
                    raise RuntimeError('unknown physical failure reason')
                result = dict(seed=seed,target=target,success=success,duration=obs.time,
                              failure_reason=reason,final_object_pose=list(obs.cubes[target]),phase=expert.phase,
                              robot='openmanipulator_x',initial_object_poses=initial_positions,
                              pick_result=expert.lifted,placement_result=success,
                              safety_events=sorted(set(safety_events)),controller='classical_only',
                              simulation_executed=True)
                recorder.close(result)
                trials.append(result)
                _save(known.output,aggregate(trials,True,False))
                if not app.is_running():
                    raise RuntimeError('simulation application stopped before campaign completed')
            except BaseException:
                recorder.file.close()
                raise
        completed = True
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        (known.output/'error.txt').write_text(traceback.format_exc())
    finally:
        report = aggregate(trials,executed,completed,error)
        _save(known.output,report)
        if app is not None:
            app.close()
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1 if report['status']=='FAIL' else 2


def _save(output,report):
    report['runtime_pin'] = {'isaacsim':SIM,'isaaclab_commit':LAB_COMMIT,'robot_commit':MANIFEST['commit']}
    report['utc'] = datetime.now(timezone.utc).isoformat()
    report['relic_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dest = output/'report.json'
    tmp = output/'report.json.tmp'
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    tmp.replace(dest)


if __name__ == '__main__':
    sys.exit(main())
