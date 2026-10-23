"""20 physical trials; offline validation and missing runtime can never earn PASS."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import traceback
from .assets import MANIFEST
from .benchmark import CAMERA_EYE, CAMERA_LOOK, COLORS, CONTROL_DT, SCENE_REVISION, inside_bin
from .expert import Expert, FAILURES
from .evaluation import metrics
from .preflight import LAB_COMMIT, SIM
from .recording import Recorder
from .scenarios import plan
from dataclasses import asdict


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


def campaign_cases(mode, split, episodes, start_index=0):
    if mode != 'collection' and start_index != 0:
        raise ValueError('scenario offset is only valid for collection')
    if mode == 'qualification':
        return [dict(seed=seed,target=COLORS[seed%3]) for seed in range(20)]
    if mode == 'smoke':
        if type(episodes) is not int or not 1 <= episodes <= 3:
            raise ValueError('smoke campaigns allow one to three trials')
        return [dict(seed=seed,target=COLORS[seed%3]) for seed in range(episodes)]
    return [asdict(case) for case in plan(split,episodes,start_index)]


def campaign_report(trials,executed,completed,error,mode,cases,robot):
    if mode == 'qualification':
        report = aggregate(trials,executed,completed,error)
    else:
        valid = executed and completed and len(trials)==len(cases) and error is None
        report = dict(status=('COMPLETE' if all(row['success'] for row in trials) else 'COMPLETE_WITH_FAILURES') if valid else 'BLOCKED',
                      simulation_executed=executed,completed=completed,trials_executed=len(trials),
                      successes=sum(row['success'] for row in trials),error=error,trials=trials,
                      statistics=metrics(trials),threshold_achieved=False)
    report.update(mode=mode,embodiment=robot,planned_trials=len(cases))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['qualification','smoke','collection'],default='qualification')
    parser.add_argument('--split',choices=['train','validation','evaluation'],default='train')
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--start-index',type=int,default=0)
    parser.add_argument('--robot',choices=['openmanipulator_x','franka_panda'],default='openmanipulator_x')
    parser.add_argument('--preflight-only',action='store_true')
    known,_ = parser.parse_known_args()
    # Refuse to overwrite historical evidence, including failed or partial attempts.
    known.output.mkdir(parents=True,exist_ok=False)
    trials,executed,completed = [],False,False
    error = None
    app = None
    cases = []
    provenance = source_provenance()
    try:
        cases = campaign_cases(known.mode,known.split,known.episodes,known.start_index)
        if known.preflight_only:
            raise RuntimeError('preflight only; real Isaac Lab simulation was not executed')
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        if args.mode == 'collection' and not args.enable_cameras:
            raise ValueError('collection requires --enable_cameras')
        (args.output/'scenarios.json').write_text(json.dumps(cases,indent=2)+'\n')
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
        backend = IsaacBackend(args.device,args.enable_cameras,args.robot)
        executed = True
        checks = backend.qualify_articulation()
        (args.output/'articulation.json').write_text(json.dumps(checks,indent=2)+'\n')
        for case in cases:
            seed,target = case['seed'],case['target']
            obs = backend.reset(seed)
            executed = True
            initial_positions = {color:list(pose) for color,pose in obs.cubes.items()}
            safety_events = []
            from .panda import GRASP_Z as PANDA_GRASP_Z
            from .benchmark import GRASP_Z
            expert = Expert(target,PANDA_GRASP_Z if args.robot == 'franka_panda' else GRASP_Z)
            recorder = Recorder(args.output/f'trial_{seed:02d}',f'Pick up the {target} cube and place it in the bin.',
                metadata={'embodiment':args.robot,'joint_names':[*backend.arm_names,*backend.finger_names],
                          'joint_units':['radian']*len(backend.arm_names)+['meter']*2,'control_dt':CONTROL_DT,
                          'camera_eye':CAMERA_EYE,'camera_look':CAMERA_LOOK,
                          'camera_enabled':args.enable_cameras,'seed':seed,'target':target,
                          'initial_object_poses':initial_positions,'physics_engine':'physx',
                          'simulation_executed':True,'controller':'classical_only','evidence_source':'isaac_lab_physx',
                          'purpose':'collection' if args.mode == 'collection' else args.mode,
                          'split':args.split if args.mode == 'collection' else 'qualification',
                          **({'scenario_id':case['scenario_id'],'region':case['region']} if args.mode == 'collection' else {}),
                          'scene_revision':SCENE_REVISION,
                          'gripper_command_joint_range':[backend.gripper_closed,backend.gripper_open],
                          'gripper_mechanical_joint_limits':backend.finger_limits,
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
                        backend.safe_stop()
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
                              robot=args.robot,initial_object_poses=initial_positions,
                              pick_result=expert.lifted,placement_result=success,
                              safety_events=sorted(set(safety_events)),controller='classical_only',
                              simulation_executed=True)
                recorder.close(result)
                trials.append(result)
                _save(known.output,campaign_report(trials,True,False,None,known.mode,cases,known.robot),provenance)
                if not app.is_running():
                    raise RuntimeError('simulation application stopped before campaign completed')
            except BaseException:
                recorder.abort()
                raise
        completed = True
    except (Exception,SystemExit,KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
        if 'backend' in locals():
            try:
                (known.output/'diagnostics.json').write_text(json.dumps(backend.diagnostics(),indent=2,allow_nan=False)+'\n')
            except Exception:
                pass
        (known.output/'error.txt').write_text(traceback.format_exc())
    finally:
        report = campaign_report(trials,executed,completed,error,known.mode,cases,known.robot)
        _save(known.output,report,provenance)
        if 'backend' in locals():
            backend.safe_stop()
        if app is not None:
            app.close(wait_for_replicator=False)
    print(json.dumps(report,indent=2))
    return 0 if report['status'] in ('PASS','COMPLETE') else 1 if report['status'] in ('FAIL','COMPLETE_WITH_FAILURES') else 2


def source_provenance():
    root = Path(__file__).resolve().parent
    return {'relic_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'relic_worktree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
            'source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.glob('*.py'))}}


def _save(output,report,provenance=None):
    report['runtime_pin'] = {'isaacsim':SIM,'isaaclab_commit':LAB_COMMIT,'robot_commit':MANIFEST['commit']}
    report['utc'] = datetime.now(timezone.utc).isoformat()
    provenance = provenance or source_provenance()
    report.update(provenance)
    root = Path(__file__).resolve().parent
    changed = [name for name,digest in provenance['source_sha256'].items()
               if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest]
    report['source_changed_during_run'] = changed
    if changed:
        report.update(status='BLOCKED',threshold_achieved=False,error='runtime source changed during campaign')
    dest = output/'report.json'
    tmp = output/'report.json.tmp'
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    tmp.replace(dest)


if __name__ == '__main__':
    sys.exit(main())
