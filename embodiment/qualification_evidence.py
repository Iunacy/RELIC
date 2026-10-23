import argparse
import hashlib
import json
from pathlib import Path
from .benchmark import COLORS, TIMEOUT, inside_bin
from .qualify import aggregate
from .recording import validate_sample


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_campaign(root):
    root = Path(root)
    report = json.loads((root/'report.json').read_text())
    if report.get('mode') != 'qualification' or report.get('source_changed_during_run') != []:
        raise ValueError('qualification with unchanged source required')
    expected = aggregate(report['trials'],report['simulation_executed'],report['completed'],report['error'])
    if expected['status'] != 'PASS' or report['status'] != 'PASS':
        raise ValueError('complete eighteen-of-twenty gate required')
    articulation = json.loads((root/'articulation.json').read_text())
    if articulation.get('fixed_base') is not True or articulation.get('physical_checks_executed') is not True:
        raise ValueError('physical articulation checks required')
    evidence = []
    for trial in report['trials']:
        seed,target = trial['seed'],trial['target']
        if target != COLORS[seed%3]:
            raise ValueError('qualification target schedule changed')
        folder = root/f'trial_{seed:02d}'
        outcome = json.loads((folder/'outcome.json').read_text())
        metadata = json.loads((folder/'metadata.json').read_text())
        if outcome != trial or metadata.get('evidence_source') != 'isaac_lab_physx' or metadata.get('purpose') != 'qualification':
            raise ValueError('trial evidence or provenance mismatch')
        if metadata.get('seed') != seed or metadata.get('target') != target or metadata.get('simulation_executed') is not True:
            raise ValueError('trial metadata mismatch')
        rows = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        telemetry = [json.loads(line) for line in (folder/'telemetry.jsonl').read_text().splitlines()]
        if not rows or len(rows) != len(telemetry):
            raise ValueError('missing aligned trial telemetry')
        previous = -1.
        lifted = False
        wrong_color = False
        for row,physical in zip(rows,telemetry,strict=True):
            previous = validate_sample(row,previous)
            if abs(row['timestamp']-physical['timestamp']) > 1e-8:
                raise ValueError('telemetry timestamps differ')
            forces = physical.get('contact_forces',{}).get(target,())
            bilateral = len(forces) == 2 and all(type(force) in (int,float) and force > .005 for force in forces)
            lifted |= bool(physical['contacts'].get(target) is True and bilateral and physical['cube_poses'][target][2] >= .055)
            wrong_color |= any(pose[2] > .045 for color,pose in physical['cube_poses'].items() if color != target)
        if abs(rows[-1]['timestamp']-trial['duration']) > 1e-6 or trial['duration'] > TIMEOUT+1/30:
            raise ValueError('trial duration differs from telemetry')
        if trial['success']:
            if not lifted or wrong_color or any(row['unsafe'] or row['safety_events'] for row in telemetry):
                raise ValueError('success lacks safe physical grasp evidence')
            if trial.get('pick_result') is not True or trial.get('placement_result') is not True or trial.get('safety_events') != []:
                raise ValueError('inconsistent success labels')
            if not inside_bin(telemetry[-1]['cube_poses'][target]) or telemetry[-1]['cube_poses'][target] != trial['final_object_pose']:
                raise ValueError('final placement evidence mismatch')
        evidence.append({'seed':seed,'target':target,'success':trial['success'],'duration':trial['duration'],
                         'failure_reason':trial['failure_reason'],'physical_lift_with_bilateral_forces':lifted,
                         'wrong_color_lift':wrong_color,'sha256':{name:digest(folder/name) for name in
                         ('metadata.json','outcome.json','samples.jsonl','telemetry.jsonl')}})
    return {'schema':'relic.qualification_evidence.v1','status':'PASS','embodiment':report['embodiment'],
            'statistics':expected['statistics'],'runtime_pin':report['runtime_pin'],
            'relic_commit':report['relic_commit'],'source_sha256':report['source_sha256'],
            'articulation':articulation,'trials':evidence,
            'report_sha256':digest(root/'report.json'),'raw_artifact_root':str(root.resolve()),
            'claim':'classical physical embodiment qualification; no learned-policy inference'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    evidence = verify_campaign(args.campaign)
    with args.output.open('x') as stream:
        json.dump(evidence,stream,indent=2,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
