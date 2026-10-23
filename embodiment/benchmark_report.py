import argparse
import hashlib
import json
import math
from pathlib import Path
from .benchmark import SCENE_REVISION, TIMEOUT, inside_bin
from .evaluation import metrics
from .scenarios import plan
from .training import validate_matched

CONTROLLERS = ('vla_only', 'fallback_assisted', 'classical_only')


def validate_run(run):
    if run.get('schema') != 'relic.evaluation.v1' or run.get('scene_revision') != SCENE_REVISION:
        raise ValueError('evaluation schema or scene revision mismatch')
    if run.get('evidence_source') != 'isaac_lab_physx' or run.get('simulation_executed') is not True:
        raise ValueError('physical simulation evidence required')
    if run.get('completed') is not True or run.get('source_changed_during_run') != []:
        raise ValueError('incomplete campaign or changed runtime source')
    if run.get('embodiment') not in ('openmanipulator_x', 'franka_panda'):
        raise ValueError('unknown embodiment')
    if not run.get('runtime_pin') or not run.get('source_sha256') or not run.get('fallback_protocol_sha256'):
        raise ValueError('frozen runtime and fallback protocol required')
    if run.get('condition') not in ('classical', 'a', 'b_only', 'a_to_b'):
        raise ValueError('unknown evaluation condition')
    expected = {case.scenario_id: case for case in plan('evaluation', 20)}
    trials = run['trials']
    if len(trials) != 20:
        raise ValueError('twenty independent held-out trials required')
    seen = set()
    for row in trials:
        case = expected.get(row.get('scenario_id'))
        if case is None or case.scenario_id in seen or row.get('seed') != case.seed or row.get('target') != case.target:
            raise ValueError('held-out scenario mismatch or duplicate')
        seen.add(case.scenario_id)
        if type(row.get('success')) is not bool or type(row.get('simulation_executed')) is not bool or not row['simulation_executed']:
            raise ValueError('invalid physical outcome')
        duration = row.get('duration')
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0 or duration > TIMEOUT + 1 / 30:
            raise ValueError('invalid trial duration')
        controller = row.get('controller')
        triggers = row.get('fallback_triggers')
        if controller not in CONTROLLERS or not isinstance(triggers, list) or not all(isinstance(v, str) and v for v in triggers):
            raise ValueError('invalid controller attribution')
        if bool(triggers) != (controller == 'fallback_assisted'):
            raise ValueError('fallback attribution mismatch')
        if run.get('condition') == 'classical' and controller != 'classical_only':
            raise ValueError('classical condition contains policy execution')
        if run.get('condition') != 'classical' and controller == 'classical_only':
            raise ValueError('policy condition contains a classical-only trial')
        if not isinstance(row.get('safety_events'), list):
            raise ValueError('missing safety outcome')
        if row['success']:
            if row.get('failure_reason') is not None or row.get('pick_result') is not True or row.get('placement_result') is not True:
                raise ValueError('success lacks pick and placement evidence')
            if row['safety_events'] or not inside_bin(row['final_object_pose']):
                raise ValueError('unsafe or invalid successful placement')
        elif not isinstance(row.get('failure_reason'), str) or not row['failure_reason']:
            raise ValueError('failure reason required')
    summary = metrics(trials)
    summary.update(successes_by_controller={name: sum(row['success'] and row['controller'] == name for row in trials) for name in CONTROLLERS},
                   fallback_trigger_rate=sum(bool(row['fallback_triggers']) for row in trials) / len(trials),
                   safety_violation_trials=sum(bool(row['safety_events']) for row in trials),
                   mean_success_duration=sum(row['duration'] for row in trials if row['success']) / summary['successes'] if summary['successes'] else None)
    return summary


def compare(baseline, transfer):
    left, right = validate_run(baseline), validate_run(transfer)
    signature = validate_matched(baseline['training_run'], transfer['training_run'])
    for key in ('embodiment', 'scene_revision', 'runtime_pin', 'source_sha256', 'fallback_protocol_sha256'):
        if baseline[key] != transfer[key]:
            raise ValueError('evaluation protocols differ: ' + key)
    if baseline.get('condition') != 'b_only' or transfer.get('condition') != 'a_to_b':
        raise ValueError('invalid comparison conditions')
    return {'schema': 'relic.transfer_result.v1', 'matched_adaptation_signature': signature,
            'b_only': left, 'a_to_b': right,
            'success_rate_difference': right['success_rate'] - left['success_rate'],
            'vla_only_success_rate_difference': (right['successes_by_controller']['vla_only'] - left['successes_by_controller']['vla_only']) / 20,
            'limitations': ['twenty trials per condition; Wilson intervals describe each rate, not the paired difference']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--transfer', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.baseline] + ([args.transfer] if args.transfer else [])
    runs = [json.loads(path.read_text()) for path in paths]
    result = compare(*runs) if len(runs) == 2 else validate_run(runs[0])
    result['input_sha256'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
