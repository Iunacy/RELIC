"""Compare hash-verified classical reference and actual learned-policy evaluation."""
import argparse
import json
from pathlib import Path
from .collection import hashes
from .dataset import read_json
from .paired import bootstrap, verify_evidence
from .replay import load_episode


def compare(reference_root, policy_report):
    reference_root, policy_report = Path(reference_root), Path(policy_report)
    policy = read_json(policy_report)
    if policy.get('observation_condition', {}).get('kind', 'baseline') != 'baseline':
        raise ValueError('classical reference comparison requires matched nominal observations')
    verify_evidence(policy_report.parent, policy)
    robot = policy['embodiment']
    configuration = read_json(reference_root/'experiment.json')
    if configuration['suite'] != 'heldout' or configuration['source_sha256'] != policy['source_sha256']:
        raise ValueError('reference and policy must use the same frozen source and held-out suite')
    report = read_json(reference_root/robot/'run/report.json')
    if not report['completed'] or report['source_changed_during_run'] != [] or report.get('error'):
        raise ValueError('classical reference campaign incomplete')
    if report['source_sha256'] != policy['source_sha256']: raise ValueError('reference worker source mismatch')
    reference = []
    for index, trial in enumerate(report['trials']):
        folder = reference_root/robot/'run'/f'trial_{index:04d}'
        if trial.get('evidence_sha256') != hashes(folder): raise ValueError('reference trial hashes changed')
        episode = load_episode(folder)
        if episode.outcome != {key: value for key, value in trial.items() if key != 'evidence_sha256'}:
            raise ValueError('classical outcome differs from raw evidence')
        if episode.metadata.get('source_sha256') != policy['source_sha256'] or episode.metadata.get('runtime_pin') != policy['runtime_pin']:
            raise ValueError('classical episode source or runtime differs')
        if trial.get('controller') != 'classical_only' or trial['perturbation']['kind'] != 'baseline':
            raise ValueError('comparison requires nominal classical controller reference')
        if any(trial.get(key) != value for key, value in configuration['cases'][index].items()):
            raise ValueError('reference scenario identity changed')
        reference.append(trial)
    result = bootstrap(reference, policy['trials'], baseline_metric='classical_controller_completion')
    return {'schema': 'viper.reference_policy_comparison.v1', **result, 'embodiment': robot,
            'condition': policy['condition'], 'raw_episode_hashes_verified': True,
            'policy_intervention_frequency': sum(bool(row.get('fallback_triggers') or row.get('rejected_policy_actions') or row.get('safety_events'))
                                                 for row in policy['trials'])/len(policy['trials']),
            'claim': 'policy autonomy versus privileged classical feasibility; no transfer-effect claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--policy-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.policy_report)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2)
