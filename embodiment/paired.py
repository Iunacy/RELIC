"""Scene-paired, deterministic analysis. Synthetic inputs never become measured evidence."""
import argparse
from collections import Counter
import hashlib
import json
import random
from pathlib import Path

RESAMPLES = 10000
BOOTSTRAP_SEED = 20261024
PRIMARY_BUDGET = 100
PRIMARY_SCENES = 50
PRIMARY_EFFECT = .20


def category(row):
    if type(row.get('success')) is not bool:
        raise ValueError('explicit Boolean trial outcome required')
    for key in ('safety_events', 'fallback_triggers', 'rejected_policy_actions', 'policy_errors'):
        if not isinstance(row.get(key, []), list):
            raise ValueError('invalid intervention ledger')
    if row.get('infrastructure_error'):
        if row['success']:
            raise ValueError('infrastructure failure cannot be a success')
        return 'infrastructure_failure'
    if row.get('safety_events') or row.get('rejected_policy_actions') or row.get('failure_reason') == 'unsafe_movement':
        return 'safety_rejection_or_stop'
    controller = row.get('controller')
    if controller not in ('vla_only', 'fallback_assisted', 'classical_only'):
        raise ValueError('unknown controller attribution')
    if bool(row.get('fallback_triggers')) != (controller == 'fallback_assisted'):
        raise ValueError('fallback attribution mismatch')
    if not row['success']:
        return 'policy_failure'
    if row.get('policy_errors') and controller == 'vla_only':
        raise ValueError('policy error cannot count as autonomous success')
    return {'vla_only': 'vla_only_success', 'fallback_assisted': 'fallback_assisted_completion',
            'classical_only': 'classical_controller_completion'}[controller]


def bootstrap(baseline, transfer, seed=BOOTSTRAP_SEED, baseline_metric='vla_only_success'):
    if baseline_metric not in ('vla_only_success', 'classical_controller_completion'):
        raise ValueError('unknown baseline success metric')
    def indexed(rows):
        result = {}
        for row in rows:
            identity = row.get('scenario_id')
            if not isinstance(identity, str) or not identity or identity in result:
                raise ValueError('missing or duplicate scene identity')
            result[identity] = category(row)
        return result
    left, right = indexed(baseline), indexed(transfer)
    if not left or set(left) != set(right):
        raise ValueError('paired analysis requires every scene in both conditions')
    identities = sorted(left)
    differences = [int(right[key] == 'vla_only_success')-int(left[key] == baseline_metric) for key in identities]
    rng = random.Random(seed)
    size = len(differences)
    samples = sorted(sum(differences[rng.randrange(size)] for _ in range(size))/size for _ in range(RESAMPLES))
    def quantile(p):
        index = (RESAMPLES-1)*p
        low = int(index)
        return samples[low]+(samples[min(low+1, RESAMPLES-1)]-samples[low])*(index-low)
    wins, losses = differences.count(1), differences.count(-1)
    interval = [quantile(.025), quantile(.975)]
    return {'baseline_metric': baseline_metric, 'scene_count': size, 'scene_ids': identities, 'difference': sum(differences)/size,
            'paired_bootstrap_95': interval, 'resamples': RESAMPLES, 'bootstrap_seed': seed,
            'wins': wins, 'losses': losses, 'ties': size-wins-losses, 'discordant_pairs': wins+losses,
            'baseline_categories': dict(Counter(left.values())), 'transfer_categories': dict(Counter(right.values())),
            'denominator': 'all assigned scenes, including safety and infrastructure failures',
            'training_variance': 'not estimated by scene bootstrap'}


def compare(baseline, transfer):
    if baseline.get('observation_condition') != transfer.get('observation_condition'):
        raise ValueError('policy observation disturbances differ across compared conditions')
    from .benchmark_report import validate_run
    from .training import validate_matched
    for run in (baseline, transfer): validate_run(run)
    signature = validate_matched(baseline['training_run'], transfer['training_run'])
    for key in ('embodiment', 'scene_revision', 'runtime_pin', 'source_sha256', 'fallback_protocol_sha256'):
        if baseline[key] != transfer[key]: raise ValueError('unmatched evaluation: '+key)
    result = bootstrap(baseline['trials'], transfer['trials'])
    budget = baseline['training_run'].get('demonstration_budget')
    eligible = budget == PRIMARY_BUDGET and result['scene_count'] == PRIMARY_SCENES
    return {'schema': 'viper.paired_result.v1', **result, 'matched_adaptation_signature': signature,
            'evidence_source': 'validated_report_claims; raw episode hashes must also be audited',
            'primary_rule': {'b_demonstrations': PRIMARY_BUDGET, 'scenes': PRIMARY_SCENES, 'effect': PRIMARY_EFFECT},
            'primary_eligible': eligible,
            'primary_threshold_met': bool(eligible and result['difference'] >= PRIMARY_EFFECT and result['paired_bootstrap_95'][0] > 0),
            'claim': 'related-robot initialization comparison; exclusive transfer attribution requires E5'}


def verify_evidence(root, report):
    from .collection import hashes
    from .replay import load_episode
    from .benchmark_report import validate_run
    validate_run(report)
    for trial in report['trials']:
        folder = Path(root)/f"trial_{trial['seed']}"
        if not trial.get('evidence_sha256') or hashes(folder) != trial['evidence_sha256']:
            raise ValueError('trial evidence missing or corrupted')
        episode = load_episode(folder)
        if episode.outcome != {key: value for key, value in trial.items() if key != 'evidence_sha256'}:
            raise ValueError('reported outcome differs from recorded episode')
        if episode.metadata.get('source_sha256') != report['source_sha256']:
            raise ValueError('episode source differs from evaluation')
        if episode.metadata.get('policy_service', {}).get('checkpoint_sha256') != report.get('policy_service', {}).get('checkpoint_sha256'):
            raise ValueError('trial checkpoint differs from served model')
        if episode.metadata.get('observation_condition') != report.get('observation_condition'):
            raise ValueError('trial observation disturbance differs from evaluation report')
        from .perception import verify_trace
        verify_trace(folder/'perception.jsonl', episode.metadata, episode.rows)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--transfer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in (args.baseline, args.transfer)]
    for path, report in zip((args.baseline, args.transfer), reports, strict=True):
        verify_evidence(path.parent, report)
    result = compare(*reports)
    result['raw_episode_hashes_verified'] = True
    result['evidence_source'] = 'hash-verified Isaac evaluation episodes'
    result['input_sha256'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.baseline, args.transfer)}
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__': main()
