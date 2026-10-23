import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path

from .collection import atomic_json, balanced_selection, hashes, inspect_job
from .dataset import DOF, file_digest, read_json
from .qualify import source_provenance
from .scenarios import SPLITS


def inspect_collection(root):
    root = Path(root).resolve()
    configuration = read_json(root/'collection.json')
    robot, split = configuration['embodiment'], configuration['split']
    if configuration.get('schema') != 'relic.collection.v1' or robot not in DOF or split not in SPLITS:
        raise ValueError('invalid collection manifest')
    if not isinstance(configuration.get('source_sha256'), dict) or not configuration['source_sha256']:
        raise ValueError('missing collection source provenance')
    accepted, failures, jobs = [], [], []
    next_index = 0
    with (root/'.lock').open('rb') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('collection is still running') from None
        for job in sorted((root/'jobs').glob('job-*')):
            if not (job/'request.json').exists():
                raise ValueError('collection job lacks its request')
            receipt = inspect_job(job, robot, split, configuration['source_sha256'])
            if receipt['request']['start'] != next_index:
                raise ValueError('job ranges overlap or contain a gap')
            next_index += receipt['request']['count']
            accepted.extend(job/'run'/item['folder'] for item in receipt['successes'])
            failures.extend({'job': str(job), **item} for item in receipt['failures'])
            jobs.append({'path': str(job), 'request_sha256': file_digest(job/'request.json'),
                         'receipt_sha256': file_digest(job/'receipt.json'),
                         'process': read_json(job/'process.json') if (job/'process.json').exists() else None,
                         'log_sha256': file_digest(job/'process.log') if (job/'process.log').exists() else None})
        target = {'train': 60, 'validation': 12, 'evaluation': 20}[split]
        selected, surplus = balanced_selection(accepted, split, target)
        episodes = []
        for path in accepted:
            metadata = read_json(path/'metadata.json')
            episodes.append({'path': str(path), 'scenario_id': metadata['scenario_id'],
                             'seed': metadata['seed'], 'target': metadata['target'], 'region': metadata['region'],
                             'selected_for_initial_target': path in selected, 'sha256': hashes(path)})
        seeds = [item['seed'] for item in episodes]
        return {'root': str(root), 'status': 'VALIDATED', 'embodiment': robot, 'split': split,
                'manifest_sha256': file_digest(root/'collection.json'),
                'frozen_source_sha256': configuration['source_sha256'],
                'successful_episodes': len(episodes), 'balanced_successful_episodes': len(selected),
                'initial_target': target, 'initial_target_complete': len(selected) == target,
                'surplus_successes': len(surplus), 'attempts': next_index,
                'seed_range': [min(seeds), max(seeds)] if seeds else None,
                'colors': dict(Counter(item['target'] for item in episodes)),
                'regions': dict(Counter(str(item['region']) for item in episodes)),
                'episodes': episodes, 'failures': failures, 'jobs': jobs}


def inventory(roots):
    collections, errors, seen = [], [], set()
    counts = {robot: {split: 0 for split in SPLITS} for robot in DOF}
    for root in roots:
        try:
            result = inspect_collection(root)
            keys = {(result['embodiment'], item['scenario_id']) for item in result['episodes']}
            if len(keys) != len(result['episodes']) or seen.intersection(keys):
                raise ValueError('duplicate accepted scenarios across collection roots')
            seen.update(keys)
            counts[result['embodiment']][result['split']] += result['successful_episodes']
            collections.append(result)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({'root': str(Path(root).resolve()), 'error': f'{type(error).__name__}: {error}'})
    return {'schema': 'relic.dataset_inventory.v1', 'status': 'BLOCKED' if errors else 'VALIDATED',
            'counts': counts, 'collections': collections, 'errors': errors,
            'training_executed': False, 'auditor': source_provenance()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ('status', 'counts', 'errors')}, indent=2))
    return 0 if result['status'] == 'VALIDATED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
