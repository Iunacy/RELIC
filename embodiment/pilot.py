"""Freeze a bounded paired pilot; collection remains an explicit separate command."""
import argparse
from dataclasses import asdict
import json
import os
import subprocess
from pathlib import Path
import sys
from .benchmark import CAMERA_EYE, CAMERA_LOOK, CONTROL_DT, SCENE_REVISION, layout
from .collection import atomic_json, locked, execute as run_process
from .contract import manifest
from .dataset import DOF
from .qualify import source_provenance
from .scenarios import plan, validate_plans


def prepare(root, count=12, python=sys.executable, data_python=sys.executable):
    if type(count) is not int or not 10 <= count <= 20:
        raise ValueError('pilot requires 10 to 20 episodes per robot')
    plans = {split: plan(split, count if split == 'train' else 6) for split in ('train', 'validation', 'evaluation')}
    validate_plans(plans)
    configuration = {'schema': 'viper.pilot.v1', 'scene_revision': SCENE_REVISION, **source_provenance(),
        'camera': {'eye': CAMERA_EYE, 'look': CAMERA_LOOK, 'color_order': 'RGB', 'observation_dt': CONTROL_DT},
        'contracts': {robot: manifest(robot) for robot in DOF},
        'scenarios': {split: [{**asdict(case), 'positions': layout(case.seed),
                              'instruction': f'Pick up the {case.target} cube and place it in the bin.'} for case in cases]
                      for split, cases in plans.items()},
        'simulation_executed': False, 'training_ready': False,
        'attempt_schedule': {split: [asdict(case) for case in plan(split, 19)] for split in plans},
        'split_commands': {robot: {split: [str(data_python), '-m', 'embodiment.collection',
            '--python', str(python), '--root', str(Path(root).resolve()/robot/split), '--robot', robot,
            '--split', split, '--target', str(len(cases)), '--batch-size', '2', '--max-jobs', '10',
            '--export', str(Path(root).resolve()/'exports'/robot/split)] for split, cases in plans.items()} for robot in DOF},
        'coverage': 'collection receipts retain every failed, incomplete and accepted attempt; replacements use frozen schedule',
        'commands': {robot: [str(data_python), '-m', 'embodiment.collection', '--python', str(python), '--root', str(Path(root).resolve()/robot/'train'),
            '--robot', robot, '--split', 'train', '--target', str(count), '--batch-size', '2', '--max-jobs', '10',
            '--export', str(Path(root).resolve()/'exports'/robot/'train')] for robot in DOF}}
    with locked(Path(root)):
        from .dataset import read_json
        path = Path(root)/'pilot.json'
        # JSON conversion canonicalizes tuple/list fields for restart comparisons.
        configuration = json.loads(json.dumps(configuration))
        if path.exists() and read_json(path) != configuration: raise ValueError('pilot configuration changed')
        atomic_json(path, configuration)
    return configuration


def run(root, count=12, python=sys.executable, data_python=sys.executable, groot_source=None,
        execute=False, reader_library_path=None):
    from .collection import hashes
    from .dataset import read_json
    root = Path(root).resolve()
    configuration = prepare(root, count, python, data_python)
    result = {'schema': 'viper.pilot_execution.v1', 'status': 'NOT_EXECUTED',
              'simulation_executed': False, 'training_ready': False, 'collections': {}, 'errors': []}
    if not execute: return {**result, 'commands': configuration['split_commands']}
    if groot_source is None: raise ValueError('actual upstream reader source required for pilot admission')
    with locked(root/'execution'):
        host_command = [str(python), '-m', 'embodiment.host', '--stage', 'simulation', '--volume', str(root)]
        host = subprocess.run(host_command, text=True, capture_output=True, timeout=30)
        try: report = json.loads(host.stdout)
        except ValueError: report = {'ready': False, 'blockers': [host.stderr or 'simulation host preflight failed']}
        if not report['ready']:
            result.update(status='BLOCKED', errors=report['blockers'], host=report)
            atomic_json(root/'execution/result.json', result)
            return result
        result['host'] = report
        source = source_provenance()
        if source['relic_worktree_dirty']: raise ValueError('physical pilot requires a clean committed source revision')
        for robot in DOF:
            for split, command in configuration['split_commands'][robot].items():
                label = robot+'/'+split
                job = root/'execution'/robot/split
                job.mkdir(parents=True, exist_ok=True)
                try:
                    dataset = root/'exports'/robot/split
                    receipt = job/'verified.json'
                    collection_root = root/robot/split
                    if receipt.exists():
                        saved = read_json(receipt)
                        if hashes(dataset) != saved['dataset_files'] or hashes(collection_root) != saved['collection_files']:
                            raise ValueError('completed pilot dataset or raw collection was changed')
                        result['collections'][label] = saved
                        continue
                    attempt = job/f'attempt-{len(list(job.glob("attempt-*"))):04d}'
                    attempt.mkdir()
                    if not dataset.exists():
                        process = run_process(command, attempt, 3600)
                        if process['returncode'] != 0: raise RuntimeError('collection incomplete; inspect preserved attempt log')
                    validation = subprocess.run([str(data_python), '-m', 'embodiment.dataset', '--validate-export', str(dataset)],
                                                capture_output=True, text=True, timeout=300)
                    if validation.returncode: raise ValueError('physical export validation failed: '+validation.stderr[-1500:])
                    rows = [json.loads(line) for line in (dataset/'meta/relic_sources.jsonl').read_text().splitlines()]
                    scheduled = {case['scenario_id'] for case in configuration['attempt_schedule'][split]}
                    if any(row['scenario_id'] not in scheduled or row['metadata'].get('source_sha256') != configuration['source_sha256'] for row in rows):
                        raise ValueError('pilot source or attempted scene differs from frozen plan')
                    result['simulation_executed'] = True
                    if any(row['metadata'].get('episode_contract') != 'viper.physical_episode.v2' for row in rows):
                        raise ValueError('pilot requires new physical terminal-stability evidence')
                    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
                    if reader_library_path: environment['LD_LIBRARY_PATH'] = str(reader_library_path)
                    upstream = subprocess.run([str(data_python), '-m', 'embodiment.upstream_validation', '--source', str(groot_source),
                        '--dataset', str(dataset), '--decode-video'], capture_output=True, text=True, timeout=300, env=environment)
                    if upstream.returncode: raise ValueError('upstream reader validation failed: '+upstream.stderr[-1500:])
                    saved = {'dataset': str(dataset), 'validation': json.loads(validation.stdout),
                             'reader_process_succeeded': True, 'reader_output': upstream.stdout,
                             'dataset_files': hashes(dataset), 'collection_files': hashes(collection_root)}
                    atomic_json(receipt, saved)
                    result['collections'][label] = saved
                except Exception as error:
                    result['errors'].append({'collection': label, 'error': f'{type(error).__name__}: {error}'})
                atomic_json(root/'execution/result.json', result)
        ready = len(result['collections']) == 6 and not result['errors']
        if source_provenance() != source: raise ValueError('source changed during pilot')
        result.update(status='VALIDATED' if ready else 'INCOMPLETE', simulation_executed=result['simulation_executed'] or bool(result['collections']), training_ready=ready)
        atomic_json(root/'execution/result.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--data-python', default=sys.executable)
    parser.add_argument('--groot-source', type=Path)
    parser.add_argument('--reader-library-path', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = run(args.root, args.episodes, args.python, args.data_python, args.groot_source, args.execute, args.reader_library_path)
    print(json.dumps(result, indent=2))
    raise SystemExit(2 if result['status'] in ('BLOCKED', 'INCOMPLETE') else 0)
