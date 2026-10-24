"""Opt-in A-to-B G3 workflow. Dry-run never launches training or a simulator."""
import argparse
import json
from pathlib import Path
import sys
import time

from .collection import atomic_json, locked
from .dataset import file_digest, read_json, validate_export
from .policy import checkpoint_digest, validate_checkpoint
from .qualify import source_provenance
from .training import dataset_digest
from .training_smoke import smoke, validate_resume
from .transfer import prepare
from .upstream_validation import validate_with_upstream, verify_source


def workflow(source, dataset_a, dataset_b, pretrained, output, execute=False, timeout=1800):
    source, dataset_a, dataset_b, pretrained, output = map(lambda path: Path(path).resolve(),
                                                         (source, dataset_a, dataset_b, pretrained, output))
    with locked(output):
        result = {'schema': 'viper.cross_robot_smoke.v1', 'status': 'BLOCKED', 'errors': [],
                  'training_executed': False, 'closed_loop_executed': False,
                  'stages': {}, 'paid_compute_go': False}
        started = time.monotonic()
        try:
            verify_source(source)
            datasets = {}
            for robot, path in (('openmanipulator_x', dataset_a), ('franka_panda', dataset_b)):
                report = validate_export(path)
                if report['embodiment'] != robot or report['split'] != 'train':
                    raise ValueError('cross smoke requires separate A/B training datasets')
                datasets[robot] = {'sha256': dataset_digest(path), 'validation': report,
                                  'upstream': validate_with_upstream(source, path, decode_video=True)}
            checked = validate_checkpoint(pretrained, require_processor=False)
            configuration = {'schema': 'viper.cross_robot_smoke_config.v1',
                             'source': source_provenance()['source_sha256'], 'upstream': str(source),
                             'datasets': datasets, 'pretrained': str(pretrained),
                             'pretrained_files': {name: file_digest(pretrained/name) for name in
                                 ['config.json', *checked['weight_paths']]}, 'timeout': timeout}
            config_path = output/'configuration.json'
            if config_path.exists() and read_json(config_path) != configuration:
                raise ValueError('smoke inputs changed; use a new run root')
            atomic_json(config_path, configuration)
            result['stages']['preflight'] = {'status': 'PASS', 'datasets': datasets, 'checkpoint': checked}
            if not execute:
                result['status'] = 'PREFLIGHT_ONLY'
                result['next_command'] = [sys.executable, '-m', 'embodiment.cross_smoke', '--source', str(source),
                    '--dataset-a', str(dataset_a), '--dataset-b', str(dataset_b), '--pretrained', str(pretrained),
                    '--output', str(output), '--execute', '--timeout', str(timeout)]
                return result
            import torch
            if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
            free, total = torch.cuda.mem_get_info()
            result['gpu'] = {'name': torch.cuda.get_device_name(), 'free_bytes': free, 'total_bytes': total,
                             'torch': torch.__version__, 'cuda': torch.version.cuda}
            if total < 40*1024**3 or free < 38*1024**3:
                raise RuntimeError('G3 requires at least 40 GiB total and 38 GiB free; run simulator separately')
            def training_stage(name, dataset, base):
                stage = output/name
                stage.mkdir(exist_ok=True)
                receipt = stage/'completed.json'
                if receipt.exists():
                    saved = read_json(receipt)
                    if saved['base_files'] != {filename: file_digest(base/filename) for filename in
                            ['config.json', *validate_checkpoint(base, require_processor=False)['weight_paths']]}:
                        raise ValueError('completed stage initialization changed')
                    if checkpoint_digest(Path(saved['checkpoint'])) != saved['checkpoint_sha256']:
                        raise ValueError('completed stage checkpoint was corrupted')
                    return saved
                attempt = stage/f'attempt-{len(list(stage.glob("attempt-*"))):04d}'
                candidates = sorted(stage.glob('attempt-*/training/checkpoint-*'),
                                    key=lambda path: (int(path.name.split('-')[-1]), path.parent.parent.name), reverse=True)
                resume = candidates[0] if candidates else None
                if resume is not None:
                    validate_resume(resume)
                measured = smoke(source, dataset, base, attempt, execute=True, timeout=timeout, resume_checkpoint=resume)
                result['stages'][name] = measured
                atomic_json(output/'result.json', result)
                if measured['status'] != 'PASS':
                    raise RuntimeError(name+' failed: '+'; '.join(measured['errors']))
                checkpoint = attempt/'training'
                saved = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_digest(checkpoint),
                         'base_files': {filename: file_digest(base/filename) for filename in
                             ['config.json', *validate_checkpoint(base, require_processor=False)['weight_paths']]},
                         'measured': measured}
                atomic_json(receipt, saved)
                return saved
            a = training_stage('robot_a', dataset_a, pretrained)
            result['stages']['robot_a'] = a
            result['training_executed'] = True
            initialization = output/'target_initialization'
            if initialization.exists():
                transfer = read_json(initialization/'initialization.json')
                if (checkpoint_digest(initialization) != transfer['checkpoint_sha256'] or
                        checkpoint_digest(Path(a['checkpoint'])) != transfer['source_checkpoint_sha256'] or
                        transfer['target_dataset_sha256'] != dataset_digest(dataset_b)):
                    raise ValueError('saved transfer initialization integrity failure')
            else:
                transfer = prepare(Path(a['checkpoint']), pretrained, dataset_b, source, initialization)
            result['stages']['transfer'] = transfer
            result['stages']['robot_b'] = training_stage('robot_b', dataset_b, initialization)
            verify_source(source)
            if source_provenance()['source_sha256'] != configuration['source']:
                raise ValueError('VIPER source changed during smoke execution')
            result.update(status='OPTIMIZER_AND_INFERENCE_PASS', training_executed=True,
                          remaining_gate='real closed-loop Robot B simulation; this status is not full G3 PASS')
        except Exception as error:
            result['errors'].append(f'{type(error).__name__}: {error}')
        finally:
            result['wall_seconds'] = time.monotonic()-started
            atomic_json(output/'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'dataset-a', 'dataset-b', 'pretrained', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='opt in on an already approved GPU host; never provisions hardware')
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    result = workflow(args.source, args.dataset_a, args.dataset_b, args.pretrained, args.output, args.execute, args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] != 'BLOCKED' else 2


if __name__ == '__main__': raise SystemExit(main())
