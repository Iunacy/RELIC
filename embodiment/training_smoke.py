import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .collection import atomic_json
from .dataset import DOF, VIDEO_KEY, read_json, validate_export
from .policy import GrootPolicy, PolicyInput, prepare_training_checkpoint, validate_checkpoint
from .training import Budget, dataset_digest, training_arguments, validate_upstream_arguments
from .upstream_validation import validate_with_upstream, verify_source


def first_observation(dataset, robot):
    import av
    import pyarrow.parquet as pq
    info = read_json(dataset/'meta/info.json')
    row = pq.read_table(dataset/info['data_path'].format(episode_chunk=0, episode_index=0)).to_pylist()[0]
    tasks = {item['task_index']: item['task'] for item in
             (json.loads(line) for line in (dataset/'meta/tasks.jsonl').read_text().splitlines())}
    video = dataset/info['video_path'].format(episode_chunk=0, episode_index=0, video_key=VIDEO_KEY)
    with av.open(str(video)) as container:
        image = next(container.decode(video=0)).to_ndarray(format='rgb24')
    joints = DOF[robot] + 2
    state = row['observation.state']
    return PolicyInput(robot, tasks[row['task_index']], tuple(state[:joints]), tuple(state[joints:2*joints]),
                       tuple(state[2*joints:2*joints+3]), state[-1], image)


def validate_resume(checkpoint, maximum_step=2):
    import zipfile
    from .dataset import file_digest
    checkpoint = Path(checkpoint).resolve()
    candidates = [path for path in checkpoint.parent.glob('checkpoint-*')
                  if path.is_dir() and path.name.removeprefix('checkpoint-').isdigit()]
    if candidates and max(candidates, key=lambda path: int(path.name.split('-')[-1])) != checkpoint:
        raise ValueError('upstream Boolean resume selects the latest checkpoint; requested checkpoint is older')
    validate_checkpoint(checkpoint)
    trainer = read_json(checkpoint/'trainer_state.json')
    step = trainer.get('global_step')
    if type(step) is not int or not 0 < step <= maximum_step:
        raise ValueError('resume checkpoint step is outside the requested training schedule')
    integrity = {}
    for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        path = checkpoint/name
        if not path.is_file() or not zipfile.is_zipfile(path):
            raise ValueError('resume requires complete upstream optimizer, scheduler and RNG archives')
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError('corrupted resume archive')
        integrity[name] = file_digest(path)
    from .policy import checkpoint_digest
    return {'checkpoint': str(checkpoint), 'step': step, 'checkpoint_sha256': checkpoint_digest(checkpoint),
            'optimizer_state_sha256': integrity}


def smoke(source, dataset, base, output, execute=False, timeout=1800, minimum_vram_gib=40, resume_checkpoint=None):
    source, dataset, base, output = (Path(path).resolve() for path in (source, dataset, base, output))
    if type(timeout) is not int or timeout <= 0 or minimum_vram_gib < 40:
        raise ValueError('positive timeout and at least 40 GiB VRAM are required')
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(steps=2, batch_size=1, save_steps=2)
    result = {'schema': 'relic.training_smoke.v1', 'status': 'BLOCKED', 'budget': asdict(budget),
              'training_executed': False, 'checkpoint_reload_verified': False,
              'inference_executed': False, 'safe_action_chunk_verified': False,
              'safety_validation_scope': 'decoded finite action dimensions and scalar limits; workspace, IK and physical tracking require closed-loop execution',
              'closed_loop_executed': False, 'peak_training_memory_bytes': None,
              'errors': [], 'dataset_digest': None}
    started = time.monotonic()
    try:
        verify_source(source)
        validation = validate_export(dataset)
        if validation['split'] != 'train':
            raise ValueError('smoke training requires a validated training split')
        result['dataset_digest'] = dataset_digest(dataset)
        result['upstream'] = validate_with_upstream(source, dataset, decode_video=True)
        result['base_checkpoint'] = validate_checkpoint(base, require_processor=False)
        training_output = Path(resume_checkpoint).resolve().parent if resume_checkpoint is not None else output/'training'
        from .weights import inspect_weights
        identity = {'dataset_sha256': result['dataset_digest'], 'base_sha256': inspect_weights(base)['sha256'],
                    'budget': asdict(budget), 'source': str(source)}
        identity_path = training_output/'viper_smoke_inputs.json'
        if resume_checkpoint is not None:
            if read_json(identity_path) != identity:
                raise ValueError('resume dataset, initialization or training configuration changed')
        arguments = training_arguments(dataset, str(base), training_output, budget)
        if resume_checkpoint is not None:
            result['resume'] = validate_resume(resume_checkpoint)
            arguments += ['--resume-from-checkpoint']
        result['checkpoint_directory'] = str(training_output)
        result['upstream_config'] = validate_upstream_arguments(source, arguments)
        result['upstream_command'] = [sys.executable, str(source/'gr00t/experiment/launch_finetune.py'), *arguments]
        result['command'] = [sys.executable, '-m', 'embodiment.training_worker', '--source', str(source),
                             '--metrics', str(output/'training_resources.json'), *arguments]
        if not execute:
            result['status'] = 'PREFLIGHT_ONLY'
            return result
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required')
        memory = torch.cuda.get_device_properties(0).total_memory
        result['gpu'] = {'name': torch.cuda.get_device_name(0), 'vram_bytes': memory}
        if memory < minimum_vram_gib * 1024**3:
            raise RuntimeError('GPU memory is below the training smoke requirement')
        training_output.mkdir(parents=True, exist_ok=True)
        atomic_json(identity_path, identity)
        environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled'}
        from .collection import execute as run_process
        already_complete = result.get('resume', {}).get('step') == budget.steps
        result['training_executed_this_attempt'] = not already_complete
        if not already_complete:
            (output/'process').mkdir()
            process = run_process(result['command'], output/'process', timeout, environment=environment)
            result['process'] = process
            if process['returncode'] != 0:
                raise RuntimeError('upstream training failed or timed out; inspect process/process.log')
            resources = read_json(output/'training_resources.json')
            if resources.get('training_process_completed') is not True or not resources.get('trainable_parameters'):
                raise ValueError('training resource and parameter inventory receipt is incomplete')
            result['training_resources'] = resources
            result['peak_training_memory_bytes'] = resources['peak_allocated_gpu_bytes']
            result['training_steps_per_second'] = (budget.steps-result.get('resume', {}).get('step', 0)) / process['wall_seconds']
        checkpoint = training_output/'checkpoint-2'
        result['saved_checkpoint'] = validate_checkpoint(checkpoint)
        trainer = read_json(checkpoint/'trainer_state.json')
        if trainer.get('global_step') != budget.steps:
            raise ValueError('checkpoint does not prove completion of the requested steps')
        result['training_executed'] = True
        inference_checkpoint = checkpoint if already_complete else training_output
        result['checkpoint_directory'] = str(inference_checkpoint)
        result['transfer_checkpoint'] = prepare_training_checkpoint(inference_checkpoint)
        policy = GrootPolicy.from_checkpoint(source, inference_checkpoint)
        result['checkpoint_reload_verified'] = True
        policy.reset()
        observation = first_observation(dataset, validation['embodiment'])
        with torch.inference_mode():
            raw_output, _ = policy.policy.get_action(policy.pack(observation))
        result['inference_executed'] = True
        chunk = policy.decode(raw_output)
        result['safe_action_chunk_verified'] = True
        result['action_chunk_length'] = len(chunk)
        result['peak_inference_memory_bytes'] = torch.cuda.max_memory_allocated()
        result['status'] = 'PASS'
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        result['wall_seconds'] = time.monotonic() - started
        atomic_json(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    result = smoke(args.source, args.dataset, args.base_checkpoint, args.output, args.execute, args.timeout, resume_checkpoint=args.resume_checkpoint)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('PASS', 'PREFLIGHT_ONLY') else 2


if __name__ == '__main__':
    sys.exit(main())
