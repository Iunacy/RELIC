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


def smoke(source, dataset, base, output, execute=False, timeout=1800, minimum_vram_gib=40):
    source, dataset, base, output = (Path(path).resolve() for path in (source, dataset, base, output))
    if type(timeout) is not int or timeout <= 0 or minimum_vram_gib < 40:
        raise ValueError('positive timeout and at least 40 GiB VRAM are required')
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(steps=2, batch_size=1, save_steps=2)
    result = {'schema': 'relic.training_smoke.v1', 'status': 'BLOCKED', 'budget': asdict(budget),
              'training_executed': False, 'checkpoint_reload_verified': False,
              'inference_executed': False, 'safe_action_chunk_verified': False,
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
        arguments = training_arguments(dataset, str(base), output/'training', budget)
        result['upstream_config'] = validate_upstream_arguments(source, arguments)
        result['command'] = [sys.executable, str(source/'gr00t/experiment/launch_finetune.py'), *arguments]
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
        environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled'}
        from .collection import execute as run_process
        (output/'process').mkdir()
        process = run_process(result['command'], output/'process', timeout, environment=environment)
        result['process'] = process
        if process['returncode'] != 0:
            raise RuntimeError('upstream training failed or timed out; inspect process/process.log')
        checkpoint = output/'training/checkpoint-2'
        result['saved_checkpoint'] = validate_checkpoint(checkpoint)
        trainer = read_json(checkpoint/'trainer_state.json')
        if trainer.get('global_step') != budget.steps:
            raise ValueError('checkpoint does not prove completion of the requested steps')
        result['training_executed'] = True
        result['training_steps_per_second'] = budget.steps / process['wall_seconds']
        result['transfer_checkpoint'] = prepare_training_checkpoint(output/'training')
        policy = GrootPolicy.from_checkpoint(source, output/'training')
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
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    result = smoke(args.source, args.dataset, args.base_checkpoint, args.output, args.execute, args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('PASS', 'PREFLIGHT_ONLY') else 2


if __name__ == '__main__':
    sys.exit(main())
