import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .dataset import DOF, export, load_episode, read_json, validate_collection
from .qualify import source_provenance
from .scenarios import SPLITS, plan


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def locked(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another collector owns this collection') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def hashes(path):
    if path.is_symlink():
        raise ValueError('symlink episode directories are not permitted')
    result = {}
    for file in sorted(path.rglob('*')):
        if file.is_symlink():
            raise ValueError('symlinks are not permitted in collected episodes')
        if file.is_file():
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                while chunk := stream.read(1024*1024):
                    digest.update(chunk)
            result[str(file.relative_to(path))] = digest.hexdigest()
    return result


def inspect_job(job, robot, split, source_hashes):
    with (job/'.running').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'simulator still owns unfinished job: {job}') from None
    request = read_json(job/'request.json')
    expected = [asdict(case) for case in plan(split, request['count'], request['start'])]
    if request['robot'] != robot or request['split'] != split or request['cases'] != expected:
        raise ValueError('collection job scenario identity changed')
    receipt_path = job/'receipt.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['request'] != request:
            raise ValueError('collection receipt request changed')
        for accepted in receipt['successes']:
            path = job/'run'/accepted['folder']
            if hashes(path) != accepted['sha256']:
                raise ValueError(f'accepted episode was corrupted: {path}')
            load_episode(path)
        return receipt
    successes, failures = [], []
    for case in expected:
        folder = f"trial_{case['seed']:02d}"
        path = job/'run'/folder
        try:
            episode = load_episode(path)
            if any(episode.metadata.get(key) != value for key, value in case.items()):
                raise ValueError('episode scenario differs from assigned case')
            if episode.metadata.get('source_sha256') != source_hashes:
                raise ValueError('episode source differs from frozen collection source')
            if episode.metadata['embodiment'] != robot or episode.metadata.get('purpose') != 'collection':
                raise ValueError('episode embodiment or purpose mismatch')
            if not episode.outcome['success']:
                failures.append({'case': case, 'reason': episode.outcome.get('failure_reason', 'physical_failure')})
                continue
            validate_collection([path])
            successes.append({'case': case, 'folder': folder, 'sha256': hashes(path)})
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append({'case': case, 'reason': f'{type(error).__name__}: {error}'})
    receipt = {'request': request, 'successes': successes, 'failures': failures}
    atomic_json(receipt_path, receipt)
    return receipt


def progress(root, robot, split):
    successes, failures, jobs = [], [], []
    next_index = 0
    source_hashes = read_json(root/'collection.json')['source_sha256']
    if source_provenance()['source_sha256'] != source_hashes:
        raise ValueError('collection source changed during execution')
    for job in sorted((root/'jobs').glob('job-*')):
        if not (job/'request.json').exists():
            continue
        receipt = inspect_job(job, robot, split, source_hashes)
        request = receipt['request']
        if request['start'] != next_index:
            raise ValueError('job ranges overlap or contain a gap')
        next_index += request['count']
        successes.extend(job/'run'/item['folder'] for item in receipt['successes'])
        failures.extend(receipt['failures'])
        jobs.append(job)
    if successes:
        validate_collection(successes)
    return successes, failures, next_index, jobs


def execute(command, job, timeout, environment=None):
    started = time.monotonic()
    with (job/'process.log').open('xb') as log, (job/'.running').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=(lease.fileno(),), cwd=Path(__file__).resolve().parents[1], env=environment)
        try:
            code = process.wait(timeout=timeout)
            result = {'returncode': code, 'timed_out': False}
        except BaseException as error:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if not isinstance(error, subprocess.TimeoutExpired):
                raise
            result = {'returncode': process.returncode, 'timed_out': True}
    result['wall_seconds'] = time.monotonic() - started
    atomic_json(job/'process.json', result)
    return result


def collect(root, robot, split, target, batch_size=10, max_jobs=30, timeout=3600,
            python=sys.executable, device='cuda:0', runner=execute):
    if robot not in DOF or split not in SPLITS:
        raise ValueError('invalid embodiment or split')
    if any(type(value) is not int or value <= 0 for value in (target, batch_size, max_jobs, timeout)):
        raise ValueError('collection counts and timeout must be positive integers')
    root = Path(root).resolve()
    with locked(root):
        configuration = {'schema': 'relic.collection.v1', 'embodiment': robot, 'split': split,
                         'source_sha256': source_provenance()['source_sha256']}
        config_path = root/'collection.json'
        if config_path.exists():
            if read_json(config_path) != configuration:
                raise ValueError('collection source, embodiment or split changed; use a new collection root')
        else:
            if (root/'jobs').exists():
                raise ValueError('existing jobs lack collection provenance')
            atomic_json(config_path, configuration)
        (root/'jobs').mkdir(exist_ok=True)
        paths, failures, offset, jobs = progress(root, robot, split)
        stalled = 0
        for _ in range(max_jobs):
            if len(paths) >= target or stalled >= 3:
                break
            count = min(batch_size, target - len(paths))
            cases = [asdict(case) for case in plan(split, count, offset)]
            sequence = max((int(path.name.split('-')[1]) for path in (root/'jobs').glob('job-*')), default=-1) + 1
            job = root/'jobs'/f'job-{sequence:06d}'
            job.mkdir(exist_ok=False)
            request = {'robot': robot, 'split': split, 'start': offset, 'count': count, 'cases': cases}
            atomic_json(job/'request.json', request)
            command = [str(python), '-m', 'embodiment.qualify', '--mode', 'collection',
                       '--robot', robot, '--split', split, '--episodes', str(count),
                       '--start-index', str(offset), '--output', str(job/'run'),
                       '--headless', '--enable_cameras', '--device', device]
            atomic_json(job/'command.json', command)
            before = len(paths)
            runner(command, job, timeout)
            paths, failures, offset, jobs = progress(root, robot, split)
            stalled = stalled + 1 if len(paths) == before else 0
            summarize(root, target, paths, failures, jobs)
        return summarize(root, target, paths, failures, jobs)


def summarize(root, target, paths, failures, jobs):
    measurements = [read_json(job/'process.json') for job in jobs if (job/'process.json').exists()]
    seconds = sum(item['wall_seconds'] for item in measurements)
    complete_timing = len(measurements) == len(jobs) and bool(jobs)
    metadata = [read_json(path/'metadata.json') for path in paths]
    summary = {'schema': 'relic.collection_progress.v1', 'target_successes': target,
               'successful_episodes': len(paths), 'failed_or_incomplete_attempts': len(failures),
               'status': 'COMPLETE' if len(paths) >= target else 'INCOMPLETE',
               'jobs': len(jobs), 'measured_process_seconds': seconds,
               'successful_colors': dict(Counter(item['target'] for item in metadata)),
               'successful_regions': dict(Counter(str(item['region']) for item in metadata)),
               'successful_episodes_per_hour': len(paths)*3600/seconds if complete_timing and seconds > 0 else None,
               'throughput_scope': 'simulator startup, physics, camera rendering and episode writing; export excluded',
               'timing_complete': complete_timing, 'failures': failures,
               'episodes': [str(path.relative_to(root)) for path in paths]}
    atomic_json(root/'progress.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('failures', 'episodes')}, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--robot', choices=DOF, required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--target', type=int)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--max-jobs', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--export', type=Path)
    args = parser.parse_args()
    target = args.target if args.target is not None else {'train': 200, 'validation': 25, 'evaluation': 50}[args.split]
    summary = collect(args.root, args.robot, args.split, target, args.batch_size, args.max_jobs,
                      args.timeout, args.python, args.device)
    if args.export:
        with locked(args.root):
            paths, _, _, _ = progress(args.root.resolve(), args.robot, args.split)
            if not paths:
                raise ValueError('no successful physical camera episodes to export')
            started = time.monotonic()
            export(paths, args.export)
            export_seconds = time.monotonic() - started
            total_seconds = summary['measured_process_seconds'] + export_seconds
            atomic_json(args.root/'export.json', {'output': str(args.export.resolve()),
                        'episodes': len(paths), 'export_wall_seconds': export_seconds,
                        'collection_and_export_seconds': total_seconds if summary['timing_complete'] else None,
                        'successful_episodes_per_hour': len(paths)*3600/total_seconds if summary['timing_complete'] else None})
    return 0 if summary['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    sys.exit(main())
