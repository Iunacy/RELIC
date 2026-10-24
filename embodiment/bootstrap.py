"""Idempotent isolated setup on an existing host; dry-run by default, never model downloads."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .collection import atomic_json, locked
from .dataset import GROOT_COMMIT, read_json
from .preflight import LAB_COMMIT

UV_VERSION = '0.13.4'


def plan(prefix, role, repository=None):
    prefix = Path(prefix).resolve()
    repo = Path(repository).resolve() if repository else Path(__file__).resolve().parents[1]
    if role not in ('simulation', 'training'): raise ValueError('unknown environment role')
    source = prefix/'sources'/('IsaacLab' if role == 'simulation' else 'GR00T')
    if role == 'simulation':
        commands = [
            ['uv', 'venv', '--python', '3.11.17', '--seed', str(prefix/'simulation')],
            [str(prefix/'simulation/bin/python'), '-m', 'pip', 'install', 'pip==25.3', 'setuptools==80.9.0', 'wheel==0.45.1'],
            ['uv', 'pip', 'install', '--python', str(prefix/'simulation/bin/python'), 'torch==2.7.0', 'torchvision==0.22.0', '--index-url', 'https://download.pytorch.org/whl/cu128'],
            ['uv', 'pip', 'install', '--python', str(prefix/'simulation/bin/python'), 'isaacsim[all,extscache]==5.1.0',
             '--extra-index-url', 'https://pypi.nvidia.com', '-c', str(repo/'embodiment/runtime_constraints.txt')],
            ['uv', 'pip', 'install', '--python', str(prefix/'simulation/bin/python'), '--no-build-isolation', 'flatdict==4.0.1'],
            ['uv', 'pip', 'install', '--python', str(prefix/'simulation/bin/python'), '-e', str(source/'source/isaaclab'),
             '-e', str(source/'source/isaaclab_assets'), '-c', str(repo/'embodiment/runtime_constraints.txt')],
            [str(prefix/'simulation/bin/python'), '-m', 'pip', 'check'],
            [str(prefix/'simulation/bin/python'), '-m', 'embodiment.assets'],
            [str(prefix/'simulation/bin/python'), '-c', 'from embodiment.replay_render import panda_asset; print(panda_asset())'],
            ['uv', 'venv', '--python', '3.12.15', str(prefix/'asset_tools')],
            ['uv', 'pip', 'install', '--python', str(prefix/'asset_tools/bin/python'), 'usd-core==25.11'],
            [str(prefix/'asset_tools/bin/python'), '-m', 'embodiment.asset_cache', '--root', str(prefix/'assets')],
        ]
        url, commit, python = 'https://github.com/isaac-sim/IsaacLab.git', LAB_COMMIT, prefix/'simulation/bin/python'
    else:
        commands = [['uv', 'sync', '--project', str(source), '--frozen', '--no-dev', '--python', '3.12.15'],
                    ['uv', 'pip', 'install', '--python', str(source/'.venv/bin/python'), 'av==14.2.0'],
                    [str(source/'.venv/bin/python'), '-m', 'embodiment.codec_runtime', '--root', str(prefix/'codec')],
                    ['uv', 'pip', 'check', '--python', str(source/'.venv/bin/python')],
                    [str(source/'.venv/bin/python'), '-m', 'embodiment.runtime_probe', '--source', str(source)]]
        url, commit, python = 'https://github.com/NVIDIA/Isaac-GR00T.git', GROOT_COMMIT, source/'.venv/bin/python'
    return {'schema': 'viper.bootstrap.v1', 'role': role, 'prefix': str(prefix), 'source': str(source),
            'url': url, 'commit': commit, 'uv_version': UV_VERSION, 'python': str(python), 'commands': commands,
            'minimum_free_disk_gib': 65 if role == 'simulation' else 40,
            'cache': str(prefix/'cache/uv'), 'downloads_models': False, 'provisions_hardware': False}


def setup(prefix, role, execute=False, offline=False, reader_library_path=None):
    configuration = plan(prefix, role)
    if reader_library_path is not None:
        configuration['reader_library_path'] = str(Path(reader_library_path).resolve())
    prefix = Path(prefix).resolve()
    with locked(prefix/'setup'):
        atomic_json(prefix/'setup'/(role+'.plan.json'), configuration)
        if not execute: return {**configuration, 'status': 'NOT_EXECUTED'}
        executable = shutil.which('uv')
        if executable is None: raise RuntimeError('install uv=='+UV_VERSION+' from PyPI in a bootstrap virtual environment')
        version = subprocess.check_output([executable, '--version'], text=True)
        if not version.startswith('uv '+UV_VERSION+' '): raise ValueError('setup requires uv=='+UV_VERSION)
        if shutil.disk_usage(prefix).free < configuration['minimum_free_disk_gib']*1024**3:
            raise RuntimeError('insufficient setup storage reserve')
        source = Path(configuration['source'])
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            if offline: raise RuntimeError('offline setup requires the pinned source checkout')
            git_environment = {**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'}
            subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none', configuration['url'], str(source)], check=True, env=git_environment)
            subprocess.run(['git', '-C', str(source), 'checkout', '--detach', configuration['commit']], check=True, env=git_environment)
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        changes = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if actual != configuration['commit'] or changes: raise ValueError('setup checkout differs from clean pinned revision')
        environment = {**os.environ, 'UV_CACHE_DIR': configuration['cache'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'UV_PYTHON_INSTALL_DIR': str(prefix/'cache/python'),
                       'VIPER_ASSET_CACHE': str(prefix/'robot_cache'),
                       'WANDB_MODE': 'disabled', 'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'MAX_JOBS': '2',
                       'NO_ALBUMENTATIONS_UPDATE': '1', 'DS_BUILD_OPS': '0',
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'UV_CONCURRENT_DOWNLOADS': '2'}
        if offline: environment.update(UV_OFFLINE='1', PIP_NO_INDEX='1', VIPER_OFFLINE='1')
        receipt = prefix/'setup'/(role+'.receipt.json')
        if receipt.exists() and read_json(receipt).get('configuration') != configuration: raise ValueError('setup configuration changed')
        started = time.monotonic()
        processes = []
        for index, command in enumerate(configuration['commands']):
            if command[:2] == ['uv', 'venv'] and (Path(command[-1])/'bin/python').is_file(): continue
            log = prefix/'setup'/f'{role}-{index}.log'
            command_environment = dict(environment)
            if 'embodiment.runtime_probe' in command:
                command_environment['LD_LIBRARY_PATH'] = str(Path(reader_library_path).resolve()) if reader_library_path else str(prefix/'codec')
                command_environment['CUDA_VISIBLE_DEVICES'] = ''
            with log.open('ab') as stream:
                process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                         cwd=Path(__file__).resolve().parents[1], env=command_environment)
            processes.append({'command': command, 'returncode': process.returncode, 'log': str(log)})
            if process.returncode:
                result = {'configuration': configuration, 'status': 'BLOCKED', 'processes': processes, 'wall_seconds': time.monotonic()-started}
                atomic_json(receipt, result)
                raise RuntimeError('setup failed; inspect '+str(log)+'; rerun the same command to resume from cache')
        result = {'configuration': configuration, 'status': 'INSTALLED', 'processes': processes, 'wall_seconds': time.monotonic()-started,
                  'simulation_executed': False, 'training_executed': False}
        atomic_json(receipt, result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--role', choices=['simulation', 'training'], required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--reader-library-path', type=Path)
    args = parser.parse_args()
    print(json.dumps(setup(args.prefix, args.role, args.execute, args.offline, args.reader_library_path), indent=2))
