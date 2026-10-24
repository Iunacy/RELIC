"""Read-only stage-specific host checks; never initializes Kit or model weights."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from .preflight import SIM, LAB_COMMIT

STAGES = {'simulation': {'total_gib': 16, 'free_gib': 12, 'ram_gib': 24, 'disk_gib': 12},
          'training': {'total_gib': 40, 'free_gib': 38, 'ram_gib': 32, 'disk_gib': 32}}


def inspect_host(stage, volume=None):
    if stage not in STAGES: raise ValueError('unknown compute stage')
    limits = STAGES[stage]
    result = {'schema': 'viper.host_preflight.v1', 'stage': stage, 'ready': False, 'blockers': [],
              'requirements': limits, 'python': sys.version, 'platform': platform.platform(),
              'compute_started': False, 'packages': {},
              'isaac_requirements_source': 'https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html'}
    packages = ('isaacsim', 'isaaclab', 'torch', 'numpy', 'pillow') if stage == 'simulation' else ('torch', 'transformers', 'safetensors', 'pyarrow', 'av', 'torchcodec', 'tyro')
    for name in packages:
        try: result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['blockers'].append('missing package: '+name)
    from packaging.version import Version
    pins = ({'torch': '2.7.0', 'numpy': '1.26.0', 'isaaclab': '0.47.2'} if stage == 'simulation' else
            {'torch': '2.9.0', 'transformers': '4.57.3', 'torchcodec': '0.8.0'})
    for name, expected in pins.items():
        installed = result['packages'].get(name)
        if installed and Version(installed).base_version != expected:
            result['blockers'].append(name+' version differs from pinned runtime '+expected)
    if stage == 'simulation':
        if sys.version_info[:2] != (3, 11): result['blockers'].append('Isaac Sim stage requires Python 3.11')
        if 'isaacsim' in result['packages'] and Version(result['packages']['isaacsim']) != Version(SIM):
            result['blockers'].append('Isaac Sim version differs from pin')
        try:
            spec = importlib.util.find_spec('isaaclab')
            if spec is None or spec.origin is None: raise ValueError('Isaac Lab source cannot be resolved')
            root = Path(spec.origin).resolve().parents[3]
            commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            changes = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--', 'source/isaaclab', 'source/isaaclab_assets'], text=True)
            result['isaaclab_commit'] = commit
            if commit != LAB_COMMIT or changes: result['blockers'].append('Isaac Lab source differs from clean pin')
        except (OSError, ValueError, subprocess.SubprocessError) as error: result['blockers'].append(str(error))
    if stage == 'training':
        if sys.version_info[:2] != (3, 12): result['blockers'].append('pinned GR00T training stage requires Python 3.12')
        try:
            import torch
            result['torch_cuda_build'] = torch.version.cuda
            if torch.version.cuda is None: result['blockers'].append('training interpreter contains a CPU-only PyTorch build')
        except ImportError: pass
    try:
        process = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, check=True, timeout=15)
        gpu = process.stdout.splitlines()[0].split(',')
        result['gpu'] = {'name': gpu[0].strip(), 'driver': gpu[1].strip(), 'total_mib': int(gpu[2]), 'free_mib': int(gpu[3])}
        if int(gpu[2]) < limits['total_gib']*1024 or int(gpu[3]) < limits['free_gib']*1024:
            result['blockers'].append('insufficient total or free GPU memory for '+stage)
        if stage == 'simulation' and 'RTX' not in gpu[0] and not any(name in gpu[0] for name in ('A10', 'A40', 'L4', 'L40')):
            result['blockers'].append('RTX-capable rendering GPU not verified for Isaac camera stage')
        driver = tuple(int(value) for value in gpu[1].strip().split('.'))
        minimum = (580, 65, 6) if stage == 'simulation' else (570, 0, 0)
        if driver < minimum:
            result['blockers'].append('NVIDIA driver is below the conservative pinned CUDA/runtime requirement')
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error: result['blockers'].append('GPU check: '+str(error))
    memory = {}
    if Path('/proc/meminfo').exists():
        memory = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
    result['memory_bytes'] = memory
    if memory.get('MemAvailable', 0) < limits['ram_gib']*1024**3: result['blockers'].append('insufficient available host RAM')
    if volume is not None:
        volume = Path(volume).resolve()
        result['volume'] = str(volume)
        if not volume.is_dir() or not os.access(volume, os.W_OK): result['blockers'].append('existing writable volume required')
        else:
            result['storage_free_bytes'] = shutil.disk_usage(volume).free
            if result['storage_free_bytes'] < limits['disk_gib']*1024**3: result['blockers'].append('insufficient storage reserve')
    result['ready'] = not result['blockers']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--volume', type=Path)
    args = parser.parse_args()
    result = inspect_host(args.stage, args.volume)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ready'] else 2)
