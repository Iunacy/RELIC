"""Read-only cloud-host readiness inspection; no provisioning, downloads or training."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .dataset import file_digest
from .policy import validate_checkpoint
from .upstream_validation import verify_source


def preflight(source, pretrained, volume):
    source, pretrained, volume = map(lambda path: Path(path).resolve(), (source, pretrained, volume))
    result = {'schema': 'viper.cloud_preflight.v1', 'billing_started': False, 'training_executed': False,
              'model_downloads': False, 'blockers': [], 'python': sys.version,
              'persistent_volume': str(volume), 'packages': {}, 'groot_lock_sha256': None}
    for name in ('torch', 'transformers', 'safetensors', 'pyarrow', 'av', 'torchcodec', 'tyro'):
        try: result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['blockers'].append('missing dependency: '+name)
    try:
        verify_source(source)
        result['groot_lock_sha256'] = file_digest(source/'uv.lock')
        result['checkpoint'] = validate_checkpoint(pretrained, require_processor=False)
        if not volume.is_dir() or not os.access(volume, os.W_OK): raise ValueError('existing writable persistent volume required')
        result['storage_free_bytes'] = shutil.disk_usage(volume).free
        result['persistent_storage_verified'] = False
        result['blockers'].append('operator must verify volume persistence, host RAM, rental cap and gated backbone cache')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result['blockers'].append(f'{type(error).__name__}: {error}')
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader'],
                             text=True, capture_output=True, timeout=15, check=True)
        result['gpu'] = gpu.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error: result['blockers'].append('GPU inspection: '+str(error))
    result['setup_command'] = ['uv', 'sync', '--project', str(source), '--locked']
    result['compute_policy'] = 'training and simulator stages run separately; cross_smoke enforces memory and explicit execution opt-in'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'pretrained', 'volume'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.source, args.pretrained, args.volume), indent=2))
