"""Pinned optional model acquisition; disabled unless explicitly enabled after license acceptance."""
import argparse
import json
from pathlib import Path
from .collection import atomic_json, locked
from .dataset import read_json, file_digest
from .weights import inspect_weights

SOURCES = read_json(Path(__file__).with_name('model_sources.json'))['models']


def acquire(prefix, execute=False, license_accepted=False):
    prefix = Path(prefix).resolve()
    result = {'schema': 'viper.model_acquisition.v1', 'status': 'NOT_EXECUTED', 'models': SOURCES,
              'downloads_started': False, 'cache': str(prefix/'cache/huggingface/hub')}
    if not execute: return result
    if not license_accepted: raise ValueError('explicit model license acceptance required before downloads')
    from huggingface_hub import snapshot_download
    with locked(prefix/'models'):
        snapshots = {}
        for name, source in SOURCES.items():
            root = Path(snapshot_download(source['repository'], revision=source['revision'], cache_dir=result['cache'],
                allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model', '*.tiktoken', 'LICENSE*']))
            if root.name != source['revision']: raise ValueError('downloaded snapshot revision differs from pin')
            if source.get('config_sha256') and file_digest(root/'config.json') != source['config_sha256']:
                raise ValueError('public model configuration differs from verified pin')
            integrity = inspect_weights(root, require_processor=name == 'backbone')
            ref = root.parent.parent/'refs/main'
            ref.parent.mkdir(exist_ok=True)
            ref.write_text(source['revision'])
            resolved = Path(snapshot_download(source['repository'], cache_dir=result['cache'], local_files_only=True))
            if resolved.resolve() != root.resolve(): raise ValueError('offline upstream lookup differs from pinned snapshot')
            snapshots[name] = {**source, 'integrity': integrity}
        link = prefix/'models/groot'
        target = Path(snapshots['groot']['integrity']['directory'])
        if link.exists() and link.resolve() != target.resolve(): raise ValueError('pretrained model location already contains another snapshot')
        if not link.exists(): link.symlink_to(target, target_is_directory=True)
        result.update(status='VERIFIED', snapshots=snapshots, pretrained=str(link), downloads_started=True)
        atomic_json(prefix/'models/receipt.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path('/workspace/viper'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--license-accepted', action='store_true')
    args = parser.parse_args()
    print(json.dumps(acquire(args.prefix, args.execute, args.license_accepted), indent=2))
