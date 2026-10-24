"""Offline safetensors snapshot integrity; never loads model tensors onto a device."""
import hashlib
import json
from pathlib import Path
from .dataset import read_json, file_digest


def inspect_weights(root, require_processor=False):
    from safetensors import safe_open
    root = Path(root).resolve()
    configuration = read_json(root/'config.json')
    index = root/'model.safetensors.index.json'
    if index.is_file():
        mapping = read_json(index).get('weight_map')
        if not isinstance(mapping, dict) or not mapping or any(not isinstance(key, str) or not key or not isinstance(name, str) for key, name in mapping.items()):
            raise ValueError('invalid weight index')
        names = sorted(set(mapping.values()))
    else:
        mapping, names = None, ['model.safetensors']
    tensors, files = set(), {}
    for name in names:
        if Path(name).name != name or not name.endswith('.safetensors'): raise ValueError('unsafe shard path')
        path = root/name
        with safe_open(path, framework='np') as stream:
            keys = set(stream.keys())
            if not keys or tensors & keys: raise ValueError('empty or duplicate shard tensors')
            if mapping is not None and keys != {key for key, filename in mapping.items() if filename == name}:
                raise ValueError('shard contents differ from index')
            tensors.update(keys)
        files[name] = file_digest(path)
    metadata = ['config.json']+(['model.safetensors.index.json'] if index.is_file() else [])
    if require_processor:
        required = ['preprocessor_config.json', 'video_preprocessor_config.json', 'tokenizer_config.json',
                    'tokenizer.json', 'chat_template.json']
        if any(not (root/name).is_file() for name in required):
            raise ValueError('backbone processor and tokenizer cache incomplete')
        metadata += required
    files.update({name: file_digest(root/name) for name in metadata})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'directory': str(root), 'model_type': configuration.get('model_type'), 'files': files,
            'sha256': digest, 'tensors': len(tensors), 'weights_loaded': False}


def cached_backbone(identifier):
    from huggingface_hub import snapshot_download
    path = Path(identifier)
    root = path if path.is_dir() else Path(snapshot_download(identifier, local_files_only=True))
    return inspect_weights(root, require_processor=True)
