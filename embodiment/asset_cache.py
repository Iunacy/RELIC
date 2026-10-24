"""Cache the versioned Panda USD dependency closure without starting Kit or a GPU."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request
from .collection import atomic_json, locked
from .dataset import file_digest, read_json

ORIGIN = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/'
PANDA = 'Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd'


def cache(root, origin=ORIGIN, entry=PANDA, maximum_bytes=512*1024**2):
    from pxr import Sdf, UsdUtils
    root = Path(root).resolve()
    origin = origin.rstrip('/')+'/'
    with locked(root):
        receipt = root/'assets.json'
        previous = read_json(receipt) if receipt.exists() else (read_json(root/'partial_assets.json') if (root/'partial_assets.json').exists() else None)
        if previous and (previous['origin'] != origin or previous['entry'] != entry): raise ValueError('asset cache inputs changed')
        pending, files, builtins, total = [entry], {}, set(), 0
        try:
            while pending:
                name = pending.pop()
                if name in files: continue
                if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts: raise ValueError('unsafe asset dependency')
                path = root/name
                if not path.exists():
                    if os.environ.get('VIPER_OFFLINE') == '1': raise RuntimeError('offline asset cache is incomplete: '+name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name+'.partial')
                    with urllib.request.urlopen(origin+urllib.parse.quote(name, safe='/'), timeout=60) as response, temporary.open('wb') as stream:
                        while chunk := response.read(1024*1024):
                            total += len(chunk)
                            if total > maximum_bytes: raise ValueError('asset download exceeded bounded storage budget')
                            stream.write(chunk)
                    temporary.replace(path)
                digest = file_digest(path)
                if previous and name in previous['files'] and previous['files'][name]['sha256'] != digest:
                    raise ValueError('cached asset changed: '+name)
                files[name] = {'sha256': digest, 'bytes': path.stat().st_size}
                if sum(item['bytes'] for item in files.values()) > maximum_bytes: raise ValueError('asset closure exceeds storage budget')
                if path.suffix in ('.usd', '.usda', '.usdc'):
                    layer = Sdf.Layer.FindOrOpen(str(path))
                    if layer is None: raise ValueError('invalid USD layer: '+name)
                    dependencies = {item for group in UsdUtils.ExtractExternalReferences(str(path)) for item in group}
                    for dependency in dependencies:
                        if dependency.startswith('Omni') and dependency.endswith('.mdl'):
                            builtins.add(dependency)
                            continue
                        url = urllib.parse.urljoin(origin+name, dependency)
                        if not url.startswith(origin): raise ValueError('asset references outside versioned origin: '+url)
                        pending.append(urllib.parse.unquote(url[len(origin):]))
            result = {'schema': 'viper.asset_cache.v1', 'origin': origin, 'entry': entry, 'files': files,
                      'runtime_builtin_materials': sorted(builtins), 'bytes': sum(item['bytes'] for item in files.values()),
                      'usd': str(root/entry), 'status': 'CACHED', 'simulation_executed': False,
                      'integrity_scope': 'hash-checked versioned-origin assets; first acquisition hashes are recorded locally'}
            atomic_json(receipt, result)
            return result
        except Exception:
            atomic_json(root/'partial_assets.json', {'origin': origin, 'entry': entry, 'files': files, 'status': 'INCOMPLETE'})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(cache(args.root), indent=2))
