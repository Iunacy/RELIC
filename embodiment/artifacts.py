"""Export checksummed run artifacts without model/optimizer tensors unless requested."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from .dataset import file_digest


def bundle(root, output, include_checkpoints=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root in output.parents: raise ValueError('archive must be outside the run root')
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('artifact symlinks are not supported')
        if not path.is_file(): continue
        if not include_checkpoints and (path.suffix in ('.safetensors', '.pt', '.pth') or path.name == 'training_args.bin'): continue
        files.append(path)
    manifest = {str(path.relative_to(root)): {'sha256': file_digest(path), 'bytes': path.stat().st_size} for path in files}
    if 'viper_bundle_manifest.json' in manifest: raise ValueError('reserved bundle filename')
    with output.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for path in files:
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                    info.uid, info.gid, info.uname, info.gname, info.mtime = 0, 0, '', '', 0
                    with path.open('rb') as source: archive.addfile(info, source)
                payload = json.dumps({'schema': 'viper.artifact_bundle.v1', 'includes_checkpoints': include_checkpoints,
                                      'files': manifest}, sort_keys=True).encode()
                info = tarfile.TarInfo('viper_bundle_manifest.json')
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    result = verify(output)
    return {**result, 'output': str(output), 'sha256': file_digest(output), 'bytes': output.stat().st_size}


def verify(path):
    actual, manifest = {}, None
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts or member.name in actual:
                raise ValueError('unsafe or duplicate archive entry')
            source = archive.extractfile(member)
            if member.name == 'viper_bundle_manifest.json':
                if manifest is not None: raise ValueError('duplicate bundle manifest')
                manifest = json.load(source)
            else:
                digest = hashlib.sha256()
                while chunk := source.read(1024*1024): digest.update(chunk)
                actual[member.name] = {'sha256': digest.hexdigest(), 'bytes': member.size}
    if manifest is None or actual != manifest['files']: raise ValueError('artifact bundle integrity mismatch')
    return {'status': 'VERIFIED', 'files': len(actual), 'includes_checkpoints': manifest['includes_checkpoints']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--include-checkpoints', action='store_true')
    args = parser.parse_args()
    if not args.verify and (not args.root or not args.output): parser.error('provide root and output, or verify')
    print(json.dumps(verify(args.verify) if args.verify else bundle(args.root, args.output, args.include_checkpoints), indent=2))
