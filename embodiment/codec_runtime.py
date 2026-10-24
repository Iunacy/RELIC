"""Expose the pinned PyAV wheel's FFmpeg 7 libraries to TorchCodec in a private directory."""
import argparse
import importlib.metadata
from pathlib import Path
import re


def prepare(root):
    distribution = importlib.metadata.distribution('av')
    if distribution.version != '14.2.0': raise ValueError('codec bridge requires pinned PyAV 14.2.0')
    libraries = Path(distribution.locate_file('av.libs'))
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not list(libraries.glob('libavcodec-*.so.61.*')): raise ValueError('pinned FFmpeg 7 library bundle missing')
    for source in libraries.iterdir():
        if not source.is_file(): continue
        names = [source.name]
        match = re.fullmatch(r'(.+)-[a-f0-9]+\.so\.(\d+)\..+', source.name)
        if match: names.append(match[1]+'.so.'+match[2])
        for name in names:
            target = root/name
            if target.is_symlink() and target.resolve() != source.resolve(): raise ValueError('codec bridge points to another environment')
            if not target.exists(): target.symlink_to(source.resolve())
    return str(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.root))
