"""Short entry point for setup and bounded stages on an existing GPU host."""
import argparse
import json
import os
from pathlib import Path
from .bootstrap import setup
from .cloud_workflow import STAGES, run


def workflow(prefix, root, pretrained, stage=None, execute=False, reader_library_path=None, runtime_prefix=None):
    prefix = Path(prefix).resolve()
    runtime = Path(runtime_prefix).resolve() if runtime_prefix else prefix
    from .asset_cache import PANDA
    panda = runtime/'assets'/PANDA
    if reader_library_path is None and (runtime/'codec').is_dir(): reader_library_path = runtime/'codec'
    return run(root, runtime/'simulation/bin/python', runtime/'sources/GR00T/.venv/bin/python',
               runtime/'sources/IsaacLab', runtime/'sources/GR00T', pretrained,
               stage, execute, reader_library_path, panda if panda.is_file() else None, runtime/'robot_cache')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path('/workspace/viper'))
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--root', type=Path, default=Path('/workspace/runs/first'))
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--setup', choices=['simulation', 'training', 'both'])
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--reader-library-path', type=Path)
    args = parser.parse_args()
    os.environ['HF_HUB_CACHE'] = str(args.prefix.resolve()/'cache/huggingface/hub')
    if args.setup:
        if args.stage: parser.error('select setup or one workflow stage')
        roles = ('simulation', 'training') if args.setup == 'both' else (args.setup,)
        result = {role: setup(args.prefix, role, args.execute, args.offline, args.reader_library_path) for role in roles}
    else:
        if args.offline: parser.error('--offline applies to cached setup')
        runtime = args.runtime or (Path('/opt/viper-runtime') if Path('/opt/viper-runtime/simulation/bin/python').exists() else None)
        result = workflow(args.prefix, args.root, args.pretrained or args.prefix/'models/groot', args.stage, args.execute, args.reader_library_path, runtime)
    print(json.dumps(result, indent=2))
    return 2 if result.get('status') == 'BLOCKED' else 0


if __name__ == '__main__': raise SystemExit(main())
