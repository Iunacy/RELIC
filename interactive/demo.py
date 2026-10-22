import argparse
import json
import uuid
from pathlib import Path

from .actions import Client, load_env
from .arm import ROOT
from .engine import comparison, scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--robot', choices=['menagerie','synthetic'], default='menagerie')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--instruction')
    group.add_argument('--target', choices=['target_left','target_center','target_right'])
    p.add_argument('--degraded', action='store_true')
    p.add_argument('--env-file', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--no-capture', action='store_true')
    args = p.parse_args()
    load_env(args.env_file)
    output = args.output or ROOT/'runs'/('demo-'+uuid.uuid4().hex)
    if output.exists():
        p.exit(2,'output already exists; choose a fresh directory\n')
    output.mkdir(parents=True)
    try:
        targets = scene(args.robot, output/'scene')
        if args.instruction:
            action = Client().select(args.instruction, targets, output/'selection.json')
        else:
            action = dict(action='reach_target',target_id=args.target)
            (output/'selection.json').write_text(json.dumps(dict(mode='manual',validated_output=action)))
        report = comparison(args.robot, action['target_id'], args.degraded, output/'execution', capture=not args.no_capture)
        print(json.dumps(report,indent=2))
    except ValueError as e:
        (output/'failure.json').write_text(json.dumps(dict(error=str(e))))
        p.exit(2,str(e)+'\n')


if __name__ == '__main__':
    main()
