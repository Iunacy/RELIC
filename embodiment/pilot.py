"""Freeze a bounded paired pilot; collection remains an explicit separate command."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from .benchmark import CAMERA_EYE, CAMERA_LOOK, CONTROL_DT, SCENE_REVISION, layout
from .collection import atomic_json, locked
from .contract import manifest
from .dataset import DOF
from .qualify import source_provenance
from .scenarios import plan, validate_plans


def prepare(root, count=12, python=sys.executable):
    if type(count) is not int or not 10 <= count <= 20:
        raise ValueError('pilot requires 10 to 20 episodes per robot')
    plans = {split: plan(split, count if split == 'train' else 6) for split in ('train', 'validation', 'evaluation')}
    validate_plans(plans)
    configuration = {'schema': 'viper.pilot.v1', 'scene_revision': SCENE_REVISION, **source_provenance(),
        'camera': {'eye': CAMERA_EYE, 'look': CAMERA_LOOK, 'color_order': 'RGB', 'observation_dt': CONTROL_DT},
        'contracts': {robot: manifest(robot) for robot in DOF},
        'scenarios': {split: [{**asdict(case), 'positions': layout(case.seed),
                              'instruction': f'Pick up the {case.target} cube and place it in the bin.'} for case in cases]
                      for split, cases in plans.items()},
        'simulation_executed': False, 'training_ready': False,
        'coverage': 'collection receipts retain every failed, incomplete and accepted attempt; replacements use frozen schedule',
        'commands': {robot: [str(python), '-m', 'embodiment.collection', '--root', str(Path(root).resolve()/robot),
            '--robot', robot, '--split', 'train', '--target', str(count), '--batch-size', '2', '--max-jobs', '10',
            '--export', str(Path(root).resolve()/(robot+'_lerobot'))] for robot in DOF}}
    with locked(Path(root)):
        from .dataset import read_json
        path = Path(root)/'pilot.json'
        # JSON conversion canonicalizes tuple/list fields for restart comparisons.
        configuration = json.loads(json.dumps(configuration))
        if path.exists() and read_json(path) != configuration: raise ValueError('pilot configuration changed')
        atomic_json(path, configuration)
    return configuration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.episodes, args.python), indent=2))
