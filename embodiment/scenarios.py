import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from .benchmark import COLORS, SCENE_REVISION, layout

SPLITS = ('train', 'validation', 'evaluation')
HELD_OUT = frozenset((('red', 2), ('green', 0), ('blue', 1)))


@dataclass(frozen=True)
class Scenario:
    seed: int
    target: str
    region: int
    split: str
    scenario_id: str


def region(position: tuple) -> int:
    angle = math.atan2(position[1], position[0] - .012)
    return 0 if angle < -.35 else 1 if angle < .1 else 2


def identity(seed: int, target: str) -> str:
    payload = json.dumps({'target': target, 'positions': layout(seed)}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def plan(split: str, count: int) -> list[Scenario]:
    if split not in SPLITS or type(count) is not int or not 0 < count <= 100000:
        raise ValueError('invalid collection request')
    pairs = [(color, sector) for color in COLORS for sector in range(3)
             if ((color, sector) in HELD_OUT) == (split == 'evaluation')]
    result = []
    for index in range(count):
        target, sector = pairs[index % len(pairs)]
        prefix = ((SPLITS.index(split) + 1) << 48) + (index << 16)
        for offset in range(65536):
            seed = prefix + offset
            if region(layout(seed)[target]) == sector:
                result.append(Scenario(seed, target, sector, split, identity(seed, target)))
                break
        else:
            raise RuntimeError('unable to sample balanced scenario')
    return result


def validate_plans(plans: dict[str, list[Scenario]]) -> None:
    seen = set()
    seeds = set()
    for split, cases in plans.items():
        if split not in SPLITS:
            raise ValueError('unknown split')
        for case in cases:
            if case.split != split or case.target not in COLORS or case.region != region(layout(case.seed)[case.target]):
                raise ValueError('scenario annotation mismatch')
            if case.scenario_id != identity(case.seed, case.target):
                raise ValueError('scenario fingerprint mismatch')
            if case.scenario_id in seen or case.seed in seeds:
                raise ValueError('duplicate scenario or seed')
            if ((case.target, case.region) in HELD_OUT) != (split == 'evaluation'):
                raise ValueError('held-out configuration leakage')
            seen.add(case.scenario_id)
            seeds.add(case.seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=60)
    parser.add_argument('--validation', type=int, default=12)
    parser.add_argument('--evaluation', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plans = {split: plan(split, getattr(args, split)) for split in SPLITS}
    validate_plans(plans)
    payload = {'schema': 'relic.scenarios.v1', 'scene_revision': SCENE_REVISION,
               'held_out_pairs': sorted(HELD_OUT), 'qualification_seeds_excluded': list(range(20)),
               'scenarios': {split: [asdict(case) for case in cases] for split, cases in plans.items()}}
    with args.output.open('x') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
