from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random

from .benchmark import COLORS, CUBE, layout, workspace

KINDS = ('baseline', 'position_shift', 'observation_bias', 'observation_noise', 'control_latency')


@dataclass(frozen=True)
class Perturbation:
    kind: str = 'baseline'
    seed: int = 0
    magnitude: float = 0.

    def __post_init__(self):
        if self.kind not in KINDS or type(self.seed) is not int or not 0 <= self.seed < 2**48:
            raise ValueError('invalid robustness condition')
        limits = {'baseline': 0., 'position_shift': .003, 'observation_bias': .001,
                  'observation_noise': .001, 'control_latency': 2.}
        if type(self.magnitude) not in (int, float) or not math.isfinite(self.magnitude) or not 0 <= self.magnitude <= limits[self.kind]:
            raise ValueError('perturbation exceeds declared validity limits')
        if self.kind == 'control_latency' and self.magnitude != int(self.magnitude):
            raise ValueError('control latency must be an integer number of control ticks')

    def offsets(self, tick=0):
        if type(tick) is not int or tick < 0:
            raise ValueError('invalid observation tick')
        rng = random.Random((self.seed << 32) + (tick if self.kind == 'observation_noise' else 0))
        return tuple(rng.uniform(-self.magnitude, self.magnitude) for _ in range(3))

    def positions(self, scene_seed):
        positions = layout(scene_seed)
        if self.kind == 'position_shift':
            dx, dy, _ = self.offsets()
            positions = {color: (pose[0]+dx, pose[1]+dy, pose[2]) for color, pose in positions.items()}
        if any(not workspace(pose) for pose in positions.values()):
            raise ValueError('perturbed objects leave the shared workspace')
        if any(math.dist(positions[a], positions[b]) < .05 for i, a in enumerate(COLORS) for b in COLORS[:i]):
            raise ValueError('perturbed objects overlap the valid task spacing')
        return positions

    def observe(self, observation, tick):
        if self.kind not in ('observation_bias', 'observation_noise'):
            return observation
        return replace(observation, ee=tuple(value+delta for value, delta in zip(observation.ee, self.offsets(tick), strict=True)))


def plan_robustness(count=3, conditions=KINDS):
    if type(count) is not int or not 1 <= count <= 1000 or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError('invalid robustness campaign size or conditions')
    cases = []
    magnitudes = {'baseline': 0., 'position_shift': .003, 'observation_bias': .001,
                  'observation_noise': .001, 'control_latency': 1.}
    for index in range(count):
        seed = (4 << 48) + (index << 16)
        target = COLORS[index % len(COLORS)]
        for kind in conditions:
            if kind not in magnitudes:
                raise ValueError('unsupported robustness condition')
            perturbation = Perturbation(kind, index, magnitudes[kind])
            payload = {'namespace': 'relic.robustness.v1', 'seed': seed, 'target': target,
                       'perturbation': asdict(perturbation)}
            identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            cases.append({**payload, 'scenario_id': identity, 'split': 'robustness'})
    return cases
