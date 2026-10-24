"""Timestamp-indexed observation disturbances; true simulator state remains available to safety."""
from dataclasses import replace
import math
import json
from .benchmark import CONTROL_DT
from .perturbations import Perturbation

KINDS = ('baseline', 'observation_bias', 'observation_noise')


def condition(kind='baseline', magnitude=None):
    if kind not in KINDS: raise ValueError('unsupported learned-policy perception disturbance')
    magnitude = (0. if kind == 'baseline' else .001) if magnitude is None else magnitude
    Perturbation(kind, 0, magnitude)
    return {'kind': kind, 'magnitude_m': magnitude, 'schedule': 'scenario seed and simulator control tick',
            'scope': 'policy Cartesian proprioception only; safety and recorded physical state use true observations'}


def verify_trace(path, metadata, samples):
    declared = metadata.get('observation_condition')
    if declared is None: return
    if declared != condition(declared['kind'], declared['magnitude_m']): raise ValueError('perception protocol changed')
    perturbation = Perturbation(declared['kind'], metadata['seed'] % 2**48, declared['magnitude_m'])
    physical = {row['timestamp']: row['ee_position'] for row in samples}
    previous = -1.
    for line in path.read_text().splitlines():
        row = json.loads(line)
        timestamp = row['timestamp']
        if timestamp < previous or timestamp not in physical: raise ValueError('perception timestamps differ from physical observations')
        tick = round(timestamp/CONTROL_DT)
        offset = perturbation.offsets(tick)
        expected = [a+b for a, b in zip(physical[timestamp], offset, strict=True)]
        if row != {'timestamp':timestamp, 'control_tick':tick, 'true_ee_m':physical[timestamp],
                   'perceived_ee_m':expected, 'offset_m':list(offset)}:
            raise ValueError('perceived policy input differs from declared disturbance and physical state')
        previous = timestamp


class DisturbedPolicy:
    def __init__(self, policy, perturbation):
        if perturbation.kind not in KINDS: raise ValueError('unsupported policy perception disturbance')
        self.policy, self.perturbation = policy, perturbation
        self.trace = []

    def __getattr__(self, name): return getattr(self.policy, name)

    def reset(self):
        self.trace = []
        self.policy.reset()

    def actions(self, observation):
        if observation.timestamp is None or not math.isfinite(observation.timestamp) or observation.timestamp < 0:
            raise ValueError('physical timestamp required for matched disturbance scheduling')
        tick = round(observation.timestamp/CONTROL_DT)
        offset = self.perturbation.offsets(tick)
        perceived = replace(observation, ee_position=tuple(a+b for a, b in zip(observation.ee_position, offset, strict=True)))
        self.trace.append({'timestamp': observation.timestamp, 'control_tick': tick,
                           'true_ee_m': list(observation.ee_position), 'perceived_ee_m': list(perceived.ee_position),
                           'offset_m': list(offset)})
        return self.policy.actions(perceived)
