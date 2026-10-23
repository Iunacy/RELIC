import math
from pathlib import Path

from .dataset import DOF, read_json


def validate_processor(checkpoint, robot=None):
    checkpoint = Path(checkpoint)
    root = checkpoint if (checkpoint/'processor_config.json').exists() else checkpoint/'processor'
    config = read_json(root/'processor_config.json')
    statistics = read_json(root/'statistics.json')
    try:
        kwargs = config['processor_kwargs']
        modalities = kwargs['modality_configs']['new_embodiment']
        stats = statistics['new_embodiment']
        count = len(stats['state']['joint_position']['mean'])
        inferred = next(name for name, dof in DOF.items() if dof+2 == count)
        if robot is not None and robot != inferred:
            raise ValueError('checkpoint statistics belong to another embodiment')
        expected = {'state': {'joint_position': count, 'joint_velocity': count, 'ee_position': 3, 'gripper': 1},
                    'action': {'delta_position': 3, 'gripper': 1}}
        for modality, groups in expected.items():
            if modalities[modality]['modality_keys'] != list(groups):
                raise ValueError('checkpoint modality order differs from RELIC')
            if modalities[modality]['delta_indices'] != (list(range(16)) if modality == 'action' else [0]):
                raise ValueError('checkpoint temporal horizon differs from RELIC')
            for group, dimension in groups.items():
                for field in ('mean', 'std', 'min', 'max', 'q01', 'q99'):
                    values = stats[modality][group][field]
                    if len(values) != dimension or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                        raise ValueError('checkpoint normalization dimensions or values invalid')
                if any(value < 0 for value in stats[modality][group]['std']):
                    raise ValueError('negative checkpoint normalization deviation')
                for low, high in (('min', 'max'), ('q01', 'q99')):
                    if any(a > b for a, b in zip(stats[modality][group][low], stats[modality][group][high], strict=True)):
                        raise ValueError('reversed checkpoint normalization bounds')
        for action in modalities['action']['action_configs']:
            if action['rep'] != 'ABSOLUTE' or action['type'] != 'NON_EEF' or action['format'] != 'DEFAULT':
                raise ValueError('checkpoint would transform world-frame increments again')
        if len(modalities['action']['action_configs']) != 2:
            raise ValueError('checkpoint shared action configuration mismatch')
        if modalities['video']['modality_keys'] != ['overhead'] or modalities['video']['delta_indices'] != [0]:
            raise ValueError('checkpoint camera mapping mismatch')
        if modalities['language']['modality_keys'] != ['annotation.human.task_description']:
            raise ValueError('checkpoint instruction mapping mismatch')
        return {'embodiment': inferred, 'joint_count': count, 'action_dim': 4, 'action_horizon': 16}
    except (KeyError, TypeError, StopIteration) as error:
        raise ValueError('checkpoint lacks compatible RELIC processor metadata') from error
