from dataclasses import dataclass
from pathlib import Path
import hashlib
import sys
from typing import Any, Protocol
from .benchmark import Action, Observation, finite
from .dataset import DOF, read_json
from .upstream_validation import verify_source
from .contract import ADAPTERS


def validate_checkpoint(checkpoint: Path, require_processor: bool = True):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir() or not (checkpoint/'config.json').is_file():
        raise ValueError('local GR00T checkpoint configuration required')
    config = read_json(checkpoint/'config.json')
    if 'gr00t' not in str(config.get('model_type', '')).lower():
        raise ValueError('checkpoint is not a GR00T model')
    index = checkpoint/'model.safetensors.index.json'
    if index.exists():
        mapping = read_json(index).get('weight_map', {})
        if not mapping or not all(isinstance(name, str) for name in mapping.values()):
            raise ValueError('invalid safe checkpoint shard index')
        weights = [checkpoint/name for name in set(mapping.values())]
    else:
        weights = [checkpoint/'model.safetensors']
    if any(path.parent != checkpoint or path.suffix != '.safetensors' or not path.is_file()
           or path.stat().st_size <= 8 for path in weights):
        raise ValueError('local safe checkpoint weights are incomplete')
    from safetensors import safe_open
    try:
        for path in weights:
            with safe_open(path, framework='np') as tensors:
                keys = set(tensors.keys())
                if not keys:
                    raise ValueError('checkpoint shard contains no tensors')
                if index.exists() and keys != {key for key, value in mapping.items() if value == path.name}:
                    raise ValueError('checkpoint tensor index differs from saved shards')
    except Exception as error:
        raise ValueError('invalid safetensors checkpoint or shard index') from error
    processor = checkpoint if (checkpoint/'processor_config.json').exists() else checkpoint/'processor'
    if require_processor and any(not (processor/name).is_file() for name in
                                 ('processor_config.json', 'statistics.json', 'embodiment_id.json')):
        raise ValueError('checkpoint processor and embodiment statistics required')
    return {'checkpoint': str(checkpoint), 'weight_files': len(weights), 'processor': str(processor),
            'weight_paths': sorted(path.name for path in weights)}


def checkpoint_digest(checkpoint: Path):
    checkpoint = Path(checkpoint).resolve()
    manifest = validate_checkpoint(checkpoint)
    files = [checkpoint/'config.json', *(checkpoint/name for name in manifest['weight_paths'])]
    if (checkpoint/'model.safetensors.index.json').exists():
        files.append(checkpoint/'model.safetensors.index.json')
    files.extend(Path(manifest['processor'])/name for name in
                 ('processor_config.json', 'statistics.json', 'embodiment_id.json'))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(checkpoint)).encode())
        with path.open('rb') as stream:
            while chunk := stream.read(1024*1024):
                digest.update(chunk)
    return digest.hexdigest()


def prepare_training_checkpoint(checkpoint: Path):
    checkpoint = Path(checkpoint).resolve()
    manifest = validate_checkpoint(checkpoint)
    processor = Path(manifest['processor'])
    names = ('processor_config.json', 'statistics.json', 'embodiment_id.json')
    for name in names:
        read_json(processor/name)
        destination = checkpoint/name
        if destination.exists() and destination.read_bytes() != (processor/name).read_bytes():
            raise ValueError('checkpoint root and saved processor disagree')
    copied = []
    if processor != checkpoint:
        for name in ('statistics.json', 'embodiment_id.json', 'processor_config.json'):
            destination = checkpoint/name
            if not destination.exists():
                with destination.open('xb') as stream:
                    stream.write((processor/name).read_bytes())
                copied.append(name)
    return {**validate_checkpoint(checkpoint), 'prepared_processor_files': copied}


@dataclass(frozen=True)
class PolicyInput:
    embodiment: str
    instruction: str
    joint_position: tuple
    joint_velocity: tuple
    ee_position: tuple
    gripper_open: float
    image: Any
    timestamp: float | None = None

    @classmethod
    def from_observation(cls, observation: Observation, image, embodiment: str, instruction: str):
        if embodiment not in DOF:
            raise ValueError('unknown policy embodiment')
        count = DOF[embodiment]+2
        finite(observation.joints,count)
        finite(observation.velocities,count)
        finite(observation.ee,3)
        finite((observation.time,), 1)
        if observation.time < 0:
            raise ValueError('negative observation timestamp')
        aperture = ADAPTERS[embodiment].aperture(observation.joints)
        return cls(embodiment,instruction,tuple(observation.joints),tuple(observation.velocities),
                   tuple(observation.ee),aperture,image,observation.time)


class Policy(Protocol):
    def reset(self) -> None: ...
    def actions(self, observation: PolicyInput) -> tuple[Action, ...]: ...


class GrootPolicy:
    def __init__(self, policy, horizon: int = 16, embodiment=None):
        if type(horizon) is not int or not 1 <= horizon <= 16:
            raise ValueError('invalid inference horizon')
        self.policy = policy
        self.horizon = horizon
        self.embodiment = embodiment

    @classmethod
    def from_checkpoint(cls, source: Path, checkpoint: Path, device: str = 'cuda:0'):
        verify_source(source)
        checkpoint = Path(checkpoint).resolve()
        validate_checkpoint(checkpoint)
        from .checkpoint_compatibility import validate_processor
        compatibility = validate_processor(checkpoint)
        sys.path.insert(0,str(Path(source).resolve()))
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        return cls(Gr00tPolicy('NEW_EMBODIMENT',str(checkpoint),device=device,strict=True), embodiment=compatibility['embodiment'])

    def reset(self) -> None:
        self.policy.reset()

    def pack(self, observation: PolicyInput) -> dict:
        import numpy as np
        if self.embodiment is not None and observation.embodiment != self.embodiment:
            raise ValueError('observation embodiment differs from checkpoint statistics')
        if observation.embodiment not in DOF:
            raise ValueError('unknown policy embodiment')
        count = DOF[observation.embodiment]+2
        finite(observation.joint_position,count)
        finite(observation.joint_velocity,count)
        finite(observation.ee_position,3)
        finite((observation.gripper_open,),1)
        if not 0 <= observation.gripper_open <= 1:
            raise ValueError('invalid observed gripper aperture')
        if not isinstance(observation.instruction,str) or not observation.instruction.strip():
            raise ValueError('missing policy instruction')
        if observation.timestamp is not None:
            finite((observation.timestamp,), 1)
            if observation.timestamp < 0:
                raise ValueError('negative policy timestamp')
        image = observation.image
        if not isinstance(image,np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('synchronized uint8 RGB camera observation required')
        values = {'joint_position':observation.joint_position,'joint_velocity':observation.joint_velocity,
                  'ee_position':observation.ee_position,'gripper':(observation.gripper_open,)}
        return {'video':{'overhead':image[None,None].copy()},
                'state':{key:np.asarray(value,dtype=np.float32)[None,None] for key,value in values.items()},
                'language':{'annotation.human.task_description':[[observation.instruction]]}}

    def actions(self, observation: PolicyInput) -> tuple[Action, ...]:
        output,info = self.policy.get_action(self.pack(observation))
        return self.decode(output)

    def decode(self, output) -> tuple[Action, ...]:
        import numpy as np
        if not isinstance(output,dict) or set(output) != {'delta_position','gripper'}:
            raise ValueError('GR00T action modality mismatch')
        delta,gripper = np.asarray(output['delta_position']),np.asarray(output['gripper'])
        if delta.ndim != 3 or delta.shape[0] != 1 or delta.shape[2] != 3:
            raise ValueError('GR00T Cartesian action shape mismatch')
        length = delta.shape[1]
        if not 1 <= length <= self.horizon or gripper.shape != (1,length,1):
            raise ValueError('GR00T chunk horizon or gripper shape mismatch')
        if not np.isfinite(delta).all() or not np.isfinite(gripper).all():
            raise ValueError('nonfinite GR00T action')
        return tuple(Action(tuple(float(v) for v in delta[0,i]),float(gripper[0,i,0])) for i in range(length))


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-training-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_training_checkpoint(args.prepare_training_checkpoint), indent=2))
