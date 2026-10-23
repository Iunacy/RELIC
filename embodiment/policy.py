from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Protocol
from .benchmark import Action, Observation, finite
from .dataset import DOF, read_json
from .upstream_validation import verify_source


@dataclass(frozen=True)
class PolicyInput:
    embodiment: str
    instruction: str
    joint_position: tuple
    joint_velocity: tuple
    ee_position: tuple
    gripper_open: float
    image: Any

    @classmethod
    def from_observation(cls, observation: Observation, image, embodiment: str, instruction: str):
        if embodiment not in DOF:
            raise ValueError('unknown policy embodiment')
        count = DOF[embodiment]+2
        finite(observation.joints,count)
        finite(observation.velocities,count)
        finite(observation.ee,3)
        closed,opened = (-.011,.02) if embodiment == 'openmanipulator_x' else (0.,.04)
        aperture = max(0.,min(1.,(observation.joints[-2]-closed)/(opened-closed)))
        return cls(embodiment,instruction,tuple(observation.joints),tuple(observation.velocities),
                   tuple(observation.ee),aperture,image)


class Policy(Protocol):
    def reset(self) -> None: ...
    def actions(self, observation: PolicyInput) -> tuple[Action, ...]: ...


class GrootPolicy:
    def __init__(self, policy, horizon: int = 16):
        if type(horizon) is not int or not 1 <= horizon <= 16:
            raise ValueError('invalid inference horizon')
        self.policy = policy
        self.horizon = horizon

    @classmethod
    def from_checkpoint(cls, source: Path, checkpoint: Path, device: str = 'cuda:0'):
        verify_source(source)
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_dir() or not (checkpoint/'config.json').is_file():
            raise ValueError('local GR00T checkpoint configuration required')
        weights = list(checkpoint.glob('*.safetensors'))
        if not weights or any(path.stat().st_size == 0 for path in weights):
            raise ValueError('local safe checkpoint weights required')
        config = read_json(checkpoint/'config.json')
        if 'gr00t' not in str(config.get('model_type','')).lower():
            raise ValueError('checkpoint is not a GR00T model')
        sys.path.insert(0,str(Path(source).resolve()))
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        return cls(Gr00tPolicy('NEW_EMBODIMENT',str(checkpoint),device=device,strict=True))

    def reset(self) -> None:
        self.policy.reset()

    def pack(self, observation: PolicyInput) -> dict:
        import numpy as np
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
        image = observation.image
        if not isinstance(image,np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('synchronized uint8 RGB camera observation required')
        values = {'joint_position':observation.joint_position,'joint_velocity':observation.joint_velocity,
                  'ee_position':observation.ee_position,'gripper':(observation.gripper_open,)}
        return {'video':{'overhead':image[None,None].copy()},
                'state':{key:np.asarray(value,dtype=np.float32)[None,None] for key,value in values.items()},
                'language':{'annotation.human.task_description':[[observation.instruction]]}}

    def actions(self, observation: PolicyInput) -> tuple[Action, ...]:
        import numpy as np
        output,info = self.policy.get_action(self.pack(observation))
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
