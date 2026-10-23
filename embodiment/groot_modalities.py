from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ACTION_HORIZON = 16

RELIC_CONFIG = {
    'video': ModalityConfig(delta_indices=[0], modality_keys=['overhead']),
    'state': ModalityConfig(delta_indices=[0], modality_keys=['joint_position','joint_velocity','ee_position','gripper']),
    'action': ModalityConfig(delta_indices=list(range(ACTION_HORIZON)), modality_keys=['delta_position','gripper'],
                             action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                                                         type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                                             ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                                                         type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)]),
    'language': ModalityConfig(delta_indices=[0], modality_keys=['annotation.human.task_description']),
}

register_modality_config(RELIC_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
