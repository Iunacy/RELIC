"""Frozen physical semantics; robot joints retain their native dimensions."""
from dataclasses import asdict, dataclass
from .assets import ARM, FINGERS
from .panda import ARM as PANDA_ARM, FINGERS as PANDA_FINGERS
from .benchmark import CONTROL_DT, DT, TASK_GRIPPER_LIMITS, finite


@dataclass(frozen=True)
class RobotAdapter:
    robot: str
    joint_names: tuple
    arm_dof: int

    @property
    def state_dimension(self):
        return 2 * len(self.joint_names) + 4

    def aperture(self, joints):
        finite(joints, len(self.joint_names))
        closed, opened = TASK_GRIPPER_LIMITS[self.robot]
        values = [(value-closed)/(opened-closed) for value in joints[-2:]]
        if any(not -.1 <= value <= 1.1 for value in values):
            raise ValueError('finger state outside calibrated task range')
        return max(0., min(1., values[0]))

    def fingers(self, aperture):
        finite((aperture,), 1)
        if not 0 <= aperture <= 1:
            raise ValueError('gripper command outside [0, 1]')
        closed, opened = TASK_GRIPPER_LIMITS[self.robot]
        return (closed + aperture*(opened-closed),)*2


ADAPTERS = {
    'openmanipulator_x': RobotAdapter('openmanipulator_x', ARM+FINGERS, 4),
    'franka_panda': RobotAdapter('franka_panda', PANDA_ARM+PANDA_FINGERS, 7),
}


@dataclass(frozen=True)
class TaskFrame:
    """Rigid right-handed task-to-world transform; positions in meters."""
    rotation: tuple = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    translation: tuple = (0., 0., 0.)

    def __post_init__(self):
        finite(self.translation, 3)
        if len(self.rotation) != 3:
            raise ValueError('rotation must be 3 by 3')
        for row in self.rotation:
            finite(row, 3)
        r = self.rotation
        if any(abs(sum(r[i][k]*r[j][k] for k in range(3))-(i == j)) > 1e-8 for i in range(3) for j in range(3)):
            raise ValueError('rotation must be orthonormal')
        determinant = sum(r[0][i]*(r[1][(i+1)%3]*r[2][(i+2)%3]-r[1][(i+2)%3]*r[2][(i+1)%3]) for i in range(3))
        if abs(determinant-1) > 1e-8:
            raise ValueError('rotation must be right handed')

    def world(self, values, displacement=False):
        finite(values, 3)
        return tuple(sum(row[j]*values[j] for j in range(3))+(0 if displacement else t)
                     for row, t in zip(self.rotation, self.translation, strict=True))

    def task(self, values, displacement=False):
        finite(values, 3)
        shifted = [v-(0 if displacement else t) for v, t in zip(values, self.translation, strict=True)]
        return tuple(sum(self.rotation[j][i]*shifted[j] for j in range(3)) for i in range(3))


def manifest(robot):
    adapter = ADAPTERS[robot]
    return {'schema': 'viper.policy_contract.v1', 'robot': robot, 'joint_names': list(adapter.joint_names),
            'joint_count': len(adapter.joint_names), 'state_dimension': adapter.state_dimension,
            'state_order': ['joint_position', 'joint_velocity', 'ee_position', 'gripper'],
            'joint_units': ['radian']*adapter.arm_dof+['meter']*2,
            'velocity_units': ['radian/second']*adapter.arm_dof+['meter/second']*2,
            'frame': asdict(TaskFrame()), 'action_order': ['dx', 'dy', 'dz', 'gripper_open'],
            'action_units': ['meter']*3+['unit_interval'], 'gripper_observation': 'first named finger, matching legacy dataset and Isaac observer; both finger channels retained', 'gripper_open': 1.,
            'gripper_closed': 0., 'gripper_joint_range': TASK_GRIPPER_LIMITS[robot],
            'integration': 'once per control tick from measured end effector; no accumulated target',
            'orientation': 'robot-specific admissible grasp pitch; no policy rotation command',
            'physics_dt': DT, 'control_dt': CONTROL_DT, 'observation_alignment': 'before commanded tick',
            'prediction_horizon': 16, 'maximum_execution_horizon': 16,
            'rgb': 'uint8 HWC RGB; synchronized overhead camera; resolution recorded per dataset',
            'model_padding': 'upstream processor pads normalized 16/22 state to configured max_state_dim; never map robot joints by padding',
            'normalization': 'checkpoint processor with embodiment-specific training-only statistics'}
