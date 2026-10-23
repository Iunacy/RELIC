"""Meters, seconds, radians; common world and VLA boundary for both embodiments."""
from dataclasses import dataclass, field
import math
import random
from typing import Protocol

COLORS = ('red', 'green', 'blue')
SCENE_REVISION = 4
TASK_GRIPPER_LIMITS = {'openmanipulator_x':(-.011,.004),'franka_panda':(0.,.020)}
CUBE = 0.025
# Official fingertips extend 20.7 mm beyond the URDF tool origin.
# Hold that origin above the cube center so fingertips clear the support plane.
GRASP_Z = 0.0285
BIN = (0.14, 0.145, 0.004)
BIN_INNER = 0.060
BIN_WALL = 0.025
TIMEOUT = 45.0
DT = 1 / 120
CONTROL_DT = 1 / 30
CAMERA_EYE = (0.55, 0.42, 0.48)
CAMERA_LOOK = (0.17, 0.02, 0.04)


def finite(values, count):
    if len(values) != count or not all(isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values):
        raise ValueError('invalid numeric vector')


def workspace(p):
    finite(p, 3)
    radius = math.hypot(p[0] - 0.012, p[1])
    return 0.10 <= radius <= 0.27 and -0.12 <= p[1] <= 0.18 and 0.008 <= p[2] <= 0.19


@dataclass(frozen=True)
class Action:
    delta: tuple
    gripper: float

    def __post_init__(self):
        finite(self.delta, 3)
        finite((self.gripper,), 1)
        if math.dist(self.delta, (0,0,0)) > 0.003 + 1e-12 or not 0 <= self.gripper <= 1:
            raise ValueError('action outside shared limits')


@dataclass
class Observation:
    time: float
    ee: tuple
    joints: tuple
    velocities: tuple
    cubes: dict  # privileged simulator poses, xyz + quaternion wxyz
    cube_speeds: dict
    unsafe: bool = False
    contacts: dict = field(default_factory=dict)
    gripper_open: float | None = None
    arm_dof: int = 4
    safety_events: tuple = ()
    contact_forces: dict = field(default_factory=dict)


class Embodiment(Protocol):
    def observe(self) -> Observation: ...
    def step(self, action: Action) -> Observation: ...


def layout(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = random.Random(seed)
    for _ in range(1000):
        result = {}
        sectors = [(-.62,-.54),(-.18,-.10),(.25,.34)]
        rng.shuffle(sectors)
        for color, (low,high) in zip(COLORS,sectors):
            radius, angle = rng.uniform(.155,.185), rng.uniform(low,high)
            result[color] = (.012 + radius*math.cos(angle), radius*math.sin(angle), CUBE/2 + .001)
        poses = list(result.values())
        if all(math.dist(poses[i],poses[j]) >= .055 for i in range(3) for j in range(i)):
            return result
    raise ValueError('could not sample separated cubes')


def inside_bin(pose):
    finite(pose, 7)
    x,y,z,w,qx,qy,qz = pose
    if abs(w*w+qx*qx+qy*qy+qz*qz-1) > 1e-3:
        raise ValueError('unnormalized object quaternion')
    # Quaternion rotation: use all eight corners, including tilted cubes.
    matrix = ((1-2*(qy*qy+qz*qz),2*(qx*qy-qz*w),2*(qx*qz+qy*w)),
              (2*(qx*qy+qz*w),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*w)),
              (2*(qx*qz-qy*w),2*(qy*qz+qx*w),1-2*(qx*qx+qy*qy)))
    ext = [CUBE/2*sum(abs(v) for v in row) for row in matrix]
    return (abs(x-BIN[0])+ext[0] < BIN_INNER/2 and
            abs(y-BIN[1])+ext[1] < BIN_INNER/2 and
            z-ext[2] >= BIN[2]-.002 and z+ext[2] <= BIN[2]+BIN_WALL+.004)
