from dataclasses import dataclass
import math
from .benchmark import Action, CONTROL_DT, Observation, finite, workspace


class SafetyViolation(ValueError):
    pass


@dataclass(frozen=True)
class SafetyConfig:
    joint_limits: tuple[tuple[float, float], ...]
    arm_dof: int
    arm_velocity: float = 1.2
    finger_velocity: float = .15
    cartesian_velocity: float = .09
    tracking_error: float = .20
    finger_tracking_error: float = .008
    tracking_timeout: float = .5
    stall_timeout: float = 2.
    progress_distance: float = .002

    def __post_init__(self):
        if type(self.arm_dof) is not int or self.arm_dof not in (4, 6, 7):
            raise ValueError('invalid arm mapping')
        if len(self.joint_limits) != self.arm_dof + 2:
            raise ValueError('missing gripper limits')
        for low, high in self.joint_limits:
            finite((low, high), 2)
            if low >= high:
                raise ValueError('invalid joint limits')
        values = (self.arm_velocity, self.finger_velocity, self.cartesian_velocity,
                  self.tracking_error, self.finger_tracking_error, self.tracking_timeout,
                  self.stall_timeout, self.progress_distance)
        finite(values, len(values))
        if min(values) <= 0:
            raise ValueError('safety thresholds must be positive')


class Guard:
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.tracking_since = None
        self.progress_since = None
        self.progress_origin = None
        self.previous_time = -1.

    def state(self, obs: Observation) -> None:
        try:
            finite((obs.time,), 1)
            finite(obs.ee, 3)
            finite(obs.joints, len(self.config.joint_limits))
            finite(obs.velocities, len(self.config.joint_limits))
        except ValueError as error:
            raise SafetyViolation('nonfinite_or_malformed_state') from error
        if obs.time < 0 or obs.time <= self.previous_time:
            raise SafetyViolation('nonmonotonic_time')
        self.previous_time = obs.time
        if obs.unsafe:
            raise SafetyViolation('unsafe_state')
        if not workspace(obs.ee):
            raise SafetyViolation('workspace_state')
        for i, (value, velocity, (low, high)) in enumerate(zip(obs.joints, obs.velocities, self.config.joint_limits, strict=True)):
            if not low - .001 <= value <= high + .001:
                raise SafetyViolation('joint_position')
            limit = self.config.arm_velocity if i < self.config.arm_dof else self.config.finger_velocity
            if abs(velocity) > limit:
                raise SafetyViolation('joint_velocity')

    def command(self, obs: Observation, raw: tuple | list | Action) -> Action:
        try:
            action = raw if isinstance(raw, Action) else Action(tuple(raw[:3]), raw[3]) if len(raw) == 4 else None
            if action is None:
                raise ValueError('action dimension')
        except (TypeError, ValueError, IndexError) as error:
            raise SafetyViolation('invalid_action') from error
        if math.dist(action.delta, (0., 0., 0.)) / CONTROL_DT > self.config.cartesian_velocity + 1e-12:
            raise SafetyViolation('cartesian_velocity')
        goal = tuple(a + b for a, b in zip(obs.ee, action.delta, strict=True))
        if not workspace(goal):
            raise SafetyViolation('workspace_command')
        return action

    def joint_command(self, current: tuple, command: tuple) -> None:
        try:
            finite(current, len(self.config.joint_limits))
            finite(command, len(self.config.joint_limits))
        except ValueError as error:
            raise SafetyViolation('invalid_joint_command') from error
        for i, (value, goal, (low, high)) in enumerate(zip(current, command, self.config.joint_limits, strict=True)):
            limit = self.config.arm_velocity if i < self.config.arm_dof else self.config.finger_velocity
            if not low <= goal <= high:
                raise SafetyViolation('joint_command_limit')
            if abs(goal - value) / CONTROL_DT > limit + 1e-12:
                raise SafetyViolation('joint_command_velocity')

    def tracking(self, obs: Observation, reference: tuple) -> str | None:
        finite(reference, len(self.config.joint_limits))
        contact = any(value is True for value in obs.contacts.values())
        exceeds = any(abs(value - goal) > (self.config.tracking_error if i < self.config.arm_dof else self.config.finger_tracking_error)
                      and not (i >= self.config.arm_dof and contact and goal <= value)
                      for i, (value, goal) in enumerate(zip(obs.joints, reference, strict=True)))
        if not exceeds:
            self.tracking_since = None
        elif self.tracking_since is None:
            self.tracking_since = obs.time
        if self.tracking_since is not None and obs.time - self.tracking_since >= self.config.tracking_timeout:
            return 'tracking_error'
        return None

    def progress(self, obs: Observation, action: Action) -> str | None:
        if math.dist(action.delta, (0., 0., 0.)) < .001:
            self.progress_since = None
            self.progress_origin = None
            return None
        if self.progress_origin is None or math.dist(obs.ee, self.progress_origin) >= self.config.progress_distance:
            self.progress_origin = obs.ee
            self.progress_since = obs.time
        if obs.time - self.progress_since >= self.config.stall_timeout:
            return 'lack_of_progress'
        return None
