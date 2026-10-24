from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from .benchmark import Action, BIN, COLORS, GRASP_Z, TIMEOUT, inside_bin
from .expert import Expert
from .policy import PolicyInput
from .safety import Guard, SafetyViolation


@dataclass(frozen=True)
class ExecutionConfig:
    fallback_enabled: bool = True
    execution_horizon: int = 16
    placement_dwell: float = 1.

    def __post_init__(self):
        if type(self.fallback_enabled) is not bool or type(self.execution_horizon) is not int or not 1 <= self.execution_horizon <= 16:
            raise ValueError('invalid execution configuration')
        if self.placement_dwell != 1.:
            raise ValueError('placement dwell is frozen at one second')


class ClosedLoop:
    def __init__(self, backend, policy, target, config=ExecutionConfig()):
        if target not in COLORS:
            raise ValueError('unknown target')
        self.backend, self.policy, self.target, self.config = backend, policy, target, config
        self.guard = Guard(backend.safety_config)
        self.queue = deque()
        self.expert = None
        self.triggers = []
        self.lifted = False
        self.stable_since = None
        self.last_time = -1.
        self.failure = None
        self.success = False
        self.safety_events = []
        self.rejected_policy_actions = []
        self.policy_errors = []
        self.action_trace = []
        self.instruction = f'Pick up the {target} cube and place it in the bin.'
        self.policy.reset()

    def protocol_digest(self):
        payload = {'execution':asdict(self.config),'safety':asdict(self.backend.safety_config),
                   'timeout':TIMEOUT,'lift_height':.055,'wrong_color_height':.045,'placement_speed':.02}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

    def _fallback(self, reason, obs):
        self.queue.clear()
        if not self.config.fallback_enabled:
            self.failure = reason
            return
        self.triggers.append(reason)
        grasp_z = GRASP_Z if self.backend.embodiment == 'openmanipulator_x' else .0135
        self.expert = Expert(self.target,grasp_z)
        self.expert.entered = obs.time
        if self.lifted and obs.contacts.get(self.target) is True:
            self.expert.phase = 'transport'
            self.expert.lifted = True
        self.guard = Guard(self.backend.safety_config)
        if hasattr(self.backend,'guard'):
            self.backend.guard = Guard(self.backend.safety_config)

    def tick(self, obs, image):
        if self.failure or self.success:
            raise RuntimeError('episode is terminal')
        requested_action = None
        try:
            self.guard.state(obs)
            if obs.time <= self.last_time:
                raise SafetyViolation('nonmonotonic_time')
            self.last_time = obs.time
            if any(pose[2] > .045 for color,pose in obs.cubes.items() if color != self.target):
                self.failure = 'wrong_color'
            if obs.time >= TIMEOUT:
                self.failure = self.failure or 'timeout'
            cube = obs.cubes[self.target]
            self.lifted |= bool(obs.contacts.get(self.target) is True and cube[2] >= .055 and math.dist(cube[:3],obs.ee) <= .035)
            stable = (self.lifted and inside_bin(cube) and obs.cube_speeds[self.target] < .02 and
                      obs.gripper_open is not None and obs.gripper_open >= .84 and
                      not obs.contacts.get(self.target,False) and obs.ee[2] >= .08)
            self.stable_since = (obs.time if self.stable_since is None else self.stable_since) if stable else None
            self.success = self.failure is None and self.stable_since is not None and obs.time-self.stable_since >= self.config.placement_dwell
            if self.failure or self.success:
                self.backend.safe_stop()
                return None
            if self.expert is None:
                try:
                    if not self.queue:
                        observation = PolicyInput.from_observation(obs,image,self.backend.embodiment,self.instruction)
                        chunk = tuple(self.policy.actions(observation))
                        if not 1 <= len(chunk) <= 16 or not all(isinstance(action,Action) for action in chunk):
                            raise ValueError('invalid policy chunk')
                        self.queue.extend(chunk[:self.config.execution_horizon])
                    requested_action = self.queue.popleft()
                    action = self.guard.command(obs,requested_action)
                    reason = self.guard.progress(obs,action)
                    if reason:
                        self._fallback(reason,obs)
                except SafetyViolation as error:
                    self.rejected_policy_actions.append(str(error))
                    self.action_trace.append({'timestamp': obs.time,
                                              'requested_action': [*requested_action.delta, requested_action.gripper] if requested_action else None,
                                              'applied_action': None,
                                              'intervention': str(error)})
                    self._fallback('unsafe_policy_action:'+str(error),obs)
                except (ValueError,RuntimeError,TypeError,TimeoutError) as error:
                    self.policy_errors.append(type(error).__name__)
                    self.action_trace.append({'timestamp': obs.time, 'applied_action': None,
                                              'intervention': 'policy_error:'+type(error).__name__})
                    self._fallback('policy_error:'+type(error).__name__,obs)
            if self.failure:
                self.backend.safe_stop()
                return None
            if self.expert is not None:
                action = self.expert.update(obs)
                if self.expert.failure:
                    self.failure = self.expert.failure
                    self.backend.safe_stop()
                    return None
            action = self.guard.command(obs,action)
            response = self.backend.step(action)
            target = tuple(value+delta for value, delta in zip(obs.ee, action.delta, strict=True))
            trace = {'timestamp': obs.time,
                     'requested_action': [*requested_action.delta, requested_action.gripper] if requested_action else None,
                     'normalized_action': None, 'normalization_scope': 'upstream processor; decoded physical action only at execution boundary',
                     'decoded_action': [*action.delta, action.gripper],
                     'safety_validated_action': [*action.delta, action.gripper],
                     'applied_action': [*action.delta, action.gripper], 'desired_ee': target,
                     'controller': 'fallback_assisted' if self.expert is not None else 'vla_only',
                     'fallback_reason': self.triggers[-1] if self.triggers else None,
                     'clipping': 'none; out-of-contract commands are rejected'}
            if hasattr(response, 'ee'):
                trace.update(response_timestamp=response.time, observed_next_ee=response.ee,
                             tracking_error_m=math.dist(response.ee, target))
            self.action_trace.append(trace)
            return response
        except SafetyViolation as error:
            self.failure = 'unsafe_movement'
            self.safety_events.append(str(error))
            self.backend.safe_stop()
            return None
        except Exception:
            self.backend.safe_stop()
            raise

    def result(self):
        if not self.success and self.failure is None:
            raise RuntimeError('episode is incomplete')
        return {'success':self.success,'failure_reason':self.failure,'pick_result':self.lifted,
                'placement_result':self.success,'controller':'fallback_assisted' if self.triggers else 'vla_only',
                'fallback_triggers':list(self.triggers),'safety_events':list(self.safety_events),
                'rejected_policy_actions':list(self.rejected_policy_actions),'policy_errors':list(self.policy_errors),
                'action_trace_samples':len(self.action_trace),
                'fallback_protocol_sha256':self.protocol_digest()}
