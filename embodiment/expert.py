"""Privileged feedback expert; no images or ground-truth poses cross the VLA API."""
import math
from .benchmark import Action, BIN, COLORS, GRASP_Z, TIMEOUT, inside_bin

PHASES = ('approach','align','descend','grasp','verify','lift','transport',
          'lower','release','verify_placement','retreat','done')
FAILURES = ('wrong_color','failed_grasp','dropped_object','timeout','unsafe_movement')


class Expert:
    def __init__(self, target):
        if target not in COLORS:
            raise ValueError('unknown target color')
        self.target = target
        self.phase = 'approach'
        self.entered = 0.0
        self.failure = None
        self.stable_since = None
        self.lifted = False
        self.released = False
        self.pick = None
        self.last_time = -1.0

    def advance(self, time):
        self.phase = PHASES[PHASES.index(self.phase)+1]
        self.entered = time
        self.stable_since = None

    def fail(self, reason):
        self.failure = reason
        return Action((0,0,0), 1.0 if not self.lifted else 0.0)

    def update(self, obs):
        if obs.time <= self.last_time:
            raise ValueError('observations must advance in simulation time')
        self.last_time = obs.time
        if self.failure or self.phase == 'done':
            return Action((0,0,0), 1)
        if obs.unsafe:
            return self.fail('unsafe_movement')
        if obs.time >= TIMEOUT or obs.time-self.entered > 10:
            return self.fail('timeout')
        cube = obs.cubes[self.target]
        if self.pick is None:
            self.pick = cube[:3]
        if any(p[2] > .045 for color,p in obs.cubes.items() if color != self.target):
            return self.fail('wrong_color')
        if self.lifted and self.phase in ('lift','transport') and math.dist(cube[:3], obs.ee) > .045:
            return self.fail('dropped_object')
        x,y,_ = self.pick
        elapsed = obs.time-self.entered
        goal = obs.ee
        grip = 1.0
        if self.phase in ('approach','align'):
            goal = (x,y,.085)
        elif self.phase == 'descend':
            goal = (x,y,GRASP_Z)
        elif self.phase in ('grasp','verify'):
            goal,grip = (x,y,GRASP_Z),0.0
        elif self.phase == 'lift':
            goal,grip = (x,y,.085),0.0
        elif self.phase == 'transport':
            goal,grip = (BIN[0],BIN[1],.085),0.0
        elif self.phase in ('lower','release','verify_placement'):
            goal = (BIN[0],BIN[1],.065)
            if self.phase == 'release' and math.dist(obs.ee,goal) <= .004:
                self.released = True
            grip = 1.0 if self.phase == 'verify_placement' or self.released else 0.0
        elif self.phase == 'retreat':
            goal = (BIN[0],BIN[1],.085)
        arrived = math.dist(obs.ee,goal) < .004
        slow = max(map(abs,obs.velocities[:obs.arm_dof]),default=0) < .12
        if self.phase == 'grasp':
            if elapsed >= 1.6:
                self.advance(obs.time)
        elif self.phase == 'verify':
            # Physical aperture must remain blocked by the cube, near the tool.
            if elapsed >= .4:
                if not obs.contacts.get(self.target, False) or math.dist(cube[:3],obs.ee) > .025:
                    return self.fail('failed_grasp')
                self.advance(obs.time)
        elif self.phase == 'lift':
            if arrived and slow:
                if not obs.contacts.get(self.target, False) or cube[2] < .055 or math.dist(cube[:3],obs.ee) > .035:
                    return self.fail('failed_grasp')
                self.lifted = True
                self.advance(obs.time)
        elif self.phase == 'release':
            if arrived and grip == 1 and elapsed > 1 and (obs.gripper_open if obs.gripper_open is not None else (obs.joints[4]+.011)/.031) >= .84:
                self.advance(obs.time)
        elif self.phase in ('verify_placement','retreat'):
            stable = inside_bin(cube) and obs.cube_speeds[self.target] < .02
            self.stable_since = (obs.time if self.stable_since is None else self.stable_since) if stable else None
            if self.phase == 'verify_placement' and elapsed > 3 and not stable:
                return self.fail('dropped_object')
            if self.stable_since is not None and obs.time-self.stable_since >= 1.0 and (self.phase != 'retreat' or arrived):
                self.advance(obs.time)
        elif arrived and slow:
            self.advance(obs.time)
        delta = tuple(b-a for a,b in zip(obs.ee,goal))
        distance = math.dist(delta,(0,0,0))
        scale = min(1, .0015/max(distance,1e-12))
        return Action(tuple(v*scale for v in delta),grip)
