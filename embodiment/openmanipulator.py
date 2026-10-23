"""Radially aligned downward grasp IK from the pinned ROBOTIS joint origins."""
import math
from .assets import ARM, validate_urdf
from .benchmark import finite, workspace

LIMITS = tuple((float(validate_urdf()[j].find('limit').attrib['lower']),
                float(validate_urdf()[j].find('limit').attrib['upper'])) for j in ARM)


def fk(q):
    finite(q,4)
    yaw,a,b,c = q
    pitch = a+b+c
    radius = .024*math.cos(a)+.128*math.sin(a)+.124*math.cos(a+b)+.126*math.cos(pitch)
    z = .0595-.024*math.sin(a)+.128*math.cos(a)-.124*math.sin(a+b)-.126*math.sin(pitch)
    return (.012+radius*math.cos(yaw), radius*math.sin(yaw), z)


def ik(p, previous=(0,0,0,0)):
    if not workspace(p):
        raise ValueError('outside shared workspace')
    finite(previous,4)
    x,y,z = p
    radius = math.hypot(x-.012,y)
    height = z-.0595+.126
    length = math.hypot(.024,.128)
    alpha = math.atan2(.128,.024)
    cosine = (radius**2+height**2-length**2-.124**2)/(2*length*.124)
    if not -1 <= cosine <= 1:
        raise ValueError('unreachable downward tool pose')
    candidates = []
    for relative in (math.acos(cosine), -math.acos(cosine)):
        theta = math.atan2(height,radius)-math.atan2(.124*math.sin(relative),length+.124*math.cos(relative))
        a = alpha-theta
        b = -relative-alpha
        c = math.pi/2-a-b
        q = (math.atan2(y,x-.012), a,b,c)
        if all(lo+1e-4 <= v <= hi-1e-4 for v,(lo,hi) in zip(q,LIMITS)):
            candidates.append(q)
    if not candidates:
        raise ValueError('downward grasp violates joint limits')
    return min(candidates, key=lambda q: math.dist(q,previous))


def bounded_reference(current, goal, dt):
    finite(current,4)
    finite(goal,4)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('invalid dt')
    if any(not lo <= value <= hi for value,(lo,hi) in zip(goal,LIMITS)):
        raise ValueError('joint limit')
    return tuple(a+max(-.5*dt,min(.5*dt,b-a)) for a,b in zip(current,goal))
