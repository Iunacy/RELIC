"""Bounded reproducible physical scene variations; execution requires simulator qualification."""
from dataclasses import asdict, dataclass
import math
import random
from .benchmark import CAMERA_EYE, CAMERA_LOOK

KINDS = ('baseline', 'mass', 'friction', 'geometry', 'camera')


@dataclass(frozen=True)
class Scene:
    geometry: str = 'box'
    mass_kg: float = .012
    static_friction: float = .8
    dynamic_friction: float = .6
    camera_eye: tuple = CAMERA_EYE
    camera_look: tuple = CAMERA_LOOK

    def __post_init__(self):
        if self.geometry not in ('box', 'cylinder'): raise ValueError('unsupported object geometry')
        values = (self.mass_kg, self.static_friction, self.dynamic_friction)
        if any(not math.isfinite(value) for value in values) or not .006 <= self.mass_kg <= .024:
            raise ValueError('mass exceeds bounded tabletop range')
        if not .3 <= self.dynamic_friction <= self.static_friction <= 1.2:
            raise ValueError('invalid bounded friction coefficients')
        for vector, reference in ((self.camera_eye, CAMERA_EYE), (self.camera_look, CAMERA_LOOK)):
            if len(vector) != 3 or any(not math.isfinite(value) for value in vector) or math.dist(vector, reference) > .035:
                raise ValueError('camera variation exceeds bounded view envelope')

    def manifest(self):
        return {**asdict(self), 'camera_eye': list(self.camera_eye), 'camera_look': list(self.camera_look)}


def variation(kind='baseline', seed=0):
    if kind not in KINDS or type(seed) is not int or seed < 0: raise ValueError('invalid physical variation schedule')
    rng = random.Random(seed)
    if kind == 'mass': return Scene(mass_kg=rng.choice((.006, .018, .024)))
    if kind == 'friction':
        static, dynamic = rng.choice(((.5, .3), (1., .8), (1.2, 1.)))
        return Scene(static_friction=static, dynamic_friction=dynamic)
    if kind == 'geometry': return Scene(geometry='cylinder')
    if kind == 'camera': return Scene(camera_eye=tuple(value+rng.uniform(-.015, .015) for value in CAMERA_EYE))
    return Scene()
