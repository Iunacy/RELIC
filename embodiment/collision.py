"""Manufacturer collision-pair exclusions from the same pinned robot revision."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from .assets import MANIFEST

SRDF = 'open_manipulator_moveit_config/config/open_manipulator_x/open_manipulator_x.srdf'


def pairs():
    path = Path(__file__).with_name('assets')/'open_manipulator_x.srdf'
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST['files'][SRDF]:
        raise ValueError('official collision matrix checksum differs from pin')
    links = {item.get('name') for item in ET.parse(Path(__file__).with_name('assets')/'open_manipulator_x.urdf').findall('link')}
    result = []
    for item in ET.parse(path).findall('disable_collisions'):
        first, second, reason = (item.get(key) for key in ('link1', 'link2', 'reason'))
        if first not in links or second not in links or first == second or reason not in ('Adjacent', 'Never'):
            raise ValueError('invalid manufacturer collision exclusion')
        result.append((first, second, reason))
    if len({tuple(sorted((first, second))) for first, second, reason in result}) != len(result):
        raise ValueError('duplicate manufacturer collision exclusion')
    return result


def apply_usd(stage, root='/World/Robot'):
    from pxr import Usd, UsdPhysics
    prims = list(Usd.PrimRange(stage.GetPrimAtPath(root)))
    result = []
    for first, second, reason in pairs():
        selected = []
        for name in (first, second):
            matches = [prim for prim in prims if prim.GetName() == name]
            if len(matches) != 1: raise ValueError('ambiguous manufacturer collision body: '+name)
            selected.append(matches[0])
        relationship = UsdPhysics.FilteredPairsAPI.Apply(selected[0]).CreateFilteredPairsRel()
        relationship.AddTarget(selected[1].GetPath())
        result.append({'first': str(selected[0].GetPath()), 'second': str(selected[1].GetPath()), 'reason': reason})
    return {'srdf_sha256': MANIFEST['files'][SRDF], 'pairs': result,
            'scope': 'manufacturer planning collision matrix applied to robot self pairs only; object and environment contacts remain enabled'}
