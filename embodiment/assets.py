"""Fetch pinned official assets and resolve package URIs without changing physics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / 'assets/manifest.json').read_text())
URDF = ROOT / 'assets/open_manipulator_x.urdf'
ARM = ('joint1', 'joint2', 'joint3', 'joint4')
FINGERS = ('gripper_left_joint', 'gripper_right_joint')


def validate_urdf(path=URDF):
    robot = ET.parse(path).getroot()
    joints = {j.attrib['name']: j for j in robot.findall('joint')}
    if set(joints) != set(ARM + FINGERS + ('world_fixed', 'end_effector_joint')):
        raise ValueError('unexpected articulation mapping')
    for name in ARM + FINGERS:
        joint = joints[name]
        limit = joint.find('limit')
        values = [float(limit.attrib[k]) for k in ('lower', 'upper', 'effort', 'velocity')]
        if not all(map(math.isfinite, values)) or values[0] >= values[1] or min(values[2:]) <= 0:
            raise ValueError(f'invalid joint limits: {name}')
        if joint.attrib['type'] != ('revolute' if name in ARM else 'prismatic'):
            raise ValueError('unexpected joint type')
    mimic = joints[FINGERS[1]].find('mimic')
    if mimic is None or mimic.attrib != {'joint': FINGERS[0], 'multiplier': '1'}:
        raise ValueError('gripper mimic must be preserved')
    if joints['world_fixed'].attrib['type'] != 'fixed':
        raise ValueError('base must be fixed')
    if joints['end_effector_joint'].find('origin').attrib['xyz'] != '0.126 0.0 0.0':
        raise ValueError('unexpected tool transform')
    for link in robot.findall('link'):
        inertia = link.find('inertial')
        if inertia is not None:
            mass = float(inertia.find('mass').attrib['value'])
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError('invalid mass')
            xx,yy,zz,xy,xz,yz = [float(inertia.find('inertia').attrib[k])
                                for k in ('ixx','iyy','izz','ixy','ixz','iyz')]
            determinant = xx*yy*zz+2*xy*xz*yz-xx*yz*yz-yy*xz*xz-zz*xy*xy
            if not all(math.isfinite(v) for v in (xx,yy,zz,xy,xz,yz)) or min(xx,yy,zz,xx*yy-xy*xy,determinant) <= 0:
                raise ValueError('invalid inertia')
    return joints


def link_masses(path=URDF):
    robot = ET.parse(path).getroot()
    return {link.attrib['name']:float(link.find('inertial/mass').attrib['value'])
            for link in robot.findall('link') if link.find('inertial/mass') is not None}


def fetch(cache=ROOT / 'cache'):
    cache = Path(cache)
    for name, expected in MANIFEST['files'].items():
        dest = cache / name
        if not dest.exists():
            url = f"https://raw.githubusercontent.com/ROBOTIS-GIT/open_manipulator/{MANIFEST['commit']}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'checksum mismatch: {name}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
            raise ValueError(f'cached asset changed: {name}')
    source = cache / 'open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf'
    validate_urdf(source)
    tree = ET.parse(source)
    for mesh in tree.findall('.//mesh'):
        uri = mesh.attrib['filename']
        if not uri.startswith('package://open_manipulator_description/'):
            raise ValueError('unexpected mesh URI')
        mesh.attrib['filename'] = str((cache / uri.removeprefix('package://')).resolve())
    resolved = cache / 'resolved.urdf'
    tree.write(resolved, encoding='utf-8', xml_declaration=True)
    return resolved.resolve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'cache')
    print(fetch(parser.parse_args().cache))
