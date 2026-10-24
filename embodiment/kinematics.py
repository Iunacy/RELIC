"""Independent pinned-URDF kinematics, verified against a CPU simulator."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from .assets import ARM, FINGERS, ROOT, URDF, MANIFEST
from .benchmark import finite
from .panda import ARM as PANDA_ARM, FINGERS as PANDA_FINGERS, BASE, TOOL_OFFSET


def robot_asset(robot):
    if robot == 'openmanipulator_x':
        return URDF
    if robot != 'franka_panda':
        raise ValueError('unknown robot')
    from .replay_render import panda_asset
    return panda_asset()


def rotation(axis, angle):
    import numpy as np
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)


def transform(origin):
    import numpy as np
    value = np.eye(4)
    xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0.]*3
    rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0.]*3
    finite(xyz, 3); finite(rpy, 3)
    value[:3, 3] = xyz
    value[:3, :3] = rotation((0., 0., 1.), rpy[2])@rotation((0., 1., 0.), rpy[1])@rotation((1., 0., 0.), rpy[0])
    return value


class Kinematics:
    def __init__(self, robot):
        self.robot = robot
        self.path = robot_asset(robot)
        tree = ET.parse(self.path).getroot()
        self.names = ARM if robot == 'openmanipulator_x' else PANDA_ARM
        self.fingers = FINGERS if robot == 'openmanipulator_x' else PANDA_FINGERS
        self.tool = 'link5' if robot == 'openmanipulator_x' else 'panda_hand'
        self.base = (0., 0., 0.) if robot == 'openmanipulator_x' else BASE
        self.offset = (.126, 0., 0.) if robot == 'openmanipulator_x' else (0., 0., TOOL_OFFSET)
        parents = {joint.find('child').get('link'): joint for joint in tree.findall('joint')}
        chain, link, seen = [], self.tool, set()
        while link in parents:
            if link in seen: raise ValueError('cyclic URDF chain')
            seen.add(link)
            joint = parents[link]
            chain.append(joint)
            link = joint.find('parent').get('link')
        self.chain = chain[::-1]
        if tuple(j.get('name') for j in self.chain if j.get('type') != 'fixed') != self.names:
            raise ValueError('URDF task chain differs from robot adapter')
        self.limits = tuple((float(j.find('limit').get('lower')), float(j.find('limit').get('upper')))
                            for j in self.chain if j.get('type') != 'fixed')
        self.asset_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def forward(self, joints, jacobian=False):
        import numpy as np
        finite(joints, len(self.names))
        if any(not lo-1e-8 <= q <= hi+1e-8 for q, (lo, hi) in zip(joints, self.limits, strict=True)):
            raise ValueError('joint exceeds URDF limit')
        pose = np.eye(4)
        pose[:3, 3] = self.base
        axes, positions, index = [], [], 0
        for joint in self.chain:
            pose = pose@transform(joint.find('origin'))
            if joint.get('type') == 'fixed': continue
            if joint.get('type') != 'revolute': raise ValueError('unsupported task-chain joint')
            axis = [float(v) for v in joint.find('axis').get('xyz').split()]
            positions.append(pose[:3, 3].copy())
            axes.append(pose[:3, :3]@axis)
            motion = np.eye(4)
            motion[:3, :3] = rotation(axis, joints[index])
            pose = pose@motion
            index += 1
        pose[:3, 3] += pose[:3, :3]@self.offset
        if not jacobian: return pose
        derivative = np.zeros((6, len(joints)))
        for i, (axis, origin) in enumerate(zip(axes, positions, strict=True)):
            derivative[:3, i] = np.cross(axis, pose[:3, 3]-origin)
            derivative[3:, i] = axis
        return pose, derivative

    def inverse(self, goal, initial, iterations=160):
        import numpy as np
        from .openmanipulator import ik
        from .panda import downward_quaternion
        finite(goal, 3)
        if self.robot == 'openmanipulator_x': return ik(goal, initial)
        finite(initial, 7)
        w, x, y, z = downward_quaternion(goal)
        desired = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        q = np.asarray(initial, dtype=float).copy()
        for _ in range(iterations):
            pose, jacobian = self.forward(q.tolist(), True)
            relative = desired@pose[:3, :3].T
            orientation = .5*np.array([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0], relative[1, 0]-relative[0, 1]])
            position = np.asarray(goal)-pose[:3, 3]
            angular_distance = math.acos(float(np.clip((np.trace(relative)-1)/2, -1, 1)))
            if np.linalg.norm(position) < 1e-5 and angular_distance < 1e-4:
                return tuple(float(v) for v in q)
            error = np.r_[position, .25*orientation]
            jacobian[3:] *= .25
            update = jacobian.T@np.linalg.solve(jacobian@jacobian.T+.0001*np.eye(6), error)
            update *= min(1., .15/max(float(np.max(np.abs(update))), 1e-12))
            q = np.clip(q+update, [lo+1e-5 for lo, hi in self.limits], [hi-1e-5 for lo, hi in self.limits])
        raise ValueError('bounded URDF inverse kinematics did not converge')


def simulator_model(robot, actuated=False):
    import mujoco
    model = Kinematics(robot)
    tree = ET.parse(model.path)
    for link in tree.findall('link'):
        for visual in link.findall('visual'): link.remove(visual)
    for mesh in tree.findall('.//mesh'):
        name = mesh.get('filename')
        if name.startswith('package://open_manipulator_description/'):
            path = ROOT/'cache'/name.removeprefix('package://')
            key = name.removeprefix('package://')
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST['files'].get(key):
                raise ValueError('pinned OpenMANIPULATOR collision asset missing or altered')
        elif name.startswith('package://franka_description/'):
            path = model.path.parents[1]/name.removeprefix('package://franka_description/')
        else: raise ValueError('unknown asset URI')
        mesh.set('filename', str(path))
    extension = ET.SubElement(tree.getroot(), 'mujoco')
    ET.SubElement(extension, 'compiler', {'fusestatic': 'false'})
    spec = mujoco.MjSpec.from_string(ET.tostring(tree.getroot(), encoding='unicode'))
    if robot == 'franka_panda': spec.body('panda_link0').pos = BASE
    spec.body(model.tool).add_site(name='viper_audit_tool', pos=model.offset, size=[.001, 0., 0.])
    if actuated:
        spec.option.timestep = 1/240
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for name, limits in zip(model.names, model.limits, strict=True):
            actuator = spec.add_actuator(name='drive_'+name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            actuator.set_to_position(kp=100. if robot == 'openmanipulator_x' else 400., kv=10. if robot == 'openmanipulator_x' else 40.)
            actuator.ctrllimited, actuator.ctrlrange = True, limits
            effort = 2. if robot == 'openmanipulator_x' else (87. if model.names.index(name) < 4 else 12.)
            actuator.forcelimited, actuator.forcerange = True, [-effort, effort]
        spec.worldbody.add_geom(name='audit_floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1., 1., .01])
        cube = spec.worldbody.add_body(name='audit_contact_cube', pos=[.5, .5, .08])
        cube.add_freejoint(name='audit_cube_free')
        cube.add_geom(name='audit_cube', type=mujoco.mjtGeom.mjGEOM_BOX, size=[.0125]*3, mass=.012)
    return model, spec.compile()


def audit(robot, samples=12):
    import mujoco
    import numpy as np
    if type(samples) is not int or not 1 <= samples <= 100: raise ValueError('bounded audit sample count required')
    chain, model = simulator_model(robot)
    data = mujoco.MjData(model)
    site = model.site('viper_audit_tool').id
    joints = [model.joint(name).id for name in chain.names]
    addresses = [model.jnt_qposadr[index] for index in joints]
    dofs = [model.jnt_dofadr[index] for index in joints]
    rng = random.Random(20261024)
    errors, derivatives, limits = [], [], []
    for joint, bounds in zip(joints, chain.limits, strict=True):
        if not np.allclose(model.jnt_range[joint], bounds, atol=1e-8): raise ValueError('simulator joint limits differ from URDF')
        limits.append(model.jnt_range[joint].tolist())
    for _ in range(samples):
        q = [rng.uniform(lo+.01, hi-.01) for lo, hi in chain.limits]
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        pose, jacobian = chain.forward(q, True)
        errors.append(float(np.linalg.norm(pose[:3, 3]-data.site_xpos[site])))
        if not np.allclose(pose[:3, :3], data.site_xmat[site].reshape(3, 3), atol=1e-8):
            raise ValueError('simulator orientation differs from URDF chain')
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        derivatives.append(float(np.max(np.abs(jacobian-np.vstack((jp[:, dofs], jr[:, dofs]))))))
    initial = (0.,)*4 if robot == 'openmanipulator_x' else (0., -.5, 0., -2., 0., 1.5, .785)
    inverse_errors = []
    for goal in ((.17, 0., .085), (.18, -.04, .06), (.14, .145, .085)):
        q = chain.inverse(goal, initial)
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        inverse_errors.append(float(np.linalg.norm(data.site_xpos[site]-goal)))
    passed = max(errors) < 1e-8 and max(derivatives) < 1e-8 and max(inverse_errors) < 1e-4
    return {'schema': 'viper.kinematic_audit.v1', 'robot': robot, 'simulator': 'mujoco',
            'version': mujoco.__version__, 'asset_sha256': chain.asset_sha256, 'passed': passed,
            'samples': samples, 'maximum_fk_error_m': max(errors), 'maximum_jacobian_error': max(derivatives),
            'inverse_simulator_errors_m': inverse_errors, 'joint_limits': limits,
            'collision_geometries': model.ngeom, 'dynamics_executed': False, 'manipulation_qualified': False,
            'scope': 'real CPU simulator FK/Jacobian and imported collision geometry; no PhysX, actuator or grasp qualification'}


def audit_dynamics(robot):
    import mujoco
    import numpy as np
    chain, model = simulator_model(robot, actuated=True)
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[model.joint(name).id] for name in chain.names]
    initial = (0.,)*4 if robot == 'openmanipulator_x' else (0., -.5, 0., -2., 0., 1.5, .785)
    home = chain.inverse((.17, 0., .085), initial)
    data.qpos[addresses] = home
    for finger in chain.fingers: data.qpos[model.jnt_qposadr[model.joint(finger).id]] = .004 if robot == 'openmanipulator_x' else .02
    desired = np.asarray(home)
    desired[0] += .01
    data.ctrl[:] = desired
    max_contact, torque, violations = 0., 0., 0
    contact_pairs = set()
    cube = model.geom('audit_cube').id
    arm_dofs = [model.jnt_dofadr[model.joint(name).id] for name in chain.names]
    for _ in range(240):
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = 0.
        data.qfrc_applied[arm_dofs] = data.qfrc_bias[arm_dofs]
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all(): raise RuntimeError('nonfinite CPU dynamics')
        torque = max(torque, float(np.max(np.abs(data.actuator_force))))
        violations += any(not lo-.001 <= value <= hi+.001 for value, (lo, hi) in zip(data.qpos[addresses], chain.limits, strict=True))
        for index in range(data.ncon):
            contact = data.contact[index]
            if cube not in (contact.geom1, contact.geom2):
                contact_pairs.add(tuple(model.body(model.geom_bodyid[geom]).name for geom in (contact.geom1, contact.geom2)))
            if cube in (contact.geom1, contact.geom2):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                max_contact = max(max_contact, float(force[0]))
    error = float(np.max(np.abs(data.qpos[addresses]-desired)))
    return {'schema': 'viper.cpu_dynamics_audit.v1', 'robot': robot, 'simulator': 'mujoco', 'version': mujoco.__version__,
            'simulation_seconds': float(data.time), 'actuator_names': ['drive_'+name for name in chain.names],
            'joint_limit_violating_ticks': violations, 'endpoint_joint_error_rad': error,
            'maximum_actuator_force': torque, 'robot_contact_pairs': sorted(contact_pairs),
            'joint_endpoint_errors_rad': {name: float(q-goal) for name, q, goal in zip(chain.names, data.qpos[addresses], desired, strict=True)}, 'maximum_cube_floor_normal_force_n': max_contact,
            'passed': violations == 0 and error < .003 and max_contact > .05,
            'scope': 'one-second CPU joint-servo dynamics and a separate cube-floor contact probe; no grasp or PhysX qualification',
            'dynamics_executed': True, 'manipulation_qualified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=['openmanipulator_x', 'franka_panda'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dynamics', action='store_true')
    args = parser.parse_args()
    result = audit_dynamics(args.robot) if args.dynamics else audit(args.robot)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 2)
