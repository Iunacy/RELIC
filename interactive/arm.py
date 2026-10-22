import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor/low_cost_robot_arm'
JOINTS = ('base_rotation', 'pitch', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper')
HOME = np.array([0., -.25, -.3, .35, 0., -.65])
POSES = {
    'target_left': [-.65, -.45, -.45, .45, 0., -.65],
    'target_center': [0., -.6, -.65, .55, 0., -.65],
    'target_right': [.65, -.45, -.45, .45, 0., -.65],
}
COLORS = {'target_left': [.27, .75, 1., .8], 'target_center': [1., .68, .22, .8],
          'target_right': [.33, .92, .65, .8]}


class Arm:
    def __init__(self, degraded=False):
        spec = mujoco.MjSpec.from_file(str(VENDOR / 'scene.xml'))
        spec.body('gripper_static_finger').add_site(name='relic_tool', pos=[-.063, .013, 0.],
                                                  size=[.004, 0., 0.], rgba=[1., .5, .1, 1.])
        spec.visual.global_.offwidth = 960
        spec.visual.global_.offheight = 640
        self.model = spec.compile()
        self.nominal = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.plan = mujoco.MjData(self.model)
        self.gravity = mujoco.MjData(self.nominal)
        self.jids = np.array([self.model.joint(x).id for x in JOINTS])
        self.aids = np.array([self.model.actuator(x).id for x in JOINTS])
        self.qids = self.model.jnt_qposadr[self.jids]
        self.dids = self.model.jnt_dofadr[self.jids]
        self.lower = np.maximum(self.model.jnt_range[self.jids, 0], self.model.actuator_ctrlrange[self.aids, 0])
        self.upper = np.minimum(self.model.jnt_range[self.jids, 1], self.model.actuator_ctrlrange[self.aids, 1])
        for j, a in zip(self.jids, self.aids):
            if (self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or
                self.model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or
                self.model.actuator_trnid[a, 0] != j or
                not self.model.jnt_limited[j] or not self.model.actuator_ctrllimited[a] or
                not self.model.actuator_forcelimited[a] or
                not np.array_equal(self.model.actuator_gear[a], [1, 0, 0, 0, 0, 0])):
                raise ValueError('unsupported actuator mapping')
        if degraded:
            self.model.actuator_gainprm[self.aids, 0] *= .4
            self.model.actuator_biasprm[self.aids, 1] *= .4
            self.model.dof_frictionloss[self.dids] += .15
        self.site = self.model.site('relic_tool').id
        self.targets = {k: self.fk(np.array(v)).tolist() for k, v in POSES.items()}
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = HOME
        self.data.ctrl[self.aids] = HOME
        mujoco.mj_forward(self.model, self.data)

    def check_q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all() or np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError('joint_or_control_limit')
        return q

    def contacts(self, data):
        return [(int(c.geom[0]), int(c.geom[1]), float(c.dist)) for c in data.contact
                if c.dist <= 0]

    def fk(self, q):
        self.plan.qpos[self.qids] = self.check_q(q)
        mujoco.mj_forward(self.model, self.plan)
        return self.plan.site_xpos[self.site].copy()

    def safe_path(self, start, goal):
        self.check_q(start)
        self.check_q(goal)
        count = max(1, int(np.ceil(np.max(np.abs(goal-start))/.01)))
        for u in np.linspace(0, 1, count+1):
            self.fk(start+u*(goal-start))
            if self.contacts(self.plan):
                raise ValueError('sampled_contact_path')

    def solve(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or np.linalg.norm(target) > .5:
            raise ValueError('invalid_cartesian_target')
        best = None
        for seed in [HOME, *[np.array(x) for x in POSES.values()]]:
            q = seed.copy()
            for _ in range(250):
                error = target-self.fk(q)
                if np.linalg.norm(error) < .0001:
                    try:
                        self.safe_path(HOME, q)
                        if best is None or np.linalg.norm(q-HOME) < np.linalg.norm(best-HOME):
                            best = q.copy()
                    except ValueError:
                        pass
                    break
                jac = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.plan, jac, None, self.site)
                jac = jac[:, self.dids[:4]]
                delta = jac.T @ np.linalg.solve(jac@jac.T + 1.e-5*np.eye(3), error)
                q[:4] = np.clip(q[:4]+np.clip(delta, -.08, .08), self.lower[:4]+.02, self.upper[:4]-.02)
                q[4:] = HOME[4:]
        if best is None:
            raise ValueError('unreachable_or_contact_rejected')
        return best

    def health(self):
        d = self.data
        if (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or
            not np.isfinite(d.qacc).all() or any(w.number for w in d.warning)):
            raise ValueError('physics_health_failure')
        self.check_q(d.qpos[self.qids])
        if np.max(np.abs(d.qvel[self.dids])) > 8:
            raise ValueError('observed_velocity_limit')
        if self.contacts(d):
            raise ValueError('observed_contact')
        effort = d.actuator_force[self.aids]
        if np.any(effort < self.model.actuator_forcerange[self.aids, 0]-1.e-6) or np.any(effort > self.model.actuator_forcerange[self.aids, 1]+1.e-6):
            raise ValueError('actuator_force_limit')

    def run(self, target_id, controller='conventional', callback=None, inject=None):
        if target_id not in self.targets:
            raise ValueError('unknown_target')
        if controller not in ('conventional', 'integral'):
            raise ValueError('unsupported_controller_no_mlp_transfer')
        goal = self.solve(self.targets[target_id])
        self.reset()
        self.health()
        dt = self.model.opt.timestep
        ticks = int(np.ceil(3./dt))
        total = ticks + int(np.ceil(2./dt))
        correction = np.zeros(6)
        integral = np.zeros(6)
        last_command = HOME.copy()
        rows = []
        status, reason, withdrawing = 'Success', 'completed', False
        withdrawal_reason = None
        failure_observation = None
        command_error = None
        fallbacks = 0
        began = time.perf_counter()
        for k in range(total+1):
            u = min(1., k/ticks)
            s = u**3*(10+u*(-15+6*u))
            desired = HOME+s*(goal-HOME)
            velocity = (goal-HOME)*30*u*u*(1-u)**2/(ticks*dt) if k <= ticks else np.zeros(6)
            try:
                self.health()
                self.gravity.qpos[:] = desired
                self.gravity.qvel[:] = 0
                mujoco.mj_forward(self.nominal, self.gravity)
                error_pre = desired-self.data.qpos[self.qids]
                raw = .10*velocity+self.gravity.qfrc_bias[self.dids]/100.+error_pre+integral
                raw[4:] = 0
                if inject is not None and k >= inject:
                    raw[0] = np.nan
                if controller == 'integral':
                    if not np.isfinite(raw).all() or np.any(np.abs(raw) > 5):
                        withdrawing = True
                        withdrawal_reason = withdrawal_reason or 'invalid_controller_output'
                    next_correction = correction + np.clip((np.zeros(6) if withdrawing else np.clip(raw, -.12, .12))-correction, -.0005, .0005)
                    command = desired+next_correction
                else:
                    next_correction = np.zeros(6)
                    command = desired
                try:
                    if np.max(np.abs(command-last_command)) > .005+1.e-12:
                        raise ValueError('command_step')
                    self.safe_path(last_command, command)
                    self.safe_path(self.data.qpos[self.qids].copy(), command)
                except ValueError as rejected:
                    if controller != 'integral':
                        raise
                    withdrawing = True
                    withdrawal_reason = withdrawal_reason or str(rejected)
                    next_correction = correction+np.clip(-correction, -.0005, .0005)
                    command = desired+next_correction
                    if np.max(np.abs(command-last_command)) > .005+1.e-12:
                        raise ValueError('withdrawal_infeasible_step')
                    try:
                        self.safe_path(last_command, command)
                        self.safe_path(self.data.qpos[self.qids].copy(), command)
                    except ValueError as e:
                        raise ValueError('withdrawal_infeasible_path') from e
                if withdrawing:
                    fallbacks += 1
                if controller == 'integral' and not withdrawing:
                    delta = 2.*error_pre*dt
                    allowed = ((np.abs(raw) < .12) & (np.abs(raw-correction) <= .0005)) | (delta*(raw-correction) < 0)
                    integral = np.clip(integral+np.where(allowed, delta, 0), -.12, .12)
                previous_observed = self.data.qpos[self.qids].copy()
                self.data.ctrl[self.aids] = command
                mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
                self.health()
                self.safe_path(previous_observed, self.data.qpos[self.qids].copy())
                correction, last_command = next_correction, command
            except ValueError as e:
                status, reason = 'Stopped', str(e)
                failure_observation = dict(time=float(self.data.time) if np.isfinite(self.data.time) else None,
                    q=[float(x) if np.isfinite(x) else None for x in self.data.qpos],
                    velocity=[float(x) if np.isfinite(x) else None for x in self.data.qvel],
                    contacts=self.contacts(self.data))
                break
            observed = self.data.qpos[self.qids].copy()
            row = dict(t=float(self.data.time), desired=desired.tolist(), observed=observed.tolist(),
                       velocity=self.data.qvel[self.dids].tolist(), command=command.tolist(),
                       effort=self.data.actuator_force[self.aids].tolist(), correction=correction.tolist(),
                       tool=self.data.site_xpos[self.site].tolist(), contacts=int(self.data.ncon),
                       phase='command' if k <= ticks else 'settling', fallback=withdrawing, fallback_reason=withdrawal_reason)
            rows.append(row)
            if k == ticks:
                command_error = float(np.linalg.norm(self.data.site_xpos[self.site]-self.targets[target_id]))
            if callback is not None and (k % 16 == 0 or k == total):
                callback(self, row)
        moving = rows[:ticks+1]
        error = np.array([np.array(r['desired'])-r['observed'] for r in moving])
        final_error = float(np.linalg.norm(self.data.site_xpos[self.site]-self.targets[target_id]))
        success = status == 'Success' and final_error <= .005 and np.max(np.abs(self.data.qvel[self.dids])) < .02
        if status == 'Success' and not success:
            status, reason = 'NonConvergence', 'final_pose_or_velocity_tolerance'
        return dict(schema='relic.interactive.v1', robot='menagerie', target_id=target_id,
                    target_m=self.targets[target_id], controller=controller, status=status, reason=reason,
                    success=bool(success), failure_observation=failure_observation, withdrawal_reason=withdrawal_reason, initial_q=HOME.tolist(), goal_q=goal.tolist(),
                    physics=dict(gain=float(self.model.actuator_gainprm[self.aids[0], 0]),
                                 friction=float(self.model.dof_frictionloss[self.dids[0]])),
                    metrics=dict(rmse_rad=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                                 peak_rad=float(np.max(np.abs(error))) if len(error) else None,
                                 command_end_m=command_error, final_m=final_error,
                                 simulation_s=float(self.data.time), wall_s=time.perf_counter()-began,
                                 measured_motion_rad=float(np.max(np.abs(self.data.qpos[self.qids]-HOME))),
                                 fallbacks=fallbacks), trace=rows)


def camera(model):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [-.07, 0, .13]
    cam.distance = .72
    cam.azimuth = 135
    cam.elevation = -23
    return cam


def markers(scene, targets, selected):
    for name, pos in targets.items():
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.009, 0, 0]), np.array(pos),
                           np.eye(3).flatten(), np.array(COLORS[name], dtype=np.float32))
        g.label = name.replace('target_', '')+(' *' if name == selected else '')
        scene.ngeom += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', choices=POSES, default='target_left')
    p.add_argument('--controller', choices=['conventional', 'integral'], default='conventional')
    p.add_argument('--degraded', action='store_true')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'runs/viewer')
    args = p.parse_args()
    arm = Arm(args.degraded)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.headless:
        result = arm.run(args.target, args.controller)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
            viewer.cam.lookat[:] = camera(arm.model).lookat
            viewer.cam.distance = .72
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -23
            start = time.perf_counter()
            def update(a, row):
                if not viewer.is_running():
                    raise KeyboardInterrupt
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    markers(viewer.user_scn, a.targets, args.target)
                viewer.sync()
                time.sleep(max(0, start+row['t']-time.perf_counter()))
            result = arm.run(args.target, args.controller, update)
            (args.output/'execution.json').write_text(json.dumps(result, indent=2))
            while viewer.is_running():
                viewer.sync()
                time.sleep(.05)
    (args.output/'execution.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'trace'}, indent=2))


if __name__ == '__main__':
    main()
