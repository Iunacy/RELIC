"""Real Isaac Lab/PhysX adapter. Import only after AppLauncher starts Kit."""
import math
import torch
import isaaclab.sim as sim
from isaaclab.sim.converters import UrdfConverter
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera, CameraCfg
from .assets import ARM, FINGERS, ROOT, fetch
from .benchmark import (BIN, BIN_INNER, BIN_WALL, CAMERA_EYE, CAMERA_LOOK, COLORS,
                        CONTROL_DT, CUBE, DT, Observation, layout, workspace)
from .openmanipulator import LIMITS, bounded_reference, fk, ik

HOME = (.17,0,.085)


class FaithfulUrdfConverter(UrdfConverter):
    def _get_urdf_import_config(self):
        config = super()._get_urdf_import_config()
        # Lab 2.3.2 passes convert_mimic_joints_to_normal_joints directly to
        # set_parse_mimic. Sim 5.1 defines True as parsing physical mimic joints.
        # Override that inversion locally, and explicitly import URDF inertias.
        config.set_parse_mimic(True)
        config.set_import_inertia_tensor(True)
        return config


class IsaacBackend:
    def __init__(self,device='cuda:0',cameras=False):
        self.sim = sim.SimulationContext(sim.SimulationCfg(dt=DT, device=device))
        self.device = self.sim.device
        self.time = 0.0
        self.camera = None
        self.abort = False
        ground = sim.GroundPlaneCfg()
        ground.func('/World/Ground',ground)
        light = sim.DomeLightCfg(intensity=2500)
        light.func('/World/Light',light)
        self.sim.set_camera_view(CAMERA_EYE,CAMERA_LOOK)
        drive = sim.UrdfConverterCfg.JointDriveCfg(
            gains=sim.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0,damping=0))
        converter = FaithfulUrdfConverter(sim.UrdfConverterCfg(asset_path=str(fetch()),fix_base=True,
            usd_dir=str(ROOT/'cache/usd'),force_usd_conversion=True,
            merge_fixed_joints=False,convert_mimic_joints_to_normal_joints=False,
            self_collision=True,collider_type='convex_decomposition',joint_drive=drive))
        spawn = sim.UsdFileCfg(usd_path=converter.usd_path,
            rigid_props=sim.RigidBodyPropertiesCfg(max_depenetration_velocity=.5),
            articulation_props=sim.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,solver_position_iteration_count=16,solver_velocity_iteration_count=4))
        # URDF effort=1000 is preserved in the source, but is not a hardware rating.
        # Conservative task drive caps (Nm/N) are explicit, pending real qualification.
        self.robot = Articulation(ArticulationCfg(prim_path='/World/Robot',spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={**dict(zip(ARM,ik(HOME))),FINGERS[0]:.020}),
            actuators={
                'arm':ImplicitActuatorCfg(joint_names_expr=list(ARM),stiffness=40.,damping=4.,
                                         effort_limit_sim=1.,velocity_limit_sim=.65),
                'gripper':ImplicitActuatorCfg(joint_names_expr=[FINGERS[0]],stiffness=200.,damping=5.,
                                             effort_limit_sim=5.,velocity_limit_sim=.04)}))
        self.cubes = {}
        for color,rgb in zip(COLORS,((.8,.03,.03),(.03,.7,.03),(.03,.1,.8))):
            self.cubes[color] = RigidObject(RigidObjectCfg(prim_path=f'/World/{color}',
                spawn=sim.CuboidCfg(size=(CUBE,)*3,
                    rigid_props=sim.RigidBodyPropertiesCfg(solver_position_iteration_count=16,solver_velocity_iteration_count=4),
                    mass_props=sim.MassPropertiesCfg(mass=.012),
                    collision_props=sim.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
                    physics_material=sim.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.6,restitution=0.),
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=rgb)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=layout(0)[color])))
        self._bin()
        if cameras:
            self.camera = Camera(CameraCfg(prim_path='/World/Camera',width=640,height=480,
                data_types=['rgb'],update_period=0.,spawn=sim.PinholeCameraCfg(focal_length=24.,
                    horizontal_aperture=20.955,clipping_range=(.01,10.))))
        self.sim.reset()
        self.arm_ids,_ = self.robot.find_joints(list(ARM),preserve_order=True)
        self.left_ids,_ = self.robot.find_joints([FINGERS[0]])
        self.right_ids,_ = self.robot.find_joints([FINGERS[1]])
        self.tool_ids,_ = self.robot.find_bodies(['end_effector_link'])
        self.base_ids,_ = self.robot.find_bodies(['link1'])
        if len(self.arm_ids)!=4 or len(self.left_ids)!=1 or len(self.right_ids)!=1 or len(self.tool_ids)!=1 or len(self.base_ids)!=1:
            raise RuntimeError('imported joint/body mappings do not match official URDF')
        if not self.robot.is_fixed_base:
            raise RuntimeError('imported robot is not fixed-base')
        self.all_ids = self.arm_ids+self.left_ids+self.right_ids
        imported = self.robot.data.joint_pos_limits[0,self.arm_ids].cpu().tolist()
        if any(abs(v-e)>1e-4 for pair,expected in zip(imported,LIMITS) for v,e in zip(pair,expected)):
            raise RuntimeError('imported joint limits changed')
        if self.camera:
            self.camera.set_world_poses_from_view(
                torch.tensor([CAMERA_EYE],device=self.device),torch.tensor([CAMERA_LOOK],device=self.device))
        self.reference = ik(HOME)
        self.grip_ref = .02
        self.base_position = self.robot.data.body_pos_w[0,self.base_ids[0]].clone()

    def _bin(self):
        x,y,z = BIN
        t = .004
        outer = BIN_INNER+2*t
        pieces = [('floor',(outer,outer,z),(x,y,z/2)),
                  ('left',(t,outer,BIN_WALL),(x-(BIN_INNER+t)/2,y,z+BIN_WALL/2)),
                  ('right',(t,outer,BIN_WALL),(x+(BIN_INNER+t)/2,y,z+BIN_WALL/2)),
                  ('front',(BIN_INNER,t,BIN_WALL),(x,y-(BIN_INNER+t)/2,z+BIN_WALL/2)),
                  ('back',(BIN_INNER,t,BIN_WALL),(x,y+(BIN_INNER+t)/2,z+BIN_WALL/2))]
        for name,size,pos in pieces:
            cfg = sim.CuboidCfg(size=size,collision_props=sim.CollisionPropertiesCfg(),
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.35,.35,.35)))
            cfg.func(f'/World/Bin_{name}',cfg,translation=pos)

    def _tick(self):
        self.robot.write_data_to_sim()
        for obj in self.cubes.values():
            obj.write_data_to_sim()
        self.sim.step()
        self.robot.update(DT)
        for obj in self.cubes.values():
            obj.update(DT)
        if self.camera:
            self.camera.update(DT)
        self.time += DT

    def reset(self,seed):
        # State writes occur ONLY at episode reset, never during expert execution.
        self.abort = False
        self.time = 0.
        self.reference = ik(HOME)
        self.grip_ref = .02
        q = self.robot.data.default_joint_pos.clone()
        q[:,self.arm_ids] = torch.tensor([self.reference],device=self.device)
        q[:,self.left_ids+self.right_ids] = .02
        self.robot.write_joint_state_to_sim(q,torch.zeros_like(q))
        self.robot.reset()
        for color,p in layout(seed).items():
            state = self.cubes[color].data.default_root_state.clone()
            state[:,:3] = torch.tensor([p],device=self.device)
            state[:,3:7] = torch.tensor([[1.,0.,0.,0.]],device=self.device)
            state[:,7:] = 0
            self.cubes[color].write_root_pose_to_sim(state[:,:7])
            self.cubes[color].write_root_velocity_to_sim(state[:,7:])
            self.cubes[color].reset()
        self._targets()
        for _ in range(120):
            self._tick()
        self.time = 0.
        obs = self.observe()
        expected = fk(obs.joints[:4])
        if math.dist(expected,obs.ee) > .003:
            raise RuntimeError('imported end-effector transform fails URDF FK check')
        quat = self.robot.data.body_quat_w[0,self.tool_ids[0]].cpu().tolist()
        w,x,y,z = quat
        if 2*(x*z-w*y) > -.995:
            raise RuntimeError('imported tool does not point downward')
        if obs.unsafe or math.dist(obs.ee,HOME) > .01:
            raise RuntimeError('articulation failed stable home check')
        return obs

    def qualify_articulation(self):
        self.reset(0)
        endpoints = []
        for target in (-.011,.020):
            for _ in range(240):
                self.grip_ref += max(-.025*DT,min(.025*DT,target-self.grip_ref))
                self._targets()
                self._tick()
                if self.observe().unsafe:
                    raise RuntimeError('unsafe articulation during empty gripper check')
            q = self.observe().joints
            if abs(q[4]-target) > .003 or abs(q[4]-q[5]) > .002:
                raise RuntimeError('physical gripper opening/closing or mimic check failed')
            endpoints.append(list(q[4:]))
        return {'fixed_base':True,'fk_error_m':math.dist(fk(self.observe().joints[:4]),self.observe().ee),
                'empty_gripper_endpoints':endpoints,'physical_checks_executed':True}

    def _targets(self):
        self.robot.set_joint_position_target(torch.tensor([self.reference],device=self.device),joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(torch.tensor([[self.grip_ref]],device=self.device),joint_ids=self.left_ids)

    def observe(self):
        q = tuple(self.robot.data.joint_pos[0,self.all_ids].cpu().tolist())
        dq = tuple(self.robot.data.joint_vel[0,self.all_ids].cpu().tolist())
        ee = tuple(self.robot.data.body_pos_w[0,self.tool_ids[0]].cpu().tolist())
        cubes = {c:tuple(o.data.root_pose_w[0].cpu().tolist()) for c,o in self.cubes.items()}
        speeds = {c:float(torch.linalg.vector_norm(o.data.root_lin_vel_w[0]) +
                          CUBE/2*torch.linalg.vector_norm(o.data.root_ang_vel_w[0])) for c,o in self.cubes.items()}
        unsafe = (self.abort or not all(math.isfinite(v) for v in (*q,*dq,*ee)) or
                  not workspace(ee) or
                  any(not lo-.015 <= v <= hi+.015 for v,(lo,hi) in zip(q,LIMITS)) or
                  max(map(abs,dq[:4])) > 1.2 or max(map(abs,dq[4:])) > .15 or
                  abs(q[4]-q[5])>.004 or
                  any(not -.013 <= v <= .022 for v in q[4:]) or
                  float(torch.linalg.vector_norm(self.robot.data.body_pos_w[0,self.base_ids[0]]-self.base_position)) > .001 or
                  any(not all(math.isfinite(v) for v in p) for p in cubes.values()))
        return Observation(self.time,ee,q,dq,cubes,speeds,unsafe)

    def step(self,action):
        obs = self.observe()
        if obs.unsafe:
            self.abort = True
            return obs
        try:
            goal = tuple(a+b for a,b in zip(obs.ee,action.delta))
            desired = ik(goal,self.reference)
            self.reference = bounded_reference(self.reference,desired,CONTROL_DT)
            target = -.011 + .031*action.gripper
            self.grip_ref += max(-.025*CONTROL_DT,min(.025*CONTROL_DT,target-self.grip_ref))
            self._targets()
            for _ in range(round(CONTROL_DT/DT)):
                self._tick()
                if self.observe().unsafe:
                    self.abort = True
                    break
        except ValueError:
            self.abort = True
        return self.observe()

    def frame(self):
        if self.camera is None:
            return None
        # RGB data and state are read after the same completed physics/render tick.
        return self.camera.data.output['rgb'][0,:,:,:3].cpu().numpy()
