"""Real Isaac Lab/PhysX adapter. Import only after AppLauncher starts Kit."""
import math
import random
import torch
import isaaclab.sim as sim
from isaaclab.sim.converters import UrdfConverter
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from .assets import ARM, FINGERS, ROOT, fetch
from .benchmark import (BIN, BIN_INNER, BIN_WALL, CAMERA_EYE, CAMERA_LOOK, COLORS,
                        CONTROL_DT, CUBE, DT, Observation, layout, workspace)
from .openmanipulator import LIMITS, bounded_reference, fk, ik
from .safety import Guard, SafetyConfig, SafetyViolation

HOME = (.17,0,.085)


class FaithfulUrdfConverter(UrdfConverter):
    def _get_urdf_import_config(self):
        config = super()._get_urdf_import_config()
        config.set_parse_mimic(True)
        config.set_import_inertia_tensor(True)
        return config


class IsaacBackend:
    def __init__(self,device='cuda:0',cameras=False,robot='openmanipulator_x'):
        if robot not in ('openmanipulator_x','franka_panda'):
            raise ValueError('unsupported embodiment')
        self.embodiment = robot
        self.initializing = False
        self.sim = sim.SimulationContext(sim.SimulationCfg(dt=DT, device=device))
        self.device = self.sim.device
        self.time = 0.0
        self.camera = None
        self.abort = False
        self.safety_events = []
        ground = sim.GroundPlaneCfg()
        ground.func('/World/Ground',ground)
        light = sim.DomeLightCfg(intensity=2500)
        light.func('/World/Light',light)
        self.sim.set_camera_view(CAMERA_EYE,CAMERA_LOOK)
        if robot == 'openmanipulator_x':
            drive = sim.UrdfConverterCfg.JointDriveCfg(
                gains=sim.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0,damping=0))
            converter = FaithfulUrdfConverter(sim.UrdfConverterCfg(asset_path=str(fetch()),fix_base=True,
                usd_dir=str(ROOT/'cache/usd'),force_usd_conversion=True,
                merge_fixed_joints=False,convert_mimic_joints_to_normal_joints=False,
                self_collision=True,collider_type='convex_decomposition',joint_drive=drive))
            spawn = sim.UsdFileCfg(usd_path=converter.usd_path,activate_contact_sensors=True,
                rigid_props=sim.RigidBodyPropertiesCfg(max_depenetration_velocity=.5),
                articulation_props=sim.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,solver_position_iteration_count=16,solver_velocity_iteration_count=4))
            self.robot = Articulation(ArticulationCfg(prim_path='/World/Robot',spawn=spawn,
                init_state=ArticulationCfg.InitialStateCfg(joint_pos={**dict(zip(ARM,ik(HOME))),FINGERS[0]:.020}),
                actuators={
                    'arm':ImplicitActuatorCfg(joint_names_expr=list(ARM),stiffness=40.,damping=4.,
                                             effort_limit_sim=1.,velocity_limit_sim=.65),
                    'gripper':ImplicitActuatorCfg(joint_names_expr=[FINGERS[0]],stiffness=200.,damping=5.,
                                                 effort_limit_sim=5.,velocity_limit_sim=.04)}))
            self.arm_names, self.finger_names = ARM, FINGERS
            self.finger_bodies = ('gripper_left_link','gripper_right_link')
            self.tool_name, self.base_name = 'link5', 'link1'
            self.gripper_closed, self.gripper_open = -.011, .02
        else:
            from isaaclab_assets import FRANKA_PANDA_CFG
            from .panda import ARM as PANDA_ARM, FINGERS as PANDA_FINGERS, BASE
            cfg = FRANKA_PANDA_CFG.copy()
            cfg.prim_path = '/World/Robot'
            cfg.spawn.activate_contact_sensors = True
            cfg.spawn.articulation_props.solver_position_iteration_count = 16
            cfg.spawn.articulation_props.solver_velocity_iteration_count = 4
            cfg.init_state.pos = BASE
            cfg.actuators['panda_shoulder'].stiffness = 400.
            cfg.actuators['panda_shoulder'].damping = 40.
            cfg.actuators['panda_forearm'].stiffness = 400.
            cfg.actuators['panda_forearm'].damping = 40.
            self.robot = Articulation(cfg)
            self.arm_names, self.finger_names = PANDA_ARM, PANDA_FINGERS
            self.finger_bodies = ('panda_leftfinger','panda_rightfinger')
            self.tool_name, self.base_name = 'panda_hand', 'panda_link0'
            self.gripper_closed, self.gripper_open = 0., .04
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
        from pxr import Usd
        self.contact_sensors = []
        for name in self.finger_bodies:
            matches = [str(prim.GetPath()) for prim in Usd.PrimRange(self.sim.stage.GetPrimAtPath('/World/Robot'))
                       if prim.GetName() == name]
            if len(matches) != 1:
                raise RuntimeError(f'contact body mapping is ambiguous: {name}')
            self.contact_sensors.append(ContactSensor(ContactSensorCfg(prim_path=matches[0],
                filter_prim_paths_expr=[f'/World/{color}' for color in COLORS],update_period=0.)))
        if cameras:
            self.camera = Camera(CameraCfg(prim_path='/World/Camera',width=640,height=480,
                data_types=['rgb'],update_period=0.,spawn=sim.PinholeCameraCfg(focal_length=24.,
                    horizontal_aperture=20.955,clipping_range=(.01,10.))))
        self.sim.reset()
        self.arm_ids,_ = self.robot.find_joints(list(self.arm_names),preserve_order=True)
        self.left_ids,_ = self.robot.find_joints([self.finger_names[0]])
        self.right_ids,_ = self.robot.find_joints([self.finger_names[1]])
        self.tool_ids,_ = self.robot.find_bodies([self.tool_name])
        self.base_ids,_ = self.robot.find_bodies([self.base_name])
        if len(self.arm_ids)!=len(self.arm_names) or len(self.left_ids)!=1 or len(self.right_ids)!=1 or len(self.tool_ids)!=1 or len(self.base_ids)!=1:
            raise RuntimeError('imported joint/body mappings do not match official URDF')
        if not self.robot.is_fixed_base:
            raise RuntimeError('imported robot is not fixed-base')
        self.all_ids = self.arm_ids+self.left_ids+self.right_ids
        imported = self.robot.data.joint_pos_limits[0,self.arm_ids].cpu().tolist()
        if robot == 'openmanipulator_x' and any(abs(v-e)>1e-4 for pair,expected in zip(imported,LIMITS) for v,e in zip(pair,expected)):
            raise RuntimeError('imported joint limits changed')
        if self.camera:
            self.camera.set_world_poses_from_view(
                torch.tensor([CAMERA_EYE],device=self.device),torch.tensor([CAMERA_LOOK],device=self.device))
        if robot == 'franka_panda':
            from .panda import PandaControl
            self.panda_control = PandaControl(self.robot,self.arm_ids,self.tool_ids[0],self.device)
        self.reference = ik(HOME) if robot == 'openmanipulator_x' else tuple(self.robot.data.joint_pos[0,self.arm_ids].cpu().tolist())
        self.grip_ref = self.gripper_open
        self.base_position = self.robot.data.body_pos_w[0,self.base_ids[0]].clone()
        limits = LIMITS if robot == 'openmanipulator_x' else self.panda_control.limits
        self.safety_config = SafetyConfig(tuple(limits)+((self.gripper_closed,self.gripper_open),)*2,len(self.arm_names))
        self.guard = Guard(self.safety_config)

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
        for sensor in self.contact_sensors:
            sensor.update(DT)
        if self.camera:
            self.camera.update(DT)
        self.time += DT

    def reset(self,seed):
        self.abort = False
        self.safety_events = []
        self.guard = Guard(self.safety_config)
        self.time = 0.
        self.reference = ik(HOME) if self.embodiment == 'openmanipulator_x' else tuple(self.robot.data.default_joint_pos[0,self.arm_ids].cpu().tolist())
        self.grip_ref = self.gripper_open
        q = self.robot.data.default_joint_pos.clone()
        q[:,self.arm_ids] = torch.tensor([self.reference],device=self.device)
        q[:,self.left_ids+self.right_ids] = self.gripper_open
        self.robot.write_joint_state_to_sim(q,torch.zeros_like(q))
        self.robot.reset()
        self.sim.stage.GetPrimAtPath('/World/Light').GetAttribute('inputs:intensity').Set(random.Random(seed).uniform(1800.,2800.))
        for sensor in self.contact_sensors:
            sensor.reset()
        for color,p in layout(seed).items():
            state = self.cubes[color].data.default_root_state.clone()
            state[:,:3] = torch.tensor([p],device=self.device)
            yaw = math.atan2(p[1],p[0]-.012)
            state[:,3:7] = torch.tensor([[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]],device=self.device)
            state[:,7:] = 0
            self.cubes[color].write_root_pose_to_sim(state[:,:7])
            self.cubes[color].write_root_velocity_to_sim(state[:,7:])
            self.cubes[color].reset()
        self._targets()
        for _ in range(120):
            self._tick()
        if self.embodiment == 'franka_panda':
            self.initializing = True
            for _ in range(900):
                self.reference = self.panda_control.solve(HOME,self.reference)
                self._targets()
                for _ in range(round(CONTROL_DT/DT)):
                    self._tick()
                if math.dist(self.observe().ee,HOME) < .003:
                    break
            self.initializing = False
        self.time = 0.
        obs = self.observe()
        expected = fk(obs.joints[:4]) if self.embodiment == 'openmanipulator_x' else HOME
        if math.dist(expected,obs.ee) > .003:
            raise RuntimeError('imported end-effector transform fails URDF FK check')
        quat = self.robot.data.body_quat_w[0,self.tool_ids[0]].cpu().tolist()
        w,x,y,z = quat
        downward = 2*(x*z-w*y) if self.embodiment == 'openmanipulator_x' else 1-2*(x*x+y*y)
        if downward > -.995:
            raise RuntimeError('imported tool does not point downward')
        if obs.unsafe or math.dist(obs.ee,HOME) > .01:
            raise RuntimeError('articulation failed stable home check')
        return obs

    def qualify_articulation(self):
        self.reset(0)
        endpoints = []
        for target in (self.gripper_closed,self.gripper_open):
            for _ in range(240):
                self.grip_ref += max(-.025*DT,min(.025*DT,target-self.grip_ref))
                self._targets()
                self._tick()
                if self.observe().unsafe:
                    raise RuntimeError('unsafe articulation during empty gripper check')
            q = self.observe().joints
            if abs(q[-2]-target) > .003 or abs(q[-2]-q[-1]) > .002:
                raise RuntimeError('physical gripper opening/closing or mimic check failed')
            endpoints.append(list(q[-2:]))
        return {'fixed_base':True,'fk_error_m':math.dist(fk(self.observe().joints[:4]),self.observe().ee) if self.embodiment == 'openmanipulator_x' else None,
                'empty_gripper_endpoints':endpoints,'physical_checks_executed':True}

    def _targets(self):
        self.robot.set_joint_position_target(torch.tensor([self.reference],device=self.device),joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(torch.tensor([[self.grip_ref] if self.embodiment == 'openmanipulator_x' else [self.grip_ref,self.grip_ref]],device=self.device),joint_ids=self.left_ids if self.embodiment == 'openmanipulator_x' else self.left_ids+self.right_ids)

    def observe(self):
        q = tuple(self.robot.data.joint_pos[0,self.all_ids].cpu().tolist())
        dq = tuple(self.robot.data.joint_vel[0,self.all_ids].cpu().tolist())
        origin = self.robot.data.body_pos_w[0,self.tool_ids[0]].cpu().tolist()
        w,x,y,z = self.robot.data.body_quat_w[0,self.tool_ids[0]].cpu().tolist()
        axis = (1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y))
        ee = tuple(a+.126*b for a,b in zip(origin,axis))
        if self.embodiment == 'franka_panda':
            ee = tuple(self.panda_control.pose()[0][0].cpu().tolist())
        dof = len(self.arm_names)
        limits = LIMITS if self.embodiment == 'openmanipulator_x' else self.panda_control.limits
        cubes = {c:tuple(o.data.root_pose_w[0].cpu().tolist()) for c,o in self.cubes.items()}
        speeds = {c:float(torch.linalg.vector_norm(o.data.root_lin_vel_w[0]) +
                          CUBE/2*torch.linalg.vector_norm(o.data.root_ang_vel_w[0])) for c,o in self.cubes.items()}
        unsafe = (self.abort or not all(math.isfinite(v) for v in (*q,*dq,*ee)) or
                  (not self.initializing and not workspace(ee)) or
                  any(not lo-.015 <= v <= hi+.015 for v,(lo,hi) in zip(q,limits)) or
                  max(map(abs,dq[:dof])) > 1.2 or max(map(abs,dq[dof:])) > .15 or
                  abs(q[-2]-q[-1])>.004 or
                  any(not self.gripper_closed-.002 <= v <= self.gripper_open+.002 for v in q[dof:]) or
                  float(torch.linalg.vector_norm(self.robot.data.body_pos_w[0,self.base_ids[0]]-self.base_position)) > .001 or
                  any(not all(math.isfinite(v) for v in p) for p in cubes.values()))
        forces = [torch.linalg.vector_norm(sensor.data.force_matrix_w[0,0],dim=-1).cpu().tolist()
                  for sensor in self.contact_sensors]
        contacts = {color:all(values[i] > .005 for values in forces) for i,color in enumerate(COLORS)}
        return Observation(self.time,ee,q,dq,cubes,speeds,unsafe,contacts,
                           max(0.,min(1.,(q[-2]-self.gripper_closed)/(self.gripper_open-self.gripper_closed))),dof,tuple(self.safety_events) or (('unsafe_state',) if unsafe else ()))

    def step(self,action):
        obs = self.observe()
        try:
            self.guard.state(obs)
            action = self.guard.command(obs,action)
            reason = self.guard.tracking(obs,(*self.reference,self.grip_ref,self.grip_ref)) or self.guard.progress(obs,action)
            if reason:
                raise SafetyViolation(reason)
            previous = (*self.reference,self.grip_ref,self.grip_ref)
            goal = tuple(a+b for a,b in zip(obs.ee,action.delta))
            if not workspace(goal):
                raise ValueError('outside shared workspace')
            if self.embodiment == 'openmanipulator_x':
                desired = ik(goal,self.reference)
                self.reference = bounded_reference(self.reference,desired,CONTROL_DT)
            else:
                self.reference = self.panda_control.solve(goal,self.reference)
            target = self.gripper_closed + (self.gripper_open-self.gripper_closed)*action.gripper
            self.grip_ref += max(-.025*CONTROL_DT,min(.025*CONTROL_DT,target-self.grip_ref))
            self.guard.joint_command(previous,(*self.reference,self.grip_ref,self.grip_ref))
            self._targets()
            for _ in range(round(CONTROL_DT/DT)):
                self._tick()
                if self.observe().unsafe:
                    self.abort = True
                    break
        except ValueError as error:
            self.safety_events.append(str(error) if isinstance(error,SafetyViolation) else 'ik_infeasible')
            self.safe_stop()
        return self.observe()

    def safe_stop(self):
        self.abort = True
        self.reference = tuple(self.robot.data.joint_pos[0,self.arm_ids].cpu().tolist())
        self.grip_ref = float(self.robot.data.joint_pos[0,self.left_ids[0]])
        self._targets()
        self.robot.set_joint_velocity_target(torch.zeros((1,len(self.all_ids)),device=self.device),joint_ids=self.all_ids)
        self.robot.write_data_to_sim()

    def frame(self):
        if self.camera is None:
            return None
        return self.camera.data.output['rgb'][0,:,:,:3].cpu().numpy()
