import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    launcher = AppLauncher(args)
    report = {'simulation_executed':False,'qualification':False}
    try:
        import carb
        import torch
        import isaaclab.sim as sim
        from isaaclab.assets import RigidObject,RigidObjectCfg
        from isaaclab.sensors import ContactSensor,ContactSensorCfg
        from .runtime_config import simulation_config
        context = sim.SimulationContext(simulation_config(args.device,1/120))
        ground = sim.GroundPlaneCfg()
        ground.func('/World/Ground',ground)
        cube = RigidObject(RigidObjectCfg(prim_path='/World/Cube',
            spawn=sim.CuboidCfg(size=(.025,)*3,activate_contact_sensors=True,
                rigid_props=sim.RigidBodyPropertiesCfg(),collision_props=sim.CollisionPropertiesCfg(),mass_props=sim.MassPropertiesCfg(mass=.012)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,.1))))
        sensor = ContactSensor(ContactSensorCfg(prim_path='/World/Cube',update_period=0.))
        before = carb.settings.get_settings().get('/physics/disableContactProcessing')
        context.reset()
        after = carb.settings.get_settings().get('/physics/disableContactProcessing')
        samples = []
        for _ in range(240):
            cube.write_data_to_sim()
            context.step()
            cube.update(1/120)
            sensor.update(1/120,force_recompute=True)
            samples.append(float(torch.linalg.vector_norm(sensor.data.net_forces_w)))
        report.update(simulation_executed=True,device=args.device,contact_processing_disabled_before=before,
                      contact_processing_disabled_after=after,maximum_force_n=max(samples),
                      final_force_n=samples[-1],expected_rest_force_n=.012*9.81,
                      final_cube_pose=cube.data.root_pose_w.cpu().tolist(),
                      passed=max(samples)>.05 and abs(samples[-1]-.012*9.81)<.03)
    except Exception as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        launcher.app.close(wait_for_replicator=False)
    print(json.dumps(report,indent=2))
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
