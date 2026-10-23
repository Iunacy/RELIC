PHYSX_PARAMETERS = {
    'gpu_max_rigid_contact_count': 2**17,
    'gpu_max_rigid_patch_count': 2**15,
    'gpu_found_lost_pairs_capacity': 2**16,
    'gpu_found_lost_aggregate_pairs_capacity': 2**16,
    'gpu_total_aggregate_pairs_capacity': 2**16,
    'gpu_collision_stack_size': 2**24,
    'gpu_heap_capacity': 2**25,
}


def simulation_config(device: str, dt: float):
    import isaaclab.sim as sim
    return sim.SimulationCfg(dt=dt,device=device,physx=sim.PhysxCfg(**PHYSX_PARAMETERS))
