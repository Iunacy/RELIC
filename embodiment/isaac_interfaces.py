"""Verify pinned source-level API presence without starting Isaac Sim."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from .preflight import LAB_COMMIT

REQUIRED = {
    'source/isaaclab/isaaclab/assets/articulation/articulation.py': ('find_joints', 'find_bodies', 'set_joint_position_target', 'write_joint_state_to_sim'),
    'source/isaaclab/isaaclab/assets/articulation/articulation_data.py': ('body_pose_w', 'joint_pos_limits'),
    'source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py': ('net_forces_w', 'force_matrix_w'),
    'source/isaaclab/isaaclab/sensors/camera/camera.py': ('set_world_poses_from_view',),
    'source/isaaclab/isaaclab/sim/converters/urdf_converter.py': ('_get_urdf_import_config',),
}


def verify(source):
    source = Path(source).resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', 'source/isaaclab', 'source/isaaclab_assets'], text=True)
    if commit != LAB_COMMIT or changes: raise ValueError('Isaac Lab checkout differs from clean pin')
    checked = {}
    for name, required in REQUIRED.items():
        path = source/name
        tree = ast.parse(path.read_text())
        names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        names.update(node.target.id for node in ast.walk(tree) if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name))
        missing = set(required)-names
        if missing: raise ValueError('pinned upstream interface missing: '+str(sorted(missing)))
        checked[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'interfaces': required}
    return {'schema': 'viper.isaac_source_interfaces.v1', 'commit': commit, 'checked': checked,
            'source_interfaces_present': True, 'simulation_executed': False, 'runtime_compatibility_verified': False,
            'remaining_gate': 'Kit initialization, URDF/USD import, contact reporters and PhysX tensor methods on qualified GPU'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.source), indent=2))
