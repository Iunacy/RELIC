import math
from .benchmark import finite

ARM = tuple(f'panda_joint{i}' for i in range(1, 8))
FINGERS = ('panda_finger_joint1', 'panda_finger_joint2')
BASE = (-.28, 0., 0.)
TOOL_OFFSET = .1034
GRASP_Z = .0135


def downward_quaternion(position: tuple) -> tuple:
    finite(position, 3)
    yaw = math.atan2(position[1], position[0] - .012)
    return 0., math.cos(yaw / 2), math.sin(yaw / 2), 0.


def bounded_reference(current: tuple, desired: tuple, limits: tuple, dt: float) -> tuple:
    finite(current, 7)
    finite(desired, 7)
    finite((dt,), 1)
    if dt <= 0 or len(limits) != 7:
        raise ValueError('invalid Panda control configuration')
    result = []
    for value, goal, (low, high) in zip(current, desired, limits, strict=True):
        finite((low, high), 2)
        if low >= high or not low <= value <= high or not low <= goal <= high:
            raise ValueError('Panda joint limit')
        result.append(value + max(-.5 * dt, min(.5 * dt, goal - value)))
    return tuple(result)


class PandaControl:
    def __init__(self, robot, arm_ids, tool_id, device):
        import torch
        self.robot = robot
        self.arm_ids = arm_ids
        self.tool_id = tool_id
        self.device = device
        self.offset = torch.tensor([[0., 0., TOOL_OFFSET]], device=device)
        self.limits = tuple(map(tuple, robot.data.joint_pos_limits[0, arm_ids].cpu().tolist()))

    def pose(self):
        from isaaclab.utils.math import quat_apply
        pose = self.robot.data.body_pose_w[:, self.tool_id]
        position = pose[:, :3] + quat_apply(pose[:, 3:7], self.offset)
        return position, pose[:, 3:7]

    def solve(self, goal, reference):
        import torch
        from isaaclab.utils.math import compute_pose_error, quat_apply, skew_symmetric_matrix
        position, quaternion = self.pose()
        target = torch.tensor([goal], device=self.device)
        rotation = torch.tensor([downward_quaternion(goal)], device=self.device)
        translation_error, rotation_error = compute_pose_error(position, quaternion, target, rotation, rot_error_type='axis_angle')
        error = torch.cat((translation_error, rotation_error), dim=-1)
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.tool_id - 1, :, self.arm_ids].clone()
        offset_w = quat_apply(quaternion, self.offset)
        jacobian[:, :3] -= torch.bmm(skew_symmetric_matrix(offset_w), jacobian[:, 3:])
        transpose = jacobian.transpose(1, 2)
        regularized = torch.bmm(jacobian, transpose) + .0025 * torch.eye(6, device=self.device).unsqueeze(0)
        update = torch.bmm(transpose, torch.linalg.solve(regularized, error.unsqueeze(-1))).squeeze(-1)
        current = self.robot.data.joint_pos[:, self.arm_ids]
        desired = (current + update)[0].cpu().tolist()
        if not all(math.isfinite(v) for v in desired):
            raise ValueError('nonfinite Panda IK')
        return bounded_reference(reference, tuple(desired), self.limits, 1 / 30)
