import numpy as np


class GraspPoseService:
    def compute_pose(self, position, tangent):
        x_axis = tangent / (np.linalg.norm(tangent) + 1e-8)

        z_axis = np.array([0.0, 0.0, 1.0])

        # if close to parallel → chose different up vector to avoid singularity
        if abs(np.dot(x_axis, z_axis)) > 0.95:
            z_axis = np.array([0.0, 1.0, 0.0])

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis) + 1e-8

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-8

        # approach direction before the in-plane tool rotation
        approach_axis = -z_axis.copy()

        # flip tool z-axis while keeping a right-handed frame
        z_axis = -z_axis
        y_axis = -y_axis

        # rotate tool frame by +90 deg around its local z-axis
        x_axis_rot = y_axis
        y_axis_rot = -x_axis

        R = np.stack([x_axis_rot, y_axis_rot, z_axis], axis=1)

        return {
            "position": position,
            "rotation": R,
            "approach_axis": approach_axis,
        }

    def compute_grasp_poses(self, grasps):
        poses = []

        for g in grasps:
            pose = self.compute_pose(g["position"], g["tangent"])
            poses.append(pose)

        return poses
