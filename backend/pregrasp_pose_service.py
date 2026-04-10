import numpy as np


class PreGraspPoseService:
    def compute_pregrasp_poses(self, grasp_poses, offset=0.08):
        pregrasp_poses = []

        for pose in grasp_poses:
            pos = np.asarray(pose["position"]).astype(float)
            R = np.asarray(pose["rotation"]).astype(float)

            # local z-axis of end effector
            z_axis = R[:, 2]
            pre_pos = pos + z_axis * offset

            pre_pose = {
                "position": pre_pos,
                "rotation": R.copy(),
                "arm": pose.get("arm", "right"),
            }

            pregrasp_poses.append(pre_pose)

        return pregrasp_poses
