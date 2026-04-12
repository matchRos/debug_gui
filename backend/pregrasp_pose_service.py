import numpy as np


class PreGraspPoseService:
    def compute_pregrasp_poses(self, grasp_poses, offset=0.08):
        pregrasp_poses = []

        for pose in grasp_poses:
            pos = np.asarray(pose["position"]).astype(float)
            R = np.asarray(pose["rotation"]).astype(float)

            print("grasp pose position:", pose["position"])

            # local z-axis of end effector

            approach_axis = np.asarray(pose["approach_axis"]).astype(float)
            approach_axis = approach_axis / (np.linalg.norm(approach_axis) + 1e-8)
            pre_pos = pos - approach_axis * offset

            pre_pose = {
                "position": pre_pos,
                "rotation": R.copy(),
                "arm": pose.get("arm", "right"),
            }

            pregrasp_poses.append(pre_pose)

        return pregrasp_poses
