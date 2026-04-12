import numpy as np


class PreGraspPoseService:
    def compute_pregrasp_poses(self, grasp_poses, offset=0.08):
        pregrasp_poses = []

        for pose in grasp_poses:
            pos = np.asarray(pose["position"]).astype(float)
            R = np.asarray(pose["rotation"]).astype(float)

            print("grasp pose position:", pose["position"])

            # approach_axis = np.asarray(pose["approach_axis"]).astype(float)
            # approach_axis = approach_axis / (np.linalg.norm(approach_axis) + 1e-8)
            # pre_pos = pos - approach_axis * offset

            # for now just move straight up, assuming grasp approach is mostly vertical
            pre_pos = pos.copy()
            pre_pos[2] += offset

            # for the right arm move slightly more to the right, for the left arm slightly more to the left,
            # to reduce collision risk during staggered descend
            if pose.get("arm") == "right":
                pre_pos[1] -= 0.04  # 3 cm to the right
            elif pose.get("arm") == "left":
                pre_pos[1] += 0.04  # 3 cm to the left

            pre_pose = {
                "position": pre_pos,
                "rotation": R.copy(),
                "arm": pose.get("arm", "right"),
                "path_index": int(pose["path_index"]),
            }

            pregrasp_poses.append(pre_pose)

        return pregrasp_poses
