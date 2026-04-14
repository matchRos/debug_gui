import numpy as np

from cable_routing.debug_gui.backend.planes import ensure_min_plane_height


class PreGraspPoseService:
    def compute_pregrasp_poses(
        self,
        grasp_poses,
        plane,
        pregrasp_offset_from_grasp_m: float = 0.08,
        routing_height_above_plane_m: float = 0.025,
    ):
        pregrasp_poses = []
        normal_axis = np.asarray(plane.normal, dtype=float).reshape(3)
        lateral_axis = np.asarray(plane.v_axis, dtype=float).reshape(3)
        forward_axis = np.asarray(plane.u_axis, dtype=float).reshape(3)

        for pose in grasp_poses:
            pos = np.asarray(pose["position"]).astype(float)
            R = np.asarray(pose["rotation"]).astype(float)

            # Move out of the routing plane along its normal.
            pre_pos = pos.copy() + normal_axis * float(pregrasp_offset_from_grasp_m)

            # Keep the existing anti-collision offsets, but in plane coordinates.
            if pose.get("arm") == "right":
                pre_pos = pre_pos - lateral_axis * 0.04
            elif pose.get("arm") == "left":
                pre_pos = pre_pos + lateral_axis * 0.04
                pre_pos = pre_pos - forward_axis * 0.02

            pre_pos = ensure_min_plane_height(
                pre_pos,
                plane,
                routing_height_above_plane_m,
            )

            pre_pose = {
                "position": pre_pos,
                "rotation": R.copy(),
                "arm": pose.get("arm", "right"),
                "path_index": int(pose["path_index"]),
            }

            pregrasp_poses.append(pre_pose)

        return pregrasp_poses
