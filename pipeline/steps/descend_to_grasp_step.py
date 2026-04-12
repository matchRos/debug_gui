from typing import Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState

from cable_routing.debug_gui.pipeline.arm_motion_utils import (
    compute_xy_shift,
    msg_pose_to_dict,
    publish_staggered_dual_arm_targets,
    quat_to_list,
    split_dual_arm_poses,
    validate_min_distance,
)


class DescendToGraspStep(BaseStep):
    name = "descend_to_grasp"
    description = "Move both arms from pre-grasp to grasp pose simultaneously."

    def __init__(self):
        super().__init__()

        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_descend_to_grasp", anonymous=True)

        self.pub_left = rospy.Publisher(
            "/yumi/robl/slowly_approach_pose",
            PoseStamped,
            queue_size=1,
        )
        self.pub_right = rospy.Publisher(
            "/yumi/robr/slowly_approach_pose",
            PoseStamped,
            queue_size=1,
        )

    def _build_msg(self, pos, rot):
        quat = R.from_matrix(rot).as_quat()  # x, y, z, w

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])

        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        return msg, quat

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not hasattr(state, "grasp_poses"):
            raise RuntimeError("No grasp poses available.")

        if not hasattr(state, "pregrasp_poses"):
            raise RuntimeError("No pregrasp poses available.")

        grasp_poses = state.grasp_poses
        pregrasp_poses = state.pregrasp_poses

        if len(grasp_poses) != 2 or len(pregrasp_poses) != 2:
            raise RuntimeError(
                "Dual-arm descend requires exactly 2 grasp and 2 pregrasp poses."
            )

        left_grasp = None
        right_grasp = None
        left_pre = None
        right_pre = None

        for pose in grasp_poses:
            if pose.get("arm") == "left":
                left_grasp = pose
            elif pose.get("arm") == "right":
                right_grasp = pose

        for pose in pregrasp_poses:
            if pose.get("arm") == "left":
                left_pre = pose
            elif pose.get("arm") == "right":
                right_pre = pose

        if left_grasp is None or right_grasp is None:
            raise RuntimeError("Need exactly one left and one right grasp pose.")

        if left_pre is None or right_pre is None:
            raise RuntimeError("Need exactly one left and one right pregrasp pose.")

        # Safety check: both arms should mainly descend vertically, not jump laterally
        # left_xy_shift = float(
        #     np.linalg.norm(
        #         np.asarray(left_grasp["position"][:2])
        #         - np.asarray(left_pre["position"][:2])
        #     )
        # )
        # right_xy_shift = float(
        #     np.linalg.norm(
        #         np.asarray(right_grasp["position"][:2])
        #         - np.asarray(right_pre["position"][:2])
        #     )
        # )

        # max_xy_shift = 0.05  #  cm tolerance
        # if left_xy_shift > max_xy_shift:
        #     raise RuntimeError(
        #         f"Left descend not vertical enough: xy shift = {left_xy_shift:.3f} m"
        #     )
        # if right_xy_shift > max_xy_shift:
        #     raise RuntimeError(
        #         f"Right descend not vertical enough: xy shift = {right_xy_shift:.3f} m"
        #     )

        left_pos = np.asarray(left_grasp["position"]).astype(float).copy()
        right_pos = np.asarray(right_grasp["position"]).astype(float).copy()

        # Collision safety check
        min_dist_xyz = 0.08  # 8 cm
        dist_xyz = float(np.linalg.norm(left_pos - right_pos))
        if dist_xyz < min_dist_xyz:
            raise RuntimeError(
                f"Grasp poses too close: distance={dist_xyz:.3f} m < {min_dist_xyz:.3f} m"
            )

        # Determine which arm is farther away from the cable start
        # Higher path progress = farther from cable start
        if "path_s" in left_grasp and "path_s" in right_grasp:
            left_progress = float(left_grasp["path_s"])
            right_progress = float(right_grasp["path_s"])
        elif "path_index" in left_grasp and "path_index" in right_grasp:
            left_progress = float(left_grasp["path_index"])
            right_progress = float(right_grasp["path_index"])
        else:
            raise RuntimeError(
                "Each grasp pose must contain either 'path_s' or 'path_index'."
            )

        left_msg, left_quat = self._build_msg(left_pos, left_grasp["rotation"])
        right_msg, right_quat = self._build_msg(right_pos, right_grasp["rotation"])

        stagger_delay_s = 0.50

        if left_progress > right_progress:
            first_arm = "left"
            second_arm = "right"

            self.pub_left.publish(left_msg)
            rospy.sleep(stagger_delay_s)
            self.pub_right.publish(right_msg)
        else:
            first_arm = "right"
            second_arm = "left"

            self.pub_right.publish(right_msg)
            rospy.sleep(stagger_delay_s)
            self.pub_left.publish(left_msg)

        rospy.sleep(2.0)

        state.descend_target_sent = True

        return {
            "descend_sent": True,
            "arms": ["left", "right"],
            "distance_xyz": dist_xyz,
            "left_xy_shift_from_pregrasp": left_xy_shift,
            "right_xy_shift_from_pregrasp": right_xy_shift,
            "left_progress": left_progress,
            "right_progress": right_progress,
            "first_arm_sent": first_arm,
            "second_arm_sent": second_arm,
            "stagger_delay_s": stagger_delay_s,
            "left_position": [
                left_msg.pose.position.x,
                left_msg.pose.position.y,
                left_msg.pose.position.z,
            ],
            "right_position": [
                right_msg.pose.position.x,
                right_msg.pose.position.y,
                right_msg.pose.position.z,
            ],
            "left_quaternion": [
                float(left_quat[0]),
                float(left_quat[1]),
                float(left_quat[2]),
                float(left_quat[3]),
            ],
            "right_quaternion": [
                float(right_quat[0]),
                float(right_quat[1]),
                float(right_quat[2]),
                float(right_quat[3]),
            ],
        }
