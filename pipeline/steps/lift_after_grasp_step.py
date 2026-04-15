from typing import Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from cable_routing.debug_gui.backend.planes import get_routing_plane
from cable_routing.debug_gui.pipeline.arm_motion_utils import (
    MOTION_FRAME_ID,
    enforce_pose_min_height,
    is_dual_arm_grasp,
    wait_until_robot_settled,
)
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class LiftAfterGraspStep(BaseStep):
    name = "lift_after_grasp"
    description = "Lift both arms 5 cm upward after grasping."

    def __init__(self):
        super().__init__()

        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_lift_after_grasp", anonymous=True)

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
        msg.header.frame_id = MOTION_FRAME_ID

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

        poses = state.grasp_poses
        plane = get_routing_plane(state.config)
        n = np.asarray(plane.normal, dtype=float).reshape(3)
        n /= np.linalg.norm(n) + 1e-8

        lift_distance = 0.02  # 2 cm along plane normal (toward robot for YZ board)
        detangle_floor = float(
            state.config.routing_height_above_plane_m
            + state.config.detangle_offset_from_routing_m
        )
        lift_vec = float(lift_distance) * n

        if is_dual_arm_grasp(state.config):
            if len(poses) != 2:
                raise RuntimeError("Lift requires exactly 2 grasp poses.")

            left_pose = None
            right_pose = None

            for pose in poses:
                if pose.get("arm") == "left":
                    left_pose = pose
                elif pose.get("arm") == "right":
                    right_pose = pose

            if left_pose is None or right_pose is None:
                raise RuntimeError("Need exactly one left and one right grasp pose.")

            left_pose = enforce_pose_min_height(left_pose, state, detangle_floor)
            right_pose = enforce_pose_min_height(right_pose, state, detangle_floor)

            left_pos = np.asarray(left_pose["position"]).astype(float).copy() + lift_vec
            right_pos = (
                np.asarray(right_pose["position"]).astype(float).copy() + lift_vec
            )

            min_dist_xyz = 0.08
            dist_xyz = float(np.linalg.norm(left_pos - right_pos))
            if dist_xyz < min_dist_xyz:
                raise RuntimeError(
                    f"Lift targets too close: distance={dist_xyz:.3f} m < {min_dist_xyz:.3f} m"
                )

            left_msg, left_quat = self._build_msg(left_pos, left_pose["rotation"])
            right_msg, right_quat = self._build_msg(right_pos, right_pose["rotation"])

            now = rospy.Time.now()
            left_msg.header.stamp = now
            right_msg.header.stamp = now

            self.pub_left.publish(left_msg)
            self.pub_right.publish(right_msg)

            wait_until_robot_settled()

            state.lift_after_grasp_done = True

            return {
                "lift_sent": True,
                "arms": ["left", "right"],
                "lift_distance_m": lift_distance,
                "distance_xyz": dist_xyz,
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

        if len(poses) != 1:
            raise RuntimeError("Single-arm lift requires exactly 1 grasp pose.")
        only = enforce_pose_min_height(poses[0], state, detangle_floor)
        arm = only.get("arm", "right")
        pos = np.asarray(only["position"], dtype=float).copy() + lift_vec
        msg, quat = self._build_msg(pos, only["rotation"])
        msg.header.stamp = rospy.Time.now()
        if arm == "left":
            self.pub_left.publish(msg)
        else:
            self.pub_right.publish(msg)
        wait_until_robot_settled()
        state.lift_after_grasp_done = True
        return {
            "lift_sent": True,
            "arms": [arm],
            "lift_distance_m": lift_distance,
            "distance_xyz": 0.0,
            "left_position": (
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                if arm == "left"
                else None
            ),
            "right_position": (
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                if arm == "right"
                else None
            ),
            "left_quaternion": (
                [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                if arm == "left"
                else None
            ),
            "right_quaternion": (
                [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                if arm == "right"
                else None
            ),
        }
