from typing import Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

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

        poses = state.grasp_poses

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

        lift_distance = 0.05  # 5 cm

        left_pos = np.asarray(left_pose["position"]).astype(float).copy()
        right_pos = np.asarray(right_pose["position"]).astype(float).copy()

        left_pos[2] += lift_distance
        right_pos[2] += lift_distance

        # basic collision sanity check
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

        rospy.sleep(2.0)

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
