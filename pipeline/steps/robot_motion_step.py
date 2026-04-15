from typing import Dict
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R


from cable_routing.debug_gui.pipeline.arm_motion_utils import (
    MOTION_FRAME_ID,
    is_dual_arm_grasp,
)
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class RobotMotionStep(BaseStep):
    name = "robot_motion"
    description = "Send pre-grasp pose to YuMi via ROS."

    def __init__(self):
        super().__init__()

        # init ROS node once
        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_robot_motion", anonymous=True)

        self.pub_left = rospy.Publisher(
            "/yumi/robl/moveit_target_pose",
            PoseStamped,
            queue_size=1,
        )
        self.pub_right = rospy.Publisher(
            "/yumi/robr/moveit_target_pose",
            PoseStamped,
            queue_size=1,
        )

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not hasattr(state, "pregrasp_poses"):
            raise RuntimeError("No pregrasp poses available.")

        poses = state.pregrasp_poses

        if is_dual_arm_grasp(state.config):
            if len(poses) != 2:
                raise RuntimeError("Dual-arm motion requires exactly 2 pregrasp poses.")

            left_pose = None
            right_pose = None

            for pose in poses:
                arm = pose.get("arm", None)
                if arm == "left":
                    left_pose = pose
                elif arm == "right":
                    right_pose = pose

            if left_pose is None or right_pose is None:
                raise RuntimeError("Need exactly one left pose and one right pose.")

            left_pos = left_pose["position"]
            right_pos = right_pose["position"]

            min_dist_xyz = 0.1  # 10 cm
            dist_xyz = float(((left_pos - right_pos) ** 2).sum() ** 0.5)

            if dist_xyz < min_dist_xyz:
                raise RuntimeError(
                    f"Pregrasp poses too close: distance={dist_xyz:.3f} m < {min_dist_xyz:.3f} m"
                )
        else:
            if len(poses) != 1:
                raise RuntimeError("Single-arm motion requires exactly 1 pregrasp pose.")
            only = poses[0]
            arm = only.get("arm", "right")
            left_pose = only if arm == "left" else None
            right_pose = only if arm == "right" else None
            dist_xyz = 0.0

        def build_msg(pos, rot):
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

        if is_dual_arm_grasp(state.config):
            left_msg, left_quat = build_msg(left_pose["position"], left_pose["rotation"])
            right_msg, right_quat = build_msg(
                right_pose["position"], right_pose["rotation"]
            )

            stagger_delay_s = 1.00
            self.pub_left.publish(left_msg)
            rospy.sleep(stagger_delay_s)
            self.pub_right.publish(right_msg)

            state.robot_target_sent = True

            return {
                "target_sent": True,
                "arms": ["left", "right"],
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

        only_pose = left_pose if left_pose is not None else right_pose
        arm = only_pose.get("arm", "right")
        msg, quat = build_msg(only_pose["position"], only_pose["rotation"])
        if arm == "left":
            self.pub_left.publish(msg)
        else:
            self.pub_right.publish(msg)

        state.robot_target_sent = True

        return {
            "target_sent": True,
            "arms": [arm],
            "distance_xyz": dist_xyz,
            "left_position": (
                [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ]
                if arm == "left"
                else None
            ),
            "right_position": (
                [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ]
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
