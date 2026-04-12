from typing import Dict
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R


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
            "/yumi/robl/cartesian_pose_command",
            PoseStamped,
            queue_size=1,
        )
        self.pub_right = rospy.Publisher(
            "/yumi/robr/cartesian_pose_command",
            PoseStamped,
            queue_size=1,
        )

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not hasattr(state, "pregrasp_poses"):
            raise RuntimeError("No pregrasp poses available.")

        poses = state.pregrasp_poses

        if len(poses) != 2:
            raise RuntimeError("Dual-arm motion requires exactly 2 pregrasp poses.")

        # sort by assigned arm
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

        # simple collision safety check
        min_dist_xyz = 0.1  # 10 cm
        dist_xyz = float(((left_pos - right_pos) ** 2).sum() ** 0.5)

        if dist_xyz < min_dist_xyz:
            raise RuntimeError(
                f"Pregrasp poses too close: distance={dist_xyz:.3f} m < {min_dist_xyz:.3f} m"
            )

        def build_msg(pos, rot):
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

        left_msg, left_quat = build_msg(left_pose["position"], left_pose["rotation"])
        right_msg, right_quat = build_msg(
            right_pose["position"], right_pose["rotation"]
        )

        stagger_delay_s = 2.50
        # publish both targets
        self.pub_left.publish(left_msg)
        rospy.sleep(
            stagger_delay_s
        )  # staggered start to reduce collision risk during approach
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
