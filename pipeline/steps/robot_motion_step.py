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

        pose = state.pregrasp_poses[0]

        pos = pose["position"]
        rot = pose["rotation"]
        arm = pose.get("arm", "right")

        quat = R.from_matrix(rot).as_quat()  # x, y, z, w

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "yumi_base_link"

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])

        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        if arm == "left":
            self.pub_left.publish(msg)
        else:
            self.pub_right.publish(msg)

        state.robot_target_sent = True

        return {
            "target_sent": True,
            "arm": arm,
            "position": [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            "quaternion": [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
        }
