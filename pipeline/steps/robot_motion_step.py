from typing import Dict
import rospy
from geometry_msgs.msg import PointStamped

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
            "/yumi/robl/moveit_target_position_facing_down",
            PointStamped,
            queue_size=1,
        )
        self.pub_right = rospy.Publisher(
            "/yumi/robr/moveit_target_position_facing_down",
            PointStamped,
            queue_size=1,
        )

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not hasattr(state, "pregrasp_poses"):
            raise RuntimeError("No pregrasp poses available.")

        pose = state.pregrasp_poses[0]

        pos = pose["position"]
        arm = pose.get("arm", "right")

        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "yumi_base_link"
        msg.point.x = float(pos[0])
        msg.point.y = float(pos[1])
        msg.point.z = float(pos[2])

        if arm == "left":
            self.pub_left.publish(msg)
        else:
            self.pub_right.publish(msg)

        state.robot_target_sent = True

        return {
            "target_sent": True,
            "arm": arm,
            "position": [msg.point.x, msg.point.y, msg.point.z],
        }
