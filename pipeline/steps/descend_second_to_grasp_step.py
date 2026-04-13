from typing import Dict

import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from cable_routing.debug_gui.pipeline.arm_motion_utils import wait_until_robot_settled
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class DescendSecondToGraspStep(BaseStep):
    name = "descend_second_to_grasp"
    description = "Descend the second arm after the first arm has already grasped."

    def __init__(self):
        super().__init__()

        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_descend_second_to_grasp", anonymous=True)

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
        quat = R.from_matrix(rot).as_quat()

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
        if not hasattr(state, "descend_second_arm"):
            raise RuntimeError("No second descend arm stored in state.")
        if not hasattr(state, "second_grasp_pose"):
            raise RuntimeError("No second grasp pose stored in state.")

        second_arm = state.descend_second_arm
        second_pose = state.second_grasp_pose

        msg, quat = self._build_msg(second_pose["position"], second_pose["rotation"])

        if second_arm == "left":
            self.pub_left.publish(msg)
        elif second_arm == "right":
            self.pub_right.publish(msg)
        else:
            raise RuntimeError(f"Invalid second arm: {second_arm}")

        wait_until_robot_settled()

        state.second_descend_done = True

        return {
            "descend_sent": True,
            "arm": second_arm,
            "position": [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            "quaternion": [
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
                float(quat[3]),
            ],
        }
