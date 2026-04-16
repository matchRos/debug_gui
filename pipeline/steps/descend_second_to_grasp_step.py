from typing import Dict

import rospy
from geometry_msgs.msg import PoseStamped

from cable_routing.debug_gui.pipeline.arm_motion_utils import (
    enforce_pose_min_height,
    is_dual_arm_grasp,
    pose_to_msg,
    wait_until_robot_settled,
)
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

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not is_dual_arm_grasp(state.config):
            return {
                "descend_sent": False,
                "skipped": True,
                "reason": "dual_arm_grasp disabled",
            }
        if not hasattr(state, "descend_second_arm"):
            raise RuntimeError("No second descend arm stored in state.")
        if not hasattr(state, "second_grasp_pose"):
            raise RuntimeError("No second grasp pose stored in state.")

        second_arm = state.descend_second_arm
        second_pose = state.second_grasp_pose

        grasp_floor = float(state.config.grasp_height_above_plane_m)
        second_pose = enforce_pose_min_height(second_pose, state, grasp_floor)

        msg, quat = pose_to_msg(
            second_pose["position"], second_pose["rotation"], config=state.config
        )
        msg.header.stamp = rospy.Time.now()

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
