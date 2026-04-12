from typing import Dict

import rospy
from std_srvs.srv import Trigger

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class CloseGripperStep(BaseStep):
    name = "close_gripper"
    description = "Close both grippers for dual-arm grasping."

    def __init__(self):
        super().__init__()

        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_close_gripper", anonymous=True)

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not hasattr(state, "grasp_poses"):
            raise RuntimeError("No grasp poses available.")

        rospy.wait_for_service("/yumi/gripper_l/close", timeout=5.0)
        rospy.wait_for_service("/yumi/gripper_r/close", timeout=5.0)

        close_left_srv = rospy.ServiceProxy("/yumi/gripper_l/close", Trigger)
        close_right_srv = rospy.ServiceProxy("/yumi/gripper_r/close", Trigger)

        resp_left = close_left_srv()
        resp_right = close_right_srv()

        if not resp_left.success:
            raise RuntimeError(f"/yumi/gripper_l/close failed: {resp_left.message}")

        if not resp_right.success:
            raise RuntimeError(f"/yumi/gripper_r/close failed: {resp_right.message}")

        rospy.sleep(1.0)

        state.grippers_closed = True

        return {
            "gripper_closed": True,
            "arms": ["left", "right"],
            "left_service": "/yumi/gripper_l/close",
            "right_service": "/yumi/gripper_r/close",
            "left_message": resp_left.message,
            "right_message": resp_right.message,
        }
