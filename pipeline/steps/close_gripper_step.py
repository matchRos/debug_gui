from typing import Dict

import rospy
from std_srvs.srv import Trigger

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class CloseGripperStep(BaseStep):
    name = "close_gripper"
    description = "Close the selected gripper."

    def __init__(self):
        super().__init__()

        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_close_gripper", anonymous=True)

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not hasattr(state, "grasp_poses"):
            raise RuntimeError("No grasp poses available.")

        arm = state.grasp_poses[0].get("arm", "right")

        if arm == "left":
            service_name = "/yumi/gripper_l/close"
        else:
            service_name = "/yumi/gripper_r/close"

        rospy.wait_for_service(service_name, timeout=5.0)
        close_srv = rospy.ServiceProxy(service_name, Trigger)
        resp = close_srv()

        if not resp.success:
            raise RuntimeError(f"{service_name} failed: {resp.message}")

        rospy.sleep(1.0)

        return {
            "gripper_closed": True,
            "arm": arm,
            "service": service_name,
            "message": resp.message,
        }
