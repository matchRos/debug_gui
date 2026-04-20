from typing import Dict

import rospy
from std_srvs.srv import Trigger

from cable_routing.debug_gui.backend.handover_pose_service import resolve_handover_arm
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class CloseSecondGripperStep(BaseStep):
    name = "close_second_gripper"
    description = "Close the gripper of the second arm."

    def __init__(self):
        super().__init__()

        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_close_second_gripper", anonymous=True)

    def run(self, state: PipelineState) -> Dict[str, object]:
        dual = bool(getattr(state.config, "dual_arm_grasp", True))
        side_done = bool(getattr(state, "second_arm_side_approach_done", False))
        # Dual sequential grasp sets descend_second_arm; side-approach path does not.
        if not dual and not side_done:
            return {
                "gripper_closed": False,
                "skipped": True,
                "reason": "dual_arm_grasp disabled (and no second_arm_side_approach_done)",
            }

        second_arm = getattr(state, "descend_second_arm", None)
        if second_arm is None and side_done:
            carrier = resolve_handover_arm(
                state, getattr(state.config, "handover_arm", None)
            )
            second_arm = "right" if carrier == "left" else "left"
        if second_arm is None:
            return {
                "gripper_closed": False,
                "skipped": True,
                "reason": "no second arm (descend_second_arm unset and no side approach)",
            }

        if second_arm == "left":
            service_name = "/yumi/gripper_l/close"
        elif second_arm == "right":
            service_name = "/yumi/gripper_r/close"
        else:
            raise RuntimeError(f"Invalid second arm: {second_arm}")

        rospy.wait_for_service(service_name, timeout=5.0)
        close_srv = rospy.ServiceProxy(service_name, Trigger)
        resp = close_srv()

        if not resp.success:
            raise RuntimeError(f"{service_name} failed: {resp.message}")

        rospy.sleep(1.0)

        state.second_gripper_closed = True

        return {
            "gripper_closed": True,
            "arm": second_arm,
            "service": service_name,
            "message": resp.message,
        }
