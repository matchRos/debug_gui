from typing import Any, Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

from cable_routing.debug_gui.backend.dual_arm_presentation_geometry import (
    rotation_second_arm_side_grasp_world,
)
from cable_routing.debug_gui.backend.handover_pose_service import resolve_handover_arm
from cable_routing.debug_gui.pipeline.arm_motion_utils import pose_to_msg, wait_until_robot_settled
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class SecondArmSideApproachStep(BaseStep):
    """
    Move the **other** arm to the same TCP position as the carrier, plus a world-Z
    offset (default -0.1 m = 10 cm down). Orientation: side grasp — tool Z along +Y
    if the second arm is **right**, along -Y if the second arm is **left``.
    """

    name = "second_arm_side_approach"
    description = "Second arm to carrier XY + delta Z; side-grasp orientation (±world Y)."

    def __init__(self) -> None:
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_second_arm_side_approach", anonymous=True)
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

    def _publish(self, arm: str, msg: PoseStamped) -> None:
        msg.header.stamp = rospy.Time.now()
        if arm == "left":
            self.pub_left.publish(msg)
        elif arm == "right":
            self.pub_right.publish(msg)
        else:
            raise RuntimeError(f"Invalid arm: {arm}")

    def run(self, state: PipelineState) -> Dict[str, Any]:
        if state.env is None:
            raise RuntimeError("Environment not initialized.")

        carrier = resolve_handover_arm(state, getattr(state.config, "handover_arm", None))
        second = "right" if carrier == "left" else "left"

        pos_c = getattr(state, "handover_carrier_tcp_world", None)
        if pos_c is None:
            pos_c = np.asarray(
                getattr(state.config, "handover_goal_world_m", (0.4, 0.0, 0.4)),
                dtype=float,
            ).reshape(3)
        else:
            pos_c = np.asarray(pos_c, dtype=float).reshape(3)

        dz = float(getattr(state.config, "dual_side_second_arm_delta_z_m", -0.1))
        pos = pos_c + np.array([0.0, 0.0, dz], dtype=float)

        R = rotation_second_arm_side_grasp_world(second_arm_is_right=(second == "right"))

        msg, quat = pose_to_msg(pos, R, config=state.config)
        self._publish(second, msg)
        wait_until_robot_settled()

        state.second_arm_side_approach_done = True

        return {
            "carrier_arm": carrier,
            "second_arm": second,
            "second_arm_tool_z_world": ("+Y" if second == "right" else "-Y"),
            "position_m": pos.tolist(),
            "delta_z_m": dz,
            "quaternion_xyzw": [float(quat[i]) for i in range(4)],
        }
