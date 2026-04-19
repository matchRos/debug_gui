from typing import Any, Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

from cable_routing.debug_gui.backend.dual_arm_presentation_geometry import (
    rotation_carrier_cable_vertical_world,
)
from cable_routing.debug_gui.backend.grasp_pose_service import _rotation_world_rx_deg
from cable_routing.debug_gui.backend.handover_pose_service import resolve_handover_arm
from cable_routing.debug_gui.backend.planes import get_routing_plane
from cable_routing.debug_gui.pipeline.arm_motion_utils import (
    pose_to_msg,
    wait_until_robot_settled,
)
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class PresentCableVerticalStep(BaseStep):
    """
    Carrying arm only (``slowly_approach_pose``): base frame from
    ``rotation_carrier_cable_vertical_world``, then an extra **world +X** rotation
    (``present_cable_extra_world_rx_deg``, default +90°) so the gripper ends horizontal
    (parallel to floor) instead of vertical. TCP position unchanged.
    """

    name = "present_cable_vertical"
    description = (
        "Rotate carrier arm so cable hangs along -Z; gripper normal to board plane."
    )

    def __init__(self) -> None:
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_present_cable_vertical", anonymous=True)
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
        if state.routing is None or len(state.routing) < 1:
            raise RuntimeError("routing required.")

        arm = resolve_handover_arm(state, getattr(state.config, "handover_arm", None))
        ridx = int(getattr(state.config, "handover_clip_routing_index", 0))
        clip_board_idx = int(state.routing[ridx])
        plane = get_routing_plane(state.config, clip_id=clip_board_idx)

        pos = getattr(state, "handover_carrier_tcp_world", None)
        if pos is None:
            pos = np.asarray(
                getattr(state.config, "handover_goal_world_m", (0.4, 0.0, 0.4)),
                dtype=float,
            ).reshape(3)
        else:
            pos = np.asarray(pos, dtype=float).reshape(3)

        R = rotation_carrier_cable_vertical_world(plane)
        rx_extra = float(getattr(state.config, "present_cable_extra_world_rx_deg", 90.0))
        if abs(rx_extra) > 1e-9:
            R = _rotation_world_rx_deg(rx_extra) @ R
        state.handover_tcp_rotation_world = R.copy()

        msg, quat = pose_to_msg(pos, R, config=state.config)
        self._publish(arm, msg)
        wait_until_robot_settled()

        state.present_cable_vertical_done = True

        return {
            "arm": arm,
            "position_m": pos.tolist(),
            "quaternion_xyzw": [float(quat[i]) for i in range(4)],
        }
