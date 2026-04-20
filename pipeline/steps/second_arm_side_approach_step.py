from typing import Any, Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

from cable_routing.debug_gui.backend.dual_arm_presentation_geometry import (
    rotation_second_arm_side_grasp_world,
    rotation_world_ry_deg,
)
from cable_routing.debug_gui.backend.handover_pose_service import resolve_handover_arm
from cable_routing.debug_gui.pipeline.arm_motion_utils import (
    pose_to_msg,
    wait_until_robot_settled,
)
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class SecondArmSideApproachStep(BaseStep):
    """
    Second arm: fast **moveit_target_pose** to a lateral prepose (from -Y for right arm,
    from +Y for left), then **slowly_approach_pose** to the final TCP pose (carrier XY,
    Z + delta_z). Same orientation for both motions.
    """

    name = "second_arm_side_approach"
    description = "Prepose (moveit) from the side, then slow approach; side-grasp orientation (±world Y)."

    def __init__(self) -> None:
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_second_arm_side_approach", anonymous=True)
        self.pub_left_slow = rospy.Publisher(
            "/yumi/robl/slowly_approach_pose",
            PoseStamped,
            queue_size=1,
        )
        self.pub_right_slow = rospy.Publisher(
            "/yumi/robr/slowly_approach_pose",
            PoseStamped,
            queue_size=1,
        )
        self.pub_left_moveit = rospy.Publisher(
            "/yumi/robl/cartesian_pose_command",
            PoseStamped,
            queue_size=1,
        )
        self.pub_right_moveit = rospy.Publisher(
            "/yumi/robr/cartesian_pose_command",
            PoseStamped,
            queue_size=1,
        )

    def _publish_moveit(self, arm: str, msg: PoseStamped) -> None:
        msg.header.stamp = rospy.Time.now()
        if arm == "left":
            self.pub_left_moveit.publish(msg)
        elif arm == "right":
            self.pub_right_moveit.publish(msg)
        else:
            raise RuntimeError(f"Invalid arm: {arm}")

    def _publish_slow(self, arm: str, msg: PoseStamped) -> None:
        msg.header.stamp = rospy.Time.now()
        if arm == "left":
            self.pub_left_slow.publish(msg)
        elif arm == "right":
            self.pub_right_slow.publish(msg)
        else:
            raise RuntimeError(f"Invalid arm: {arm}")

    def run(self, state: PipelineState) -> Dict[str, Any]:
        if state.env is None:
            raise RuntimeError("Environment not initialized.")

        carrier = resolve_handover_arm(
            state, getattr(state.config, "handover_arm", None)
        )
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
        pos_final = pos_c + np.array([0.0, 0.0, dz], dtype=float)

        lateral = float(
            getattr(state.config, "dual_side_second_arm_prepose_offset_y_m", 0.08)
        )
        if second == "right":
            # Approach from -Y: prepose further in -Y.
            dy_pre = -abs(lateral)
        else:
            # Left arm: from +Y.
            dy_pre = abs(lateral)

        pos_pre = pos_final + np.array([0.0, dy_pre, 0.0], dtype=float)

        R = rotation_second_arm_side_grasp_world(
            second_arm_is_right=(second == "right")
        )
        ry_extra = float(getattr(state.config, "second_arm_extra_world_ry_deg", 90.0))
        if abs(ry_extra) > 1e-9:
            R = rotation_world_ry_deg(ry_extra) @ R

        pause_s = float(
            getattr(state.config, "dual_side_second_arm_prepose_pause_s", 0.5)
        )

        msg_pre, _ = pose_to_msg(pos_pre, R, config=state.config)
        self._publish_moveit(second, msg_pre)
        rospy.sleep(max(0.0, pause_s))

        msg_fin, quat = pose_to_msg(pos_final, R, config=state.config)
        self._publish_slow(second, msg_fin)
        wait_until_robot_settled()

        state.second_arm_side_approach_done = True

        return {
            "carrier_arm": carrier,
            "second_arm": second,
            "second_arm_tool_z_world": ("+Y" if second == "right" else "-Y"),
            "prepose_position_m": pos_pre.tolist(),
            "final_position_m": pos_final.tolist(),
            "prepose_delta_y_m": float(dy_pre),
            "delta_z_m": dz,
            "second_arm_extra_world_ry_deg": ry_extra,
            "quaternion_xyzw": [float(quat[i]) for i in range(4)],
        }
