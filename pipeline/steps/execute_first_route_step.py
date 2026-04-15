from typing import Any, Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

from cable_routing.debug_gui.backend.first_route_targets import (
    build_c_clip_centering_poses,
    build_first_route_execution_poses,
)
from cable_routing.debug_gui.pipeline.arm_motion_utils import (
    enforce_pose_min_height,
    is_dual_arm_grasp,
    pose_to_published_pose_stamped,
    wait_until_robot_settled,
)
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class ExecuteFirstRouteStep(BaseStep):
    name = "execute_first_route"
    description = (
        "Execute planned first-route targets with slowly_approach_pose (both arms "
        "published together; peg keeps secondary arm at grasp)."
    )

    def __init__(self) -> None:
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("debug_gui_execute_first_route", anonymous=True)

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

    def _publish_route_pair(
        self, state: PipelineState, left_msg: PoseStamped, right_msg: PoseStamped
    ) -> str:
        """Publish both arms, or only ``current_primary_arm`` when ``dual_arm_grasp`` is False."""
        if is_dual_arm_grasp(state.config):
            self.pub_left.publish(left_msg)
            self.pub_right.publish(right_msg)
            return "both"
        primary = getattr(state, "current_primary_arm", None)
        if primary is None and hasattr(state, "grasp_poses") and state.grasp_poses:
            primary = state.grasp_poses[0].get("arm", "left")
        if primary == "right":
            self.pub_right.publish(right_msg)
            return "right"
        self.pub_left.publish(left_msg)
        return "left"

    def run(self, state: PipelineState) -> Dict[str, Any]:
        left_pose, right_pose, mode = build_first_route_execution_poses(state)
        routing_floor = float(state.config.routing_height_above_plane_m)
        left_pose = enforce_pose_min_height(left_pose, state, routing_floor)
        right_pose = enforce_pose_min_height(right_pose, state, routing_floor)

        left_msg, _ = pose_to_published_pose_stamped(
            left_pose["position"], left_pose["rotation"], state.config
        )
        right_msg, _ = pose_to_published_pose_stamped(
            right_pose["position"], right_pose["rotation"], state.config
        )

        now = rospy.Time.now()
        left_msg.header.stamp = now
        right_msg.header.stamp = now

        publish_mode = self._publish_route_pair(state, left_msg, right_msg)

        wait_until_robot_settled()

        second_phase_executed = False
        second_phase_mode = None
        if mode == "c_clip_entry":
            left_center, right_center, second_phase_mode = build_c_clip_centering_poses(
                state
            )
            left_center = enforce_pose_min_height(left_center, state, routing_floor)
            right_center = enforce_pose_min_height(right_center, state, routing_floor)

            left_center_msg, _ = pose_to_published_pose_stamped(
                left_center["position"], left_center["rotation"], state.config
            )
            right_center_msg, _ = pose_to_published_pose_stamped(
                right_center["position"], right_center["rotation"], state.config
            )
            now2 = rospy.Time.now()
            left_center_msg.header.stamp = now2
            right_center_msg.header.stamp = now2
            self._publish_route_pair(state, left_center_msg, right_center_msg)
            wait_until_robot_settled()
            second_phase_executed = True

        state.first_route_executed = True

        return {
            "executed": True,
            "mode": mode,
            "publish_mode": publish_mode,
            "second_phase_executed": second_phase_executed,
            "second_phase_mode": second_phase_mode,
            "left_position": [
                left_msg.pose.position.x,
                left_msg.pose.position.y,
                left_msg.pose.position.z,
            ],
            "right_position": [
                right_msg.pose.position.x,
                right_msg.pose.position.y,
                right_msg.pose.position.z,
            ],
        }
