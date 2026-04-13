from typing import Dict, Optional

import cv2
import numpy as np

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.env.robots.misc import calculate_sequence


class PlanFirstRouteStep(BaseStep):
    name = "plan_first_route"
    description = (
        "Prepare and visualize the first routing move around the first target clip."
    )

    def __init__(self):
        super().__init__()

    def _clip_px(self, clip):
        return np.array([float(clip.x), float(clip.y)], dtype=float)

    def _clip_to_dict(self, clip):
        return {
            "x": float(clip.x),
            "y": float(clip.y),
            "type": clip.clip_type,
            "orientation": clip.orientation,
        }

    def _pose_for_arm(self, poses, arm_name) -> Optional[dict]:
        for pose in poses:
            if pose.get("arm") == arm_name:
                return pose
        return None

    def _world_to_pixel(self, world_point, arm, state: PipelineState):
        env = state.env
        if env is None or env.camera is None:
            raise RuntimeError(
                "Camera/environment not available for world->pixel projection."
            )

        if not hasattr(env, "T_CAM_BASE") or arm not in env.T_CAM_BASE:
            raise RuntimeError(f"T_CAM_BASE for arm '{arm}' not available.")

        T_cam_base = env.T_CAM_BASE[arm]

        p_base = np.asarray(world_point, dtype=float).reshape(3)
        p_cam = T_cam_base.inverse().apply(p_base)

        if p_cam[2] <= 1e-6:
            raise RuntimeError(f"Projected point is behind camera for arm '{arm}'.")

        uv = env.camera.intrinsic.project(p_cam.reshape(3, 1))
        uv = np.asarray(uv).reshape(-1)

        return np.array([float(uv[0]), float(uv[1])], dtype=float)

    def _compute_preview_target_px(
        self,
        prev_clip,
        curr_clip,
        next_clip,
        clockwise_direction,
        offset_px=60.0,
    ):
        p_prev = self._clip_px(prev_clip)
        p_curr = self._clip_px(curr_clip)
        p_next = self._clip_px(next_clip)

        direction = p_next - p_curr
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            direction = p_curr - p_prev
            norm = np.linalg.norm(direction)

        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=float)
            norm = 1.0

        direction = direction / norm

        perp = np.array([-direction[1], direction[0]], dtype=float)
        side_sign = -1.0 if clockwise_direction else 1.0

        target_px = (
            p_curr + direction * offset_px + perp * side_sign * (0.25 * offset_px)
        )
        return target_px

    def _draw_overlay(
        self,
        image,
        clips,
        prev_clip_id,
        curr_clip_id,
        next_clip_id,
        clockwise_direction,
        primary_arm,
        start_px,
        target_px,
    ):
        overlay = image.copy()

        for clip_id, clip in clips.items():
            p = self._clip_px(clip).astype(int)
            cv2.circle(overlay, tuple(p), 6, (180, 180, 180), -1)
            cv2.putText(
                overlay,
                str(clip_id),
                (int(p[0]) + 8, int(p[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        prev_clip = clips[prev_clip_id]
        curr_clip = clips[curr_clip_id]
        next_clip = clips[next_clip_id]

        p_prev = self._clip_px(prev_clip).astype(int)
        p_curr = self._clip_px(curr_clip).astype(int)
        p_next = self._clip_px(next_clip).astype(int)

        cv2.line(overlay, tuple(p_prev), tuple(p_curr), (255, 255, 0), 2)
        cv2.line(overlay, tuple(p_curr), tuple(p_next), (255, 255, 0), 2)

        cv2.circle(overlay, tuple(p_prev), 10, (255, 0, 0), 2)
        cv2.circle(overlay, tuple(p_curr), 12, (255, 0, 255), 2)
        cv2.circle(overlay, tuple(p_next), 10, (0, 255, 0), 2)

        s = np.asarray(start_px, dtype=float).astype(int)
        t = np.asarray(target_px, dtype=float).astype(int)

        cv2.circle(overlay, tuple(s), 8, (0, 255, 255), -1)
        cv2.circle(overlay, tuple(t), 8, (255, 100, 100), -1)
        cv2.line(overlay, tuple(s), tuple(t), (255, 120, 120), 2)

        cv2.putText(
            overlay,
            "start",
            (int(s[0]) + 8, int(s[1]) + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "target",
            (int(t[0]) + 8, int(t[1]) + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 120, 120),
            1,
            cv2.LINE_AA,
        )

        direction_text = "CW" if clockwise_direction else "CCW"
        info_text = (
            f"{prev_clip_id} -> {curr_clip_id} -> {next_clip_id} | "
            f"{direction_text} | arm={primary_arm}"
        )

        cv2.putText(
            overlay,
            info_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 100, 100),
            2,
            cv2.LINE_AA,
        )

        return overlay

    def run(self, state: PipelineState) -> Dict[str, object]:
        if state.routing is None or len(state.routing) < 3:
            raise RuntimeError("Routing must contain at least 3 clip IDs.")

        if state.clips is None:
            raise RuntimeError("Clip data not available.")

        if state.rgb_image is None:
            raise RuntimeError("No RGB image available for visualization.")

        if not hasattr(state, "grasp_poses"):
            raise RuntimeError("No grasp poses available.")

        prev_clip_id = state.routing[0]
        curr_clip_id = state.routing[1]
        next_clip_id = state.routing[2]

        clips = state.clips
        prev_clip = clips[prev_clip_id]
        curr_clip = clips[curr_clip_id]
        next_clip = clips[next_clip_id]

        prev_clip_dict = self._clip_to_dict(prev_clip)
        curr_clip_dict = self._clip_to_dict(curr_clip)
        next_clip_dict = self._clip_to_dict(next_clip)

        sequence, clockwise_direction = calculate_sequence(
            curr_clip_dict,
            prev_clip_dict,
            next_clip_dict,
        )

        primary_arm = getattr(state, "descend_first_arm", None)
        if primary_arm is None:
            primary_arm = "left"

        primary_pose = self._pose_for_arm(state.grasp_poses, primary_arm)
        if primary_pose is None:
            raise RuntimeError(f"No grasp pose found for primary arm '{primary_arm}'.")

        start_px = self._world_to_pixel(primary_pose["position"], primary_arm, state)
        target_px = self._compute_preview_target_px(
            prev_clip=prev_clip,
            curr_clip=curr_clip,
            next_clip=next_clip,
            clockwise_direction=clockwise_direction,
            offset_px=60.0,
        )

        state.current_primary_arm = primary_arm
        state.first_route_prev_clip_id = prev_clip_id
        state.first_route_curr_clip_id = curr_clip_id
        state.first_route_next_clip_id = next_clip_id
        state.first_route_clockwise = clockwise_direction
        state.first_route_sequence = sequence
        state.first_route_start_px = start_px
        state.first_route_target_px = target_px

        overlay = self._draw_overlay(
            image=state.rgb_image,
            clips=clips,
            prev_clip_id=prev_clip_id,
            curr_clip_id=curr_clip_id,
            next_clip_id=next_clip_id,
            clockwise_direction=clockwise_direction,
            primary_arm=primary_arm,
            start_px=start_px,
            target_px=target_px,
        )

        state.routing_overlay = overlay
        state.grasp_overlay = None

        return {
            "route_ready": True,
            "overlay_updated": True,
            "primary_arm": primary_arm,
            "prev_clip_id": prev_clip_id,
            "curr_clip_id": curr_clip_id,
            "next_clip_id": next_clip_id,
            "clockwise_direction": clockwise_direction,
            "sequence": sequence,
            "curr_clip_type": curr_clip.clip_type,
            "curr_clip_orientation": curr_clip.orientation,
            "start_px": [float(start_px[0]), float(start_px[1])],
            "target_px": [float(target_px[0]), float(target_px[1])],
        }
