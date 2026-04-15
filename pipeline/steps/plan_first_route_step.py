from typing import Dict, Optional

import cv2
import numpy as np

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.debug_gui.backend.board_projection import pixel_from_world_debug
from cable_routing.debug_gui.backend.clip_types import CLIP_TYPE_PEG
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
        if env is None:
            raise RuntimeError("Environment not available for world->pixel projection.")

        intrinsic = env.camera.intrinsic if env.camera is not None else None
        T_cam_base = None
        if hasattr(env, "T_CAM_BASE") and env.T_CAM_BASE and arm in env.T_CAM_BASE:
            T_cam_base = env.T_CAM_BASE[arm]

        if getattr(env, "board_yz_calibration", None) is None and (
            intrinsic is None or T_cam_base is None
        ):
            raise RuntimeError(
                "Camera / T_CAM_BASE not available for pinhole world->pixel projection."
            )

        uv = pixel_from_world_debug(
            env,
            state.config,
            np.asarray(world_point, dtype=float),
            arm=arm,
            intrinsic=intrinsic,
            T_cam_base=T_cam_base,
        )
        if uv is None:
            raise RuntimeError(
                f"Grasp point projects behind camera or invalid for arm '{arm}'."
            )

        return np.array([float(uv[0]), float(uv[1])], dtype=float)

    def _compute_secondary_support_px(
        self,
        prev_clip,
        curr_clip,
        clockwise_direction: int,
        img_shape,
        extension_factor: float = 50.0,
        secondary_along_prev_normal: float = 0.5,
    ):
        """
        Pixel-space helper target for the second arm, aligned with
        env_new.execute_dual_slide_to_cable_node (normal + prev_to_normal).

        Not used for peg clips (type == CLIP_TYPE_PEG): second arm is omitted there.
        """
        curr_clip_pos = self._clip_px(curr_clip)
        prev_clip_pos = self._clip_px(prev_clip)

        clip_vector = curr_clip_pos - prev_clip_pos
        norm_cv = np.linalg.norm(clip_vector)
        if norm_cv < 1e-6:
            return None

        if clockwise_direction < 0:
            normal = np.array([-clip_vector[1], clip_vector[0]], dtype=float)
        else:
            normal = np.array([clip_vector[1], -clip_vector[0]], dtype=float)

        normal = normal / np.linalg.norm(normal)
        normal_point = curr_clip_pos + normal * extension_factor

        prev_to_normal = normal_point - prev_clip_pos
        prev_to_normal = prev_to_normal / (np.linalg.norm(prev_to_normal) + 1e-8)

        clip_distance = float(np.linalg.norm(curr_clip_pos - prev_clip_pos))

        # Same construction as execute_dual_slide_to_cable_node for the "B" secondary point.
        target_secondary = prev_clip_pos + prev_to_normal * (
            clip_distance * secondary_along_prev_normal
        )

        h, w = int(img_shape[0]), int(img_shape[1])
        target_secondary[0] = float(np.clip(target_secondary[0], 0, w - 1))
        target_secondary[1] = float(np.clip(target_secondary[1], 0, h - 1))

        return target_secondary

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
        secondary_arm=None,
        secondary_start_px=None,
        secondary_target_px=None,
        show_secondary=False,
        secondary_skipped_peg=False,
    ):
        overlay = image.copy()

        # state.clips is a List[DebugClip] (indices in routing refer to this list).
        for clip in clips:
            p = self._clip_px(clip).astype(int)
            cv2.circle(overlay, tuple(p), 6, (180, 180, 180), -1)
            cv2.putText(
                overlay,
                str(clip.clip_id),
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

        if show_secondary and secondary_start_px is not None and secondary_target_px is not None:
            ss = np.asarray(secondary_start_px, dtype=float).astype(int)
            st = np.asarray(secondary_target_px, dtype=float).astype(int)
            cv2.circle(overlay, tuple(ss), 8, (180, 255, 120), -1)
            cv2.circle(overlay, tuple(st), 8, (120, 200, 255), -1)
            cv2.line(overlay, tuple(ss), tuple(st), (120, 220, 180), 2)
            cv2.putText(
                overlay,
                "2nd start",
                (int(ss[0]) + 8, int(ss[1]) + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 120),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                "2nd target",
                (int(st[0]) + 8, int(st[1]) + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (120, 200, 255),
                1,
                cv2.LINE_AA,
            )

        # calculate_sequence: -1 == clockwise, +1 == counter-clockwise
        direction_text = "CW" if clockwise_direction < 0 else "CCW"
        if show_secondary and secondary_arm:
            sec_note = f" | 2nd={secondary_arm}"
        elif secondary_skipped_peg:
            sec_note = " | 2nd=— (peg)"
        else:
            sec_note = " | 2nd=—"
        info_text = (
            f"{prev_clip_id} -> {curr_clip_id} -> {next_clip_id} | "
            f"{direction_text} | arm={primary_arm}{sec_note}"
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
            raise RuntimeError("Routing must contain at least 3 clip indices.")

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

        secondary_arm = "right" if primary_arm == "left" else "left"
        secondary_skipped_peg = curr_clip.clip_type == CLIP_TYPE_PEG
        show_secondary = not secondary_skipped_peg
        secondary_start_px = None
        secondary_target_px = None
        if show_secondary:
            secondary_pose = self._pose_for_arm(state.grasp_poses, secondary_arm)
            if secondary_pose is None:
                show_secondary = False
            else:
                secondary_start_px = self._world_to_pixel(
                    secondary_pose["position"], secondary_arm, state
                )
                secondary_target_px = self._compute_secondary_support_px(
                    prev_clip=prev_clip,
                    curr_clip=curr_clip,
                    clockwise_direction=int(clockwise_direction),
                    img_shape=state.rgb_image.shape,
                )
                if secondary_target_px is None:
                    show_secondary = False

        state.current_primary_arm = primary_arm
        state.first_route_prev_clip_id = prev_clip_id
        state.first_route_curr_clip_id = curr_clip_id
        state.first_route_next_clip_id = next_clip_id
        state.first_route_clockwise = clockwise_direction
        state.first_route_sequence = sequence
        state.first_route_start_px = start_px
        state.first_route_target_px = target_px
        state.first_route_secondary_arm = secondary_arm if show_secondary else None
        state.first_route_secondary_start_px = secondary_start_px
        state.first_route_secondary_target_px = secondary_target_px
        state.first_route_secondary_shown = show_secondary

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
            secondary_arm=secondary_arm,
            secondary_start_px=secondary_start_px,
            secondary_target_px=secondary_target_px,
            show_secondary=show_secondary,
            secondary_skipped_peg=secondary_skipped_peg,
        )

        state.routing_overlay = overlay
        state.first_route_overlay = overlay
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
            "secondary_arm": secondary_arm if show_secondary else None,
            "secondary_shown": show_secondary,
            "secondary_start_px": (
                [float(secondary_start_px[0]), float(secondary_start_px[1])]
                if show_secondary and secondary_start_px is not None
                else None
            ),
            "secondary_target_px": (
                [float(secondary_target_px[0]), float(secondary_target_px[1])]
                if show_secondary and secondary_target_px is not None
                else None
            ),
        }
