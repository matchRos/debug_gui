"""
World-frame targets for executing the first-route preview from plan_first_route.

The original monolithic pipeline branches by clip type inside slideto_cable_node /
execute_dual_slide_*; here we keep one default path and peg (single routing arm,
second arm holds grasp). Add per-type modules later and dispatch from
build_first_route_execution_poses.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from cable_routing.debug_gui.backend.clip_types import CLIP_TYPE_PEG
from cable_routing.debug_gui.pipeline.arm_motion_utils import validate_min_distance
from cable_routing.env.ext_camera.utils.img_utils import get_world_coord_from_pixel_coord


def _pose_for_arm(grasp_poses: List[Dict[str, Any]], arm_name: str) -> Dict[str, Any]:
    for pose in grasp_poses:
        if pose.get("arm") == arm_name:
            return pose
    raise RuntimeError(f"No grasp pose for arm '{arm_name}'.")


def _pixel_to_world_clip(
    uv: np.ndarray,
    state: Any,
    arm: str,
) -> np.ndarray:
    env = state.env
    if arm not in env.T_CAM_BASE:
        raise RuntimeError(f"T_CAM_BASE missing for arm '{arm}'.")
    intrinsic = env.camera.intrinsic
    T = env.T_CAM_BASE[arm]
    img_shape = state.rgb_image.shape
    w = get_world_coord_from_pixel_coord(
        (float(uv[0]), float(uv[1])),
        intrinsic,
        T,
        image_shape=img_shape,
        is_clip=True,
        arm=arm,
    )
    return np.asarray(w, dtype=float).reshape(3)


def build_first_route_execution_poses(
    state: Any,
    min_dist_xyz: float = 0.08,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Build left/right pose dicts with 'position' (3,) and 'rotation' (3,3).

    Expects plan_first_route to have set first_route_target_px and related fields.

    Returns:
        left_pose_dict, right_pose_dict, mode ('peg_hold' | 'dual_slide')
    """
    if state.env is None or state.env.camera is None:
        raise RuntimeError("Environment / camera not available.")
    if not hasattr(state.env, "T_CAM_BASE"):
        raise RuntimeError("T_CAM_BASE not available.")
    if state.rgb_image is None:
        raise RuntimeError("No rgb_image in state.")
    if not hasattr(state, "grasp_poses"):
        raise RuntimeError("No grasp_poses in state.")
    if getattr(state, "first_route_target_px", None) is None:
        raise RuntimeError(
            "No first_route_target_px. Run plan_first_route before execute_first_route."
        )

    primary_arm = getattr(state, "current_primary_arm", None) or "left"
    secondary_arm = "right" if primary_arm == "left" else "left"

    curr_idx = getattr(state, "first_route_curr_clip_id", None)
    if curr_idx is None or state.clips is None:
        raise RuntimeError("Missing first_route_curr_clip_id or clips.")
    curr_clip = state.clips[curr_idx]
    clip_type = int(curr_clip.clip_type)

    primary_px = np.asarray(state.first_route_target_px, dtype=float).reshape(2)
    primary_pos = _pixel_to_world_clip(primary_px, state, primary_arm)
    primary_rot = _pose_for_arm(state.grasp_poses, primary_arm)["rotation"]
    primary_pose = {"position": primary_pos, "rotation": np.asarray(primary_rot)}

    secondary_pose = _pose_for_arm(state.grasp_poses, secondary_arm)
    secondary_hold = {
        "position": np.asarray(secondary_pose["position"], dtype=float).copy(),
        "rotation": np.asarray(secondary_pose["rotation"]),
    }

    if clip_type == CLIP_TYPE_PEG:
        # Primary executes routing; secondary stays at current grasp (hold).
        if primary_arm == "left":
            left, right = primary_pose, secondary_hold
        else:
            left, right = secondary_hold, primary_pose
        validate_min_distance(left, right, min_dist_xyz, label="First route (peg)")
        return left, right, "peg_hold"

    if not getattr(state, "first_route_secondary_shown", False):
        raise RuntimeError(
            "first_route_secondary_shown is False but clip is not a peg; "
            "re-run plan_first_route."
        )
    sec_px = getattr(state, "first_route_secondary_target_px", None)
    if sec_px is None:
        raise RuntimeError("Missing first_route_secondary_target_px for dual first route.")

    secondary_px = np.asarray(sec_px, dtype=float).reshape(2)
    secondary_pos = _pixel_to_world_clip(secondary_px, state, secondary_arm)
    secondary_rot = secondary_pose["rotation"]
    secondary_target = {"position": secondary_pos, "rotation": np.asarray(secondary_rot)}

    if primary_arm == "left":
        left, right = primary_pose, secondary_target
    else:
        left, right = secondary_target, primary_pose

    validate_min_distance(left, right, min_dist_xyz, label="First route (dual)")
    return left, right, "dual_slide"
