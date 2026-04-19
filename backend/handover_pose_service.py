"""
Handover helpers: small tool-frame adjustments relative to the current grasp rotation.

Avoids full re-solving of the tool frame from clip geometry — the robot is already
near the right pose; tune a few degrees with ``handover_fine_tool_*_deg`` in config.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from cable_routing.debug_gui.backend.board_projection import world_from_pixel_debug


def routing_clip_world_m(
    state: Any,
    clip_index: int,
    arm: str = "right",
) -> np.ndarray:
    """World position (3,) for clip centre from board projection (same as grasp_planning)."""
    if state.env is None:
        raise RuntimeError("Environment not available.")
    clips = state.clips
    if clips is None:
        raise RuntimeError("clips not available.")
    clip = clips[int(clip_index)]
    img_shape = state.rgb_image.shape if state.rgb_image is not None else None
    return world_from_pixel_debug(
        state.env,
        state.config,
        (float(clip.x), float(clip.y)),
        arm=arm,
        is_clip=True,
        image_shape=img_shape,
    ).reshape(3)


def resolve_handover_arm(state: Any, override: Optional[str]) -> str:
    """
    Which arm performs handover. ``override`` 'left'/'right' forces;
    None uses descend_first_arm (first arm that grasped).
    """
    if override in ("left", "right"):
        return str(override)
    first = getattr(state, "descend_first_arm", None)
    if first in ("left", "right"):
        return str(first)
    if hasattr(state, "grasp_poses") and state.grasp_poses:
        a = state.grasp_poses[0].get("arm", "right")
        if a in ("left", "right"):
            return str(a)
    raise RuntimeError(
        "Cannot resolve handover arm: set handover_arm to 'left'/'right' or run descend/grasp first."
    )


def grasp_pose_for_arm(grasp_poses: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    for p in grasp_poses:
        if p.get("arm") == arm:
            return p
    if len(grasp_poses) == 1:
        return grasp_poses[0]
    raise RuntimeError(f"No grasp pose for arm '{arm}'.")


def lift_offset_along_plane_normal(
    plane: Any,
    lift_distance_m: float,
) -> np.ndarray:
    """Same convention as lift_after_grasp_step: offset along plane normal."""
    n = np.asarray(plane.normal, dtype=float).reshape(3)
    n /= np.linalg.norm(n) + 1e-8
    return float(lift_distance_m) * n


def _rot_axis_deg(axis: str, deg: float) -> np.ndarray:
    t = np.deg2rad(float(deg))
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    raise ValueError(axis)


def fine_orient_on_grasp_rotation(
    R_grasp: np.ndarray,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> np.ndarray:
    """
    Small intrinsic rotations in the tool frame, applied after the grasp orientation:
    ``R = R_grasp @ Rz(rz) @ Ry(ry) @ Rx(rx)`` (degrees, right-handed).

    Tune a few degrees at a time; large values can trigger large wrist motion on YuMi.
    """
    Rg = np.asarray(R_grasp, dtype=float).reshape(3, 3)
    d_rx = float(rx_deg)
    d_ry = float(ry_deg)
    d_rz = float(rz_deg)
    return Rg @ _rot_axis_deg("z", d_rz) @ _rot_axis_deg("y", d_ry) @ _rot_axis_deg("x", d_rx)


class HandoverPoseService:
    """Thin facade for tests / future extensions."""

    def routing_clip_world_m(self, state: Any, clip_index: int, arm: str) -> np.ndarray:
        return routing_clip_world_m(state, clip_index, arm=arm)

    def fine_orient_on_grasp_rotation(
        self, R_grasp: np.ndarray, rx_deg: float, ry_deg: float, rz_deg: float
    ) -> np.ndarray:
        return fine_orient_on_grasp_rotation(R_grasp, rx_deg, ry_deg, rz_deg)
