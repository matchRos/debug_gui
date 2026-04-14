from typing import Any, Tuple

import numpy as np


def _quantize_orientation_deg(orientation_deg: float) -> int:
    val = int(round(float(orientation_deg) / 90.0) * 90) % 360
    return val


def _clip_forward_axis_px(orientation_deg: float) -> np.ndarray:
    """
    Return clip forward axis in image pixel coordinates (x right, y down).
    """
    ori = _quantize_orientation_deg(orientation_deg)
    lookup = {
        0: np.array([1.0, 0.0], dtype=float),
        90: np.array([0.0, -1.0], dtype=float),
        180: np.array([-1.0, 0.0], dtype=float),
        270: np.array([0.0, 1.0], dtype=float),
    }
    return lookup[ori]


def build_c_clip_entry_pixels(
    curr_clip: Any,
    primary_arm: str,
    config: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build separated entry targets for C-clip routing in pixel space.

    Targets are intentionally offset left/right in the clip-local frame to
    avoid immediate arm collisions near the clip center.
    """
    center = np.array([float(curr_clip.x), float(curr_clip.y)], dtype=float)
    forward = _clip_forward_axis_px(float(curr_clip.orientation))
    right = np.array([forward[1], -forward[0]], dtype=float)

    primary_lateral_px = float(getattr(config, "c_clip_primary_lateral_px", 70.0))
    secondary_lateral_px = float(getattr(config, "c_clip_secondary_lateral_px", 50.0))
    primary_forward_px = float(getattr(config, "c_clip_primary_forward_px", 25.0))
    secondary_forward_px = float(getattr(config, "c_clip_secondary_forward_px", -10.0))

    # Keep primary on clip-right side by default.
    primary_sign = 1.0
    secondary_sign = -1.0

    primary_px = (
        center
        + right * primary_sign * primary_lateral_px
        + forward * primary_forward_px
    )
    secondary_px = (
        center
        + right * secondary_sign * secondary_lateral_px
        + forward * secondary_forward_px
    )

    # Optional arm swap for symmetry experiments.
    if primary_arm == "right" and bool(
        getattr(config, "c_clip_swap_sides_when_primary_right", False)
    ):
        primary_px, secondary_px = secondary_px, primary_px

    return primary_px, secondary_px


def build_c_clip_center_pixels(
    curr_clip: Any,
    primary_arm: str,
    config: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build second-phase C-clip targets (move toward clip center lane).
    """
    center = np.array([float(curr_clip.x), float(curr_clip.y)], dtype=float)
    forward = _clip_forward_axis_px(float(curr_clip.orientation))
    right = np.array([forward[1], -forward[0]], dtype=float)

    primary_lateral_px = float(getattr(config, "c_clip_center_primary_lateral_px", 20.0))
    secondary_lateral_px = float(
        getattr(config, "c_clip_center_secondary_lateral_px", 45.0)
    )
    primary_forward_px = float(getattr(config, "c_clip_center_primary_forward_px", 5.0))
    secondary_forward_px = float(
        getattr(config, "c_clip_center_secondary_forward_px", -10.0)
    )

    primary_sign = 1.0
    secondary_sign = -1.0

    primary_px = (
        center
        + right * primary_sign * primary_lateral_px
        + forward * primary_forward_px
    )
    secondary_px = (
        center
        + right * secondary_sign * secondary_lateral_px
        + forward * secondary_forward_px
    )

    if primary_arm == "right" and bool(
        getattr(config, "c_clip_swap_sides_when_primary_right", False)
    ):
        primary_px, secondary_px = secondary_px, primary_px

    return primary_px, secondary_px
