"""
Project between image pixels and world frame for the debug GUI.

When ``env.board_yz_calibration`` is set, the board lies in the world YZ plane
at fixed X (``board_plane_x_m``): homography maps pixel (u,v) -> (Y,Z) in
``base_link``; the full point is ``[board_plane_x_m, Y, Z]``.

**Consistency with overlays:** The same ``world_from_pixel_debug`` /
``pixel_from_world_debug`` pair is used for trace overlays, grasp overlay, and
clip positions. If the GUI looks correct but the robot misses in space, check:
(1) ``board_plane_x_m`` matches the physical board, (2) the homography YAML was
calibrated for the same camera image size / ROI as live frames, (3) ROS poses
use ``frame_id=yumi_base_link`` (or the same frame as the homography) / your driver.

Otherwise fall back to pinhole + ``T_cam_base`` (horizontal table assumption).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from cable_routing.env.ext_camera.utils.img_utils import get_world_coord_from_pixel_coord


def world_from_pixel_debug(
    env: Any,
    config: Any,
    pixel_xy: Tuple[float, float],
    arm: str = "right",
    is_clip: bool = False,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Return world position (3,) in base_link for a pixel (u, v) = (x_img, y_img).
    """
    cal = getattr(env, "board_yz_calibration", None)
    if cal is not None:
        u, v = float(pixel_xy[0]), float(pixel_xy[1])
        bx = float(getattr(config, "board_plane_x_m", 0.56))
        return cal.pixel_to_world(u, v, bx)

    if not hasattr(env, "T_CAM_BASE") or arm not in env.T_CAM_BASE:
        raise RuntimeError(f"T_CAM_BASE missing for arm '{arm}' in pinhole projection mode.")
    intrinsic = env.camera.intrinsic
    T = env.T_CAM_BASE[arm]
    shape = image_shape
    if shape is None and getattr(env, "camera", None) is not None:
        img = None
        for name in ("get_rgb", "get_rgb_image", "get_image", "get_frame", "read"):
            if hasattr(env.camera, name):
                try:
                    img = getattr(env.camera, name)()
                    break
                except Exception:
                    pass
        if img is not None:
            shape = img.shape

    w = get_world_coord_from_pixel_coord(
        (float(pixel_xy[0]), float(pixel_xy[1])),
        intrinsic,
        T,
        image_shape=shape,
        is_clip=is_clip,
        arm=arm,
    )
    return np.asarray(w, dtype=float).reshape(3)


def pixel_from_world_debug(
    env: Any,
    config: Any,
    world_xyz: np.ndarray,
    arm: str = "right",
    intrinsic: Any = None,
    T_cam_base: Any = None,
) -> Optional[Tuple[int, int]]:
    """
    Project a world point to pixel (u, v). Returns None if invalid (e.g. behind camera).
    """
    cal = getattr(env, "board_yz_calibration", None)
    if cal is not None:
        p = np.asarray(world_xyz, dtype=float).reshape(3)
        y, z = float(p[1]), float(p[2])
        u, v = cal.yz_to_pixel(y, z)
        return int(round(u)), int(round(v))

    from cable_routing.debug_gui.backend.visualization_service import VisualizationService

    viz = VisualizationService()
    if intrinsic is None:
        intrinsic = env.camera.intrinsic
    if T_cam_base is None:
        if not hasattr(env, "T_CAM_BASE") or arm not in env.T_CAM_BASE:
            return None
        T_cam_base = env.T_CAM_BASE[arm]
    return viz.project_world_to_pixel(
        np.asarray(world_xyz, dtype=float).reshape(3),
        intrinsic,
        T_cam_base,
    )
