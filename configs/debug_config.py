from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class DebugConfig:
    """
    Minimal standalone config for the debug GUI pipeline.
    """

    board_cfg_path: str = "cable_routing/configs/board/board_config.json"
    default_routing: tuple = (0, 1, 2, 3)

    fallback_image_width: int = 1500
    fallback_image_height: int = 800
    grasp_offset_px: float = 140.0

    # Optional fallback image if no camera is available
    # debug_image_path: Optional[str] = None
    debug_image_path: Optional[str] = "/ABSOLUTER/PFAD/ZU/DEINEM/BILD.png"

    # Manual tracing seed points for the debug pipeline
    # Format: (x, y)
    trace_start_points: Tuple[Tuple[int, int], ...] = ((100, 100),)
    trace_end_points: Optional[Tuple[Tuple[int, int], ...]] = None
    # "auto_from_config": use configured trace_start_points
    # "manual_two_clicks": ask user for (start, direction) and derive 3 tracer points
    trace_start_mode: str = "auto_from_config"

    # Plane model (future-proof: multiple planes, clip->plane assignment).
    routing_plane_default_id: str = "main"
    routing_planes: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "main": {
                "origin": [0.0, 0.0, 0.15],
                "normal": [0.0, 0.0, 1.0],
                "u_axis": [1.0, 0.0, 0.0],
                "v_axis": [0.0, 1.0, 0.0],
            }
        }
    )
    clip_plane_assignments: Dict[int, str] = field(default_factory=dict)

    # Heights relative to routing plane.
    routing_height_above_plane_m: float = 0.025
    grasp_height_above_plane_m: float = 0.025
    pregrasp_offset_from_grasp_m: float = 0.07
    detangle_offset_from_routing_m: float = 0.03

    # First route: move primary arm farther along route to reduce collisions near clip.
    first_route_primary_extra_along_route_px: float = 60.0

    # C-clip primitive (pixel-space offsets in clip-local frame).
    c_clip_primary_lateral_px: float = 90.0
    c_clip_secondary_lateral_px: float = 70.0
    c_clip_primary_forward_px: float = 30.0
    c_clip_secondary_forward_px: float = -15.0
    c_clip_swap_sides_when_primary_right: bool = False
    c_clip_center_primary_lateral_px: float = 20.0
    c_clip_center_secondary_lateral_px: float = 45.0
    c_clip_center_primary_forward_px: float = 5.0
    c_clip_center_secondary_forward_px: float = -10.0

    cam_to_robot_left_trans_path = os.path.join(
        BASE_DIR, "configs/cameras/zed_to_world_left.tf"
    )
    cam_to_robot_right_trans_path = os.path.join(
        BASE_DIR, "configs/cameras/zed_to_world_right.tf"
    )
