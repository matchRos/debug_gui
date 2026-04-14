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
    pregrasp_offset_from_grasp_m: float = 0.08
    detangle_offset_from_routing_m: float = 0.04

    cam_to_robot_left_trans_path = os.path.join(
        BASE_DIR, "configs/cameras/zed_to_world_left.tf"
    )
    cam_to_robot_right_trans_path = os.path.join(
        BASE_DIR, "configs/cameras/zed_to_world_right.tf"
    )
