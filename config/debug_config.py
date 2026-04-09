from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DebugConfig:
    """
    Minimal standalone config for the debug GUI pipeline.
    """

    board_cfg_path: str = "cable_routing/configs/board/board_config.json"
    default_routing: tuple = (0, 1, 2, 3)

    fallback_image_width: int = 1500
    fallback_image_height: int = 800

    # Optional fallback image if no camera is available
    # debug_image_path: Optional[str] = None
    debug_image_path: Optional[str] = "/ABSOLUTER/PFAD/ZU/DEINEM/BILD.png"

    # Manual tracing seed points for the debug pipeline
    # Format: (x, y)
    trace_start_points: Tuple[Tuple[int, int], ...] = ((100, 100),)
    trace_end_points: Optional[Tuple[Tuple[int, int], ...]] = None
