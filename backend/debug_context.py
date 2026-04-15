from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class DebugContext:
    """
    Minimal backend context for the new debug pipeline.

    This class is intentionally small and can later be extended with
    additional services and helper methods.
    """

    config: Any
    robot: Optional[Any] = None
    camera: Optional[Any] = None
    board: Optional[Any] = None
    tracer: Optional[Any] = None

    t_cam_base_left: Optional[np.ndarray] = None
    t_cam_base_right: Optional[np.ndarray] = None

    T_CAM_BASE: Optional[dict] = None

    # Optional: homography pixel -> (Y,Z) at fixed board X (see board_calibration_yaml).
    board_yz_calibration: Optional[Any] = None
