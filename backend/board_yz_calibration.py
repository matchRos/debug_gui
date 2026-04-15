"""
2D homography from camera pixels to robot base (Y, Z) for a board in the YZ plane.

Calibration file format: cable_routing/debug_gui/configs/cameras/camera_robot_2d_calibration.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_yaml_dict(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load board calibration YAML "
            "(pip install pyyaml)."
        ) from e
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Board calibration file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top level: {path}")
    return data


@dataclass
class BoardYZCalibration:
    """
    Pixel (u, v) in full image coordinates -> world (y, z) in base_link,
    with board plane at fixed world X (see board_plane_x_m in config).
    """

    homography: np.ndarray  # 3x3, maps [u, v, 1] -> [y, z, w] homogeneous
    homography_inv: np.ndarray
    base_frame: str = "yumi_base_link"
    tcp_frame: str = ""

    @classmethod
    def from_yaml_path(cls, path: str) -> "BoardYZCalibration":
        data = _load_yaml_dict(path)
        h = data.get("homography_matrix_3x3")
        if h is None:
            raise KeyError(
                f"'homography_matrix_3x3' missing in board calibration: {path}"
            )
        H = np.asarray(h, dtype=float).reshape(3, 3)
        det = float(np.linalg.det(H))
        if abs(det) < 1e-12:
            raise ValueError(f"Singular homography in {path}")
        H_inv = np.linalg.inv(H)
        return cls(
            homography=H,
            homography_inv=H_inv,
            base_frame=str(data.get("base_frame", "yumi_base_link")),
            tcp_frame=str(data.get("tcp_frame", "")),
        )

    def pixel_to_yz(self, u: float, v: float) -> Tuple[float, float]:
        p = self.homography @ np.array([float(u), float(v), 1.0], dtype=float)
        if abs(p[2]) < 1e-12:
            raise ValueError(f"Degenerate homogeneous coordinate for pixel ({u}, {v})")
        return float(p[0] / p[2]), float(p[1] / p[2])

    def yz_to_pixel(self, y: float, z: float) -> Tuple[float, float]:
        q = self.homography_inv @ np.array([float(y), float(z), 1.0], dtype=float)
        if abs(q[2]) < 1e-12:
            raise ValueError(f"Degenerate homogeneous coordinate for yz ({y}, {z})")
        return float(q[0] / q[2]), float(q[1] / q[2])

    def pixel_to_world(self, u: float, v: float, board_plane_x_m: float) -> np.ndarray:
        y, z = self.pixel_to_yz(u, v)
        return np.array([float(board_plane_x_m), y, z], dtype=float)


def load_board_yz_calibration_optional(path: Optional[str]) -> Optional[BoardYZCalibration]:
    if path is None or str(path).strip() == "":
        return None
    try:
        return BoardYZCalibration.from_yaml_path(str(path))
    except Exception as exc:
        print(f"Warning: could not load board YZ calibration from {path!r}: {exc}")
        return None
