"""
Debug GUI configuration.

Runtime config is built from YAML fragments in ``configs/parts/`` (merged in order:
core → routing_plane → trace → first_route). Missing files fall back to the dataclass
defaults defined below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_PARTS_DIR = Path(__file__).resolve().parent / "parts"
_PART_FILES = ("core.yaml", "routing_plane.yaml", "trace.yaml", "first_route.yaml")


def _load_yaml_merged(parts_dir: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to load debug GUI config parts.") from e

    merged: Dict[str, Any] = {}
    for name in _PART_FILES:
        path = parts_dir / name
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            merged.update(data)
    return merged


def _expand_paths_if_relative(cfg: Dict[str, Any]) -> None:
    """Join package-relative camera / calibration paths with BASE_DIR (same as before)."""
    keys = (
        "cam_to_robot_left_trans_path",
        "cam_to_robot_right_trans_path",
        "board_calibration_yaml",
    )
    for key in keys:
        val = cfg.get(key)
        if not val or not isinstance(val, str):
            continue
        if os.path.isabs(val):
            continue
        cfg[key] = os.path.normpath(os.path.join(BASE_DIR, val))


def _coerce_for_dataclass(name: str, value: Any, fallback: Any) -> Any:
    if name == "trace_end_points" and value is None:
        return None
    if name == "default_routing" and isinstance(value, list):
        return tuple(int(x) for x in value)
    if name == "trace_start_points" and isinstance(value, list):
        return tuple(tuple(int(x) for x in row) for row in value)
    if name == "trace_end_points" and isinstance(value, list):
        return tuple(tuple(int(x) for x in row) for row in value)
    if name in (
        "single_arm_nominal_tcp_left_m",
        "single_arm_nominal_tcp_right_m",
        "cartesian_targets_world_position_offset_m",
    ) and isinstance(value, list):
        return tuple(float(x) for x in value)
    if name == "routing_planes" and isinstance(value, dict):
        return {str(k): dict(v) for k, v in value.items()}
    if name == "clip_plane_assignments" and isinstance(value, dict):
        return {int(k): str(v) for k, v in value.items()}
    return value


def load_debug_config(parts_dir: Optional[Path] = None) -> "DebugConfig":
    """
    Load merged config from ``parts_dir`` (default: ``configs/parts`` next to this file).
    """
    parts_dir = parts_dir or _PARTS_DIR
    merged = _load_yaml_merged(parts_dir)
    _expand_paths_if_relative(merged)

    base = DebugConfig()
    kwargs: Dict[str, Any] = {}
    for f in fields(DebugConfig):
        name = f.name
        if name in merged:
            kwargs[name] = _coerce_for_dataclass(name, merged[name], getattr(base, name))
        else:
            kwargs[name] = getattr(base, name)
    return DebugConfig(**kwargs)


@dataclass
class DebugConfig:
    """
    Single flat config object used across the debug GUI (backward compatible).

    Prefer editing YAML files under ``configs/parts/`` instead of large edits here.
    """

    board_cfg_path: str = "cable_routing/configs/board/board_config.json"
    default_routing: tuple = (0, 1, 2, 3)

    fallback_image_width: int = 1500
    fallback_image_height: int = 800

    debug_image_path: Optional[str] = "/ABSOLUTER/PFAD/ZU/DEINEM/BILD.png"

    trace_start_points: Tuple[Tuple[int, int], ...] = ((100, 100),)
    trace_end_points: Optional[Tuple[Tuple[int, int], ...]] = None
    trace_start_mode: str = "auto_from_config"
    trace_anchor_max_start_dist_px: float = 90.0
    trace_candidate_min_route_dot: float = 0.25
    trace_anchor_outward_min_delta_px: float = 8.0
    trace_auto_clip_a_p1_offset_px: float = 20.0
    trace_auto_clip_a_p2_offset_px: float = 40.0
    # Ring radii are 1x, 2x, 3x this step (pixels) for auto_white_rings_from_clip.
    trace_white_ring_step_px: float = 20.0
    trace_seed_order_descending_from_anchor: bool = True

    routing_plane_default_id: str = "main"
    routing_planes: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "main": {
                "origin": [0.56, 0.0, 0.0],
                "normal": [-1.0, 0.0, 0.0],
                "u_axis": [0.0, 0.0, 1.0],
                "v_axis": [0.0, 1.0, 0.0],
            }
        }
    )
    clip_plane_assignments: Dict[int, str] = field(default_factory=dict)

    routing_height_above_plane_m: float = 0.025
    grasp_height_above_plane_m: float = 0.025
    grasp_min_clearance_from_first_peg_m: float = 0.05
    grasp_second_min_arc_from_first_grasp_m: float = 0.08

    pregrasp_offset_from_grasp_m: float = 0.08
    # Extra rotation R <- Rx(angle)*R for YZ routing plane (YuMi TCP vs cable/tangent frame).
    # Degrees about world +X; 0 disables. Tweak sign (±90) if the gripper is still twisted.
    grasp_extra_world_rx_deg: float = 90.0
    # Poses are computed in yumi_base_link; publish in ``cartesian_targets_world_frame_id``.
    publish_cartesian_targets_in_world_frame: bool = True
    cartesian_targets_world_frame_id: str = "world"
    # If True, use tf2 (can crash in some Python/GUI setups). If False, apply only the
    # translation below (no tf2 / no native heap issues).
    publish_cartesian_targets_use_tf: bool = False
    # Extra offset on publish (usually zero if ``world_from_pixel_z_offset_m`` is used).
    cartesian_targets_world_position_offset_m: Tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    detangle_offset_from_routing_m: float = 0.03

    first_route_primary_extra_along_route_px: float = 60.0

    c_clip_primary_lateral_px: float = 90.0
    c_clip_secondary_lateral_px: float = 70.0
    c_clip_primary_forward_px: float = 30.0
    c_clip_secondary_forward_px: float = -15.0
    c_clip_swap_sides_when_primary_right: bool = False
    c_clip_center_primary_lateral_px: float = 20.0
    c_clip_center_secondary_lateral_px: float = 45.0
    c_clip_center_primary_forward_px: float = 5.0
    c_clip_center_secondary_forward_px: float = -10.0

    cam_to_robot_left_trans_path: str = os.path.join(
        BASE_DIR, "configs/cameras/zed_to_world_left.tf"
    )
    cam_to_robot_right_trans_path: str = os.path.join(
        BASE_DIR, "configs/cameras/zed_to_world_right.tf"
    )

    board_calibration_yaml: str = os.path.join(
        BASE_DIR,
        "debug_gui",
        "configs",
        "cameras",
        "camera_robot_2d_calibration.yaml",
    )
    board_plane_x_m: float = 0.56
    # Added to Z for every point from ``world_from_pixel_debug`` (homography / pinhole).
    world_from_pixel_z_offset_m: float = 0.1

    dual_arm_grasp: bool = False
    single_arm_nominal_tcp_left_m: Tuple[float, float, float] = (0.35, 0.22, 0.14)
    single_arm_nominal_tcp_right_m: Tuple[float, float, float] = (0.35, -0.22, 0.14)
