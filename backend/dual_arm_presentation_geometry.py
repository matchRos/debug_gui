"""
Tool orientations for dual-arm cable presentation (world frame).

- Carrier: cable along -world Z, gripper perpendicular to routing plane (tool Z ~ -n).
- Second arm (side grasp): tool Z along ±world Y (+Y if second arm is right, -Y if left),
  cable still along -Z in the tool frame (column 0).
"""

from __future__ import annotations

import numpy as np

from cable_routing.debug_gui.backend.planes import RoutingPlane


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("zero vector")
    return v / n


def rotation_carrier_cable_vertical_world(plane: RoutingPlane) -> np.ndarray:
    """
    Right-handed tool matrix R (columns = tool X,Y,Z in world).

    - Tool X: cable exits along **-world Z** (down).
    - Tool Z: **-plane.normal** (into the board; gripper normal to board plane).
    - Tool Y: completes right-handed frame.
    """
    n = _unit(np.asarray(plane.normal, dtype=float).reshape(3))
    x_axis = np.array([0.0, 0.0, -1.0], dtype=float)
    z_axis = -n
    if abs(float(np.dot(x_axis, z_axis))) > 0.995:
        raise RuntimeError(
            "Cable-down axis nearly parallel to board normal; check routing plane."
        )
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _unit(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = _unit(z_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def rotation_world_ry_deg(theta_deg: float) -> np.ndarray:
    """Right-handed rotation about world +Y (radians internally)."""
    t = np.deg2rad(float(theta_deg))
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rotation_second_arm_side_grasp_world(second_arm_is_right: bool) -> np.ndarray:
    """
    Side approach: tool Z along **+world Y** (right second arm) or **-world Y** (left),
    tool X along **-world Z** (cable direction through fingers).
    """
    x_axis = np.array([0.0, 0.0, -1.0], dtype=float)
    if second_arm_is_right:
        z_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        z_axis = np.array([0.0, -1.0, 0.0], dtype=float)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _unit(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = _unit(z_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)
