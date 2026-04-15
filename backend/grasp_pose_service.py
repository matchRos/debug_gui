import numpy as np

from cable_routing.debug_gui.backend.planes import (
    point_at_plane_height,
    routing_plane_is_world_yz,
)


def _rotation_world_rx_deg(theta_deg: float) -> np.ndarray:
    """Right-handed rotation about world +X."""
    t = np.deg2rad(float(theta_deg))
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float
    )


class GraspPoseService:
    def compute_pose(
        self,
        position,
        tangent,
        plane,
        grasp_height_above_plane_m: float,
        extra_world_rx_deg: float = 0.0,
    ):
        """
        Build a right-handed tool frame in world coordinates.

        - Tool Z is anti-aligned with the routing plane normal (typical approach into
          the board from the robot side).
        - Tool X is aligned with the cable tangent projected into the board plane.
        """
        n = np.asarray(plane.normal, dtype=float).reshape(3)
        n /= np.linalg.norm(n) + 1e-8

        t = np.asarray(tangent, dtype=float).reshape(3)
        t_in = t - float(np.dot(t, n)) * n
        if np.linalg.norm(t_in) < 1e-6:
            t_in = np.asarray(plane.u_axis, dtype=float).reshape(3)
            t_in = t_in - float(np.dot(t_in, n)) * n
        t_in /= np.linalg.norm(t_in) + 1e-8

        x_axis = t_in
        z_axis = -n.copy()
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis) + 1e-8
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-8

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        approach_axis = -z_axis.copy()

        # YZ board: YuMi MoveIt TCP often expects a different in-plane axis than
        # (tangent=X, approach=Z). Rotate the whole tool frame about world +X so
        # approach (column 2) stays along ±world X; only Y/Z components of the
        # gripper twist change.
        if routing_plane_is_world_yz(plane) and abs(float(extra_world_rx_deg)) > 1e-6:
            Rx = _rotation_world_rx_deg(extra_world_rx_deg)
            R = Rx @ R
            approach_axis = -R[:, 2].copy()

        # Legacy horizontal-table grasp.
        if not routing_plane_is_world_yz(plane):
            z_world = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(x_axis, z_world)) > 0.95:
                z_world = np.array([0.0, 1.0, 0.0])
            y_legacy = np.cross(z_world, x_axis)
            y_legacy /= np.linalg.norm(y_legacy) + 1e-8
            z_legacy = np.cross(x_axis, y_legacy)
            z_legacy /= np.linalg.norm(z_legacy) + 1e-8
            approach_axis = -z_legacy.copy()
            z_legacy = -z_legacy
            y_legacy = -y_legacy
            x_rot = y_legacy
            y_rot = -x_axis
            R = np.stack([x_rot, y_rot, z_legacy], axis=1)
            approach_axis = -R[:, 2].copy()

        position = np.asarray(position).astype(float)
        position = point_at_plane_height(
            position,
            plane,
            grasp_height_above_plane_m,
        )

        return {
            "position": position,
            "rotation": R,
            "approach_axis": approach_axis,
        }

    def compute_grasp_poses(
        self,
        grasps,
        plane,
        grasp_height_above_plane_m: float,
        extra_world_rx_deg: float = 0.0,
    ):
        poses = []

        for g in grasps:
            pose = self.compute_pose(
                g["position"],
                g["tangent"],
                plane,
                grasp_height_above_plane_m,
                extra_world_rx_deg=extra_world_rx_deg,
            )

            pose["path_index"] = int(g["index"])
            # Cable direction in the board plane (for overlays); not always R[:,0] after Rx fix.
            tw = np.asarray(g["tangent"], dtype=float).reshape(3)
            n = np.asarray(plane.normal, dtype=float).reshape(3)
            n /= np.linalg.norm(n) + 1e-8
            tw = tw - float(np.dot(tw, n)) * n
            ln = np.linalg.norm(tw)
            if ln < 1e-6:
                u = np.asarray(plane.u_axis, dtype=float).reshape(3)
                tw = u - float(np.dot(u, n)) * n
                ln = np.linalg.norm(tw) + 1e-8
            pose["tangent_world"] = tw / ln

            poses.append(pose)

        return poses
