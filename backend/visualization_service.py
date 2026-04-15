import numpy as np
import cv2

from cable_routing.debug_gui.backend.board_projection import pixel_from_world_debug


class VisualizationService:
    def project_world_to_pixel(self, point, intrinsic, T_cam_base):
        # base -> cam
        T_base_cam = T_cam_base.inverse()

        point = np.asarray(point).reshape(3)
        R = T_base_cam.rotation
        t = T_base_cam.translation

        point_cam = R @ point + t
        x, y, z = point_cam

        if z <= 0:
            return None

        K = intrinsic._K

        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]

        return int(u), int(v)

    def draw_grasps(self, image, poses, env, config, arm="right"):
        img = image.copy()
        intrinsic = env.camera.intrinsic if env.camera is not None else None
        T_cam_base = None
        if hasattr(env, "T_CAM_BASE") and env.T_CAM_BASE and arm in env.T_CAM_BASE:
            T_cam_base = env.T_CAM_BASE[arm]

        for i, pose in enumerate(poses):
            pos = pose["position"]
            R = pose["rotation"]

            px = pixel_from_world_debug(
                env,
                config,
                np.asarray(pos, dtype=float),
                arm=arm,
                intrinsic=intrinsic,
                T_cam_base=T_cam_base,
            )

            if px is None:
                continue

            u, v = px

            # draw point
            cv2.circle(img, (u, v), 6, (0, 0, 255), -1)

            # Cable direction in plane (stable); after grasp_extra_world_rx_deg, R[:,0]
            # may not match the drawn tangent.
            direction = pose.get("tangent_world")
            if direction is None:
                direction = R[:, 0]
            else:
                direction = np.asarray(direction, dtype=float).reshape(3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            tip = np.asarray(pos) + np.asarray(direction) * 0.05

            tip_px = pixel_from_world_debug(
                env,
                config,
                np.asarray(tip, dtype=float),
                arm=arm,
                intrinsic=intrinsic,
                T_cam_base=T_cam_base,
            )

            if tip_px is not None:
                cv2.arrowedLine(img, (u, v), tip_px, (255, 0, 0), 2)

            # label
            cv2.putText(
                img,
                f"{pose.get('arm', arm)}_{i}",
                (u + 5, v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return img
