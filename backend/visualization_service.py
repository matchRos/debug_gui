import numpy as np
import cv2


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

    def draw_grasps(self, image, poses, intrinsic, T_cam_base, arm="right"):
        img = image.copy()

        for i, pose in enumerate(poses):
            pos = pose["position"]
            R = pose["rotation"]

            px = self.project_world_to_pixel(pos, intrinsic, T_cam_base)

            if px is None:
                continue

            u, v = px

            # draw point
            cv2.circle(img, (u, v), 6, (0, 0, 255), -1)

            # draw direction (x-axis)
            direction = R[:, 0]  # tangent
            tip = np.asarray(pos) + np.asarray(direction) * 0.05

            tip_px = self.project_world_to_pixel(tip, intrinsic, T_cam_base)

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
