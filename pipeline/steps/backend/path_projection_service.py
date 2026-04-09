import numpy as np

from cable_routing.env.ext_camera.utils.img_utils import (
    get_world_coord_from_pixel_coord,
)


class PathProjectionService:
    def convert_path_to_world(self, env, path_pixels, arm="right"):
        """
        Convert a pixel path to world coordinates using the camera intrinsics
        and the camera-to-base transform stored in the debug context.
        """

        if path_pixels is None or len(path_pixels) == 0:
            raise RuntimeError("No pixel path available for projection.")

        if env is None:
            raise RuntimeError("Environment not initialized.")

        if env.camera is None:
            raise RuntimeError("Camera not available in debug context.")

        if not hasattr(env, "T_CAM_BASE"):
            raise RuntimeError("DebugContext has no T_CAM_BASE transform.")

        if arm not in env.T_CAM_BASE:
            raise RuntimeError(f"Arm '{arm}' not found in T_CAM_BASE.")

        intrinsic = env.camera.intrinsic
        T_cam_base = env.T_CAM_BASE[arm]

        world_path = []
        for pixel_coord in path_pixels:
            pixel_coord = np.asarray(pixel_coord).squeeze().reshape(-1)
            pixel_xy = (int(pixel_coord[0]), int(pixel_coord[1]))

            world_coord = get_world_coord_from_pixel_coord(
                pixel_xy,
                intrinsic,
                T_cam_base,
            )
            world_path.append(world_coord)

        return np.array(world_path)
