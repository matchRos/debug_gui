import numpy as np


class PathProjectionService:
    def convert_path_to_world(self, env, path_pixels, arm="right"):
        """
        Convert pixel path to world coordinates using existing env functionality.
        """

        if path_pixels is None or len(path_pixels) == 0:
            raise RuntimeError("No pixel path available for projection.")

        if env is None:
            raise RuntimeError("Environment not initialized.")

        # Use existing functionality from original pipeline
        path_world = env.convert_path_to_world_coord(path_pixels, arm)

        return np.array(path_world)
