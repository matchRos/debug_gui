import numpy as np


class GraspPlanningService:
    def sample_grasps(self, path_world, tangents, num_grasps=5):
        indices = np.linspace(0, len(path_world) - 1, num_grasps).astype(int)

        grasps = []

        for idx in indices:
            pos = path_world[idx]
            tangent = tangents[idx]

            grasps.append(
                {
                    "position": pos,
                    "tangent": tangent,
                    "index": int(idx),
                }
            )

        return grasps
