import numpy as np


class CableOrientationService:
    def compute_tangents(self, path_world):
        tangents = []

        for i in range(len(path_world)):
            if i == 0:
                t = path_world[1] - path_world[0]
            elif i == len(path_world) - 1:
                t = path_world[-1] - path_world[-2]
            else:
                t = path_world[i + 1] - path_world[i - 1]

            t = t / (np.linalg.norm(t) + 1e-8)
            tangents.append(t)

        return np.array(tangents)
