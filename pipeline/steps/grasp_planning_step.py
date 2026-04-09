from typing import Dict
import numpy as np
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.debug_gui.backend.grasp_planning_service import (
    GraspPlanningService,
)


class GraspPlanningStep(BaseStep):
    name = "grasp_planning"
    description = "Sample grasp points along cable."

    def __init__(self):
        super().__init__()
        self.service = GraspPlanningService()

    def run(self, state: PipelineState) -> Dict[str, object]:
        if state.path_in_world is None:
            raise RuntimeError("No world path available.")

        if not hasattr(state, "path_tangents"):
            raise RuntimeError("No tangents available.")

        routing = state.routing
        clips = state.clips

        if routing is None or len(routing) < 2:
            raise RuntimeError("Routing not available or too short.")

        first_target_id = routing[1]
        first_clip = clips[first_target_id]

        target_px = np.array([first_clip.x, first_clip.y], dtype=float)

        path_px = [
            np.asarray(p).squeeze()[:2].astype(float) for p in state.path_in_pixels
        ]

        # find cable point closest to first routing object in pixel space
        dists = [np.linalg.norm(p - target_px) for p in path_px]
        closest_idx = int(np.argmin(dists))

        # grasp a bit before the first routing object along the traced cable
        grasp_idx = max(0, closest_idx - 10)

        pos = state.path_in_world[grasp_idx]
        tangent = state.path_tangents[grasp_idx]

        grasps = [
            {
                "position": pos,
                "tangent": tangent,
                "index": grasp_idx,
            }
        ]

        print(
            f"First clip: {first_clip.clip_id}, closest_idx: {closest_idx}, grasp_idx: {grasp_idx}"
        )

        state.grasps = grasps

        return {
            "grasps_available": True,
            "num_grasps": len(grasps),
            "first_grasp": {
                "position": grasps[0]["position"].tolist(),
                "tangent": grasps[0]["tangent"].tolist(),
            },
        }
