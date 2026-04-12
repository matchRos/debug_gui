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

        import numpy as np

        first_target_id = routing[1]
        first_clip = clips[first_target_id]

        target_px = np.array([first_clip.x, first_clip.y], dtype=float)
        path_px = [
            np.asarray(p).squeeze()[:2].astype(float) for p in state.path_in_pixels
        ]

        # 1) find cable point closest to first routing clip in pixel space
        dists = [np.linalg.norm(p - target_px) for p in path_px]
        closest_idx = int(np.argmin(dists))

        # 2) walk backwards along the cable and integrate arc length
        grasp_offset_px = state.config.grasp_offset_px
        arc_len = 0.0
        grasp_idx = closest_idx

        for i in range(closest_idx, 0, -1):
            seg_len = np.linalg.norm(path_px[i] - path_px[i - 1])
            arc_len += seg_len
            grasp_idx = i - 1

            if arc_len >= grasp_offset_px:
                break

                # --- FIRST GRASP (near first routing object) ---
        pos1 = state.path_in_world[grasp_idx]
        tan1 = state.path_tangents[grasp_idx]

        # --- SECOND GRASP (further back along cable) ---
        second_offset_px = 2.0 * grasp_offset_px
        arc_len2 = 0.0
        grasp_idx2 = grasp_idx

        for i in range(grasp_idx, 0, -1):
            seg_len = np.linalg.norm(path_px[i] - path_px[i - 1])
            arc_len2 += seg_len
            grasp_idx2 = i - 1

            if arc_len2 >= second_offset_px:
                break

        pos2 = state.path_in_world[grasp_idx2]
        tan2 = state.path_tangents[grasp_idx2]

        grasps = [
            {
                "position": pos1,
                "tangent": tan1,
                "index": grasp_idx,
            },
            {
                "position": pos2,
                "tangent": tan2,
                "index": grasp_idx2,
            },
        ]

        print(
            f"First clip: {first_clip.clip_id}, "
            f"closest_idx: {closest_idx}, grasp_idx: {grasp_idx}, "
            f"arc_len_px: {arc_len:.1f}"
        )

        state.grasps = grasps

        return {
            "grasps_available": True,
            "num_grasps": len(grasps),
            "grasp_indices": [g["index"] for g in grasps],
        }
