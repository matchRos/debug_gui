from typing import Dict

import numpy as np

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.debug_gui.backend.grasp_planning_service import (
    GraspPlanningService,
)


class GraspPlanningStep(BaseStep):
    name = "grasp_planning"
    description = "Sample two grasp points along cable for dual-arm grasping."

    def __init__(self):
        super().__init__()
        self.service = GraspPlanningService()

    def run(self, state: PipelineState) -> Dict[str, object]:
        if state.path_in_world is None:
            raise RuntimeError("No world path available.")

        if not hasattr(state, "path_tangents"):
            raise RuntimeError("No tangents available.")

        if state.path_in_pixels is None:
            raise RuntimeError("No pixel path available.")

        routing = state.routing
        clips = state.clips

        if routing is None or len(routing) < 2:
            raise RuntimeError("Routing not available or too short.")

        # Use the first routing object after the start object
        first_target_id = routing[1]
        first_clip = clips[first_target_id]

        target_px = np.array([first_clip.x, first_clip.y], dtype=float)
        path_px = [
            np.asarray(p).squeeze()[:2].astype(float) for p in state.path_in_pixels
        ]

        # 1) Find cable point closest to the first routing object in pixel space
        dists = [np.linalg.norm(p - target_px) for p in path_px]
        closest_idx = int(np.argmin(dists))

        # 2) First grasp: move backwards along cable from first routing object
        grasp_offset_px = state.config.grasp_offset_px
        arc_len = 0.0
        grasp_idx1 = closest_idx

        for i in range(closest_idx, 0, -1):
            seg_len = np.linalg.norm(path_px[i] - path_px[i - 1])
            arc_len += seg_len
            grasp_idx1 = i - 1

            if arc_len >= grasp_offset_px:
                break

        # 3) Second grasp: move forwards along cable from first grasp
        second_grasp_spacing_px = grasp_offset_px * 2.0
        arc_len2 = 0.0
        grasp_idx2 = grasp_idx1

        for i in range(grasp_idx1, len(path_px) - 1):
            seg_len = np.linalg.norm(path_px[i + 1] - path_px[i])
            arc_len2 += seg_len
            grasp_idx2 = i + 1

            if arc_len2 >= second_grasp_spacing_px:
                break

        pos1 = state.path_in_world[grasp_idx1]
        tan1 = state.path_tangents[grasp_idx1]

        pos2 = state.path_in_world[grasp_idx2]
        tan2 = state.path_tangents[grasp_idx2]

        grasps = [
            {
                "position": pos1,
                "tangent": tan1,
                "index": grasp_idx1,
            },
            {
                "position": pos2,
                "tangent": tan2,
                "index": grasp_idx2,
            },
        ]

        print(
            f"First clip: {first_clip.clip_id}, "
            f"closest_idx: {closest_idx}, "
            f"grasp1_idx: {grasp_idx1}, grasp2_idx: {grasp_idx2}, "
            f"grasp_offset_px: {grasp_offset_px}, "
            f"second_spacing_px: {second_grasp_spacing_px}"
        )

        state.grasps = grasps

        return {
            "grasps_available": True,
            "num_grasps": len(grasps),
            "grasp_indices": [g["index"] for g in grasps],
            "first_grasp": {
                "position": grasps[0]["position"].tolist(),
                "tangent": grasps[0]["tangent"].tolist(),
            },
            "second_grasp": {
                "position": grasps[1]["position"].tolist(),
                "tangent": grasps[1]["tangent"].tolist(),
            },
        }
