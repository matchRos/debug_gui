from typing import Dict

import numpy as np

from cable_routing.debug_gui.backend.board_projection import world_from_pixel_debug
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


def _first_routing_clip_world_xy(
    state: PipelineState, clip_index: int
) -> np.ndarray:
    """Board clip centre in world (3,) from pixel (x,y) on image."""
    if state.env is None:
        raise RuntimeError("Environment not available for peg world position.")
    clips = state.clips
    if clips is None:
        raise RuntimeError("clips not available.")
    clip = clips[clip_index]
    img_shape = state.rgb_image.shape if state.rgb_image is not None else None
    return world_from_pixel_debug(
        state.env,
        state.config,
        (float(clip.x), float(clip.y)),
        arm="right",
        is_clip=True,
        image_shape=img_shape,
    ).reshape(3)


def _first_path_index_clear_of_anchor(
    path_world: np.ndarray,
    anchor_world: np.ndarray,
    min_dist_m: float,
) -> int:
    """
    Walk the trace in order (index 0 -> end); return first index whose position is
    at least ``min_dist_m`` from ``anchor_world`` (Euclidean in base frame).
    """
    for i in range(len(path_world)):
        d = float(np.linalg.norm(path_world[i] - anchor_world))
        if d >= min_dist_m:
            return i
    return int(len(path_world) - 1)


def _path_index_after_arc_length(
    path_world: np.ndarray,
    start_idx: int,
    min_arc_m: float,
) -> int:
    """From ``start_idx``, walk forward along the polyline until arc length >= ``min_arc_m``."""
    if start_idx >= len(path_world) - 1:
        return int(len(path_world) - 1)
    arc = 0.0
    for i in range(start_idx, len(path_world) - 1):
        arc += float(np.linalg.norm(path_world[i + 1] - path_world[i]))
        if arc >= min_arc_m:
            return int(i + 1)
    return int(len(path_world) - 1)


class GraspPlanningStep(BaseStep):
    name = "grasp_planning"
    description = "Choose grasp point(s) along the traced cable in world frame."

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

        path_w = np.asarray(state.path_in_world, dtype=float)
        peg_id = int(routing[0])
        peg_world = _first_routing_clip_world_xy(state, peg_id)

        min_clear = float(
            getattr(state.config, "grasp_min_clearance_from_first_peg_m", 0.05)
        )
        grasp_idx1 = _first_path_index_clear_of_anchor(path_w, peg_world, min_clear)

        dual = bool(getattr(state.config, "dual_arm_grasp", True))
        if dual:
            second_arc = float(
                getattr(
                    state.config,
                    "grasp_second_min_arc_from_first_grasp_m",
                    0.08,
                )
            )
            grasp_idx2 = _path_index_after_arc_length(
                path_w, grasp_idx1, second_arc
            )
            if grasp_idx2 <= grasp_idx1 and len(path_w) > grasp_idx1 + 1:
                grasp_idx2 = min(len(path_w) - 1, grasp_idx1 + 1)

            pos1 = path_w[grasp_idx1]
            tan1 = state.path_tangents[grasp_idx1]
            pos2 = path_w[grasp_idx2]
            tan2 = state.path_tangents[grasp_idx2]
            grasps = [
                {"position": pos1, "tangent": tan1, "index": grasp_idx1},
                {"position": pos2, "tangent": tan2, "index": grasp_idx2},
            ]
            print(
                f"Peg clip index {peg_id} world (for clearance): {peg_world.tolist()}, "
                f"min_clearance_m={min_clear}, grasp1_idx={grasp_idx1}, grasp2_idx={grasp_idx2}"
            )
        else:
            pos1 = path_w[grasp_idx1]
            tan1 = state.path_tangents[grasp_idx1]
            grasps = [{"position": pos1, "tangent": tan1, "index": grasp_idx1}]
            print(
                f"Peg clip index {peg_id} world (for clearance): {peg_world.tolist()}, "
                f"min_clearance_m={min_clear}, single_grasp_idx={grasp_idx1}"
            )

        state.grasps = grasps

        out = {
            "grasps_available": True,
            "num_grasps": len(grasps),
            "grasp_indices": [g["index"] for g in grasps],
            "peg_world_for_grasp_clearance": peg_world.tolist(),
            "first_grasp": {
                "position": grasps[0]["position"].tolist(),
                "tangent": grasps[0]["tangent"].tolist(),
            },
        }
        if len(grasps) > 1:
            out["second_grasp"] = {
                "position": grasps[1]["position"].tolist(),
                "tangent": grasps[1]["tangent"].tolist(),
            }
        return out
