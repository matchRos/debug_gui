from typing import Dict
import numpy as np

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.debug_gui.backend.grasp_pose_service import (
    GraspPoseService,
)
from cable_routing.debug_gui.backend.planes import get_routing_plane, routing_plane_is_world_yz


class GraspPoseStep(BaseStep):
    name = "grasp_pose"
    description = "Compute grasp poses (position + rotation)."

    def __init__(self):
        super().__init__()
        self.service = GraspPoseService()

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not hasattr(state, "grasps"):
            raise RuntimeError("No grasps available.")

        plane = get_routing_plane(state.config)
        grasp_height = float(state.config.grasp_height_above_plane_m)
        extra_rx = float(
            getattr(state.config, "grasp_extra_world_rx_deg", 0.0),
        )
        poses = self.service.compute_grasp_poses(
            state.grasps,
            plane=plane,
            grasp_height_above_plane_m=grasp_height,
            extra_world_rx_deg=extra_rx,
        )

        dual = bool(getattr(state.config, "dual_arm_grasp", True))
        if dual:
            if len(poses) != 2:
                raise RuntimeError("Dual-arm grasp requires exactly 2 grasp poses.")
            # Assign arms by world Y (YuMi convention: left is +Y).
            if poses[0]["position"][1] > poses[1]["position"][1]:
                poses[0]["arm"] = "left"
                poses[1]["arm"] = "right"
            else:
                poses[0]["arm"] = "right"
                poses[1]["arm"] = "left"

            # Legacy horizontal-table bias for the left arm only.
            if not routing_plane_is_world_yz(plane):
                theta = np.deg2rad(180.0)
                Rz_bias = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                for pose in poses:
                    if pose["arm"] == "left":
                        pose["rotation"] = pose["rotation"] @ Rz_bias
        else:
            if len(poses) != 1:
                raise RuntimeError("Single-arm grasp requires exactly 1 grasp pose.")
            gpos = np.asarray(poses[0]["position"], dtype=float).reshape(3)
            left_nom = np.asarray(
                getattr(state.config, "single_arm_nominal_tcp_left_m", (0.35, 0.22, 0.14)),
                dtype=float,
            ).reshape(3)
            right_nom = np.asarray(
                getattr(
                    state.config, "single_arm_nominal_tcp_right_m", (0.35, -0.22, 0.14)
                ),
                dtype=float,
            ).reshape(3)
            d_l = float(np.linalg.norm(gpos - left_nom))
            d_r = float(np.linalg.norm(gpos - right_nom))
            poses[0]["arm"] = "left" if d_l <= d_r else "right"

        state.grasp_poses = poses

        print("Assigned arms:", [p["arm"] for p in poses])

        out: Dict[str, object] = {
            "poses_available": True,
            "num_poses": len(poses),
            "arms": [p["arm"] for p in poses],
        }
        out["first_pose_pos"] = poses[0]["position"].tolist()
        if len(poses) > 1:
            out["second_pose_pos"] = poses[1]["position"].tolist()
        return out
