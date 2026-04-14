from typing import Dict
import numpy as np

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.debug_gui.backend.grasp_pose_service import (
    GraspPoseService,
)
from cable_routing.debug_gui.backend.planes import get_routing_plane


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
        poses = self.service.compute_grasp_poses(
            state.grasps,
            plane=plane,
            grasp_height_above_plane_m=grasp_height,
        )

        if len(poses) != 2:
            raise RuntimeError("Dual-arm grasp requires exactly 2 grasp poses.")

        # Assign arms exactly as before
        if poses[0]["position"][1] > poses[1]["position"][1]:
            poses[0]["arm"] = "left"
            poses[1]["arm"] = "right"
        else:
            poses[0]["arm"] = "right"
            poses[1]["arm"] = "left"

        # Rotate LEFT arm tool by +90 deg around its local z-axis
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
        state.grasp_poses = poses

        print("Assigned arms:", [p["arm"] for p in poses])

        return {
            "poses_available": True,
            "num_poses": len(poses),
            "first_pose_pos": poses[0]["position"].tolist(),
            "second_pose_pos": poses[1]["position"].tolist(),
            "arms": [p["arm"] for p in poses],
        }
