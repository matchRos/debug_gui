from typing import Dict

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.debug_gui.backend.grasp_pose_service import (
    GraspPoseService,
)


class GraspPoseStep(BaseStep):
    name = "grasp_pose"
    description = "Compute grasp poses (position + rotation)."

    def __init__(self):
        super().__init__()
        self.service = GraspPoseService()

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not hasattr(state, "grasps"):
            raise RuntimeError("No grasps available.")

        poses = self.service.compute_grasp_poses(state.grasps)

        # assign exactly one pose to each arm (dual-arm)
        if len(poses) != 2:
            raise RuntimeError("Dual-arm grasp requires exactly 2 grasp poses.")

        # keep original grasp order:
        # poses[0] = grasp near first peg
        # poses[1] = grasp further back along cable
        if poses[0]["position"][1] > poses[1]["position"][1]:
            poses[0]["arm"] = "left"
            poses[1]["arm"] = "right"
        else:
            poses[0]["arm"] = "right"
            poses[1]["arm"] = "left"

        state.grasp_poses = poses

        print("Assigned arms:", [p["arm"] for p in poses])

        return {
            "poses_available": True,
            "num_poses": len(poses),
            "first_pose_pos": poses[0]["position"].tolist(),
        }
