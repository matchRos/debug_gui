from typing import Dict

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.debug_gui.backend.pregrasp_pose_service import (
    PreGraspPoseService,
)


class PreGraspPoseStep(BaseStep):
    name = "pregrasp_pose"
    description = "Compute pre-grasp poses above grasp poses."

    def __init__(self):
        super().__init__()
        self.service = PreGraspPoseService()

    def run(self, state: PipelineState) -> Dict[str, object]:
        if not hasattr(state, "grasp_poses"):
            raise RuntimeError("No grasp poses available.")

        pregrasp_poses = self.service.compute_pregrasp_poses(
            state.grasp_poses,
            offset=0.08,
        )

        state.pregrasp_poses = pregrasp_poses

        return {
            "pregrasp_poses_available": True,
            "num_pregrasp_poses": len(pregrasp_poses),
            "first_pregrasp_pos": pregrasp_poses[0]["position"].tolist(),
            "first_arm": pregrasp_poses[0]["arm"],
        }
