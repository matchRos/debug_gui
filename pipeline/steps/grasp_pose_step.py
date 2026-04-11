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

        # assign arm based on y-position
        for pose in poses:
            y = pose["position"][1]

            if y > 0:
                pose["arm"] = "left"
            else:
                pose["arm"] = "right"

        state.grasp_poses = poses

        print("Assigned arms:", [p["arm"] for p in poses])

        return {
            "poses_available": True,
            "num_poses": len(poses),
            "first_pose_pos": poses[0]["position"].tolist(),
        }
