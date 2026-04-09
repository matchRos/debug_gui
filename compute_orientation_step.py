from typing import Dict

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.debug_gui.backend.cable_orientation_service import (
    CableOrientationService,
)


class ComputeOrientationStep(BaseStep):
    name = "compute_orientation"
    description = "Compute cable tangents along the path."

    def __init__(self):
        super().__init__()
        self.service = CableOrientationService()

    def run(self, state: PipelineState) -> Dict[str, object]:
        if state.path_in_world is None:
            raise RuntimeError("No world path available.")

        tangents = self.service.compute_tangents(state.path_in_world)

        state.path_tangents = tangents

        return {
            "tangents_available": True,
            "num_tangents": len(tangents),
            "first_tangent": tangents[0].tolist(),
        }
