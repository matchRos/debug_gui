from typing import Dict

from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState
from cable_routing.debug_gui.backend.path_projection_service import (
    PathProjectionService,
)


class TraceToWorldStep(BaseStep):
    name = "trace_to_world"
    description = "Convert traced pixel path to world coordinates."

    def __init__(self):
        super().__init__()
        self.service = PathProjectionService()

    def run(self, state: PipelineState) -> Dict[str, object]:
        if state.env is None:
            raise RuntimeError("Environment not initialized.")

        if state.path_in_pixels is None:
            raise RuntimeError("No pixel path available. Run trace_cable first.")

        # For now: fixed arm (can later come from config)
        arm = "right"

        path_world = self.service.convert_path_to_world(
            env=state.env,
            path_pixels=state.path_in_pixels,
            arm=arm,
        )

        state.path_in_world = path_world

        return {
            "path_world_available": True,
            "num_world_points": len(path_world),
            "first_point": path_world[0].tolist(),
            "last_point": path_world[-1].tolist(),
            "arm": arm,
        }
