from typing import Any, Dict, List, Optional

from cable_routing.debug_gui.backend.board_service import BoardService
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class PrepareRoutingStep(BaseStep):
    name = "prepare_routing"
    description = "Load clips, define routing, and build a routing overlay."

    def __init__(self, routing: Optional[List[int]] = None) -> None:
        super().__init__()
        self.board_service = BoardService()
        self.default_routing = routing or [0, 1, 2, 3]

    def _resolve_routing(self, state: PipelineState) -> List[int]:
        """
        Routing priority:
        1. Existing state.routing
        2. default routing passed into step
        """
        if state.routing is not None:
            return state.routing
        return list(self.default_routing)

    def run(self, state: PipelineState) -> Dict[str, Any]:
        if state.env is None:
            raise RuntimeError(
                "Debug context not initialized. Run init_environment first."
            )

        if state.env.board is None:
            raise RuntimeError("Board is not available in debug context.")

        routing = self._resolve_routing(state)
        debug_data = self.board_service.prepare_routing_debug_data(
            board=state.env.board,
            routing=routing,
        )

        state.routing = routing
        state.clips = debug_data["clips"]
        state.crossing_fixture_id_list = debug_data["crossing_fixture_id_list"]
        state.routing_overlay = debug_data["routing_overlay"]

        return {
            "routing": routing,
            "num_clips": debug_data["num_clips"],
            "num_crossing_fixtures": len(debug_data["crossing_fixture_id_list"]),
        }
