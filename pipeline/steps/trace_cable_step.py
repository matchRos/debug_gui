from typing import Dict, List, Optional, Tuple

from cable_routing.debug_gui.backend.tracing_service import TracingService
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class TraceCableStep(BaseStep):
    name = "trace_cable"
    description = "Acquire image, run tracer if available, and visualize the result."

    def __init__(self) -> None:
        super().__init__()
        self.tracing_service = TracingService()

    def _resolve_start_points(self, state: PipelineState) -> List[Tuple[int, int]]:
        if state.config is not None and hasattr(state.config, "trace_start_points"):
            return [tuple(p) for p in state.config.trace_start_points]
        raise RuntimeError("No trace_start_points configured.")

    def _resolve_end_points(
        self, state: PipelineState
    ) -> Optional[List[Tuple[int, int]]]:
        if state.config is not None and hasattr(state.config, "trace_end_points"):
            points = state.config.trace_end_points
            if points is None:
                return None
            return [tuple(p) for p in points]
        return None

    def run(self, state: PipelineState) -> Dict[str, object]:
        if state.env is None:
            raise RuntimeError(
                "Debug context not initialized. Run init_environment first."
            )

        image_rgb, image_source = self.tracing_service.acquire_image(
            camera=state.env.camera,
            fallback_image_path=state.config.debug_image_path,
        )

        if image_rgb is None:
            raise RuntimeError(
                "No image available. Neither camera nor debug_image_path provided a valid image."
            )

        state.rgb_image = image_rgb

        start_points = self._resolve_start_points(state)
        end_points = self._resolve_end_points(state)

        if state.env.tracer is None:
            overlay = self.tracing_service.create_no_trace_overlay(
                image_rgb=image_rgb,
                start_points=start_points,
                end_points=end_points,
                message="Tracer unavailable - showing only start/end points",
            )
            state.trace_overlay = overlay
            state.path_in_pixels = None
            state.path_in_world = None
            state.cable_orientations = None

            return {
                "image_source": image_source,
                "trace_executed": False,
                "reason": "tracer_unavailable",
                "start_points": start_points,
                "end_points": end_points,
            }

        trace_result = self.tracing_service.run_trace(
            tracer=state.env.tracer,
            image_rgb=image_rgb,
            start_points=start_points,
            end_points=end_points,
            viz=False,
        )

        path_in_pixels = trace_result["path_in_pixels"]
        trace_status = trace_result["trace_status"]

        overlay = self.tracing_service.create_trace_overlay(
            image_rgb=image_rgb,
            start_points=start_points,
            end_points=end_points,
            path_in_pixels=path_in_pixels,
        )

        state.trace_overlay = overlay
        state.path_in_pixels = path_in_pixels
        state.path_in_world = None
        state.cable_orientations = None

        num_points = 0 if path_in_pixels is None else len(path_in_pixels)

        return {
            "image_source": image_source,
            "trace_executed": True,
            "trace_status": str(trace_status),
            "num_path_points": num_points,
            "start_points": start_points,
            "end_points": end_points,
        }
