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
        anchor_point = None
        clip_points = None
        preferred_direction_xy = None
        configured_clip_positions = None

        try:
            board = getattr(state.env, "board", None)
            if board is not None and hasattr(board, "get_clips"):
                configured_clip_positions = []
                for clip in board.get_clips():
                    configured_clip_positions.append(
                        (str(clip.clip_id), int(clip.x), int(clip.y))
                    )
        except Exception:
            configured_clip_positions = None
        if state.clips is not None and len(state.clips) > 0:
            try:
                clip_idx = 0
                if state.routing is not None and len(state.routing) > 0:
                    clip_idx = int(state.routing[0])
                clip = state.clips[clip_idx]
                anchor_point = (int(clip.x), int(clip.y))
                clip_points = [(int(c.x), int(c.y)) for c in state.clips]
                if state.routing is not None and len(state.routing) > 1:
                    nxt = state.clips[int(state.routing[1])]
                    preferred_direction_xy = (
                        float(nxt.x) - float(clip.x),
                        float(nxt.y) - float(clip.y),
                    )
            except Exception:
                anchor_point = None
                clip_points = None
                preferred_direction_xy = None

        if state.env.tracer is None:
            overlay = self.tracing_service.create_no_trace_overlay(
                image_rgb=image_rgb,
                start_points=start_points,
                end_points=end_points,
                message="Tracer unavailable - showing only start/end points",
                configured_clip_positions=configured_clip_positions,
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
            start_mode=getattr(state.config, "trace_start_mode", "auto_from_config"),
            anchor_point=anchor_point,
            clip_points=clip_points,
            preferred_direction_xy=preferred_direction_xy,
            max_start_dist_px=float(
                getattr(state.config, "trace_anchor_max_start_dist_px", 260.0)
            ),
            min_route_dot=float(
                getattr(state.config, "trace_candidate_min_route_dot", -0.15)
            ),
            outward_min_delta_px=float(
                getattr(state.config, "trace_anchor_outward_min_delta_px", 8.0)
            ),
            seed_order_descending_from_anchor=bool(
                getattr(state.config, "trace_seed_order_descending_from_anchor", True)
            ),
            clip_a_p1_offset_px=float(
                getattr(state.config, "trace_auto_clip_a_p1_offset_px", 20.0)
            ),
            clip_a_p2_offset_px=float(
                getattr(state.config, "trace_auto_clip_a_p2_offset_px", 40.0)
            ),
        )

        path_in_pixels = trace_result["path_in_pixels"]
        trace_status = trace_result["trace_status"]
        tracer_start_points_used = trace_result.get("tracer_start_points_used")
        tracer_start_point_count = int(trace_result.get("tracer_start_point_count", 0))

        overlay = self.tracing_service.create_trace_overlay(
            image_rgb=image_rgb,
            start_points=start_points,
            end_points=end_points,
            path_in_pixels=path_in_pixels,
            tracer_start_points_used=tracer_start_points_used,
            configured_clip_positions=configured_clip_positions,
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
            "tracer_start_point_count": tracer_start_point_count,
            "tracer_start_points_used": tracer_start_points_used,
            "end_points": end_points,
        }
