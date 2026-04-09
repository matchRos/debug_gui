from typing import Any, Dict, Optional, Tuple

import numpy as np

from cable_routing.debug_gui.backend.debug_board import DebugBoard
from cable_routing.debug_gui.backend.debug_context import DebugContext
from cable_routing.debug_gui.config.debug_config import DebugConfig
from cable_routing.debug_gui.pipeline.base_step import BaseStep
from cable_routing.debug_gui.pipeline.state import PipelineState


class InitEnvironmentStep(BaseStep):
    name = "init_environment"
    description = (
        "Initialize debug config, standalone debug board, and optional camera preview."
    )

    def __init__(self) -> None:
        super().__init__()

    def _create_debug_config(self) -> DebugConfig:
        return DebugConfig()

    def _try_create_board(
        self, config: DebugConfig
    ) -> Tuple[Optional[Any], Optional[str]]:
        try:
            return DebugBoard(config_path=config.board_cfg_path), None
        except Exception as exc:
            return None, str(exc)

    def _try_create_camera(self) -> Tuple[Optional[Any], Optional[str]]:
        try:
            import rospy
            from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber

            if not rospy.core.is_initialized():
                rospy.init_node(
                    "debug_gui_camera", anonymous=True, disable_signals=True
                )
            rospy.sleep(1.0)
            return ZedCameraSubscriber(), None
        except Exception as exc:
            return None, str(exc)

    def _try_create_tracer(self) -> Tuple[Optional[Any], Optional[str]]:
        try:
            from cable_routing.handloom.handloom_pipeline.single_tracer import (
                CableTracer,
            )

            return CableTracer(), None
        except Exception as exc:
            return None, str(exc)

    def _safe_get_camera_image(self, camera: Any) -> Optional[np.ndarray]:
        if camera is None:
            return None

        candidate_methods = [
            "get_rgb",
            "get_rgb_image",
            "get_image",
            "get_frame",
            "read",
        ]

        for method_name in candidate_methods:
            if hasattr(camera, method_name):
                try:
                    result = getattr(camera, method_name)()
                    if isinstance(result, np.ndarray):
                        return result
                except Exception:
                    pass

        return None

    def run(self, state: PipelineState) -> Dict[str, Any]:
        config = self._create_debug_config()

        board, board_error = self._try_create_board(config)
        camera, camera_error = self._try_create_camera()
        tracer, tracer_error = self._try_create_tracer()

        context = DebugContext(
            config=config,
            robot=None,
            camera=camera,
            board=board,
            tracer=tracer,
        )

        state.config = config
        state.env = context

        preview = self._safe_get_camera_image(camera)
        if preview is not None:
            state.rgb_image = preview

        result = {
            "config_loaded": config is not None,
            "board_loaded": board is not None,
            "camera_loaded": camera is not None,
            "tracer_loaded": tracer is not None,
            "robot_loaded": False,
            "camera_preview_available": preview is not None,
            "board_cfg_path": config.board_cfg_path,
            "board_num_clips": board.num_clips() if board is not None else 0,
            "board_error": board_error,
            "camera_error": camera_error,
            "tracer_error": tracer_error,
            "debug_image_path": config.debug_image_path,
            "trace_start_points": config.trace_start_points,
            "trace_end_points": config.trace_end_points,
        }

        return result
