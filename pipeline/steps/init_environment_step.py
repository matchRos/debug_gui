from typing import Any, Dict, Optional

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

    def _try_create_board(self, config: DebugConfig) -> Optional[Any]:
        """
        Create the new standalone debug board.
        """
        try:
            return DebugBoard(config_path=config.board_cfg_path)
        except Exception:
            return None

    def _try_create_camera(self) -> Optional[Any]:
        """
        Camera stays optional for now.
        """
        try:
            from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber

            return ZedCameraSubscriber()
        except Exception:
            return None

    def _try_create_tracer(self) -> Optional[Any]:
        """
        Tracer stays optional for now.
        """
        try:
            from cable_routing.handloom.handloom_pipeline.single_tracer import (
                CableTracer,
            )

            return CableTracer()
        except Exception:
            return None

    def _safe_get_camera_image(self, camera: Any) -> Optional[np.ndarray]:
        if camera is None:
            return None

        candidate_methods = [
            "get_rgb_image",
            "get_image",
            "get_rgb",
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
        board = self._try_create_board(config)
        camera = self._try_create_camera()
        tracer = self._try_create_tracer()

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
        }

        return result
