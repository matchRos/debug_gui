from dataclasses import dataclass


@dataclass
class DebugConfig:
    """
    Minimal standalone config for the debug GUI pipeline.
    """

    board_cfg_path: str = "cable_routing/configs/board/board_config.json"
    default_routing: tuple = (0, 1, 2, 3)
    fallback_image_width: int = 1500
    fallback_image_height: int = 800
