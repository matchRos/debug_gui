from dataclasses import dataclass


@dataclass
class DebugConfig:
    """
    Minimal config for the standalone debug pipeline.
    """

    board_cfg_path: str = "cable_routing/configs/board/board_config.json"
