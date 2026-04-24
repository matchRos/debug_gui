from typing import Any, Mapping, Tuple

import numpy as np

from cable_routing.debug_gui.motion_primitives.c_clip import _clip_forward_axis_px


def build_u_clip_entry_pixels(
    curr_clip: Any,
    clip_type_config: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build clip-local approach targets for a U-clip.

    The primary arm is placed at the clip entrance (before the clip along the
    negative forward axis), while the secondary arm is placed behind the clip.
    """
    center = np.array([float(curr_clip.x), float(curr_clip.y)], dtype=float)
    forward = _clip_forward_axis_px(float(curr_clip.orientation))

    entry_offset = float(clip_type_config.get("entry_offset_px", 55.0))
    exit_offset = float(clip_type_config.get("exit_offset_px", 55.0))

    primary_px = center - forward * entry_offset
    secondary_px = center + forward * exit_offset

    return primary_px, secondary_px
