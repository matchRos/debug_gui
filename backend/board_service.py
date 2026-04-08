from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class BoardService:
    """
    Utility methods for preparing routing overlays and clip-related debug output.
    """

    def __init__(self) -> None:
        pass

    def get_clips(self, board: Any) -> Any:
        """
        Return clips from the board object.
        """
        return board.get_clips()

    def create_base_board_image(
        self,
        board: Any,
        fallback_shape: Tuple[int, int, int] = (720, 1280, 3),
    ) -> np.ndarray:
        """
        Try to get a board image from the board object. If that is not possible,
        create a blank fallback canvas.
        """
        possible_attrs = [
            "board_image",
            "img",
            "image",
            "base_img",
            "base_image",
        ]

        for attr in possible_attrs:
            if hasattr(board, attr):
                value = getattr(board, attr)
                if isinstance(value, np.ndarray):
                    image = value.copy()
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    return image

        return np.zeros(fallback_shape, dtype=np.uint8)

    def extract_clip_centers(self, clips: Any) -> List[Tuple[int, int]]:
        """
        Best-effort extraction of clip center coordinates.

        This method is intentionally defensive because the exact board/clip data
        structure may differ.
        """
        centers: List[Tuple[int, int]] = []

        if clips is None:
            return centers

        for clip in clips:
            center = self._extract_single_clip_center(clip)
            if center is not None:
                centers.append(center)

        return centers

    def _extract_single_clip_center(self, clip: Any) -> Optional[Tuple[int, int]]:
        """
        Best-effort extraction of one clip center.
        """
        candidate_keys = [
            ("x", "y"),
            ("cx", "cy"),
            ("center_x", "center_y"),
            ("px", "py"),
        ]

        # dict-like
        if isinstance(clip, dict):
            for key_x, key_y in candidate_keys:
                if key_x in clip and key_y in clip:
                    return int(clip[key_x]), int(clip[key_y])

            if "center" in clip:
                center = clip["center"]
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    return int(center[0]), int(center[1])

        # object-like
        for key_x, key_y in candidate_keys:
            if hasattr(clip, key_x) and hasattr(clip, key_y):
                return int(getattr(clip, key_x)), int(getattr(clip, key_y))

        if hasattr(clip, "center"):
            center = getattr(clip, "center")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                return int(center[0]), int(center[1])

        return None

    def draw_clip_centers(
        self,
        image: np.ndarray,
        clip_centers: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Draw all clip centers with numeric labels.
        """
        out = image.copy()
        for idx, (x, y) in enumerate(clip_centers):
            cv2.circle(out, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(
                out,
                str(idx),
                (x + 12, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return out

    def draw_routing_path(
        self,
        image: np.ndarray,
        clip_centers: List[Tuple[int, int]],
        routing: List[int],
    ) -> np.ndarray:
        """
        Draw routing arrows between clips.
        """
        out = image.copy()

        if routing is None or len(routing) < 2:
            return out

        for i in range(len(routing) - 1):
            idx_a = routing[i]
            idx_b = routing[i + 1]

            if idx_a >= len(clip_centers) or idx_b >= len(clip_centers):
                continue

            pt1 = clip_centers[idx_a]
            pt2 = clip_centers[idx_b]

            cv2.arrowedLine(
                out,
                pt1,
                pt2,
                (255, 0, 0),
                3,
                tipLength=0.04,
            )

        return out

    def estimate_crossing_fixture_ids(
        self,
        routing: List[int],
    ) -> List[int]:
        """
        Placeholder for crossing fixture estimation.

        For now this returns an empty list to keep the new debug pipeline simple.
        Later we can replace it with logic ported from the old code.
        """
        return []

    def prepare_routing_debug_data(
        self,
        board: Any,
        routing: List[int],
    ) -> Dict[str, Any]:
        """
        High-level helper for the prepare_routing step.
        """
        clips = self.get_clips(board)
        clip_centers = self.extract_clip_centers(clips)

        base_image = self.create_base_board_image(board)
        overlay = self.draw_clip_centers(base_image, clip_centers)
        overlay = self.draw_routing_path(overlay, clip_centers, routing)

        crossing_fixture_id_list = self.estimate_crossing_fixture_ids(routing)

        return {
            "clips": clips,
            "clip_centers": clip_centers,
            "routing_overlay": overlay,
            "crossing_fixture_id_list": crossing_fixture_id_list,
            "num_clips": len(clip_centers),
        }
