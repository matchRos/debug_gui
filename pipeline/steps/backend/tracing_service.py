from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import traceback


def pick_two_points_on_image(image):
    import cv2

    picked = []
    vis = image.copy()

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 2:
            picked.append((x, y))
            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Pick 2 trace start points", vis)

    cv2.imshow("Pick 2 trace start points", vis)
    cv2.setMouseCallback("Pick 2 trace start points", cb)

    while len(picked) < 2:
        cv2.waitKey(10)

    cv2.destroyWindow("Pick 2 trace start points")
    return picked


def build_three_start_points_from_two_clicks(image_rgb, pt1_xy, pt2_xy):
    pt1_xy = snap_to_bright_pixel(image_rgb, pt1_xy)
    pt2_xy = snap_to_bright_pixel(image_rgb, pt2_xy)

    mid_xy = (
        int(round((pt1_xy[0] + pt2_xy[0]) / 2)),
        int(round((pt1_xy[1] + pt2_xy[1]) / 2)),
    )
    mid_xy = snap_to_bright_pixel(image_rgb, mid_xy)

    # tracer expects (y, x)
    pt1_yx = (pt1_xy[1], pt1_xy[0])
    mid_yx = (mid_xy[1], mid_xy[0])
    pt2_yx = (pt2_xy[1], pt2_xy[0])

    return [pt1_yx, mid_yx, pt2_yx]


def build_three_start_points_from_start_and_direction(
    image_rgb,
    start_xy,
    direction_xy,
    step_px=10,
):
    import numpy as np

    start_xy = snap_to_bright_pixel(image_rgb, start_xy)
    direction_xy = snap_to_bright_pixel(image_rgb, direction_xy)

    start = np.array(start_xy, dtype=float)
    direction_point = np.array(direction_xy, dtype=float)

    direction = direction_point - start
    norm = np.linalg.norm(direction)

    if norm < 1e-6:
        # fallback: arbitrary horizontal direction
        direction = np.array([1.0, 0.0], dtype=float)
        norm = 1.0

    direction = direction / norm

    p0_xy = start
    p1_xy = start + direction * step_px
    p2_xy = start + direction * (2 * step_px)

    # optional: snap generated points back to bright cable pixels
    p0_xy = snap_to_bright_pixel(
        image_rgb, (int(round(p0_xy[0])), int(round(p0_xy[1])))
    )
    p1_xy = snap_to_bright_pixel(
        image_rgb, (int(round(p1_xy[0])), int(round(p1_xy[1])))
    )
    p2_xy = snap_to_bright_pixel(
        image_rgb, (int(round(p2_xy[0])), int(round(p2_xy[1])))
    )

    # tracer expects (y, x)
    return [
        (p0_xy[1], p0_xy[0]),
        (p1_xy[1], p1_xy[0]),
        (p2_xy[1], p2_xy[0]),
    ]


def snap_to_bright_pixel(image, pt, radius=15):
    import cv2
    import numpy as np

    x, y = pt
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    y0 = max(0, y - radius)
    y1 = min(gray.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(gray.shape[1], x + radius + 1)

    patch = gray[y0:y1, x0:x1]
    ys, xs = np.where(patch > 150)

    if len(xs) == 0:
        return pt

    d2 = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
    i = np.argmin(d2)

    return (int(xs[i] + x0), int(ys[i] + y0))


class TracingService:
    """
    Standalone helper for image acquisition, trace execution, and visualization.
    """

    def __init__(self) -> None:
        pass

    def get_image_from_camera(self, camera: Any) -> Optional[np.ndarray]:
        """
        Try common camera getter names.
        """
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
                        return self._ensure_rgb_uint8(result)
                except Exception:
                    pass

        return None

    def load_image_from_disk(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an RGB image from disk.
        """
        if not image_path:
            return None

        path = Path(image_path)
        if not path.exists():
            return None

        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

    def acquire_image(
        self,
        camera: Any = None,
        fallback_image_path: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Acquire an image from camera first, otherwise from disk.
        """
        image = self.get_image_from_camera(camera)
        if image is not None:
            return image, "camera"

        image = self.load_image_from_disk(fallback_image_path or "")
        if image is not None:
            return image, "disk"

        return None, "none"

    def run_trace(
        self,
        tracer: Any,
        image_rgb: np.ndarray,
        start_points: List[Tuple[int, int]],
        end_points: Optional[List[Tuple[int, int]]] = None,
        viz: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the legacy CableTracer wrapper if available.
        """
        if tracer is None:
            raise RuntimeError("Tracer object is not available.")

        # path, status = tracer.trace(
        #     img=image_rgb,
        #     start_points=start_points,
        #     end_points=end_points,
        #     viz=viz,
        # )

        picked_xy = pick_two_points_on_image(image_rgb)
        print("clicked points (x,y):", picked_xy)

        start_points = build_three_start_points_from_start_and_direction(
            image_rgb,
            picked_xy[0],  # first click = start
            picked_xy[1],  # second click = direction
            step_px=20,
        )

        print("generated tracer start points (y,x):", start_points)

        try:
            result = tracer.trace(
                img=image_rgb,
                start_points=start_points,
                end_points=end_points,
                viz=viz,
            )

            if result is None:
                raise RuntimeError(
                    f"Tracing failed. The start point is likely not on the cable or the analytic tracer could not initialize from start_points={start_points}."
                )

            path, status = result
        except Exception as e:
            print("\n=== TRACE ERROR ===")
            print(f"type: {type(e).__name__}")
            print(f"message: {e}")
            print(f"start_points type: {type(start_points)}")
            print(f"start_points value: {start_points}")
            if start_points is not None:
                for i, p in enumerate(start_points):
                    try:
                        arr = np.asarray(p)
                        print(
                            f"start_points[{i}] -> type={type(p)}, shape={arr.shape}, value={p}"
                        )
                    except Exception:
                        print(f"start_points[{i}] -> type={type(p)}, value={p}")
            print(f"end_points type: {type(end_points)}")
            print(f"end_points value: {end_points}")
            if end_points is not None:
                for i, p in enumerate(end_points):
                    try:
                        arr = np.asarray(p)
                        print(
                            f"end_points[{i}] -> type={type(p)}, shape={arr.shape}, value={p}"
                        )
                    except Exception:
                        print(f"end_points[{i}] -> type={type(p)}, value={p}")
            traceback.print_exc()
            raise

        return {
            "path_in_pixels": path,
            "trace_status": status,
        }

    def create_trace_overlay(
        self,
        image_rgb: np.ndarray,
        start_points: List[Tuple[int, int]],
        end_points: Optional[List[Tuple[int, int]]] = None,
        path_in_pixels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draw start/end points and traced path onto the image.
        """
        overlay = image_rgb.copy()

        # Start points in green
        for idx, pt in enumerate(start_points):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(
                overlay,
                f"S{idx}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # End points in red
        if end_points is not None:
            for idx, pt in enumerate(end_points):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(overlay, (x, y), 8, (255, 0, 0), -1)
                cv2.putText(
                    overlay,
                    f"E{idx}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Path in yellow
        if path_in_pixels is not None and len(path_in_pixels) > 1:
            pts = np.asarray(path_in_pixels).astype(np.int32)

            for i in range(len(pts) - 1):
                p1 = tuple(pts[i])
                p2 = tuple(pts[i + 1])
                cv2.line(overlay, p1, p2, (255, 255, 0), 2)

            # Highlight first and last path point
            cv2.circle(overlay, tuple(pts[0]), 6, (255, 255, 255), -1)
            cv2.circle(overlay, tuple(pts[-1]), 6, (255, 255, 255), -1)

        return overlay

    def create_no_trace_overlay(
        self,
        image_rgb: np.ndarray,
        start_points: List[Tuple[int, int]],
        end_points: Optional[List[Tuple[int, int]]] = None,
        message: str = "Tracer unavailable",
    ) -> np.ndarray:
        """
        Draw only debug markers if tracing cannot yet be executed.
        """
        overlay = self.create_trace_overlay(
            image_rgb=image_rgb,
            start_points=start_points,
            end_points=end_points,
            path_in_pixels=None,
        )

        cv2.putText(
            overlay,
            message,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 100, 100),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _ensure_rgb_uint8(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image shape/type for GUI display.
        """
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image
