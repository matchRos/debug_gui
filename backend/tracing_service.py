from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import traceback
from cable_routing.env.ext_camera.utils.img_utils import find_nearest_white_pixel


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


def snap_to_bright_pixel(image, pt, radius=5):
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


def nearest_bright_pixel_global(
    image_rgb: np.ndarray,
    anchor_xy: Tuple[int, int],
    threshold: int = 150,
) -> Optional[Tuple[int, int]]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > threshold)
    if len(xs) == 0:
        return None

    pts = np.stack([xs, ys], axis=1).astype(float)
    anchor = np.asarray(anchor_xy, dtype=float).reshape(1, 2)
    d2 = np.sum((pts - anchor) ** 2, axis=1)
    best = pts[int(np.argmin(d2))]
    return (int(best[0]), int(best[1]))


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
        start_mode: str = "auto_from_config",
        anchor_point: Optional[Tuple[int, int]] = None,
        clip_points: Optional[List[Tuple[int, int]]] = None,
        preferred_direction_xy: Optional[Tuple[float, float]] = None,
        max_start_dist_px: float = 260.0,
        min_route_dot: float = -0.15,
        outward_min_delta_px: float = 8.0,
        seed_order_descending_from_anchor: bool = True,
        clip_a_p1_offset_px: float = 20.0,
        clip_a_p2_offset_px: float = 40.0,
    ) -> Dict[str, Any]:
        """
        Execute the legacy CableTracer wrapper if available.
        """
        if tracer is None:
            raise RuntimeError("Tracer object is not available.")

        tracer_start_points = start_points

        # path, status = tracer.trace(
        #     img=image_rgb,
        #     start_points=start_points,
        #     end_points=end_points,
        #     viz=viz,
        # )

        if start_mode == "manual_two_clicks":
            picked_xy = pick_two_points_on_image(image_rgb)
            print("clicked points (x,y):", picked_xy)

            tracer_start_points = build_three_start_points_from_start_and_direction(
                image_rgb,
                picked_xy[0],  # first click = start
                picked_xy[1],  # second click = direction
                step_px=20,
            )
            print("generated tracer start points (y,x):", tracer_start_points)
        elif start_mode == "auto_from_clip_a":
            if anchor_point is None:
                raise RuntimeError(
                    "trace_start_mode=auto_from_clip_a requires an anchor_point."
                )

            nearest = nearest_bright_pixel_global(image_rgb, anchor_point, threshold=150)

            # Choose direction from anchor using cable-near candidates (same spirit as env_new).
            direction = None
            clip_dict = {"x": int(anchor_point[0]), "y": int(anchor_point[1])}
            try:
                valid_points = find_nearest_white_pixel(
                    image_rgb,
                    clip_dict,
                    num_options=25,
                    display=False,
                )
            except Exception:
                valid_points = []

            pref = None
            if preferred_direction_xy is not None:
                pref = np.asarray(preferred_direction_xy, dtype=float).reshape(2)
                n_pref = float(np.linalg.norm(pref))
                if n_pref > 1e-6:
                    pref = pref / n_pref
                else:
                    pref = None

            best_score = float("inf")
            best_point = None
            for p in valid_points:
                vec = np.asarray(p, dtype=float) - np.asarray(anchor_point, dtype=float)
                d = float(np.linalg.norm(vec))
                if d < 3.0:
                    continue
                v = vec / (d + 1e-8)
                dot_penalty = 0.0
                if pref is not None:
                    dot = float(np.dot(v, pref))
                    dot_penalty = 40.0 * (1.0 - dot)
                # Prefer points around ~20px from anchor, similar to manual start scale.
                dist_penalty = abs(d - 20.0)
                score = dist_penalty + dot_penalty
                if score < best_score:
                    best_score = score
                    best_point = p

            if best_point is not None:
                direction = np.asarray(best_point, dtype=float) - np.asarray(
                    anchor_point, dtype=float
                )
            elif nearest is not None:
                direction = np.asarray(nearest, dtype=float) - np.asarray(
                    anchor_point, dtype=float
                )
            elif pref is not None:
                direction = pref.copy()
            else:
                direction = np.array([1.0, 0.0], dtype=float)

            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                direction = np.array([1.0, 0.0], dtype=float)
                norm = 1.0
            direction = direction / norm

            # Requested strategy:
            # P0 = anchor clip; P1/P2 generated automatically along chosen direction.
            p0_xy = (int(anchor_point[0]), int(anchor_point[1]))
            p1_xy = (
                int(round(anchor_point[0] + direction[0] * float(clip_a_p1_offset_px))),
                int(round(anchor_point[1] + direction[1] * float(clip_a_p1_offset_px))),
            )
            p2_xy = (
                int(round(anchor_point[0] + direction[0] * float(clip_a_p2_offset_px))),
                int(round(anchor_point[1] + direction[1] * float(clip_a_p2_offset_px))),
            )
            p0_xy = snap_to_bright_pixel(image_rgb, p0_xy, radius=5)
            p1_xy = snap_to_bright_pixel(image_rgb, p1_xy, radius=7)
            p2_xy = snap_to_bright_pixel(image_rgb, p2_xy, radius=7)

            tracer_start_points = [
                (int(p0_xy[1]), int(p0_xy[0])),
                (int(p1_xy[1]), int(p1_xy[0])),
                (int(p2_xy[1]), int(p2_xy[0])),
            ]
            print(
                "auto_from_clip_a tracer points (y,x):",
                tracer_start_points,
                "anchor:",
                anchor_point,
                "nearest:",
                nearest,
            )
        else:
            # auto_from_config (default): robustly derive 3 tracer points from config.
            cfg_pts = [tuple(np.asarray(p).reshape(-1)[:2].astype(int)) for p in start_points]
            if len(cfg_pts) >= 2:
                tracer_start_points = build_three_start_points_from_start_and_direction(
                    image_rgb,
                    cfg_pts[0],  # start
                    cfg_pts[1],  # direction hint
                    step_px=20,
                )
            elif len(cfg_pts) == 1:
                p0 = cfg_pts[0]
                p1 = (int(p0[0] + 20), int(p0[1]))
                tracer_start_points = build_three_start_points_from_start_and_direction(
                    image_rgb,
                    p0,
                    p1,
                    step_px=20,
                )
            else:
                raise RuntimeError(
                    "trace_start_points is empty. Provide at least one point in DebugConfig."
                )
            print("auto-generated tracer start points (y,x):", tracer_start_points)

        def _build_auto_candidates() -> List[List[Tuple[int, int]]]:
            """
            Build several (y,x) candidate triplets for robust tracer initialization.
            """
            candidates: List[List[Tuple[int, int]]] = []
            cfg_pts = [tuple(np.asarray(p).reshape(-1)[:2].astype(int)) for p in start_points]

            if len(cfg_pts) >= 2:
                p0 = cfg_pts[0]
                p1 = cfg_pts[1]
            elif len(cfg_pts) == 1:
                p0 = cfg_pts[0]
                p1 = (int(p0[0] + 20), int(p0[1]))
            else:
                return candidates

            v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                v = np.array([1.0, 0.0], dtype=float)
                n = 1.0
            v = v / n

            for step in (20, 35, 50, 70):
                # forward direction
                fwd = (
                    int(round(p0[0] + v[0] * step)),
                    int(round(p0[1] + v[1] * step)),
                )
                cand_fwd = build_three_start_points_from_start_and_direction(
                    image_rgb,
                    p0,
                    fwd,
                    step_px=step,
                )
                candidates.append(cand_fwd)
                # 2-point seed to trigger analytic tracer bootstrap in CableTracer.
                candidates.append([cand_fwd[0], cand_fwd[-1]])
                candidates.append([cand_fwd[-1], cand_fwd[0]])

                # reverse direction
                rev = (
                    int(round(p0[0] - v[0] * step)),
                    int(round(p0[1] - v[1] * step)),
                )
                cand_rev = build_three_start_points_from_start_and_direction(
                    image_rgb,
                    p0,
                    rev,
                    step_px=step,
                )
                candidates.append(cand_rev)
                candidates.append([cand_rev[0], cand_rev[-1]])
                candidates.append([cand_rev[-1], cand_rev[0]])
            return candidates

        def _build_anchor_white_candidates() -> List[List[Tuple[int, int]]]:
            """
            Build start candidates from nearest cable pixels around anchor clip
            (same spirit as env_new trace_cable).
            """
            if anchor_point is None:
                return []

            clip_dict = {"x": int(anchor_point[0]), "y": int(anchor_point[1])}
            try:
                valid_points = find_nearest_white_pixel(
                    image_rgb,
                    clip_dict,
                    num_options=20,
                    display=False,
                )
            except Exception:
                return []

            if clip_points:
                filtered_points = []
                for p in valid_points:
                    ok = True
                    for c in clip_points:
                        # Allow points near the anchor clip itself.
                        if anchor_point is not None and np.linalg.norm(
                            np.asarray(c, dtype=float) - np.asarray(anchor_point, dtype=float)
                        ) < 1.0:
                            continue
                        if float(np.linalg.norm(np.asarray(p) - np.asarray(c))) < 20.0:
                            ok = False
                            break
                    if ok:
                        filtered_points.append(p)
            else:
                filtered_points = list(valid_points)

            # Strongly prefer nearest anchor-adjacent cable pixels first.
            filtered_points.sort(
                key=lambda p: float(
                    np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(anchor_point, dtype=float))
                )
            )

            candidates: List[List[Tuple[int, int]]] = []
            for p in filtered_points[:6]:
                direction_xy = (
                    int(round(2 * p[0] - anchor_point[0])),
                    int(round(2 * p[1] - anchor_point[1])),
                )
                cand = build_three_start_points_from_start_and_direction(
                    image_rgb,
                    p,
                    direction_xy,
                    step_px=20,
                )
                candidates.append(cand)
                candidates.append([cand[0], cand[-1]])
                candidates.append([cand[-1], cand[0]])
            return candidates

        def _rank_and_filter_candidates(
            candidates: List[List[Tuple[int, int]]],
        ) -> List[List[Tuple[int, int]]]:
            """
            Keep candidates near anchor and aligned with route direction, then sort.
            Candidate points are expected in tracer format (y, x).
            """
            if not candidates:
                return candidates

            pref = None
            if preferred_direction_xy is not None:
                pref = np.asarray(preferred_direction_xy, dtype=float).reshape(2)
                n = float(np.linalg.norm(pref))
                if n > 1e-6:
                    pref = pref / n
                else:
                    pref = None

            scored = []
            for cand in candidates:
                if cand is None or len(cand) < 2:
                    continue

                # Optional reordering hypothesis: P2 should be P0 (farther->nearer to anchor).
                if anchor_point is not None and len(cand) >= 3:
                    pts_xy = []
                    for pt in cand:
                        arr = np.asarray(pt, dtype=float).reshape(-1)[:2]  # (y,x)
                        pts_xy.append(np.array([arr[1], arr[0]], dtype=float))
                    dists = [
                        float(
                            np.linalg.norm(
                                pxy - np.asarray(anchor_point, dtype=float).reshape(2)
                            )
                        )
                        for pxy in pts_xy
                    ]
                    order = np.argsort(dists)
                    if seed_order_descending_from_anchor:
                        order = order[::-1]
                    cand = [cand[int(i)] for i in order.tolist()]

                p0 = np.asarray(cand[0], dtype=float).reshape(-1)[:2]  # (y,x)
                p1 = np.asarray(cand[1], dtype=float).reshape(-1)[:2]
                p0_xy = np.array([p0[1], p0[0]], dtype=float)
                p1_xy = np.array([p1[1], p1[0]], dtype=float)

                d_anchor = 0.0
                if anchor_point is not None:
                    anchor_arr = np.asarray(anchor_point, dtype=float)
                    d0 = float(np.linalg.norm(p0_xy - anchor_arr))
                    d1 = float(np.linalg.norm(p1_xy - anchor_arr))
                    d_anchor = d0
                    if d_anchor > max_start_dist_px:
                        continue
                    # Enforce the configured seed order relative to anchor.
                    delta = d1 - d0
                    if seed_order_descending_from_anchor:
                        # P0 farther than P1 by at least outward_min_delta_px.
                        if (-delta) < outward_min_delta_px:
                            continue
                    else:
                        # P0 nearer than P1 by at least outward_min_delta_px.
                        if delta < outward_min_delta_px:
                            continue

                dir_xy = p1_xy - p0_xy
                dir_n = float(np.linalg.norm(dir_xy))
                dot = 0.0
                if pref is not None and dir_n > 1e-6:
                    dot = float(np.dot(dir_xy / dir_n, pref))
                    if dot < min_route_dot:
                        continue

                # Lower is better: close to anchor and aligned with preferred direction.
                score = d_anchor + 80.0 * (1.0 - dot)
                scored.append((score, cand))

            if not scored:
                return candidates
            scored.sort(key=lambda x: x[0])
            return [c for _, c in scored]

        try:
            result = None
            last_exc: Optional[Exception] = None
            if start_mode == "auto_from_clip_a":
                # Keep this mode strictly clip-anchored to avoid unrelated config fallbacks.
                candidate_pool = [tracer_start_points]
                candidate_pool.extend(_build_anchor_white_candidates())
            elif start_mode == "manual_two_clicks":
                candidate_pool = [tracer_start_points]
            else:
                candidate_pool = [tracer_start_points] + _build_auto_candidates()
                candidate_pool.extend(_build_anchor_white_candidates())
            candidate_pool = _rank_and_filter_candidates(candidate_pool)

            print(f"trace candidate pool size: {len(candidate_pool)} (mode={start_mode})")

            for candidate_idx, candidate in enumerate(candidate_pool):
                tracer_start_points = candidate
                try:
                    result = tracer.trace(
                        img=image_rgb,
                        start_points=tracer_start_points,
                        end_points=end_points,
                        viz=viz,
                    )
                    if result is not None:
                        if candidate_idx > 0:
                            print(
                                f"trace succeeded with fallback candidate #{candidate_idx}: {tracer_start_points}"
                            )
                        # Show candidate in both conventions for easier debugging.
                        if len(tracer_start_points) >= 2:
                            p0_yx = np.asarray(tracer_start_points[0], dtype=float).reshape(-1)[:2]
                            p1_yx = np.asarray(tracer_start_points[1], dtype=float).reshape(-1)[:2]
                            p0_xy = (int(round(p0_yx[1])), int(round(p0_yx[0])))
                            p1_xy = (int(round(p1_yx[1])), int(round(p1_yx[0])))
                            print(
                                "selected trace candidate p0/p1: "
                                f"yx={tracer_start_points[0]},{tracer_start_points[1]} "
                                f"xy={p0_xy},{p1_xy}"
                            )
                        break
                except Exception as e_try:
                    msg = str(e_try)
                    # Retry only for the known sparse-start condition.
                    if "Not enough starting points" in msg:
                        last_exc = e_try
                        continue
                    raise

            if result is None and last_exc is not None:
                raise last_exc

            if result is None:
                raise RuntimeError(
                    f"Tracing failed. The start point is likely not on the cable or the analytic tracer could not initialize from start_points={start_points}."
                )

            path, status = result
        except Exception as e:
            print("\n=== TRACE ERROR ===")
            print(f"type: {type(e).__name__}")
            print(f"message: {e}")
            print(f"start_points(type/raw): {type(start_points)} -> {start_points}")
            print(
                f"tracer_start_points(type/used): {type(tracer_start_points)} -> {tracer_start_points}"
            )
            if tracer_start_points is not None:
                for i, p in enumerate(tracer_start_points):
                    try:
                        arr = np.asarray(p)
                        print(
                            f"tracer_start_points[{i}] -> type={type(p)}, shape={arr.shape}, value={p}"
                        )
                    except Exception:
                        print(
                            f"tracer_start_points[{i}] -> type={type(p)}, value={p}"
                        )
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
            "tracer_start_points_used": tracer_start_points,
            "tracer_start_point_count": len(tracer_start_points)
            if tracer_start_points is not None
            else 0,
        }

    def create_trace_overlay(
        self,
        image_rgb: np.ndarray,
        start_points: List[Tuple[int, int]],
        end_points: Optional[List[Tuple[int, int]]] = None,
        path_in_pixels: Optional[np.ndarray] = None,
        tracer_start_points_used: Optional[List[Tuple[int, int]]] = None,
        configured_clip_positions: Optional[List[Tuple[str, int, int]]] = None,
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

        # Clip target positions from board config in cyan.
        if configured_clip_positions is not None:
            for clip_id, x, y in configured_clip_positions:
                xi = int(x)
                yi = int(y)
                if xi < 0 or yi < 0 or yi >= overlay.shape[0] or xi >= overlay.shape[1]:
                    continue
                cv2.circle(overlay, (xi, yi), 10, (0, 255, 255), 2)
                cv2.drawMarker(
                    overlay,
                    (xi, yi),
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=14,
                    thickness=2,
                )
                cv2.putText(
                    overlay,
                    f"C{clip_id}",
                    (xi + 10, yi + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    f"C{clip_id}",
                    (xi + 10, yi + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
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

        # Actual tracer seed points (P0/P1) in tracer convention (y, x).
        if tracer_start_points_used is not None and len(tracer_start_points_used) >= 2:
            pxy_list = []
            for idx, pt in enumerate(tracer_start_points_used[:2]):
                arr = np.asarray(pt).reshape(-1)
                if arr.size < 2:
                    continue
                y = int(arr[0])
                x = int(arr[1])

                if x < 0 or y < 0 or y >= overlay.shape[0] or x >= overlay.shape[1]:
                    continue

                pxy_list.append((x, y))
                color = (255, 0, 255) if idx == 0 else (255, 165, 0)
                label = f"P{idx}"
                cv2.circle(overlay, (x, y), 16, color, 3)
                cv2.circle(overlay, (x, y), 5, color, -1)
                cv2.circle(overlay, (x, y), 22, (0, 0, 0), 1)
                cv2.putText(
                    overlay,
                    label,
                    (x + 18, y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    label,
                    (x + 18, y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            if len(pxy_list) >= 2:
                cv2.arrowedLine(
                    overlay,
                    pxy_list[0],
                    pxy_list[1],
                    (255, 255, 255),
                    3,
                    tipLength=0.22,
                )

            # Readable seed info panel.
            lines = [f"Tracer seeds used: {len(tracer_start_points_used)}"]
            for i, pt in enumerate(tracer_start_points_used[:3]):
                arr = np.asarray(pt).reshape(-1)
                if arr.size < 2:
                    continue
                y = int(arr[0])
                x = int(arr[1])
                lines.append(f"P{i}: xy=({x},{y}) yx=({y},{x})")

            panel_x, panel_y = 20, 20
            panel_w = 460
            panel_h = 36 + 30 * len(lines)
            cv2.rectangle(
                overlay,
                (panel_x, panel_y),
                (panel_x + panel_w, panel_y + panel_h),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                overlay,
                (panel_x, panel_y),
                (panel_x + panel_w, panel_y + panel_h),
                (255, 255, 255),
                1,
            )
            for i, line in enumerate(lines):
                y = panel_y + 28 + i * 28
                cv2.putText(
                    overlay,
                    line,
                    (panel_x + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return overlay

    def create_no_trace_overlay(
        self,
        image_rgb: np.ndarray,
        start_points: List[Tuple[int, int]],
        end_points: Optional[List[Tuple[int, int]]] = None,
        message: str = "Tracer unavailable",
        configured_clip_positions: Optional[List[Tuple[str, int, int]]] = None,
    ) -> np.ndarray:
        """
        Draw only debug markers if tracing cannot yet be executed.
        """
        overlay = self.create_trace_overlay(
            image_rgb=image_rgb,
            start_points=start_points,
            end_points=end_points,
            path_in_pixels=None,
            configured_clip_positions=configured_clip_positions,
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
