from pathlib import Path
from typing import Iterable

import csv
import numpy as np


class CableTraceIO:
    def save_csv(self, filepath: str, path_in_pixels: Iterable) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        pts = np.asarray(path_in_pixels, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise RuntimeError("path_in_pixels must be an Nx2 array-like object.")

        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for p in pts:
                writer.writerow([float(p[0]), float(p[1])])

    def load_csv(self, filepath: str) -> np.ndarray:
        path = Path(filepath)
        if not path.exists():
            raise RuntimeError(f"Trace file does not exist: {filepath}")

        pts = []
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            raise RuntimeError("Trace CSV is empty.")

        start_idx = 0
        first = [c.strip().lower() for c in rows[0]]
        if len(first) >= 2 and first[0] == "x" and first[1] == "y":
            start_idx = 1

        for row in rows[start_idx:]:
            if len(row) < 2:
                continue
            x = float(row[0])
            y = float(row[1])
            pts.append([x, y])

        if len(pts) < 2:
            raise RuntimeError("Loaded trace has too few points.")

        return np.asarray(pts, dtype=float)
