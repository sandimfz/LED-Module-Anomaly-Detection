"""
Dark region detection module.

Detects dark regions using local contrast analysis.
"""

from typing import Dict, List

import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_dark_regions_by_local_contrast(
    gray: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
) -> List[LEDAnomaly]:
    """Detect dark regions using local contrast analysis.

    Divides panel into 8x8 grid and compares each cell to global mean.
    Cells significantly darker than panel average are flagged.

    Args:
        gray: Grayscale LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.

    Returns:
        List of detected dark regions.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = gray.shape

    led_pixels = gray[led_mask > 0]
    if len(led_pixels) == 0:
        return []

    global_mean = float(np.mean(led_pixels))
    global_std = float(np.std(led_pixels))

    if global_mean < 80:
        return []

    grid_r, grid_c = 8, 8
    cell_h, cell_w = h // grid_r, w // grid_c
    if cell_h < 8 or cell_w < 8:
        return []

    dark_threshold = global_mean - 2.5 * global_std

    dark_cells = _find_dark_cells(
        gray, led_mask, grid_r, grid_c, cell_h, cell_w, dark_threshold
    )

    if not dark_cells:
        return []

    return _merge_dark_cells(dark_cells, panel, global_mean)


def _find_dark_cells(
    gray: np.ndarray,
    led_mask: np.ndarray,
    grid_r: int,
    grid_c: int,
    cell_h: int,
    cell_w: int,
    dark_threshold: float,
) -> List[Dict]:
    """Find cells that are darker than threshold.

    Args:
        gray: Grayscale image.
        led_mask: Binary mask of LED content.
        grid_r: Number of grid rows.
        grid_c: Number of grid columns.
        cell_h: Cell height.
        cell_w: Cell width.
        dark_threshold: Brightness threshold.

    Returns:
        List of dark cell dictionaries.
    """
    dark_cells: List[Dict] = []

    for r in range(grid_r):
        for c in range(grid_c):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w

            cell_mask = led_mask[y0:y1, x0:x1]
            if np.mean(cell_mask) < 0.5:
                continue

            cell_gray = gray[y0:y1, x0:x1]
            cell_mean = float(np.mean(cell_gray))

            if cell_mean < dark_threshold:
                severity = min(
                    (dark_threshold - cell_mean) / (dark_threshold + 1e-6),
                    1.0,
                )
                dark_cells.append({
                    "r": r,
                    "c": c,
                    "x": x0,
                    "y": y0,
                    "w": cell_w,
                    "h": cell_h,
                    "mean": cell_mean,
                    "severity": severity,
                })

    return dark_cells


def _merge_dark_cells(
    dark_cells: List[Dict],
    panel: LEDPanelInfo,
    global_mean: float,
) -> List[LEDAnomaly]:
    """Merge adjacent dark cells into regions.

    Args:
        dark_cells: List of dark cell dictionaries.
        panel: LED panel information.
        global_mean: Global mean brightness.

    Returns:
        List of merged dark region anomalies.
    """
    anomalies: List[LEDAnomaly] = []
    used = [False] * len(dark_cells)

    for i, cell in enumerate(dark_cells):
        if used[i]:
            continue

        group = [cell]
        used[i] = True

        for j, other in enumerate(dark_cells):
            if used[j]:
                continue
            if (
                abs(cell["r"] - other["r"]) <= 1
                and abs(cell["c"] - other["c"]) <= 1
            ):
                group.append(other)
                used[j] = True

        if len(group) < 2:
            continue

        min_x = min(g["x"] for g in group)
        min_y = min(g["y"] for g in group)
        max_x = max(g["x"] + g["w"] for g in group)
        max_y = max(g["y"] + g["h"] for g in group)
        avg_sev = float(np.mean([g["severity"] for g in group]))
        avg_mean = float(np.mean([g["mean"] for g in group]))

        anomalies.append(
            LEDAnomaly(
                x=min_x + panel.x,
                y=min_y + panel.y,
                width=max_x - min_x,
                height=max_y - min_y,
                anomaly_type="dark_region",
                severity=avg_sev,
                description=(
                    f"Dark region ({len(group)} cells): "
                    f"mean={avg_mean:.0f} vs panel={global_mean:.0f}"
                ),
            )
        )

    return anomalies
