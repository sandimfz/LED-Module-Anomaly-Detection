"""
Module error detection module.

Detects corrupted or glitchy sections in LED content that indicate
module hardware failures. These typically appear as sections with:
- Random colored horizontal lines
- Pixel chaos / noise
- Content that doesn't match surrounding areas
"""

from typing import List

import cv2
import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_module_errors(
    gray: np.ndarray,
    hsv: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
    grid_size: int = 16,
    chaos_threshold: float = 0.3,
) -> List[LEDAnomaly]:
    """Detect module error anomalies.

    Module errors appear as sections with corrupted content -
    random colored lines, pixel chaos, or inconsistent texture
    compared to surrounding areas.

    Args:
        gray: Grayscale LED region.
        hsv: HSV LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.
        grid_size: Number of grid cells per dimension.
        chaos_threshold: Threshold for chaos detection.

    Returns:
        List of detected module error anomalies.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = gray.shape

    if h < grid_size * 2 or w < grid_size * 2:
        return []

    cell_h, cell_w = h // grid_size, w // grid_size

    # Analyze each cell for module error patterns
    cell_chaos_scores = _compute_chaos_scores(
        gray, hsv, led_mask, grid_size, cell_h, cell_w
    )

    # Find cells with high chaos scores
    anomalous_cells = _find_chaotic_cells(
        cell_chaos_scores, grid_size, chaos_threshold
    )

    if not anomalous_cells:
        return []

    # Merge adjacent chaotic cells
    regions = _merge_chaotic_cells(anomalous_cells, grid_size)

    # Create anomalies for each region
    for region in regions:
        if len(region) < 2:
            continue

        anomaly = _create_module_error_anomaly(
            region, panel, cell_h, cell_w
        )
        if anomaly:
            anomalies.append(anomaly)

    return anomalies


def _compute_chaos_scores(
    gray: np.ndarray,
    hsv: np.ndarray,
    led_mask: np.ndarray,
    grid_size: int,
    cell_h: int,
    cell_w: int,
) -> List[float]:
    """Compute chaos score for each grid cell.

    Chaos score measures how "glitchy" or corrupted a cell looks.
    High chaos = lots of random color variations, inconsistent texture.

    Args:
        gray: Grayscale image.
        hsv: HSV image.
        led_mask: Binary mask.
        grid_size: Number of grid cells.
        cell_h: Cell height.
        cell_w: Cell width.

    Returns:
        List of chaos scores for each cell.
    """
    scores = []

    for r in range(grid_size):
        for c in range(grid_size):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w

            cell_mask = led_mask[y0:y1, x0:x1]
            led_ratio = float(np.mean(cell_mask))

            if led_ratio < 0.3:
                scores.append(0.0)
                continue

            cell_gray = gray[y0:y1, x0:x1]
            cell_hsv = hsv[y0:y1, x0:x1]

            chaos = _calculate_cell_chaos(cell_gray, cell_hsv, cell_mask)
            scores.append(chaos)

    return scores


def _calculate_cell_chaos(
    cell_gray: np.ndarray,
    cell_hsv: np.ndarray,
    cell_mask: np.ndarray,
) -> float:
    """Calculate chaos score for a single cell.

    Chaos is measured by:
    1. High local variance in grayscale (noise)
    2. High color variance in HSV (random colors)
    3. Inconsistent texture (lots of edges)

    Args:
        cell_gray: Grayscale cell.
        cell_hsv: HSV cell.
        cell_mask: Binary mask for cell.

    Returns:
        Chaos score between 0.0 and 1.0.
    """
    # Method 1: Grayscale variance (noise indicator)
    pixels = cell_gray[cell_mask > 0]
    if len(pixels) == 0:
        return 0.0

    gray_var = float(np.var(pixels))
    # Normalize: high variance = high chaos
    gray_chaos = min(gray_var / 2000.0, 1.0)

    # Method 2: Color variance in HSV
    h_channel = cell_hsv[:, :, 0][cell_mask > 0]
    s_channel = cell_hsv[:, :, 1][cell_mask > 0]
    v_channel = cell_hsv[:, :, 2][cell_mask > 0]

    if len(h_channel) == 0:
        return 0.0

    # High hue variance = random colors = chaos
    hue_var = float(np.var(h_channel))
    hue_chaos = min(hue_var / 1000.0, 1.0)

    # High saturation variance = inconsistent color intensity
    sat_var = float(np.var(s_channel))
    sat_chaos = min(sat_var / 1500.0, 1.0)

    # Method 3: Edge density (lots of edges = noisy)
    edges = cv2.Canny(cell_gray, 50, 150)
    edge_pixels = np.sum(edges[cell_mask > 0] > 0)
    total_pixels = np.sum(cell_mask > 0)
    edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0
    edge_chaos = min(edge_ratio * 3.0, 1.0)

    # Combine scores
    chaos = (
        gray_chaos * 0.3 +
        hue_chaos * 0.3 +
        sat_chaos * 0.2 +
        edge_chaos * 0.2
    )

    return min(chaos, 1.0)


def _find_chaotic_cells(
    scores: List[float],
    grid_size: int,
    threshold: float,
) -> List[tuple]:
    """Find cells with high chaos scores.

    Args:
        scores: List of chaos scores.
        grid_size: Number of grid cells.
        threshold: Chaos threshold.

    Returns:
        List of (row, col) tuples for chaotic cells.
    """
    chaotic = []

    for i, score in enumerate(scores):
        if score > threshold:
            r = i // grid_size
            c = i % grid_size
            chaotic.append((r, c))

    return chaotic


def _merge_chaotic_cells(
    cells: List[tuple],
    grid_size: int,
) -> List[List[tuple]]:
    """Merge adjacent chaotic cells into regions.

    Args:
        cells: List of (row, col) tuples.
        grid_size: Number of grid cells.

    Returns:
        List of regions.
    """
    if not cells:
        return []

    visited = set()
    regions = []

    for start_cell in cells:
        if start_cell in visited:
            continue

        region = []
        queue = [start_cell]

        while queue:
            cell = queue.pop(0)

            if cell in visited:
                continue

            visited.add(cell)
            region.append(cell)

            r, c = cell
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = r + dr, c + dc
                    neighbor = (nr, nc)

                    if (
                        neighbor in cells
                        and neighbor not in visited
                        and 0 <= nr < grid_size
                        and 0 <= nc < grid_size
                    ):
                        queue.append(neighbor)

        regions.append(region)

    return regions


def _create_module_error_anomaly(
    region: List[tuple],
    panel: LEDPanelInfo,
    cell_h: int,
    cell_w: int,
) -> LEDAnomaly:
    """Create anomaly from a region of chaotic cells.

    Args:
        region: List of (row, col) tuples.
        panel: LED panel information.
        cell_h: Cell height.
        cell_w: Cell width.

    Returns:
        LEDAnomaly or None.
    """
    if not region:
        return None

    # Calculate bounding box
    min_r = min(r for r, c in region)
    max_r = max(r for r, c in region)
    min_c = min(c for r, c in region)
    max_c = max(c for r, c in region)

    x = min_c * cell_w + panel.x
    y = min_r * cell_h + panel.y
    width = (max_c - min_c + 1) * cell_w
    height = (max_r - min_r + 1) * cell_h

    # Severity based on region size
    region_area = len(region)
    total_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
    fill_ratio = region_area / total_cells if total_cells > 0 else 0

    severity = min(fill_ratio * 1.5, 1.0)

    # Determine specific type based on shape
    aspect_ratio = width / height if height > 0 else 1.0

    if aspect_ratio > 3.0:
        anomaly_type = "horizontal_glitch"
        description = f"Horizontal glitch region ({region_area} cells)"
    elif aspect_ratio < 0.33:
        anomaly_type = "vertical_glitch"
        description = f"Vertical glitch region ({region_area} cells)"
    else:
        anomaly_type = "module_glitch"
        description = f"Module glitch ({width}x{height}px, {region_area} cells)"

    return LEDAnomaly(
        x=x,
        y=y,
        width=width,
        height=height,
        anomaly_type=anomaly_type,
        severity=severity,
        description=description,
    )
