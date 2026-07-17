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
    chaos_threshold: float = 0.75,
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

    Two parallel detection signals:
    1. Chaos score: random color variations, corrupted rows (existing)
    2. Flat stuck score: cell is flat (std<5) but neighbors are varied
       → module is stuck/dead while surrounding content changes

    Final score per cell = max(chaos_score, flat_stuck_score).

    Args:
        gray: Grayscale image.
        hsv: HSV image.
        led_mask: Binary mask.
        grid_size: Number of grid cells.
        cell_h: Cell height.
        cell_w: Cell width.

    Returns:
        List of combined scores for each cell.
    """
    # Pass 1: compute per-cell stats (mean, std, led_ratio)
    cell_stats = []
    for r in range(grid_size):
        for c in range(grid_size):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w

            cell_mask = led_mask[y0:y1, x0:x1]
            led_ratio = float(np.mean(cell_mask))

            if led_ratio < 0.3:
                cell_stats.append({"valid": False, "mean": 0.0, "std": 0.0})
                continue

            cell_gray = gray[y0:y1, x0:x1]
            cell_hsv = hsv[y0:y1, x0:x1]
            pixels = cell_gray[cell_mask > 0]

            if len(pixels) == 0:
                cell_stats.append({"valid": False, "mean": 0.0, "std": 0.0})
                continue

            cell_stats.append({
                "valid": True,
                "mean": float(np.mean(pixels)),
                "std": float(np.std(pixels)),
                "r": r, "c": c,
                "y0": y0, "x0": x0, "y1": y1, "x1": x1,
            })

    # Pass 2: compute chaos + flat_stuck per cell
    scores = []
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            stat = cell_stats[idx]
            idx += 1

            if not stat["valid"]:
                scores.append(0.0)
                continue

            # --- Signal 1: chaos score (existing) ---
            cell_gray = gray[stat["y0"]:stat["y1"], stat["x0"]:stat["x1"]]
            cell_hsv_cell = hsv[stat["y0"]:stat["y1"], stat["x0"]:stat["x1"]]
            cell_mask = led_mask[stat["y0"]:stat["y1"], stat["x0"]:stat["x1"]]
            chaos = _calculate_cell_chaos(cell_gray, cell_hsv_cell, cell_mask)

            # --- Signal 2: flat stuck score (new) ---
            # Cell is flat (std < 5) BUT neighbors have high variance
            # → module is stuck while surrounding content is active
            flat_stuck = 0.0
            if stat["std"] < 5.0:
                # Get neighbor stats (8-connected)
                neighbor_stds = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            n_idx = nr * grid_size + nc
                            if n_idx < len(cell_stats) and cell_stats[n_idx]["valid"]:
                                neighbor_stds.append(cell_stats[n_idx]["std"])

                if neighbor_stds:
                    avg_neighbor_std = float(np.mean(neighbor_stds))
                    # Neighbors must have HIGH variance (truly active content)
                    # 25+ means neighbors are colorful/varied, not just dark bg
                    if avg_neighbor_std > 25.0:
                        flat_stuck = 0.5

            # Combined: take the stronger signal
            scores.append(max(chaos, flat_stuck))

    return scores


def _calculate_cell_chaos(
    cell_gray: np.ndarray,
    cell_hsv: np.ndarray,
    cell_mask: np.ndarray,
) -> float:
    """Calculate chaos score for a single cell.

    Detects module error patterns by measuring:
    1. Brightness uniformity (dark/bright anomalies)
    2. Color consistency (random color variations)
    3. Horizontal line patterns (corrupted rows)

    Args:
        cell_gray: Grayscale cell.
        cell_hsv: HSV cell.
        cell_mask: Binary mask for cell.

    Returns:
        Chaos score between 0.0 and 1.0.
    """
    pixels = cell_gray[cell_mask > 0]
    if len(pixels) == 0:
        return 0.0

    # Method 1: Brightness uniformity
    # Module errors often show as dark or bright patches
    mean_brightness = float(np.mean(pixels))
    std_brightness = float(np.std(pixels))
    # Low brightness with low variance = dark module (potential defect)
    # Normal content has moderate brightness and variance
    brightness_anomaly = 0.0
    if mean_brightness < 80 and std_brightness < 30:
        # Dark patch with uniform color - potential dead module
        brightness_anomaly = min((80 - mean_brightness) / 80, 1.0)
    elif mean_brightness > 220 and std_brightness < 20:
        # Bright patch - potential stuck pixel
        brightness_anomaly = min((mean_brightness - 220) / 35, 1.0)

    # Method 2: Color consistency in HSV
    h_channel = cell_hsv[:, :, 0][cell_mask > 0]
    s_channel = cell_hsv[:, :, 1][cell_mask > 0]

    if len(h_channel) == 0:
        return brightness_anomaly

    # Module errors show as random color variations
    # Normal content has consistent hue in dominant color
    hue_var = float(np.var(h_channel))
    # High hue variance = random colors = anomaly
    color_anomaly = min(hue_var / 3000.0, 1.0)

    # Method 3: Horizontal line pattern detection
    # Module errors often show as corrupted horizontal lines
    h, w = cell_gray.shape
    if h >= 4:
        # Compute row-wise variance
        row_vars = []
        for row_idx in range(h):
            row = cell_gray[row_idx, :]
            mask_row = cell_mask[row_idx, :]
            if np.sum(mask_row > 0) > w * 0.3:
                row_vars.append(float(np.var(row[mask_row > 0])))
            else:
                row_vars.append(0.0)

        if row_vars:
            row_var_mean = np.mean(row_vars)
            row_var_std = np.std(row_vars)
            # High row variance variation = random horizontal lines
            line_anomaly = min(row_var_std / (row_var_mean + 1e-6) / 3.0, 1.0)
        else:
            line_anomaly = 0.0
    else:
        line_anomaly = 0.0

    # Combine scores
    chaos = (
        brightness_anomaly * 0.4 +
        color_anomaly * 0.3 +
        line_anomaly * 0.3
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
