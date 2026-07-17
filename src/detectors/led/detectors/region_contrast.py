"""
Region-based contrast detection module.

Detects anomalies by comparing each region with its neighbors.
This approach is more robust than absolute thresholds because it
adapts to varying content brightness.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_region_contrast_anomalies(
    gray: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
    panel_mask: Optional[np.ndarray] = None,
    grid_size: int = 8,
    contrast_threshold: float = 2.5,
    min_region_cells: int = 6,
) -> List[LEDAnomaly]:
    """Detect anomalies using region-based contrast comparison.

    Divides LED panel into grid cells and compares each cell
    with its neighbors using robust MAD-based z-score.
    Uses panel_mask as base to include white flat blocks.

    Args:
        gray: Grayscale LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.
        panel_mask: Binary mask of panel area (preferred base).
        grid_size: Number of grid rows/columns.
        contrast_threshold: Multiplier of neighbor MAD for threshold.
        min_region_cells: Minimum adjacent cells to form a region.

    Returns:
        List of detected anomalies.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = gray.shape

    if h < grid_size * 2 or w < grid_size * 2:
        return []

    base_mask = panel_mask if panel_mask is not None else led_mask
    cell_h, cell_w = h // grid_size, w // grid_size

    # Calculate brightness for each cell
    cell_stats = _compute_cell_stats(gray, base_mask, grid_size, cell_h, cell_w)

    # Find cells that are significantly different from neighbors
    anomalous_cells = _find_anomalous_cells(cell_stats, grid_size)

    if not anomalous_cells:
        return []

    # Merge adjacent anomalous cells into regions
    regions = _merge_cells_to_regions(anomalous_cells, grid_size)

    # Filter small regions and create anomalies
    for region in regions:
        if len(region) < min_region_cells:
            continue

        anomaly = _create_region_anomaly(region, panel, cell_h, cell_w)
        if anomaly:
            anomalies.append(anomaly)

    return anomalies


def _compute_cell_stats(
    gray: np.ndarray,
    led_mask: np.ndarray,
    grid_size: int,
    cell_h: int,
    cell_w: int,
) -> List[Dict]:
    """Compute brightness statistics for each grid cell.

    Args:
        gray: Grayscale image.
        led_mask: Binary mask.
        grid_size: Number of grid rows/columns.
        cell_h: Cell height.
        cell_w: Cell width.

    Returns:
        List of cell statistics dictionaries.
    """
    stats = []

    for r in range(grid_size):
        for c in range(grid_size):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w

            cell_mask = led_mask[y0:y1, x0:x1]
            led_ratio = float(np.mean(cell_mask))

            if led_ratio < 0.3:
                stats.append({
                    "r": r, "c": c,
                    "mean": 0.0, "std": 0.0,
                    "led_ratio": led_ratio, "valid": False
                })
                continue

            cell_gray = gray[y0:y1, x0:x1]
            cell_pixels = cell_gray[cell_mask > 0]

            if len(cell_pixels) == 0:
                stats.append({
                    "r": r, "c": c,
                    "mean": 0.0, "std": 0.0,
                    "led_ratio": led_ratio, "valid": False
                })
                continue

            stats.append({
                "r": r, "c": c,
                "mean": float(np.mean(cell_pixels)),
                "std": float(np.std(cell_pixels)),
                "led_ratio": led_ratio,
                "valid": True
            })

    return stats


def _find_anomalous_cells(
    cell_stats: List[Dict],
    grid_size: int,
) -> List[Tuple[int, int]]:
    """Find cells significantly different from neighbors.

    Uses robust MAD-based z-score instead of absolute threshold.
    Old code used max(std*2.5, 35) which killed signals when
    neighbor std was small (diff 28 vs threshold 35 -> skip).

    New: MAD is robust to outlier neighbors, and z > 3.0 catches
    cells that decorrelate from local context.

    Args:
        cell_stats: List of cell statistics.
        grid_size: Number of grid rows/columns.

    Returns:
        List of (row, col) tuples for anomalous cells.
    """
    anomalous = []

    for i, cell in enumerate(cell_stats):
        if not cell["valid"]:
            continue

        r, c = cell["r"], cell["c"]

        # Get valid neighbors
        neighbors = _get_neighbors(cell_stats, r, c, grid_size)
        valid_neighbors = [n for n in neighbors if n["valid"]]

        if len(valid_neighbors) < 2:
            continue

        neighbor_means = np.array([n["mean"] for n in valid_neighbors])
        neighbor_median = float(np.median(neighbor_means))

        # Robust spread using MAD (Median Absolute Deviation)
        mad = float(np.median(np.abs(neighbor_means - neighbor_median)))
        # Convert MAD to equivalent std (MAD * 1.4826 ~ std for normal dist)
        robust_std = max(mad * 1.4826, 5.0)  # eps=5 to avoid div/0

        # Z-score: how many robust-stds away from neighbor median
        z_score = abs(cell["mean"] - neighbor_median) / robust_std

        # Threshold: z > 4.5 catches significant decorrelation
        # 4.5 is strict to avoid flagging normal ad content
        # (dark sections of ads, text blocks, etc.)
        if z_score > 4.5:
            anomalous.append((r, c))

    return anomalous


def _get_neighbors(
    cell_stats: List[Dict],
    r: int,
    c: int,
    grid_size: int,
) -> List[Dict]:
    """Get neighboring cells (8-connectivity).

    Args:
        cell_stats: List of cell statistics.
        r: Current row.
        c: Current column.
        grid_size: Number of grid rows/columns.

    Returns:
        List of neighbor cell statistics.
    """
    neighbors = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue

            nr, nc = r + dr, c + dc

            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                idx = nr * grid_size + nc
                if idx < len(cell_stats):
                    neighbors.append(cell_stats[idx])

    return neighbors


def _merge_cells_to_regions(
    cells: List[Tuple[int, int]],
    grid_size: int,
) -> List[List[Tuple[int, int]]]:
    """Merge adjacent anomalous cells into regions.

    Args:
        cells: List of (row, col) tuples.
        grid_size: Number of grid rows/columns.

    Returns:
        List of regions, each region is a list of (row, col).
    """
    if not cells:
        return []

    visited = set()
    regions = []

    for start_cell in cells:
        if start_cell in visited:
            continue

        # BFS to find connected component
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


def _create_region_anomaly(
    region: List[Tuple[int, int]],
    panel: LEDPanelInfo,
    cell_h: int,
    cell_w: int,
) -> LEDAnomaly:
    """Create anomaly from a region of anomalous cells.

    Args:
        region: List of (row, col) tuples.
        panel: LED panel information.
        cell_h: Cell height.
        cell_w: Cell width.

    Returns:
        LEDAnomaly or None if region is invalid.
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

    # Calculate severity based on region size
    region_area = len(region)
    total_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
    fill_ratio = region_area / total_cells if total_cells > 0 else 0

    severity = min(fill_ratio * 1.5, 1.0)

    # Determine anomaly type based on shape
    aspect_ratio = width / height if height > 0 else 1.0

    if aspect_ratio > 3.0:
        anomaly_type = "horizontal_dark_bar"
        description = f"Horizontal dark region ({region_area} cells)"
    elif aspect_ratio < 0.33:
        anomaly_type = "vertical_dark_bar"
        description = f"Vertical dark region ({region_area} cells)"
    elif region_area >= 4:
        anomaly_type = "dark_region"
        description = f"Dark region ({region_area} cells, {width}x{height}px)"
    else:
        anomaly_type = "dark_spot"
        description = f"Dark spot ({width}x{height}px)"

    return LEDAnomaly(
        x=x,
        y=y,
        width=width,
        height=height,
        anomaly_type=anomaly_type,
        severity=severity,
        description=description,
    )
