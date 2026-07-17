"""
Module Grid calibration for LED panels.

Maps the physical LED module grid (cabinet/module structure) to image coordinates.
This enables per-module analysis instead of arbitrary grid sizes.

LED videotron panels are composed of physical modules (cabinets) in a fixed grid.
Each module is typically 16x16 or 32x32 pixels in the cropped LED image.
Knowing the grid structure allows:
- Per-module feature extraction
- Spatial decorrelation detection (defect vs neighbor)
- Consistent analysis across images/resolutions
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ModuleGrid:
    """Grid mapping for LED module structure.

    Attributes:
        rows: Number of module rows.
        cols: Number of module columns.
        module_width: Width of each module in pixels.
        module_height: Height of each module in pixels.
        origin_x: X offset of grid origin.
        origin_y: Y offset of grid origin.
    """
    rows: int
    cols: int
    module_width: int
    module_height: int
    origin_x: int = 0
    origin_y: int = 0

    def get_module_bounds(
        self, row: int, col: int
    ) -> Tuple[int, int, int, int]:
        """Get pixel bounds for a module.

        Args:
            row: Module row index.
            col: Module column index.

        Returns:
            Tuple of (x0, y0, x1, y1) pixel coordinates.
        """
        x0 = self.origin_x + col * self.module_width
        y0 = self.origin_y + row * self.module_height
        x1 = x0 + self.module_width
        y1 = y0 + self.module_height
        return (x0, y0, x1, y1)

    def get_neighbors(
        self, row: int, col: int, k: int = 1
    ) -> List[Tuple[int, int]]:
        """Get neighboring module coordinates (8-connected).

        Args:
            row: Module row index.
            col: Module column index.
            k: Connectivity radius (1 = immediate neighbors).

        Returns:
            List of (row, col) tuples for valid neighbors.
        """
        neighbors = []
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def get_module_mask(self, row: int, col: int, h: int, w: int) -> np.ndarray:
        """Get binary mask for a module.

        Args:
            row: Module row index.
            col: Module column index.
            h: Image height.
            w: Image width.

        Returns:
            Binary mask (1 = module area).
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0, x1, y1 = self.get_module_bounds(row, col)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        mask[y0:y1, x0:x1] = 1
        return mask

    def all_modules(self) -> List[Tuple[int, int]]:
        """Get all module coordinates.

        Returns:
            List of (row, col) tuples.
        """
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]


def calibrate_grid(
    panel_width: int,
    panel_height: int,
    rows: int = 12,
    cols: int = 16,
) -> ModuleGrid:
    """Calibrate module grid by dividing panel evenly.

    Default grid is 12x16 (matching LocationConfig defaults).
    Can be overridden per-location via config.

    Args:
        panel_width: Width of LED panel in pixels.
        panel_height: Height of LED panel in pixels.
        rows: Number of module rows.
        cols: Number of module columns.

    Returns:
        Calibrated ModuleGrid.
    """
    module_width = panel_width // cols
    module_height = panel_height // rows

    return ModuleGrid(
        rows=rows,
        cols=cols,
        module_width=module_width,
        module_height=module_height,
        origin_x=0,
        origin_y=0,
    )


def auto_refine_grid(
    gray: np.ndarray,
    grid: ModuleGrid,
    panel_mask: np.ndarray,
) -> ModuleGrid:
    """Refine grid using projection profile to detect bezel lines.

    LED cabinets have thin bezel gaps between modules that appear as
    dark lines in the projection profile. By finding these lines, we
    can refine the grid to match physical module boundaries.

    Args:
        gray: Grayscale LED panel image.
        grid: Initial calibrated grid.
        panel_mask: Binary mask of LED panel area.

    Returns:
        Refined ModuleGrid (or original if refinement fails).
    """
    h, w = gray.shape

    # Only refine if panel is large enough
    if h < grid.rows * 16 or w < grid.cols * 16:
        return grid

    # Apply panel mask
    masked = gray.copy()
    masked[panel_mask == 0] = 0

    # Horizontal projection (detect horizontal bezel lines)
    h_proj = np.mean(masked, axis=1)

    # Vertical projection (detect vertical bezel lines)
    v_proj = np.mean(masked, axis=0)

    # Find local minima in projections (bezel lines are darker)
    h_mins = _find_projection_minima(h_proj, grid.rows)
    v_mins = _find_projection_minima(v_proj, grid.cols)

    # If we found enough lines, refine the grid
    if len(h_mins) >= grid.rows - 1 and len(v_mins) >= grid.cols - 1:
        # Use detected lines to set module boundaries
        # Lines are at module boundaries, not centers
        if len(h_mins) >= 2:
            avg_module_h = (h_mins[-1] - h_mins[0]) / (len(h_mins) - 1)
            if avg_module_h > 10:
                grid = ModuleGrid(
                    rows=grid.rows,
                    cols=grid.cols,
                    module_width=grid.module_width,
                    module_height=int(avg_module_h),
                    origin_x=grid.origin_x,
                    origin_y=max(0, int(h_mins[0] - avg_module_h * 0.5)),
                )

        if len(v_mins) >= 2:
            avg_module_w = (v_mins[-1] - v_mins[0]) / (len(v_mins) - 1)
            if avg_module_w > 10:
                grid = ModuleGrid(
                    rows=grid.rows,
                    cols=grid.cols,
                    module_width=int(avg_module_w),
                    module_height=grid.module_height,
                    origin_x=max(0, int(v_mins[0] - avg_module_w * 0.5)),
                    origin_y=grid.origin_y,
                )

    return grid


def _find_projection_minima(
    projection: np.ndarray, expected_count: int
) -> List[int]:
    """Find local minima in projection profile.

    Minima correspond to bezel lines (darker gaps between modules).

    Args:
        projection: 1D projection array.
        expected_count: Expected number of minima.

    Returns:
        List of indices where minima occur.
    """
    n = len(projection)
    if n < 10:
        return []

    # Smooth the projection
    kernel_size = max(3, n // (expected_count * 2))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 15)

    smoothed = np.convolve(
        projection,
        np.ones(kernel_size) / kernel_size,
        mode='same',
    )

    # Find local minima
    minima = []
    for i in range(2, n - 2):
        if (smoothed[i] < smoothed[i - 1]
                and smoothed[i] < smoothed[i + 1]
                and smoothed[i] < smoothed[i - 2]
                and smoothed[i] < smoothed[i + 2]):
            minima.append(i)

    # Filter: keep only minima that are significantly below mean
    if minima:
        proj_mean = float(np.mean(smoothed))
        minima = [m for m in minima if smoothed[m] < proj_mean * 0.85]

    # If too many minima, keep the strongest ones
    if len(minima) > expected_count + 2:
        strengths = [(smoothed[m], m) for m in minima]
        strengths.sort()
        minima = [m for _, m in strengths[:expected_count + 2]]
        minima.sort()

    return minima
