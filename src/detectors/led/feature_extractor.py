"""
Feature Extractor for per-module LED analysis.

Computes normalized features per module using global statistics.
Key insight: use RELATIVE features (delta from global median) instead
of ABSOLUTE features (raw mean brightness).

This is the foundation for:
- Spatial decorrelation detection (module vs neighbors)
- Temporal correlation (module signal vs global signal)
- Baseline comparison (module vs historical normal)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detectors.led.module_grid import ModuleGrid


@dataclass
class ModuleFeatures:
    """Features for a single LED module.

    All brightness values are RELATIVE to global median (normalized).
    """
    row: int
    col: int
    # Relative brightness (delta from global median)
    brightness: float = 0.0
    # Absolute brightness
    brightness_abs: float = 0.0
    # Brightness standard deviation (flatness indicator)
    std: float = 0.0
    # Flatness flag (std < 8 = very flat)
    flat: bool = False
    # HSV features
    hue_median: float = 0.0
    sat_mean: float = 0.0
    # Edge density (0-1)
    edge_density: float = 0.0
    # Texture (Laplacian variance)
    texture: float = 0.0
    # Saturation low flag (potential defect)
    sat_low: bool = False
    # White stuck flag (bright + flat + low sat)
    white_stuck: bool = False
    # Validity
    valid: bool = True


class FeatureExtractor:
    """Extract per-module features with global normalization.

    Usage:
        extractor = FeatureExtractor(gray, hsv, panel_mask, module_grid)
        features = extractor.extract()
        # features is Dict[(row, col), ModuleFeatures]
    """

    def __init__(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        panel_mask: np.ndarray,
        grid: ModuleGrid,
    ):
        """Initialize feature extractor.

        Args:
            gray: Grayscale LED panel image.
            hsv: HSV LED panel image.
            panel_mask: Binary mask of LED panel area.
            grid: Calibrated ModuleGrid.
        """
        self.gray = gray
        self.hsv = hsv
        self.panel_mask = panel_mask
        self.grid = grid
        self.h, self.w = gray.shape

        # Compute global robust statistics
        panel_pixels = gray[panel_mask > 0]
        if len(panel_pixels) > 0:
            self.global_median = float(np.median(panel_pixels))
            self.global_mad = float(
                np.median(np.abs(panel_pixels - self.global_median))
            )
            self.global_std = float(np.std(panel_pixels))
        else:
            self.global_median = 128.0
            self.global_mad = 30.0
            self.global_std = 50.0

        # Edge map for edge density computation
        self.edges = cv2.Canny(gray, 50, 150)

    def extract(self) -> Dict[Tuple[int, int], ModuleFeatures]:
        """Extract features for all modules.

        Returns:
            Dict mapping (row, col) to ModuleFeatures.
        """
        features: Dict[Tuple[int, int], ModuleFeatures] = {}

        for row, col in self.grid.all_modules():
            feat = self._extract_module(row, col)
            features[(row, col)] = feat

        return features

    def _extract_module(self, row: int, col: int) -> ModuleFeatures:
        """Extract features for a single module.

        Args:
            row: Module row index.
            col: Module column index.

        Returns:
            ModuleFeatures for this module.
        """
        x0, y0, x1, y1 = self.grid.get_module_bounds(row, col)

        # Clip to image bounds
        x0 = max(0, min(x0, self.w))
        y0 = max(0, min(y0, self.h))
        x1 = max(0, min(x1, self.w))
        y1 = max(0, min(y1, self.h))

        if x1 <= x0 or y1 <= y0:
            return ModuleFeatures(row=row, col=col, valid=False)

        # Module masks
        module_mask = self.panel_mask[y0:y1, x0:x1]
        module_ratio = float(np.mean(module_mask))

        if module_ratio < 0.2:
            return ModuleFeatures(row=row, col=col, valid=False)

        # Extract pixel data
        module_gray = self.gray[y0:y1, x0:x1]
        module_hsv = self.hsv[y0:y1, x0:x1]

        pixels = module_gray[module_mask > 0]
        if len(pixels) == 0:
            return ModuleFeatures(row=row, col=col, valid=False)

        # Brightness features (relative + absolute)
        bmean = float(np.mean(pixels))
        brightness_rel = bmean - self.global_median

        # Brightness std (flatness)
        bstd = float(np.std(pixels))

        # HSV features
        hue_pixels = module_hsv[:, :, 0][module_mask > 0]
        sat_pixels = module_hsv[:, :, 1][module_mask > 0]

        hue_med = float(np.median(hue_pixels)) if len(hue_pixels) > 0 else 0.0
        sat_mean = float(np.mean(sat_pixels)) if len(sat_pixels) > 0 else 0.0

        # Edge density
        module_edges = self.edges[y0:y1, x0:x1]
        edge_pixels = np.sum(module_edges[module_mask > 0] > 0)
        total_pixels = np.sum(module_mask > 0)
        edge_density = float(edge_pixels / total_pixels) if total_pixels > 0 else 0.0

        # Texture (Laplacian variance)
        laplacian = cv2.Laplacian(module_gray, cv2.CV_64F)
        texture = float(np.var(laplacian[module_mask > 0])) if np.sum(module_mask > 0) > 0 else 0.0

        # Flags
        flat = bstd < 8.0
        sat_low = sat_mean < 30.0
        white_stuck = bmean > 200.0 and bstd < 10.0 and sat_low

        return ModuleFeatures(
            row=row,
            col=col,
            brightness=brightness_rel,
            brightness_abs=bmean,
            std=bstd,
            flat=flat,
            hue_median=hue_med,
            sat_mean=sat_mean,
            edge_density=edge_density,
            texture=texture,
            sat_low=sat_low,
            white_stuck=white_stuck,
            valid=True,
        )

    def get_global_signal(self) -> float:
        """Get global panel brightness (for temporal correlation).

        Returns:
            Global median brightness.
        """
        return self.global_median

    def get_global_stats(self) -> Dict[str, float]:
        """Get global panel statistics.

        Returns:
            Dict with global_median, global_mad, global_std.
        """
        return {
            "global_median": self.global_median,
            "global_mad": self.global_mad,
            "global_std": self.global_std,
        }
