"""
Spatial Analyzer for LED module defect detection.

Compares each module to its neighbors to detect spatial decorrelation.
Defect modules (stuck, blocking, module error) are decorrelated from
their neighbors — they have significantly different brightness/color
while neighbors follow the content pattern.

Key insight from analysis:
- Defect stuck modules are isolated among colorful content
- Normal content transitions are smooth between modules
- MAD-based z-score is robust to outlier neighbors
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.detectors.led.feature_extractor import FeatureExtractor, ModuleFeatures
from src.detectors.led.module_grid import ModuleGrid
from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


@dataclass
class SpatialResult:
    """Spatial analysis result for a single module."""
    row: int
    col: int
    # Z-score: how many MADs away from neighbor median
    z_score: float = 0.0
    # Neighbor median brightness (relative)
    neighbor_median: float = 0.0
    # Neighbor MAD
    neighbor_mad: float = 0.0
    # Is anomalous
    is_anomaly: bool = False
    # Anomaly type
    anomaly_type: str = ""
    # Severity
    severity: float = 0.0


class SpatialAnalyzer:
    """Detect defects via spatial decorrelation analysis.

    For each module, computes z-score against neighbor median.
    High z-score = module is decorrelated from neighbors = potential defect.

    Also detects flat stuck modules by checking if a cluster of
    flat modules is surrounded by varied content.
    """

    def __init__(
        self,
        z_threshold: float = 4.5,
        min_cluster_size: int = 3,
    ):
        """Initialize spatial analyzer.

        Args:
            z_threshold: Z-score threshold for anomaly detection.
            min_cluster_size: Minimum cluster size for defect detection.
        """
        self.z_threshold = z_threshold
        self.min_cluster_size = min_cluster_size

    def analyze(
        self,
        features: Dict[Tuple[int, int], ModuleFeatures],
        grid: ModuleGrid,
        panel: LEDPanelInfo,
    ) -> Tuple[List[SpatialResult], List[LEDAnomaly]]:
        """Analyze spatial decorrelation across all modules.

        Args:
            features: Per-module features from FeatureExtractor.
            grid: Calibrated ModuleGrid.
            panel: LED panel information.

        Returns:
            Tuple of (per-module results, merged anomalies).
        """
        results: List[SpatialResult] = []
        anomaly_cells: Set[Tuple[int, int]] = set()

        for row, col in grid.all_modules():
            feat = features.get((row, col))
            if feat is None or not feat.valid:
                results.append(SpatialResult(row=row, col=col))
                continue

            result = self._analyze_module(row, col, feat, features, grid)
            results.append(result)

            if result.is_anomaly:
                anomaly_cells.add((row, col))

        # Merge adjacent anomalies into regions
        anomalies = self._merge_anomalies(anomaly_cells, features, grid, panel)

        return results, anomalies

    def _analyze_module(
        self,
        row: int,
        col: int,
        feat: ModuleFeatures,
        features: Dict[Tuple[int, int], ModuleFeatures],
        grid: ModuleGrid,
    ) -> SpatialResult:
        """Analyze a single module against its neighbors.

        Args:
            row: Module row.
            col: Module column.
            feat: Module features.
            features: All module features.
            grid: Module grid.

        Returns:
            SpatialResult for this module.
        """
        result = SpatialResult(row=row, col=col)

        # Get valid neighbor features
        neighbor_feats = []
        for nr, nc in grid.get_neighbors(row, col):
            nf = features.get((nr, nc))
            if nf is not None and nf.valid:
                neighbor_feats.append(nf)

        if len(neighbor_feats) < 2:
            return result

        # Robust comparison using median + MAD
        neighbor_brightness = np.array([nf.brightness for nf in neighbor_feats])
        neighbor_median = float(np.median(neighbor_brightness))
        neighbor_mad = float(np.median(np.abs(neighbor_brightness - neighbor_median)))
        robust_std = max(neighbor_mad * 1.4826, 5.0)

        z_score = abs(feat.brightness - neighbor_median) / robust_std

        result.neighbor_median = neighbor_median
        result.neighbor_mad = neighbor_mad
        result.z_score = z_score

        # Check if anomaly
        if z_score > self.z_threshold:
            result.is_anomaly = True
            result.anomaly_type = "spatial_decorrelation"
            result.severity = min(z_score / 8.0, 1.0)

        # White stuck special case: very flat + bright + low sat + high z
        if (feat.white_stuck and z_score > 2.5):
            result.is_anomaly = True
            result.anomaly_type = "flat_white_stuck"
            result.severity = min(0.8 + z_score / 10.0, 1.0)

        return result

    def _merge_anomalies(
        self,
        anomaly_cells: Set[Tuple[int, int]],
        features: Dict[Tuple[int, int], ModuleFeatures],
        grid: ModuleGrid,
        panel: LEDPanelInfo,
    ) -> List[LEDAnomaly]:
        """Merge adjacent anomalous modules into regions.

        Args:
            anomaly_cells: Set of (row, col) for anomalous modules.
            features: Per-module features.
            grid: Module grid.
            panel: LED panel information.

        Returns:
            List of merged LEDAnomaly objects.
        """
        if not anomaly_cells:
            return []

        visited: Set[Tuple[int, int]] = set()
        anomalies: List[LEDAnomaly] = []

        for start in anomaly_cells:
            if start in visited:
                continue

            # BFS to find connected component
            cluster: List[Tuple[int, int]] = []
            queue = [start]
            while queue:
                cell = queue.pop(0)
                if cell in visited or cell not in anomaly_cells:
                    continue
                visited.add(cell)
                cluster.append(cell)
                for nr, nc in grid.get_neighbors(cell[0], cell[1]):
                    if (nr, nc) not in visited and (nr, nc) in anomaly_cells:
                        queue.append((nr, nc))

            if len(cluster) < self.min_cluster_size:
                continue

            # Compute bounding box
            rows_c = [r for r, _ in cluster]
            cols_c = [c for _, c in cluster]
            min_r, max_r = min(rows_c), max(rows_c)
            min_c, max_c = min(cols_c), max(cols_c)

            x0 = min_c * grid.module_width + grid.origin_x + panel.x
            y0 = min_r * grid.module_height + grid.origin_y + panel.y
            w = (max_c - min_c + 1) * grid.module_width
            h = (max_r - min_r + 1) * grid.module_height

            # Severity: based on cluster size and average z-score
            avg_z = np.mean([
                features[(r, c)].brightness
                for r, c in cluster
                if (r, c) in features
            ]) if cluster else 0.0

            severity = min(len(cluster) / 10.0 + abs(avg_z) / 50.0, 1.0)

            # Determine type based on shape
            aspect = w / h if h > 0 else 1.0
            if aspect > 3.0:
                atype = "horizontal_defect"
                desc = f"Spatial defect horizontal ({len(cluster)} modules)"
            elif aspect < 0.33:
                atype = "vertical_defect"
                desc = f"Spatial defect vertical ({len(cluster)} modules)"
            else:
                atype = "spatial_defect"
                desc = f"Spatial defect ({len(cluster)} modules, {w}x{h}px)"

            anomalies.append(LEDAnomaly(
                x=x0, y=y0, width=w, height=h,
                anomaly_type=atype,
                severity=severity,
                description=desc,
            ))

        return anomalies
