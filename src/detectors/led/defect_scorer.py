"""
Defect Scorer for LED anomaly detection.

Combines signals from multiple detectors into a unified per-module score
and global panel score. Uses voting/consensus for robust classification.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detectors.led.feature_extractor import ModuleFeatures
from src.detectors.led.spatial_analyzer import SpatialResult
from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


@dataclass
class ModuleScore:
    """Score for a single module."""
    row: int
    col: int
    # Combined score (0-1)
    score: float = 0.0
    # Individual signal scores
    spatial_score: float = 0.0
    flat_score: float = 0.0
    blocking_score: float = 0.0
    # Classification
    level: str = "normal"  # normal, warning, critical


class DefectScorer:
    """Combine detector signals into unified scores.

    Per-module scoring:
    - Spatial decorrelation z-score (primary)
    - Flat white stuck detection (high weight)
    - Multiple detector agreement boost

    Global scoring:
    - Area ratio of anomalous modules
    - Average severity
    - Type diversity boost
    """

    def __init__(
        self,
        spatial_weight: float = 0.5,
        flat_weight: float = 0.3,
        other_weight: float = 0.2,
    ):
        """Initialize scorer.

        Args:
            spatial_weight: Weight for spatial decorrelation signal.
            flat_weight: Weight for flat/stuck detection.
            other_weight: Weight for other detectors.
        """
        self.spatial_weight = spatial_weight
        self.flat_weight = flat_weight
        self.other_weight = other_weight

    def score_modules(
        self,
        spatial_results: List[SpatialResult],
        features: Dict[Tuple[int, int], ModuleFeatures],
        module_count: int,
    ) -> List[ModuleScore]:
        """Score each module based on multiple signals.

        Args:
            spatial_results: Results from SpatialAnalyzer.
            features: Per-module features.
            module_count: Total number of modules.

        Returns:
            List of ModuleScore for each module.
        """
        scores: List[ModuleScore] = []

        for sr in spatial_results:
            feat = features.get((sr.row, sr.col))
            ms = ModuleScore(row=sr.row, col=sr.col)

            # Spatial signal
            ms.spatial_score = min(sr.z_score / 6.0, 1.0) if sr.is_anomaly else 0.0

            # Flat stuck signal
            if feat and feat.white_stuck:
                ms.flat_score = 0.9
            elif feat and feat.flat and sr.z_score > 2.0:
                ms.flat_score = min(sr.z_score / 5.0, 0.7)

            # Combined score
            ms.score = (
                ms.spatial_score * self.spatial_weight
                + ms.flat_score * self.flat_weight
            )

            # Level classification
            if ms.score > 0.6:
                ms.level = "critical"
            elif ms.score > 0.3:
                ms.level = "warning"
            else:
                ms.level = "normal"

            scores.append(ms)

        return scores

    def compute_global_score(
        self,
        module_scores: List[ModuleScore],
        anomalies: List[LEDAnomaly],
        panel: LEDPanelInfo,
    ) -> Tuple[float, str]:
        """Compute global panel score from module scores.

        Args:
            module_scores: Per-module scores.
            anomalies: Anomalies from sub-detectors.
            panel: LED panel information.

        Returns:
            Tuple of (score, level).
        """
        if not module_scores:
            return 0.0, "normal"

        # Module-based score
        scores_arr = np.array([ms.score for ms in module_scores])
        anomaly_ratio = float(np.mean(scores_arr > 0.3))
        avg_score = float(np.mean(scores_arr))
        max_score = float(np.max(scores_arr))

        # Module-based component
        module_score = min(anomaly_ratio * 3.0 + avg_score * 0.5, 1.0)

        # Detector-based component (from existing detectors)
        if anomalies:
            det_area = sum(a.width * a.height for a in anomalies)
            det_ratio = det_area / (panel.width * panel.height) if panel.width * panel.height > 0 else 0
            det_severity = float(np.mean([a.severity for a in anomalies]))
            det_score = min(det_ratio / 0.3 * 0.5 + det_severity * 0.5, 1.0)
        else:
            det_score = 0.0

        # Combine: max of module-based and detector-based
        # (either one can catch defects the other misses)
        final_score = max(module_score, det_score)

        # Boost if both agree (multiple signals)
        if module_score > 0.3 and det_score > 0.3:
            final_score = max(final_score, (module_score + det_score) / 2 * 1.2)

        final_score = min(final_score, 1.0)

        # Level
        if final_score >= 0.55:
            level = "critical"
        elif final_score >= 0.30:
            level = "warning"
        else:
            level = "normal"

        return final_score, level
