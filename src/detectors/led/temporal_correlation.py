"""
Temporal Correlation Analyzer for LED defect detection.

Compares per-module signal to global panel signal across N frames.
Key insight: defect modules don't change with content, normal modules do.

signal_M = [brightness_norm_M per frame]
global_signal = [global_median per frame]
corr = pearson(signal_M, global_signal)

- corr < 0.3 AND spatial anomaly → defect (module doesn't follow content)
- corr > 0.7 → normal (module follows content changes)
- std(signal_M) < 2 AND std(global) > 10 → frozen
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.types import AnomalyLevel, DetectionResult
from src.detectors.led.feature_extractor import FeatureExtractor
from src.detectors.led.module_grid import ModuleGrid, calibrate_grid


class TemporalCorrelationAnalyzer:
    """Detect defects via temporal correlation analysis.

    Buffers N frames per location and computes per-module correlation
    with global panel signal. Low correlation = potential defect.
    """

    def __init__(
        self,
        buffer_size: int = 8,
        corr_threshold_low: float = 0.3,
        corr_threshold_high: float = 0.7,
        frozen_threshold: float = 2.0,
    ):
        """Initialize temporal correlation analyzer.

        Args:
            buffer_size: Maximum frames to keep.
            corr_threshold_low: Below this = likely defect.
            corr_threshold_high: Above this = likely normal.
            frozen_threshold: Below this std = frozen detection.
        """
        self.buffer_size = buffer_size
        self.corr_threshold_low = corr_threshold_low
        self.corr_threshold_high = corr_threshold_high
        self.frozen_threshold = frozen_threshold

        # Per-location buffers
        self.brightness_buffers: Dict[str, List[Dict[Tuple[int, int], float]]] = defaultdict(list)
        self.global_buffers: Dict[str, List[float]] = defaultdict(list)

    def add_frame(
        self,
        location: str,
        gray: np.ndarray,
        hsv: np.ndarray,
        panel_mask: np.ndarray,
        rows: int = 12,
        cols: int = 16,
    ) -> None:
        """Add a frame to the temporal buffer.

        Extracts per-module brightness and global panel brightness.
        Stores relative values (delta from global median).

        Args:
            location: Location name.
            gray: Grayscale image.
            hsv: HSV image.
            panel_mask: Panel mask.
            rows: Module grid rows.
            cols: Module grid cols.
        """
        h, w = gray.shape
        grid = calibrate_grid(w, h, rows=rows, cols=cols)
        extractor = FeatureExtractor(gray, hsv, panel_mask, grid)
        features = extractor.extract()

        # Store relative brightness per module
        module_data = {}
        for (r, c), feat in features.items():
            if feat.valid:
                module_data[(r, c)] = feat.brightness  # relative to global

        self.brightness_buffers[location].append(module_data)
        self.global_buffers[location].append(extractor.get_global_signal())

        # Keep only last N frames
        if len(self.brightness_buffers[location]) > self.buffer_size:
            self.brightness_buffers[location].pop(0)
            self.global_buffers[location].pop(0)

    def analyze(
        self,
        location: str,
        rows: int = 12,
        cols: int = 16,
    ) -> DetectionResult:
        """Analyze temporal patterns for a location.

        Args:
            location: Location name.
            rows: Module grid rows.
            cols: Module grid cols.

        Returns:
            DetectionResult with temporal analysis findings.
        """
        frames = self.brightness_buffers.get(location, [])
        global_frames = self.global_buffers.get(location, [])

        if len(frames) < 3:
            return DetectionResult(
                location=location,
                image_path="",
                anomaly_score=0.0,
                level=AnomalyLevel.NORMAL,
                message=f"Temporal: Need more frames ({len(frames)}/{self.buffer_size})",
            )

        n_frames = len(frames)

        # Compute per-module temporal correlation
        module_corrs: Dict[Tuple[int, int], float] = {}
        module_stds: Dict[Tuple[int, int], float] = {}
        module_frozen: Dict[Tuple[int, int], bool] = {}

        # Get all module keys across all frames
        all_modules = set()
        for frame_data in frames:
            all_modules.update(frame_data.keys())

        global_signal = np.array(global_frames[:n_frames])
        global_std = float(np.std(global_signal))

        for module in all_modules:
            # Build signal for this module across frames
            signal = []
            for frame_data in frames:
                signal.append(frame_data.get(module, 0.0))

            signal_arr = np.array(signal)
            module_stds[module] = float(np.std(signal_arr))

            # Frozen detection: module doesn't change at all
            is_frozen = (
                module_stds[module] < self.frozen_threshold
                and global_std > 10  # panel is changing
            )
            module_frozen[module] = is_frozen

            # Correlation with global signal
            if n_frames >= 3 and global_std > 5 and module_stds[module] > 1:
                # Pearson correlation
                corr = np.corrcoef(signal_arr, global_signal)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                module_corrs[module] = float(corr)
            else:
                # Can't compute correlation (no variation)
                module_corrs[module] = 0.0 if is_frozen else 0.5

        # Count anomalies
        low_corr_count = sum(
            1 for c in module_corrs.values()
            if c < self.corr_threshold_low
        )
        frozen_count = sum(1 for f in module_frozen.values() if f)
        total_modules = len(all_modules)

        # Score based on anomaly ratio
        anomaly_ratio = (low_corr_count + frozen_count) / max(total_modules, 1)

        # Calculate overall score
        score = min(anomaly_ratio * 3.0, 1.0)

        # Level
        if score >= 0.55:
            level = AnomalyLevel.CRITICAL
        elif score >= 0.30:
            level = AnomalyLevel.WARNING
        else:
            level = AnomalyLevel.NORMAL

        # Generate message
        parts = []
        if frozen_count > 0:
            parts.append(f"{frozen_count} frozen modules")
        if low_corr_count > 0:
            parts.append(f"{low_corr_count} decorrelated modules")

        if parts:
            message = f"Temporal: {', '.join(parts)} ({n_frames} frames)"
        else:
            message = f"Temporal: Normal content variation ({n_frames} frames)"

        return DetectionResult(
            location=location,
            image_path="",
            anomaly_score=round(score, 4),
            level=level,
            message=message,
        )

    def get_module_correlations(
        self, location: str
    ) -> Dict[Tuple[int, int], float]:
        """Get per-module correlation values for external use.

        Args:
            location: Location name.

        Returns:
            Dict mapping (row, col) to correlation value.
        """
        frames = self.brightness_buffers.get(location, [])
        global_frames = self.global_buffers.get(location, [])

        if len(frames) < 3:
            return {}

        n_frames = len(frames)
        global_signal = np.array(global_frames[:n_frames])

        all_modules = set()
        for frame_data in frames:
            all_modules.update(frame_data.keys())

        corrs = {}
        for module in all_modules:
            signal = np.array([frame_data.get(module, 0.0) for frame_data in frames])
            g_std = float(np.std(global_signal))
            s_std = float(np.std(signal))

            if n_frames >= 3 and g_std > 5 and s_std > 1:
                c = np.corrcoef(signal, global_signal)[0, 1]
                corrs[module] = float(c) if not np.isnan(c) else 0.0
            else:
                corrs[module] = 0.0

        return corrs

    def get_stats(self, location: str) -> Dict:
        """Get temporal analysis statistics.

        Args:
            location: Location name.

        Returns:
            Dictionary with statistics.
        """
        return {
            "location": location,
            "buffer_size": len(self.brightness_buffers.get(location, [])),
            "max_buffer_size": self.buffer_size,
            "total_frames_processed": len(self.brightness_buffers.get(location, [])),
        }
