"""
Temporal analyzer for LED anomaly detection.

Compares multiple frames over time to detect:
1. Frozen/stuck displays (content doesn't change)
2. Persistent defects (appear across multiple content changes)
3. Normal content variation (expected changes)

This solves the main problem: single-image analysis cannot distinguish
normal uniform content from frozen/stuck displays.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.types import AnomalyLevel, DetectionResult


class TemporalAnalyzer:
    """Temporal analyzer for LED anomaly detection.

    Compares multiple frames over time to detect anomalies that
    single-image analysis cannot catch.
    """

    def __init__(
        self,
        buffer_size: int = 10,
        frozen_threshold: float = 0.01,
        defect_persistence_threshold: float = 0.8,
    ):
        """Initialize temporal analyzer.

        Args:
            buffer_size: Maximum number of frames to keep per location.
            frozen_threshold: Threshold for detecting frozen content.
                Lower = more sensitive to frozen detection.
            defect_persistence_threshold: Threshold for detecting persistent defects.
                Higher = require more frames to confirm defect.
        """
        self.buffer_size = buffer_size
        self.frozen_threshold = frozen_threshold
        self.defect_persistence_threshold = defect_persistence_threshold

        # Frame buffers per location
        self.buffers: Dict[str, List[np.ndarray]] = defaultdict(list)

        # Defect tracking per location
        self.defect_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.frame_counts: Dict[str, int] = defaultdict(int)

    def add_frame(self, location: str, frame: np.ndarray) -> None:
        """Add a frame to the buffer for a location.

        Args:
            location: Location name.
            frame: BGR image frame.
        """
        # Resize for consistent comparison
        resized = cv2.resize(frame, (640, 480))

        # Add to buffer
        self.buffers[location].append(resized)

        # Keep only last N frames
        if len(self.buffers[location]) > self.buffer_size:
            self.buffers[location].pop(0)

        # Increment frame count
        self.frame_counts[location] += 1

    def analyze(self, location: str) -> DetectionResult:
        """Analyze temporal patterns for a location.

        Args:
            location: Location name.

        Returns:
            DetectionResult with temporal analysis findings.
        """
        frames = self.buffers.get(location, [])

        if len(frames) < 3:
            return DetectionResult(
                location=location,
                image_path="",
                anomaly_score=0.0,
                level=AnomalyLevel.NORMAL,
                message=f"Temporal: Need more frames ({len(frames)}/{self.buffer_size})"
            )

        # Check for frozen content
        is_frozen, frozen_score = self._detect_frozen(frames)

        # Check for persistent defects
        has_persistent_defect, defect_score = self._detect_persistent_defect(frames)

        # Calculate overall score
        if is_frozen:
            score = frozen_score
            level = AnomalyLevel.CRITICAL
            message = f"Temporal: FROZEN/STUCK detected (score={frozen_score:.4f})"
        elif has_persistent_defect:
            score = defect_score
            level = AnomalyLevel.WARNING
            message = f"Temporal: Persistent defect detected (score={defect_score:.4f})"
        else:
            score = 0.0
            level = AnomalyLevel.NORMAL
            message = f"Temporal: Normal content variation ({len(frames)} frames)"

        return DetectionResult(
            location=location,
            image_path="",
            anomaly_score=round(score, 4),
            level=level,
            message=message
        )

    def _detect_frozen(self, frames: List[np.ndarray]) -> Tuple[bool, float]:
        """Detect if content is frozen/stuck.

        Args:
            frames: List of frames.

        Returns:
            Tuple of (is_frozen, score).
        """
        if len(frames) < 3:
            return False, 0.0

        # Calculate differences between consecutive frames
        diffs = []
        for i in range(1, len(frames)):
            # Convert to grayscale for comparison
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            mean_diff = float(np.mean(diff))
            diffs.append(mean_diff)

        # Calculate statistics
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))

        # Frozen detection: very low variation between frames
        # Also check if all differences are very small
        all_small = all(d < self.frozen_threshold * 100 for d in diffs)
        mean_low = mean_diff < self.frozen_threshold * 100

        is_frozen = mean_low and all_small

        # Score: inverse of mean difference (higher = more frozen)
        score = max(0.0, 1.0 - (mean_diff / (self.frozen_threshold * 200)))

        return is_frozen, min(score, 1.0)

    def _detect_persistent_defect(self, frames: List[np.ndarray]) -> Tuple[bool, float]:
        """Detect persistent defects that appear across multiple frames.

        Args:
            frames: List of frames.

        Returns:
            Tuple of (has_defect, score).
        """
        if len(frames) < 3:
            return False, 0.0

        # Convert all frames to grayscale
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # Calculate mean frame (average of all frames)
        mean_frame = np.mean(gray_frames, axis=0).astype(np.uint8)

        # Calculate deviation from mean for each frame
        deviations = []
        for gray in gray_frames:
            # Calculate how much each pixel deviates from mean
            deviation = np.abs(gray.astype(float) - mean_frame.astype(float))
            deviations.append(float(np.mean(deviation)))

        # Persistent defect: areas that are consistently different from mean
        # This could indicate dead modules, stuck pixels, etc.
        mean_deviation = float(np.mean(deviations))

        # Check for dark spots that persist across frames
        dark_persistence = self._check_dark_persistence(gray_frames)

        # Check for bright spots that persist across frames
        bright_persistence = self._check_bright_persistence(gray_frames)

        # Calculate score based on persistence
        persistence_score = max(dark_persistence, bright_persistence)

        has_defect = persistence_score > self.defect_persistence_threshold

        return has_defect, persistence_score

    def _check_dark_persistence(self, gray_frames: List[np.ndarray]) -> float:
        """Check for persistent dark areas across frames.

        Args:
            gray_frames: List of grayscale frames.

        Returns:
            Persistence score (0-1).
        """
        if len(gray_frames) < 3:
            return 0.0

        # Find dark pixels in each frame
        dark_masks = []
        for gray in gray_frames:
            # Dark pixels (threshold can be adjusted)
            dark_mask = (gray < 80).astype(np.uint8)
            dark_masks.append(dark_mask)

        # Calculate persistence: how often each pixel is dark
        persistence = np.mean(dark_masks, axis=0)

        # High persistence = consistently dark = potential defect
        high_persistence = np.mean(persistence > 0.7)

        return float(high_persistence)

    def _check_bright_persistence(self, gray_frames: List[np.ndarray]) -> float:
        """Check for persistent bright areas across frames.

        Args:
            gray_frames: List of grayscale frames.

        Returns:
            Persistence score (0-1).
        """
        if len(gray_frames) < 3:
            return 0.0

        # Find bright pixels in each frame
        bright_masks = []
        for gray in gray_frames:
            # Bright pixels (threshold can be adjusted)
            bright_mask = (gray > 200).astype(np.uint8)
            bright_masks.append(bright_mask)

        # Calculate persistence: how often each pixel is bright
        persistence = np.mean(bright_masks, axis=0)

        # High persistence = consistently bright = potential stuck pixels
        high_persistence = np.mean(persistence > 0.7)

        return float(high_persistence)

    def get_stats(self, location: str) -> Dict:
        """Get temporal analysis statistics for a location.

        Args:
            location: Location name.

        Returns:
            Dictionary with statistics.
        """
        frames = self.buffers.get(location, [])

        return {
            "location": location,
            "buffer_size": len(frames),
            "max_buffer_size": self.buffer_size,
            "total_frames_processed": self.frame_counts.get(location, 0),
            "frozen_threshold": self.frozen_threshold,
            "defect_persistence_threshold": self.defect_persistence_threshold,
        }
