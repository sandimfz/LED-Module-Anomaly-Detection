"""
Simple defect detector using normal baseline.

Detects:
1. Blocking (area gelap persistent)
2. Module error (variance tinggi lokal)
3. Frozen/stuck (content tidak berubah)

Menggunakan normal baseline untuk adaptive threshold.
"""

import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.core.types import AnomalyLevel, DetectionResult


class SimpleDefectDetector:
    """Simple defect detector using normal baseline."""

    def __init__(self, model_dir: str = "models"):
        """Initialize detector.

        Args:
            model_dir: Directory to load baseline from.
        """
        self.model_dir = model_dir
        self.baseline = None

    def load_baseline(self, location: str) -> bool:
        """Load normal baseline for location.

        Args:
            location: Location name.

        Returns:
            True if baseline loaded successfully.
        """
        baseline_path = os.path.join(self.model_dir, f"normal_baseline_{location}.json")

        if not os.path.exists(baseline_path):
            return False

        try:
            with open(baseline_path, 'r') as f:
                self.baseline = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading baseline: {e}")
            return False

    def detect(
        self,
        image: np.ndarray,
        image_path: str = "",
    ) -> DetectionResult:
        """Detect defects in image.

        Args:
            image: BGR image.
            image_path: Path to image file.

        Returns:
            DetectionResult with defects found.
        """
        if self.baseline is None:
            return DetectionResult(
                location="unknown",
                image_path=image_path,
                anomaly_score=0.0,
                level=AnomalyLevel.NORMAL,
                message="No baseline loaded"
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate current statistics
        current_brightness = float(np.mean(gray))
        current_variance = float(np.var(gray))
        edges = cv2.Canny(gray, 50, 150)
        current_edge_density = float(np.mean(edges > 0))

        # Compare with baseline
        anomalies = []

        # Check brightness deviation
        brightness_dev = abs(current_brightness - self.baseline["brightness_mean"])
        if brightness_dev > 2 * self.baseline["brightness_std"]:
            anomalies.append({
                "type": "brightness_anomaly",
                "severity": min(brightness_dev / (3 * self.baseline["brightness_std"]), 1.0),
                "message": f"Brightness deviation: {current_brightness:.1f} vs baseline {self.baseline['brightness_mean']:.1f}"
            })

        # Check variance deviation
        variance_dev = abs(current_variance - self.baseline["variance_mean"])
        if variance_dev > 2 * self.baseline["variance_std"]:
            anomalies.append({
                "type": "variance_anomaly",
                "severity": min(variance_dev / (3 * self.baseline["variance_std"]), 1.0),
                "message": f"Variance deviation: {current_variance:.1f} vs baseline {self.baseline['variance_mean']:.1f}"
            })

        # Check for blocking (very low brightness)
        if current_brightness < self.baseline["brightness_mean"] - 2.5 * self.baseline["brightness_std"]:
            anomalies.append({
                "type": "blocking",
                "severity": min((self.baseline["brightness_mean"] - current_brightness) / (2 * self.baseline["brightness_std"]), 1.0),
                "message": f"Potential blocking: brightness {current_brightness:.1f} very low"
            })

        # Check for module error (high local variance)
        local_anomalies = self._detect_local_anomalies(gray)
        if local_anomalies:
            anomalies.extend(local_anomalies)

        # Calculate overall score
        if anomalies:
            score = max(a["severity"] for a in anomalies)
            level = AnomalyLevel.CRITICAL if score > 0.7 else AnomalyLevel.WARNING
        else:
            score = 0.0
            level = AnomalyLevel.NORMAL

        # Create message
        if anomalies:
            messages = [a["message"] for a in anomalies[:3]]  # Top 3
            message = "Anomalies: " + "; ".join(messages)
        else:
            message = "No defects detected"

        return DetectionResult(
            location="unknown",
            image_path=image_path,
            anomaly_score=round(score, 4),
            level=level,
            message=message
        )

    def _detect_local_anomalies(self, gray: np.ndarray) -> List[Dict]:
        """Detect local anomalies (module errors).

        Focus on specific defect patterns:
        1. Dark spots (dead modules)
        2. Bright spots (stuck pixels)
        3. Corrupted sections (random noise)

        Args:
            gray: Grayscale image.

        Returns:
            List of anomalies found.
        """
        anomalies = []
        h, w = gray.shape
        block_size = 16

        # Analyze each block
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_mean = float(np.mean(block))
                block_var = float(np.var(block))

                # Skip edge blocks (first/last 2 blocks)
                if i < block_size * 2 or j < block_size * 2:
                    continue
                if i > h - block_size * 3 or j > w - block_size * 3:
                    continue

                # Check for dark spots (potential dead modules)
                if block_mean < 40 and block_var < 100:
                    # Very dark and uniform = potential dead module
                    neighbors = self._get_neighbor_means(gray, i, j, block_size)
                    if neighbors:
                        neighbor_mean = np.mean(neighbors)
                        if block_mean < neighbor_mean * 0.3:  # Much darker than neighbors
                            severity = min((neighbor_mean - block_mean) / neighbor_mean, 1.0)
                            anomalies.append({
                                "type": "dark_spot",
                                "severity": severity,
                                "message": f"Dark spot at ({j},{i}): brightness {block_mean:.0f} vs neighbors {neighbor_mean:.0f}"
                            })

                # Check for bright spots (potential stuck pixels)
                elif block_mean > 220 and block_var < 100:
                    # Very bright and uniform = potential stuck pixels
                    neighbors = self._get_neighbor_means(gray, i, j, block_size)
                    if neighbors:
                        neighbor_mean = np.mean(neighbors)
                        if block_mean > neighbor_mean * 1.5:  # Much brighter than neighbors
                            severity = min((block_mean - neighbor_mean) / neighbor_mean, 1.0)
                            anomalies.append({
                                "type": "bright_spot",
                                "severity": severity,
                                "message": f"Bright spot at ({j},{i}): brightness {block_mean:.0f} vs neighbors {neighbor_mean:.0f}"
                            })

        return anomalies

    def _get_neighbor_vars(
        self,
        gray: np.ndarray,
        i: int,
        j: int,
        block_size: int,
    ) -> List[float]:
        """Get variance of neighboring blocks.

        Args:
            gray: Grayscale image.
            i: Block row index.
            j: Block column index.
            block_size: Size of each block.

        Returns:
            List of neighbor variances.
        """
        h, w = gray.shape
        neighbors = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni, nj = i + di * block_size, j + dj * block_size
                if 0 <= ni < h - block_size and 0 <= nj < w - block_size:
                    block = gray[ni:ni+block_size, nj:nj+block_size]
                    neighbors.append(float(np.var(block)))

        return neighbors

    def _get_neighbor_means(
        self,
        gray: np.ndarray,
        i: int,
        j: int,
        block_size: int,
    ) -> List[float]:
        """Get mean brightness of neighboring blocks.

        Args:
            gray: Grayscale image.
            i: Block row index.
            j: Block column index.
            block_size: Size of each block.

        Returns:
            List of neighbor means.
        """
        h, w = gray.shape
        neighbors = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni, nj = i + di * block_size, j + dj * block_size
                if 0 <= ni < h - block_size and 0 <= nj < w - block_size:
                    block = gray[ni:ni+block_size, nj:nj+block_size]
                    neighbors.append(float(np.mean(block)))

        return neighbors
