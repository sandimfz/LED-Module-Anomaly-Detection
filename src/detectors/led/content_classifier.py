"""
Content classification module for LED anomaly detection.

Classifies LED content type to enable adaptive thresholds
per content type. This reduces false positives by understanding
what type of content is being displayed.
"""

from enum import Enum
from typing import Tuple

import cv2
import numpy as np


class ContentType(Enum):
    """LED content types."""
    SOCCER = "soccer"
    TEXT_AD = "text_ad"
    VIDEO = "video"
    DARK_SCREEN = "dark_screen"
    SOLID_COLOR = "solid_color"
    UNKNOWN = "unknown"


class ContentClassifier:
    """Classify LED content type for adaptive thresholds.

    Uses rule-based analysis to determine content type:
    - Soccer: uniform green field, low variation
    - Text ad: high contrast, many edges
    - Video: varied colors, moderate variation
    - Dark screen: very low brightness
    - Solid color: very uniform, single color
    """

    def __init__(self):
        """Initialize classifier."""
        pass

    def classify(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        led_mask: np.ndarray,
    ) -> Tuple[ContentType, float]:
        """Classify content type.

        Args:
            gray: Grayscale LED region.
            hsv: HSV LED region.
            led_mask: Binary mask of LED content.

        Returns:
            Tuple of (content_type, confidence).
        """
        led_pixels = gray[led_mask > 0]
        if len(led_pixels) == 0:
            return ContentType.DARK_SCREEN, 1.0

        mean_brightness = float(np.mean(led_pixels))
        std_brightness = float(np.std(led_pixels))

        # Dark screen
        if mean_brightness < 40:
            return ContentType.DARK_SCREEN, 1.0

        # Solid color (very uniform)
        if std_brightness < 8:
            return ContentType.SOLID_COLOR, 0.8

        # Check for green dominant content (soccer field)
        hue = hsv[:, :, 0][led_mask > 0]
        sat = hsv[:, :, 1][led_mask > 0]

        # Green hue range in OpenCV (35-85)
        green_mask = (hue > 35) & (hue < 85)
        green_ratio = float(np.mean(green_mask))

        # Soccer field: lots of green, moderate variation
        if green_ratio > 0.4 and std_brightness < 15:
            return ContentType.SOCCER, 0.7

        # Check for high edge density (text/ad)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges[led_mask > 0] > 0)
        total_pixels = np.sum(led_mask > 0)
        edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0

        # Text ad: high edge density
        if edge_density > 0.1:
            return ContentType.TEXT_AD, 0.7

        # Check for varied colors (video content)
        hue_std = float(np.std(hue))
        sat_std = float(np.std(sat))

        # Video: varied colors and saturation
        if hue_std > 30 and sat_std > 30:
            return ContentType.VIDEO, 0.6

        # Default: video content
        return ContentType.VIDEO, 0.5

    def get_adaptive_thresholds(
        self,
        content_type: ContentType,
    ) -> dict:
        """Get adaptive thresholds based on content type.

        Args:
            content_type: Classified content type.

        Returns:
            Dictionary of threshold adjustments.
        """
        # Base thresholds (from v2.2)
        thresholds = {
            "uniformity_std": 10,
            "uniformity_ratio": 0.10,
            "blocking_contrast": 0.4,
            "region_contrast": 2.5,
            "color_hue_shift": 55,
        }

        # Adjust based on content type
        if content_type == ContentType.SOCCER:
            # Soccer field is naturally uniform - raise thresholds
            thresholds["uniformity_std"] = 8  # Even lower to catch truly frozen
            thresholds["uniformity_ratio"] = 0.08
            thresholds["region_contrast"] = 3.0  # Higher to avoid soccer field borders

        elif content_type == ContentType.TEXT_AD:
            # Text ads have high contrast - lower thresholds
            thresholds["uniformity_std"] = 12
            thresholds["uniformity_ratio"] = 0.12
            thresholds["color_hue_shift"] = 50  # More sensitive to color errors

        elif content_type == ContentType.DARK_SCREEN:
            # Dark screen - different thresholds
            thresholds["uniformity_std"] = 5
            thresholds["uniformity_ratio"] = 0.05
            thresholds["blocking_contrast"] = 0.2

        elif content_type == ContentType.SOLID_COLOR:
            # Solid color - very sensitive to variations
            thresholds["uniformity_std"] = 6
            thresholds["uniformity_ratio"] = 0.06

        return thresholds
