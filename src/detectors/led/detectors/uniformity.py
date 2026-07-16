"""
Uniformity detection module.

Detects LED panels with abnormally uniform content.
A normal LED ad has natural variation (text, images, colors).
A frozen or stuck panel often shows unnaturally uniform content
(low standard deviation across the panel).
"""

from typing import List

import cv2
import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_uniform_content(
    gray: np.ndarray,
    hsv: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
) -> List[LEDAnomaly]:
    """Detect abnormally uniform LED content.

    A normal LED ad has natural brightness variation (std typically
    30-80+ on the cropped panel). A frozen/stuck module will show
    content with very low variation (std < 30).

    However, some normal content (soccer field, solid color ads) can
    have low variation. We add edge detection check to distinguish:
    - Normal uniform content: has edges/shapes (soccer field, text)
    - Frozen/stuck: no edges at all (truly stuck pixels)

    Args:
        gray: Grayscale LED region.
        hsv: HSV LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.

    Returns:
        List of detected uniformity anomalies.
    """
    anomalies: List[LEDAnomaly] = []

    led_pixels = gray[led_mask > 0]
    if len(led_pixels) == 0:
        return []

    mean_brightness = float(np.mean(led_pixels))
    std_brightness = float(np.std(led_pixels))

    # Panel must have enough brightness to analyze
    if mean_brightness < 60:
        return []

    # Brightness variation threshold: std < 10 and less than 10% of mean.
    # Raised from (std < 15, ratio < 0.15) to reduce false positives on
    # normal uniform content (soccer field, solid color ads).
    # A truly frozen/stuck display has very low variation: std < 10.
    std_ratio = std_brightness / mean_brightness if mean_brightness > 0 else 1.0

    if std_brightness < 10 and std_ratio < 0.10:
        # Additional check: edge detection
        # Normal content has edges (soccer field lines, text, shapes)
        # Frozen/stuck has no edges at all
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges[led_mask > 0] > 0)
        total_pixels = np.sum(led_mask > 0)
        edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0

        # If there are edges, this is likely normal content (not frozen)
        if edge_ratio > 0.05:
            return []

        severity = min(1.0 - (std_brightness / 20.0), 1.0)

        # Also check hue variation — uniform hue = more likely frozen
        hue = hsv[:, :, 0].astype(np.float32)
        hue_pixels = hue[led_mask > 0]
        hue_std = float(np.std(hue_pixels))

        hue_factor = 1.0 - min(hue_std / 40.0, 0.5)  # 0.5 if hue also uniform

        severity = min(severity + hue_factor * severity, 1.0)

        anomalies.append(
            LEDAnomaly(
                x=panel.x,
                y=panel.y,
                width=panel.width,
                height=panel.height,
                anomaly_type="uniform_content",
                severity=severity,
                description=(
                    f"Konten terlalu seragam: "
                    f"brightness std={std_brightness:.0f} "
                    f"(ratio={std_ratio:.2f}) — kemungkinan frozen/stuck"
                ),
            )
        )

    return anomalies
