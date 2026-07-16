"""
Flat content detection module.

Detects blank or abnormally flat regions in LED content.
"""

from typing import List

import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_flat_content(
    gray: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
) -> List[LEDAnomaly]:
    """Detect flat or blank content regions.

    Args:
        gray: Grayscale LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.

    Returns:
        List of detected flat content anomalies.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = gray.shape

    led_pixels = gray[led_mask > 0]
    if len(led_pixels) == 0:
        return []

    led_mean = float(np.mean(led_pixels))
    block_size = 64

    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            block_mask = led_mask[by:by + block_size, bx:bx + block_size]
            led_ratio = np.mean(block_mask)

            if led_ratio < 0.7:
                continue

            block = gray[by:by + block_size, bx:bx + block_size]
            block_var = float(np.var(block))
            block_mean = float(np.mean(block))

            # Threshold untuk blank detection: var < 15 untuk menangkap
            # area yang benar-benar seragam (blank stuck color).
            is_blank_white = block_var < 15.0 and block_mean > 200
            is_blank_black = block_var < 15.0 and block_mean < 15
            is_abnormal_flat = (
                block_var < 5.0 and block_mean > led_mean * 1.5
            )

            if is_blank_white or is_blank_black or is_abnormal_flat:
                severity = 1.0 - min(block_var / 5.0, 1.0)
                desc = (
                    "Blank white" if is_blank_white
                    else "Blank black" if is_blank_black
                    else "Abnormal flat"
                )
                anomalies.append(
                    LEDAnomaly(
                        x=bx + panel.x,
                        y=by + panel.y,
                        width=block_size,
                        height=block_size,
                        anomaly_type="flat_content",
                        severity=severity,
                        description=f"{desc} at ({bx},{by})",
                    )
                )

    return merge_flat_anomalies(anomalies, panel)


def merge_flat_anomalies(
    anomalies: List[LEDAnomaly],
    panel: LEDPanelInfo,
) -> List[LEDAnomaly]:
    """Merge adjacent flat anomalies into larger regions.

    Args:
        anomalies: List of flat content anomalies.
        panel: LED panel information.

    Returns:
        List of merged anomalies.
    """
    if not anomalies:
        return []

    merged: List[LEDAnomaly] = []
    used = [False] * len(anomalies)
    block_size = 64

    for i, a1 in enumerate(anomalies):
        if used[i]:
            continue

        group = [a1]
        used[i] = True

        for j, a2 in enumerate(anomalies):
            if used[j]:
                continue
            if (
                abs(a1.x - a2.x) < block_size * 2
                and abs(a1.y - a2.y) < block_size * 2
            ):
                group.append(a2)
                used[j] = True

        if len(group) >= 3:
            min_x = min(a.x for a in group)
            min_y = min(a.y for a in group)
            max_x = max(a.x + a.width for a in group)
            max_y = max(a.y + a.height for a in group)
            avg_severity = np.mean([a.severity for a in group])

            merged.append(
                LEDAnomaly(
                    x=min_x,
                    y=min_y,
                    width=max_x - min_x,
                    height=max_y - min_y,
                    anomaly_type="flat_content",
                    severity=avg_severity,
                    description=f"No content area ({len(group)} blocks)",
                )
            )

    return merged
