"""
Flat content detection module.

Detects blank or abnormally flat regions in LED content.
Uses panel_mask as fallback so white flat blocks (sat < 30) are included.
"""

from typing import List, Optional

import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_flat_content(
    gray: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
    panel_mask: Optional[np.ndarray] = None,
    hsv: Optional[np.ndarray] = None,
    baseline_median: Optional[float] = None,
) -> List[LEDAnomaly]:
    """Detect flat or blank content regions.

    Uses adaptive thresholds from baseline when available.
    Falls back to absolute thresholds if baseline not ready.

    Args:
        gray: Grayscale LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.
        panel_mask: Binary mask of panel area (preferred base).
        hsv: HSV image (unused, kept for API compat).
        baseline_median: Rolling median brightness from baseline (optional).

    Returns:
        List of detected flat content anomalies.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = gray.shape

    # Use panel_mask as fallback so white flat blocks are included
    base_mask = panel_mask if panel_mask is not None else led_mask
    base_pixels = gray[base_mask > 0]
    if len(base_pixels) == 0:
        return []

    panel_mean = float(np.mean(base_pixels))
    block_size = 64

    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            block_mask = base_mask[by:by + block_size, bx:bx + block_size]
            base_ratio = float(np.mean(block_mask))

            if base_ratio < 0.5:
                continue

            block = gray[by:by + block_size, bx:bx + block_size]
            block_var = float(np.var(block))
            block_mean = float(np.mean(block))

            # Adaptive thresholds: use baseline if available
            if baseline_median is not None and baseline_median > 0:
                # Relative: flat white = significantly brighter than baseline median
                white_threshold = baseline_median + 30
                is_blank_white = block_var < 15.0 and block_mean > white_threshold
            else:
                # Fallback: absolute threshold
                is_blank_white = block_var < 15.0 and block_mean > 200

            is_blank_black = block_var < 15.0 and block_mean < 15
            is_abnormal_flat = (
                block_var < 5.0 and block_mean > panel_mean * 1.5
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
            avg_severity = float(np.mean([a.severity for a in group]))

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


def merge_flat_anomalies(
    anomalies: List[LEDAnomaly],
    panel: LEDPanelInfo,
) -> List[LEDAnomaly]:
    """Merge adjacent flat anomalies into larger regions.

    Groups anomalies that are within 2 block-sizes of each other.
    Requires >=2 adjacent blocks to form a region (reduced from 3
    since we use 32px blocks now).

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
    block_size = 32

    for i, a1 in enumerate(anomalies):
        if used[i]:
            continue

        group = [a1]
        used[i] = True

        for j, a2 in enumerate(anomalies):
            if used[j]:
                continue
            if (
                abs(a1.x - a2.x) < block_size * 3
                and abs(a1.y - a2.y) < block_size * 3
            ):
                group.append(a2)
                used[j] = True

        if len(group) >= 2:
            min_x = min(a.x for a in group)
            min_y = min(a.y for a in group)
            max_x = max(a.x + a.width for a in group)
            max_y = max(a.y + a.height for a in group)
            avg_severity = float(np.mean([a.severity for a in group]))

            merged.append(
                LEDAnomaly(
                    x=min_x,
                    y=min_y,
                    width=max_x - min_x,
                    height=max_y - min_y,
                    anomaly_type="flat_content",
                    severity=avg_severity,
                    description=(
                        f"Flat stuck area ({len(group)} blocks, "
                        f"{max_x-min_x}x{max_y-min_y}px)"
                    ),
                )
            )

    return merged
