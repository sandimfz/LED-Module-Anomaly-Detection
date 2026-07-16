"""
Scoring module.

Functions for calculating anomaly scores and filtering anomalies.
"""

from typing import List

import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def calculate_score(
    anomalies: List[LEDAnomaly],
    panel: LEDPanelInfo,
) -> float:
    """Calculate anomaly score based on anomalies and panel info.

    Args:
        anomalies: List of detected anomalies.
        panel: LED panel information.

    Returns:
        Anomaly score between 0.0 and 1.0.
    """
    if not anomalies:
        return 0.0

    total_area = panel.width * panel.height

    # Priority anomaly types get higher weight
    priority_types = {
        "blocking", "led_off", "led_mostly_off",
        "dark_region", "horizontal_dark_bar", "vertical_dark_bar",
        "module_glitch", "horizontal_glitch", "vertical_glitch",
        "dead_pixel_block",
    }

    # Separate priority and normal anomalies
    priority_anomalies = [a for a in anomalies if a.anomaly_type in priority_types]
    normal_anomalies = [a for a in anomalies if a.anomaly_type not in priority_types]

    # Calculate base score from all anomalies
    anomaly_area = sum(a.width * a.height for a in anomalies)
    avg_severity = np.mean([a.severity for a in anomalies])

    area_ratio = anomaly_area / total_area if total_area > 0 else 0
    area_contribution = min(area_ratio / 0.3, 1.0) * 0.5
    base_score = area_contribution + avg_severity * 0.5

    # Type boost: priority anomalies get additional score
    type_boost = 0.0
    if priority_anomalies:
        # Count unique priority types found
        unique_types = set(a.anomaly_type for a in priority_anomalies)
        # Boost based on how many priority types found (max 0.3)
        type_boost = min(len(unique_types) * 0.15, 0.3)

        # Additional boost for high-severity priority anomalies
        max_priority_severity = max(a.severity for a in priority_anomalies)
        type_boost += max_priority_severity * 0.1

    final_score = base_score + type_boost

    return min(final_score, 1.0)


def filter_anomalies_in_panel(
    anomalies: List[LEDAnomaly],
    panel: LEDPanelInfo,
) -> List[LEDAnomaly]:
    """Filter anomalies to only include those within panel.

    Args:
        anomalies: List of anomalies.
        panel: LED panel information.

    Returns:
        Filtered list of anomalies.
    """
    filtered: List[LEDAnomaly] = []
    panel_area = panel.width * panel.height

    px1 = panel.x
    py1 = panel.y
    px2 = panel.x + panel.width
    py2 = panel.y + panel.height

    for a in anomalies:
        ax1 = a.x
        ay1 = a.y
        ax2 = a.x + a.width
        ay2 = a.y + a.height

        # Check if anomaly overlaps with panel (not just fully inside)
        # Allow some overlap for large blocking anomalies
        if ax2 < px1 or ax1 > px2 or ay2 < py1 or ay1 > py2:
            continue

        anomaly_area = a.width * a.height
        # Only skip very tiny anomalies (< 0.1% panel)
        if anomaly_area < panel_area * 0.001:
            continue

        # Longgarkan: tidak ada filter untuk area besar
        # Blocking dan module error bisa mencakup banyak area

        filtered.append(a)

    return filtered
