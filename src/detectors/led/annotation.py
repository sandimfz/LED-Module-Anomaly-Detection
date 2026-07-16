"""
Annotation module.

Functions for annotating images with detected anomalies.
"""

from typing import List

import cv2
import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo

# Color constants (BGR format)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# Severity color thresholds
SEVERITY_LOW = 0.3
SEVERITY_MEDIUM = 0.6
SEVERITY_HIGH = 0.8


def annotate_image(
    image: np.ndarray,
    panel: LEDPanelInfo,
    anomalies: List[LEDAnomaly],
) -> np.ndarray:
    """Annotate image with detected anomalies.

    Args:
        image: Original image.
        panel: LED panel information.
        anomalies: List of detected anomalies.

    Returns:
        Annotated image.
    """
    annotated = image.copy()

    _draw_panel_border(annotated, panel)
    _draw_anomalies(annotated, anomalies)
    _draw_summary(annotated, panel, anomalies)

    return annotated


def _draw_panel_border(
    image: np.ndarray,
    panel: LEDPanelInfo,
) -> None:
    """Draw border around detected LED panel.

    Args:
        image: Image to draw on.
        panel: LED panel information.
    """
    cv2.rectangle(
        image,
        (panel.x, panel.y),
        (panel.x + panel.width, panel.y + panel.height),
        COLOR_GREEN,
        2,
    )


def _draw_anomalies(
    image: np.ndarray,
    anomalies: List[LEDAnomaly],
) -> None:
    """Draw bounding boxes for all anomalies.

    Args:
        image: Image to draw on.
        anomalies: List of anomalies.
    """
    for anomaly in anomalies:
        color = _get_severity_color(anomaly.severity)
        cv2.rectangle(
            image,
            (anomaly.x, anomaly.y),
            (anomaly.x + anomaly.width, anomaly.y + anomaly.height),
            color,
            2,
        )
        _draw_label(image, anomaly)


def _get_severity_color(severity: float) -> tuple:
    """Get color based on severity level.

    Args:
        severity: Severity value (0-1).

    Returns:
        BGR color tuple.
    """
    if severity >= SEVERITY_HIGH:
        return COLOR_RED
    elif severity >= SEVERITY_MEDIUM:
        return COLOR_ORANGE
    elif severity >= SEVERITY_LOW:
        return COLOR_YELLOW
    return COLOR_GREEN


def _draw_label(
    image: np.ndarray,
    anomaly: LEDAnomaly,
) -> None:
    """Draw text label for anomaly.

    Args:
        image: Image to draw on.
        anomaly: Anomaly to label.
    """
    label = f"{anomaly.anomaly_type}: {anomaly.severity:.2f}"
    label_size = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )[0]

    label_x = anomaly.x
    label_y = anomaly.y - 10 if anomaly.y > 20 else anomaly.y + 20

    cv2.rectangle(
        image,
        (label_x, label_y - label_size[1] - 5),
        (label_x + label_size[0] + 5, label_y + 5),
        COLOR_BLACK,
        -1,
    )

    cv2.putText(
        image,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        1,
    )


def _draw_summary(
    image: np.ndarray,
    panel: LEDPanelInfo,
    anomalies: List[LEDAnomaly],
) -> None:
    """Draw summary text at top of image.

    Args:
        image: Image to draw on.
        panel: LED panel information.
        anomalies: List of anomalies.
    """
    summary_lines = [
        f"Panel: {panel.width}x{panel.height}",
        f"Anomalies: {len(anomalies)}",
    ]

    y_offset = 30
    for line in summary_lines:
        cv2.putText(
            image,
            line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_WHITE,
            2,
        )
        y_offset += 25
