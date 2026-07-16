"""
Line defect detection module.

Detects horizontal and vertical line defects in LED content.
"""

from typing import List

import numpy as np

from src.detectors.led.helpers import col_means_safe, row_means_safe
from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_line_defects(
    gray: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
) -> List[LEDAnomaly]:
    """Detect horizontal and vertical line defects.

    Args:
        gray: Grayscale LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.

    Returns:
        List of detected line defects.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = gray.shape

    led_pixels = gray[led_mask > 0]
    if len(led_pixels) == 0:
        return []

    led_mean = float(np.mean(led_pixels))
    led_std = float(np.std(led_pixels))
    threshold = max(led_mean - led_std * 2, 30)

    anomalies.extend(
        _detect_horizontal_lines(gray, panel, led_mask, w, threshold)
    )
    anomalies.extend(
        _detect_vertical_lines(gray, panel, led_mask, h, threshold)
    )

    return anomalies


def _detect_horizontal_lines(
    gray: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
    w: int,
    threshold: float,
) -> List[LEDAnomaly]:
    """Detect horizontal line defects.

    Args:
        gray: Grayscale LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.
        w: Image width.
        threshold: Brightness threshold.

    Returns:
        List of horizontal line defects.
    """
    anomalies: List[LEDAnomaly] = []
    h = gray.shape[0]

    # Skip edges: bezels at the top/bottom of the LED cabinet
    # should not be flagged as line defects.
    edge_margin = int(h * 0.03)  # 3% of height
    for y in range(edge_margin, h - edge_margin):
        row = gray[y, :]
        row_led = led_mask[y, :] > 0

        dark_in_led = np.sum((row < threshold) & row_led)
        total_led_in_row = np.sum(row_led)

        if total_led_in_row <= w * 0.3:
            continue
        if dark_in_led <= total_led_in_row * 0.4:
            continue

        if not _is_local_minimum_y(y, gray, led_mask):
            continue

        dark_cols = np.where((row < threshold) & row_led)[0]
        if len(dark_cols) == 0:
            continue

        x_start = dark_cols[0]
        x_end = dark_cols[-1]
        line_width = x_end - x_start

        if line_width > w * 0.3:
            severity = min(dark_in_led / total_led_in_row, 1.0)
            anomalies.append(
                LEDAnomaly(
                    x=panel.x + x_start,
                    y=panel.y + y,
                    width=line_width,
                    height=1,
                    anomaly_type="line_defect",
                    severity=severity,
                    description=f"H-line at row {y} (len={line_width})",
                )
            )

    return anomalies


def _detect_vertical_lines(
    gray: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
    h: int,
    threshold: float,
) -> List[LEDAnomaly]:
    """Detect vertical line defects.

    Args:
        gray: Grayscale LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.
        h: Image height.
        threshold: Brightness threshold.

    Returns:
        List of vertical line defects.
    """
    anomalies: List[LEDAnomaly] = []
    w = gray.shape[1]

    # Skip edges: bezels on left/right of the LED cabinet
    edge_margin = int(w * 0.02)  # 2% of width
    for x in range(edge_margin, w - edge_margin):
        col = gray[:, x]
        col_led = led_mask[:, x] > 0

        dark_in_led = np.sum((col < threshold) & col_led)
        total_led_in_col = np.sum(col_led)

        if total_led_in_col <= h * 0.3:
            continue
        if dark_in_led <= total_led_in_col * 0.4:
            continue

        if not _is_local_minimum_x(x, gray, led_mask):
            continue

        dark_rows = np.where((col < threshold) & col_led)[0]
        if len(dark_rows) == 0:
            continue

        y_start = dark_rows[0]
        y_end = dark_rows[-1]
        line_height = y_end - y_start

        if line_height > h * 0.3:
            severity = min(dark_in_led / total_led_in_col, 1.0)
            anomalies.append(
                LEDAnomaly(
                    x=panel.x + x,
                    y=panel.y + y_start,
                    width=1,
                    height=line_height,
                    anomaly_type="line_defect",
                    severity=severity,
                    description=f"V-line at col {x} (len={line_height})",
                )
            )

    return anomalies


def _is_local_minimum_y(
    y: int,
    gray: np.ndarray,
    led_mask: np.ndarray,
) -> bool:
    """Check if row y is a local brightness minimum.

    Args:
        y: Row index.
        gray: Grayscale image.
        led_mask: Binary mask of LED content.

    Returns:
        True if y is a local minimum.
    """
    return (
        row_means_safe(y, gray, led_mask)
        < row_means_safe(y - 1, gray, led_mask)
        and row_means_safe(y, gray, led_mask)
        < row_means_safe(y + 1, gray, led_mask)
    )


def _is_local_minimum_x(
    x: int,
    gray: np.ndarray,
    led_mask: np.ndarray,
) -> bool:
    """Check if column x is a local brightness minimum.

    Args:
        x: Column index.
        gray: Grayscale image.
        led_mask: Binary mask of LED content.

    Returns:
        True if x is a local minimum.
    """
    return (
        col_means_safe(x, gray, led_mask)
        < col_means_safe(x - 1, gray, led_mask)
        and col_means_safe(x, gray, led_mask)
        < col_means_safe(x + 1, gray, led_mask)
    )
