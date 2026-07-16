"""
Horizontal line pattern detection module.

Detects abnormal horizontal line patterns in LED content.
"""

from typing import List

import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_horizontal_line_pattern(
    gray: np.ndarray,
    hsv: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
) -> List[LEDAnomaly]:
    """Detect abnormal horizontal line patterns.

    Detects rows with unusually high variance that indicate
    color noise or module glitch (random colored lines).

    Args:
        gray: Grayscale LED region.
        hsv: HSV LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.

    Returns:
        List of detected horizontal line pattern anomalies.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = gray.shape

    if h < 10 or w < 10:
        return []

    row_variances = _compute_row_variances(gray, led_mask, h, w)

    if len(row_variances) == 0:
        return []

    # Smooth row variances to reduce noise from content transitions
    # (e.g., soccer field → text area border)
    row_variances = _smooth_row_variances(row_variances, window=5)

    mean_var = float(np.mean(row_variances))
    std_var = float(np.std(row_variances))

    # Require higher threshold for full image analysis
    # to avoid false positives on normal content
    if std_var < 5.0:
        return []

    # Use 5x std for more conservative detection (was 4x)
    # Konten iklan normal punya row variance tinggi karena teks,
    # grafis, dan transisi konten — perlu margin lebih besar.
    threshold = mean_var + 5 * std_var

    # Require much higher minimum variance to avoid triggering
    # on natural content transitions (soccer→text, sky→ground, etc.)
    # 3500 avoids false positives on normal ad content with varied regions.
    min_variance_threshold = 3500.0

    # Edge margin: skip top/bottom 3% — bezels bukan defect
    edge_margin = int(h * 0.03)

    # Hanya flag jika ada ≥ 2 consecutive row yang exceed threshold
    # (single row spike biasanya noise konten alami, bukan defect)
    for y in range(edge_margin, h - edge_margin):
        if (
            row_variances[y] > threshold
            and row_variances[y] > min_variance_threshold
            and _is_local_maximum_variance(y, row_variances)
        ):
            # Check for consecutive rows
            consecutive = 1
            for dy in range(1, 3):
                ny = y + dy
                if ny < h and row_variances[ny] > threshold * 0.8:
                    consecutive += 1
                else:
                    break

            if consecutive < 2:
                continue

            anomalies.append(
                LEDAnomaly(
                    x=panel.x,
                    y=panel.y + y,
                    width=w,
                    height=consecutive,
                    anomaly_type="horizontal_pattern",
                    severity=min(
                        row_variances[y] / (threshold + 1e-6),
                        1.0,
                    ),
                    description=(
                        f"H-line pattern at row {y} "
                        f"(var={row_variances[y]:.1f}, "
                        f"consec={consecutive})"
                    ),
                )
            )

    return anomalies


def _compute_row_variances(
    gray: np.ndarray,
    led_mask: np.ndarray,
    h: int,
    w: int,
) -> List[float]:
    """Compute variance for each row.

    Args:
        gray: Grayscale image.
        led_mask: Binary mask.
        h: Image height.
        w: Image width.

    Returns:
        List of row variances.
    """
    row_variances: List[float] = []

    for y in range(h):
        row = gray[y, :]
        mask = led_mask[y, :] > 0
        if np.sum(mask) < w * 0.3:
            row_variances.append(0.0)
        else:
            row_variances.append(float(np.var(row[mask])))

    return row_variances


def _smooth_row_variances(
    row_variances: List[float],
    window: int = 5,
) -> List[float]:
    """Smooth row variances using moving average.

    Reduces noise from single-row content transitions while
    preserving genuine high-variance patterns from defects.

    Args:
        row_variances: Raw row variances.
        window: Smoothing window size.

    Returns:
        Smoothed row variances.
    """
    n = len(row_variances)
    if n <= window:
        return row_variances

    smoothed = []
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed.append(float(np.mean(row_variances[start:end])))
    return smoothed


def _is_local_maximum_variance(
    y: int,
    row_variances: List[float],
) -> bool:
    """Check if row has locally maximum variance.

    Args:
        y: Row index.
        row_variances: List of row variances.

    Returns:
        True if y is a local maximum.
    """
    h = len(row_variances)
    if y <= 0 or y >= h - 1:
        return False

    return (
        row_variances[y] > row_variances[y - 1]
        and row_variances[y] > row_variances[y + 1]
    )
