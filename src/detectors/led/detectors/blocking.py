"""
Blocking detection module.

Detects large dark areas that block LED content.
"""

from typing import List, Optional

import cv2
import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_blocking(
    gray: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
    panel_mask: Optional[np.ndarray] = None,
) -> List[LEDAnomaly]:
    """Detect blocking anomalies in LED panel.

    Args:
        gray: Grayscale LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.

    Returns:
        List of detected blocking anomalies.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = gray.shape

    led_pixels = gray[led_mask > 0]
    if len(led_pixels) == 0:
        # If content mask is empty (e.g. very dim LED content),
        # fall back to panel mask if available.
        if panel_mask is not None:
            led_pixels = gray[panel_mask > 0]
            if len(led_pixels) == 0:
                return []
        else:
            return []

    led_mean = np.mean(led_pixels)
    led_std = np.std(led_pixels)

    threshold_relative = max(led_mean - 2 * led_std, 30)
    threshold_absolute = 60
    threshold = max(threshold_relative, threshold_absolute)

    # Use panel_mask as base to ensure dark areas outside
    # content_mask (e.g. black blocks, dead modules) are still
    # detectable. The content mask filters out dark regions,
    # so we need the broader panel mask to catch blocking.
    if panel_mask is not None:
        dark_mask = (gray < threshold) & (panel_mask > 0)
    else:
        dark_mask = (gray < threshold) & (led_mask > 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dark_mask = cv2.morphologyEx(
        dark_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        area_ratio = area / (h * w)

        # Longgarkan filter: hanya skip noise sangat kecil (< 0.2%)
        # dan skip jika hampir seluruh panel (bukan blocking, tapi LED off)
        if area_ratio < 0.002 or area_ratio > 0.85:
            continue

        # Hitung contrast dengan area sekitar (relative comparison)
        contrast_score = _calculate_neighborhood_contrast(
            gray, x, y, cw, ch, panel_mask, led_mask
        )

        is_blocking, desc = _classify_blocking(
            cw, ch, w, h, area_ratio, contrast_score
        )
        if is_blocking:
            # Severity: kombinasi area_ratio dan contrast
            severity = min(area_ratio * 4 + contrast_score * 0.3, 1.0)
            anomalies.append(
                LEDAnomaly(
                    x=x + panel.x,
                    y=y + panel.y,
                    width=cw,
                    height=ch,
                    anomaly_type="blocking",
                    severity=severity,
                    description=desc,
                )
            )

    return anomalies


def _classify_blocking(
    cw: int,
    ch: int,
    w: int,
    h: int,
    area_ratio: float,
    contrast_score: float = 0.0,
) -> tuple[bool, str]:
    """Classify type of blocking anomaly.

    Args:
        cw: Contour width.
        ch: Contour height.
        w: Image width.
        h: Image height.
        area_ratio: Ratio of contour area to image area.
        contrast_score: Contrast score with neighborhood.

    Returns:
        Tuple of (is_blocking, description).
    """
    # Large area with high contrast = definitely blocking
    if area_ratio > 0.15 and contrast_score > 0.3:
        return True, f"Large blocking ({area_ratio*100:.0f}% panel, contrast={contrast_score:.2f})"

    # Horizontal bar pattern
    if cw > w * 0.4 and ch > h * 0.05:
        return True, f"Horizontal blocking ({cw}x{ch}px)"

    # Vertical bar pattern
    if ch > h * 0.4 and cw > w * 0.05:
        return True, f"Vertical blocking ({cw}x{ch}px)"

    # Large dark area (> 3% panel)
    if area_ratio > 0.03:
        return True, f"Large dark area ({area_ratio*100:.0f}% panel)"

    # Partial blocking with decent contrast
    if contrast_score > 0.4:
        if cw > w * 0.15 and ch > h * 0.08:
            return True, f"Partial blocking (contrast={contrast_score:.2f})"
        if ch > h * 0.15 and cw > w * 0.08:
            return True, f"Partial blocking (contrast={contrast_score:.2f})"

    # Moderate area with very high contrast
    if area_ratio > 0.01 and contrast_score > 0.5:
        return True, f"Dark region (contrast={contrast_score:.2f})"

    return False, ""


def _calculate_neighborhood_contrast(
    gray: np.ndarray,
    x: int, y: int, cw: int, ch: int,
    panel_mask: np.ndarray,
    led_mask: np.ndarray,
) -> float:
    """Calculate contrast between dark region and its neighborhood.

    Args:
        gray: Grayscale image.
        x, y, cw, ch: Bounding box of dark region.
        panel_mask: Panel mask.
        led_mask: LED content mask.

    Returns:
        Contrast score between 0.0 and 1.0.
    """
    h, w = gray.shape

    # Define neighborhood: expand bounding box by 50%
    margin_x = int(cw * 0.5)
    margin_y = int(ch * 0.5)

    nx1 = max(0, x - margin_x)
    ny1 = max(0, y - margin_y)
    nx2 = min(w, x + cw + margin_x)
    ny2 = min(h, y + ch + margin_y)

    # Get pixels in the dark region (inside bounding box)
    region_mask = np.zeros((h, w), dtype=np.uint8)
    region_mask[y:y+ch, x:x+cw] = 1
    if panel_mask is not None:
        region_mask = region_mask & (panel_mask > 0)

    region_pixels = gray[region_mask > 0]
    if len(region_pixels) == 0:
        return 0.0

    # Get pixels in neighborhood (outside bounding box, inside margin)
    neighborhood_mask = np.zeros((h, w), dtype=np.uint8)
    neighborhood_mask[ny1:ny2, nx1:nx2] = 1
    neighborhood_mask[y:y+ch, x:x+cw] = 0  # Exclude the region itself
    if panel_mask is not None:
        neighborhood_mask = neighborhood_mask & (panel_mask > 0)
    # Prefer LED content pixels for neighborhood
    if led_mask is not None:
        led_neighborhood = neighborhood_mask & (led_mask > 0)
        if np.sum(led_neighborhood) > 50:
            neighborhood_mask = led_neighborhood

    neighborhood_pixels = gray[neighborhood_mask > 0]
    if len(neighborhood_pixels) == 0:
        return 0.0

    region_mean = float(np.mean(region_pixels))
    neighbor_mean = float(np.mean(neighborhood_pixels))

    # Contrast = how much darker is region vs neighborhood
    if neighbor_mean < 10:
        return 0.0

    contrast = (neighbor_mean - region_mean) / neighbor_mean
    return max(0.0, min(contrast, 1.0))
