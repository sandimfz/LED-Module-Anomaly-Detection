"""
Content mask module.

Functions for creating LED content masks.
"""

from typing import List, Optional

import cv2
import numpy as np

from src.detectors.led.types import LEDPanelInfo


def create_led_content_mask(
    gray: np.ndarray,
    hsv: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    """Create mask for LED content area.

    Uses simple thresholding for bright and colorful areas
    since image is already cropped to LED panel.

    Args:
        gray: Grayscale LED region.
        hsv: HSV LED region.
        h: Image height.
        w: Image width.

    Returns:
        Binary mask (1 = LED content, 0 = non-LED).
    """
    value_channel = hsv[:, :, 2]
    sat_channel = hsv[:, :, 1]

    bright_mask = value_channel > 80
    sat_mask = sat_channel > 30

    led_mask = (bright_mask & sat_mask).astype(np.uint8)

    led_mask = _apply_morphological_cleanup(led_mask)
    led_mask = _extract_largest_component(led_mask)

    return led_mask


def _apply_morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """Apply morphological operations to clean up mask.

    Hanya OPEN (remove noise) + DILATE ringan.
    TIDAK pakai CLOSE karena akan nge-fill gap antara LED aktif
    (misal gap antar huruf di background hitam), menyebabkan
    background yang sebenarnya normal terdeteksi sebagai bagian
    dari LED content — lalu sub-detector flag sebagai anomali.

    Args:
        mask: Input binary mask.

    Returns:
        Cleaned mask.
    """
    # OPEN dulu untuk hapus noise kecil
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Dilate ringan untuk sambung pixel LED yang berdekatan
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    # Exclude edges to avoid bezel/edge false positives
    h, w = mask.shape
    edge_margin = 8
    if h > edge_margin * 2 and w > edge_margin * 2:
        mask[:edge_margin, :] = 0
        mask[-edge_margin:, :] = 0
        mask[:, :edge_margin] = 0
        mask[:, -edge_margin:] = 0

    return mask


def _extract_largest_component(mask: np.ndarray) -> np.ndarray:
    """Extract the largest connected component.

    Args:
        mask: Input binary mask.

    Returns:
        Mask with only largest component.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 1, -1)

    return mask


def create_full_image_mask(
    image_shape: tuple,
    panel: LEDPanelInfo,
    screen_points: Optional[List[List[int]]] = None,
) -> np.ndarray:
    """Create mask for LED content in full image.

    Creates a binary mask covering the LED content area
    without cropping the image.

    Args:
        image_shape: Shape of the full image (H, W, C).
        panel: LED panel information.
        screen_points: Optional 4 corner points for
            perspective-correct mask.

    Returns:
        Binary mask (1 = LED content, 0 = non-LED).
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    if screen_points:
        pts = np.array(screen_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    else:
        mask[
            panel.y:panel.y + panel.height,
            panel.x:panel.x + panel.width,
        ] = 1

    return mask


def refine_led_mask(
    gray: np.ndarray,
    hsv: np.ndarray,
    panel_mask: np.ndarray,
) -> np.ndarray:
    """Refine panel mask to detect actual LED content.

    Applies brightness and saturation thresholding within
    the panel area to identify active LED content.

    Args:
        gray: Grayscale full image.
        hsv: HSV full image.
        panel_mask: Binary mask of panel area.

    Returns:
        Refined mask of LED content.
    """
    value_channel = hsv[:, :, 2]
    sat_channel = hsv[:, :, 1]

    bright_mask = value_channel > 80
    sat_mask = sat_channel > 30

    content_mask = (bright_mask & sat_mask).astype(np.uint8)

    content_mask = cv2.bitwise_and(content_mask, panel_mask)

    # Same conservative approach: OPEN only, no CLOSE.
    # CLOSE fills gaps between lit areas (text, graphics) causing
    # dark background to be classified as LED content.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    content_mask = cv2.morphologyEx(
        content_mask, cv2.MORPH_OPEN, kernel
    )

    # Erode the mask to exclude bezels/edges of the LED cabinet.
    # This prevents the edges from being analyzed by sub-detectors
    # where bezels would be falsely detected as anomalies.
    edge_margin = 10
    h, w = content_mask.shape
    if h > edge_margin * 2 and w > edge_margin * 2:
        content_mask[:edge_margin, :] = 0
        content_mask[-edge_margin:, :] = 0
        content_mask[:, :edge_margin] = 0
        content_mask[:, -edge_margin:] = 0

    return content_mask
