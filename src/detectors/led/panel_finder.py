"""
Panel finder module.

Functions for detecting LED panel location and perspective correction.
"""

from typing import List, Optional

import cv2
import numpy as np

from src.detectors.led.types import LEDPanelInfo


def find_led_panel(
    image: np.ndarray,
    min_panel_area: int = 50000,
) -> Optional[LEDPanelInfo]:
    """Find LED panel bounding box in image.

    When multiple LED panels are present (e.g. two screens side by
    side), their union bounding box is used so all areas are covered.

    Args:
        image: Input BGR image.
        min_panel_area: Minimum area for panel detection.

    Returns:
        LEDPanelInfo if found, None otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_img, w_img = gray.shape

    value_channel = hsv[:, :, 2]
    sat_channel = hsv[:, :, 1]

    led_mask = _create_initial_mask(value_channel, sat_channel)
    led_mask = _cleanup_mask(led_mask)

    contours, _ = cv2.findContours(
        led_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    candidates = _evaluate_candidates(
        contours, gray, sat_channel, value_channel,
        h_img, w_img, min_panel_area
    )

    if not candidates:
        return None

    # If multiple strong candidates cover a large fraction of the image,
    # merge them into a single bounding box so no area is skipped.
    if len(candidates) >= 2:
        merged = _try_merge_candidates(candidates, h_img, w_img)
        if merged is not None:
            return merged

    best = max(candidates, key=lambda c: c["score"])

    return LEDPanelInfo(
        x=best["x"],
        y=best["y"],
        width=best["w"],
        height=best["h"],
        confidence=min(best["score"] * 2, 1.0),
        mean_brightness=float(best["brightness"]),
        mean_saturation=float(best["saturation"]),
    )


def _try_merge_candidates(
    candidates: List[dict],
    h_img: int,
    w_img: int,
) -> Optional["LEDPanelInfo"]:
    """Merge multiple panel candidates into one bounding box.

    Only merges when candidates collectively cover >40% of the image,
    suggesting multiple LED screens rather than noise/artifacts.

    Args:
        candidates: List of candidate dictionaries.
        h_img: Image height.
        w_img: Image width.

    Returns:
        Merged LEDPanelInfo or None if merge is not appropriate.
    """
    image_area = h_img * w_img
    total_area = sum(c["w"] * c["h"] for c in candidates)

    if total_area < image_area * 0.40:
        return None

    x_min = min(c["x"] for c in candidates)
    y_min = min(c["y"] for c in candidates)
    x_max = max(c["x"] + c["w"] for c in candidates)
    y_max = max(c["y"] + c["h"] for c in candidates)

    avg_brightness = float(
        np.mean([c["brightness"] for c in candidates])
    )
    avg_saturation = float(
        np.mean([c["saturation"] for c in candidates])
    )
    best_score = max(c["score"] for c in candidates)

    return LEDPanelInfo(
        x=x_min,
        y=y_min,
        width=x_max - x_min,
        height=y_max - y_min,
        confidence=min(best_score * 2, 1.0),
        mean_brightness=avg_brightness,
        mean_saturation=avg_saturation,
    )


def perspective_crop(
    image: np.ndarray,
    screen_points: List[List[int]],
) -> np.ndarray:
    """Crop LED screen using perspective transform.

    Args:
        image: Original image.
        screen_points: 4 corner points [[x,y], ...].

    Returns:
        Corrected LED screen image.
    """
    pts_src = np.array(screen_points, dtype="float32")

    tl, tr, br, bl = pts_src
    max_width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl),
    ))
    max_height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl),
    ))

    pts_dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def _create_initial_mask(
    value_channel: np.ndarray,
    sat_channel: np.ndarray,
) -> np.ndarray:
    """Create initial LED mask from HSV channels.

    Args:
        value_channel: HSV value channel.
        sat_channel: HSV saturation channel.

    Returns:
        Binary mask.
    """
    # Threshold lebih rendah dari sebelumnya (was: 120) agar panel
    # dengan konten redup tetap terdeteksi. Harmonized dengan content_mask.
    bright_mask = value_channel > 80
    sat_mask = sat_channel > 30
    return (bright_mask & sat_mask).astype(np.uint8)


def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """Apply morphological cleanup to mask.

    Args:
        mask: Input binary mask.

    Returns:
        Cleaned mask.
    """
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask


def _evaluate_candidates(
    contours: np.ndarray,
    gray: np.ndarray,
    sat_channel: np.ndarray,
    value_channel: np.ndarray,
    h_img: int,
    w_img: int,
    min_panel_area: int,
) -> List[dict]:
    """Evaluate contour candidates for panel detection.

    Args:
        contours: Detected contours.
        gray: Grayscale image.
        sat_channel: Saturation channel.
        value_channel: Value channel.
        h_img: Image height.
        w_img: Image width.
        min_panel_area: Minimum panel area.

    Returns:
        List of candidate dictionaries.
    """
    candidates: List[dict] = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_panel_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        if not (0.4 < aspect_ratio < 4.0):
            continue

        rectangularity = _compute_rectangularity(contour, area)
        if rectangularity < 0.4:
            continue

        stats = _compute_region_stats(
            gray, sat_channel, value_channel, x, y, w, h
        )

        if stats["brightness"] < 60 or stats["saturation"] < 25:
            continue

        score = _compute_candidate_score(
            rectangularity, area, h_img, w_img,
            stats["brightness"], stats["saturation"]
        )

        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "score": score,
            "brightness": stats["brightness"],
            "saturation": stats["saturation"],
            "area": area,
        })

    return candidates


def _compute_rectangularity(contour: np.ndarray, area: float) -> float:
    """Compute rectangularity score using rotated rectangle.

    Args:
        contour: Input contour.
        area: Contour area.

    Returns:
        Rectangularity score (0-1).
    """
    rot_rect = cv2.minAreaRect(contour)
    rot_w, rot_h = rot_rect[1]
    rot_area = rot_w * rot_h
    return area / rot_area if rot_area > 0 else 0


def _compute_region_stats(
    gray: np.ndarray,
    sat_channel: np.ndarray,
    value_channel: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> dict:
    """Compute statistics for a region.

    Args:
        gray: Grayscale image.
        sat_channel: Saturation channel.
        value_channel: Value channel.
        x: Region x coordinate.
        y: Region y coordinate.
        w: Region width.
        h: Region height.

    Returns:
        Dictionary with brightness and saturation.
    """
    roi_gray = gray[y:y + h, x:x + w]
    roi_sat = sat_channel[y:y + h, x:x + w]

    return {
        "brightness": float(np.mean(roi_gray)),
        "saturation": float(np.mean(roi_sat)),
    }


def _compute_candidate_score(
    rectangularity: float,
    area: float,
    h_img: int,
    w_img: int,
    brightness: float,
    saturation: float,
) -> float:
    """Compute candidate score.

    Args:
        rectangularity: Rectangularity score.
        area: Contour area.
        h_img: Image height.
        w_img: Image width.
        brightness: Mean brightness.
        saturation: Mean saturation.

    Returns:
        Combined score.
    """
    size_score = area / (h_img * w_img)
    return (
        rectangularity * 0.25
        + min(size_score * 2, 0.25)
        + (brightness / 255.0) * 0.25
        + (saturation / 255.0) * 0.25
    )
