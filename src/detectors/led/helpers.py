"""
Helper functions for LED analysis.

Utility functions for safe brightness calculations.
"""

import numpy as np


def row_means_safe(
    y: int,
    gray: np.ndarray,
    led_mask: np.ndarray,
) -> float:
    """Calculate mean brightness for row y safely.

    Args:
        y: Row index.
        gray: Grayscale image.
        led_mask: Binary mask of LED region.

    Returns:
        Mean brightness or 999.0 if invalid.
    """
    h, w = gray.shape
    if y < 0 or y >= h:
        return 999.0
    row = gray[y, :]
    mask = led_mask[y, :] > 0
    if np.sum(mask) == 0:
        return 999.0
    return float(np.mean(row[mask]))


def col_means_safe(
    x: int,
    gray: np.ndarray,
    led_mask: np.ndarray,
) -> float:
    """Calculate mean brightness for column x safely.

    Args:
        x: Column index.
        gray: Grayscale image.
        led_mask: Binary mask of LED region.

    Returns:
        Mean brightness or 999.0 if invalid.
    """
    h, w = gray.shape
    if x < 0 or x >= w:
        return 999.0
    col = gray[:, x]
    mask = led_mask[:, x] > 0
    if np.sum(mask) == 0:
        return 999.0
    return float(np.mean(col[mask]))
