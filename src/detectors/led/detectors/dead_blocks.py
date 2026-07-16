"""
Dead block detection module.

Detects dead pixel blocks in LED content.
"""

from typing import List

import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_dead_blocks(
    gray: np.ndarray,
    panel: LEDPanelInfo,
) -> List[LEDAnomaly]:
    """Detect dead pixel blocks without mask.

    Args:
        gray: Grayscale LED region.
        panel: LED panel information.

    Returns:
        List of detected dead blocks.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = gray.shape
    block_size = 32

    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            block = gray[by:by + block_size, bx:bx + block_size]
            block_mean = np.mean(block)

            if block_mean < panel.mean_brightness * 0.2:
                severity = 1.0 - (block_mean / panel.mean_brightness)
                anomalies.append(
                    LEDAnomaly(
                        x=bx + panel.x,
                        y=by + panel.y,
                        width=block_size,
                        height=block_size,
                        anomaly_type="dead_pixel_block",
                        severity=severity,
                        description=f"Dead block at ({bx},{by})",
                    )
                )

    return anomalies


def detect_dead_blocks_in_mask(
    gray: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
) -> List[LEDAnomaly]:
    """Detect dead pixel blocks with mask.

    Args:
        gray: Grayscale LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.

    Returns:
        List of detected dead blocks.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = gray.shape

    active_pixels = gray[led_mask > 0]
    if len(active_pixels) == 0:
        return []
    active_mean = float(np.mean(active_pixels))

    block_size = 32
    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            block_mask = led_mask[by:by + block_size, bx:bx + block_size]
            led_ratio = np.mean(block_mask)

            if led_ratio < 0.7:
                continue

            block = gray[by:by + block_size, bx:bx + block_size]
            block_mean = float(np.mean(block))

            is_dead_relative = (
                active_mean > 80 and block_mean < active_mean * 0.2
            )
            is_dead_absolute = block_mean < 50

            if is_dead_relative or is_dead_absolute:
                ref = active_mean if active_mean > 0 else 1
                severity = 1.0 - (block_mean / ref)
                anomalies.append(
                    LEDAnomaly(
                        x=bx + panel.x,
                        y=by + panel.y,
                        width=block_size,
                        height=block_size,
                        anomaly_type="dead_pixel_block",
                        severity=min(severity, 1.0),
                        description=(
                            f"Dead block at ({bx},{by}) "
                            f"mean={block_mean:.0f}"
                        ),
                    )
                )

    return anomalies
