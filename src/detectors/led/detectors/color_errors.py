"""
Color error detection module.

Detects color anomalies by comparing blocks to their neighbors.
"""

from typing import Dict, List, Tuple

import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_color_errors(
    hsv: np.ndarray,
    panel: LEDPanelInfo,
) -> List[LEDAnomaly]:
    """Detect color errors based on saturation comparison.

    Args:
        hsv: HSV LED region.
        panel: LED panel information.

    Returns:
        List of detected color errors.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = hsv.shape[:2]

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    block_size = 64

    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            block_hue = hue[by:by + block_size, bx:bx + block_size]
            block_sat = sat[by:by + block_size, bx:bx + block_size]
            hue_mean = np.mean(block_hue)
            sat_mean = np.mean(block_sat)

            if sat_mean < 30 and panel.mean_saturation > 50:
                severity = 1.0 - (sat_mean / panel.mean_saturation)
                anomalies.append(
                    LEDAnomaly(
                        x=bx + panel.x,
                        y=by + panel.y,
                        width=block_size,
                        height=block_size,
                        anomaly_type="color_error",
                        severity=severity,
                        description=f"Low saturation at ({bx},{by})",
                    )
                )

    return anomalies


def detect_color_errors_in_mask(
    hsv: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
) -> List[LEDAnomaly]:
    """Detect color errors by comparing blocks to their neighbors.

    Two anomaly types are detected:
    1. **Saturation drop** — block nearly greyscale while neighbors
       are highly saturated (low-sat module error).
    2. **Hue shift** — block's dominant hue differs greatly from
       neighbors (wrong-color module error, e.g. magenta/purple block
       in a yellow/white scene).

    Args:
        hsv: HSV full image.
        panel: LED panel information.
        led_mask: Binary mask of LED content.

    Returns:
        List of detected color errors.
    """
    anomalies: List[LEDAnomaly] = []
    h, w = hsv.shape[:2]

    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    block_size = 48
    rows = h // block_size
    cols = w // block_size

    block_stats = _compute_block_stats_full(
        hue, sat, val, led_mask, block_size, rows, cols
    )

    for (r, c), stat in block_stats.items():
        neighbors = _get_neighbor_stats(block_stats, r, c)
        if len(neighbors) < 3:
            continue

        n_sat = float(np.mean([n["sat"] for n in neighbors]))

        # 1. Saturation drop: block grey while neighbors colorful
        sat_drop = (
            n_sat > 70
            and stat["sat"] < n_sat * 0.3
            and stat["val"] > 60
        )

        # 2. Hue shift: dominant hue differs greatly from neighbors.
        # Hue is 0-180 in OpenCV. Circular distance used.
        # Require ≥3 saturated neighbors to reduce false positives
        # on content with varied colors (green soccer vs white text).
        n_hues = [n["hue"] for n in neighbors if n["sat"] > 40]
        hue_shift = False
        if len(n_hues) >= 3 and stat["sat"] > 40 and stat["val"] > 60:
            n_hue_median = float(np.median(n_hues))
            diff = abs(stat["hue"] - n_hue_median)
            circ_diff = min(diff, 180.0 - diff)
            # 70° threshold avoids false positives on normal ad content
            # (white text vs blue bg = ~70-80°, which is normal design)
            # Actual module errors typically show >80° hue shift.
            if circ_diff > 70:
                hue_shift = True

        if sat_drop or hue_shift:
            by, bx = r * block_size, c * block_size
            if sat_drop:
                sev = min(1.0 - (stat["sat"] / (n_sat + 1e-6)), 1.0)
                desc = (
                    f"Sat drop ({bx},{by}): "
                    f"blok={stat['sat']:.0f} vs neighbor={n_sat:.0f}"
                )
            else:
                n_hue_median = float(np.median(n_hues))
                diff = abs(stat["hue"] - n_hue_median)
                circ_diff = min(diff, 180.0 - diff)
                sev = min(circ_diff / 90.0, 1.0)
                desc = (
                    f"Hue shift ({bx},{by}): "
                    f"blok={stat['hue']:.0f} vs neighbor={n_hue_median:.0f}"
                )

            anomalies.append(
                LEDAnomaly(
                    x=bx + panel.x,
                    y=by + panel.y,
                    width=block_size,
                    height=block_size,
                    anomaly_type="color_error",
                    severity=sev,
                    description=desc,
                )
            )

    return _merge_color_anomalies(anomalies, block_size)


def _compute_block_stats(
    hue: np.ndarray,
    sat: np.ndarray,
    led_mask: np.ndarray,
    block_size: int,
    rows: int,
    cols: int,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Compute statistics for each block (legacy, sat+hue only).

    Args:
        hue: Hue channel.
        sat: Saturation channel.
        led_mask: Binary mask of LED content.
        block_size: Size of each block.
        rows: Number of block rows.
        cols: Number of block columns.

    Returns:
        Dictionary mapping (row, col) to block statistics.
    """
    block_stats: Dict[Tuple[int, int], Dict[str, float]] = {}

    for r in range(rows):
        for c in range(cols):
            by, bx = r * block_size, c * block_size
            block_mask = led_mask[by:by + block_size, bx:bx + block_size]
            led_ratio = float(np.mean(block_mask))

            if led_ratio < 0.7:
                continue

            block_hue = hue[by:by + block_size, bx:bx + block_size]
            block_sat = sat[by:by + block_size, bx:bx + block_size]

            block_stats[(r, c)] = {
                "hue": float(np.mean(block_hue)),
                "sat": float(np.mean(block_sat)),
            }

    return block_stats


def _compute_block_stats_full(
    hue: np.ndarray,
    sat: np.ndarray,
    val: np.ndarray,
    led_mask: np.ndarray,
    block_size: int,
    rows: int,
    cols: int,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Compute per-block hue, sat, val statistics within LED mask.

    Args:
        hue: Hue channel (float32).
        sat: Saturation channel (float32).
        val: Value channel (float32).
        led_mask: Binary mask of LED content.
        block_size: Block size in pixels.
        rows: Grid rows.
        cols: Grid columns.

    Returns:
        Dict mapping (row, col) → {hue, sat, val}.
    """
    block_stats: Dict[Tuple[int, int], Dict[str, float]] = {}

    for r in range(rows):
        for c in range(cols):
            by, bx = r * block_size, c * block_size
            block_mask = led_mask[by:by + block_size, bx:bx + block_size]
            if float(np.mean(block_mask)) < 0.6:
                continue

            b_hue = hue[by:by + block_size, bx:bx + block_size]
            b_sat = sat[by:by + block_size, bx:bx + block_size]
            b_val = val[by:by + block_size, bx:bx + block_size]

            block_stats[(r, c)] = {
                "hue": float(np.mean(b_hue)),
                "sat": float(np.mean(b_sat)),
                "val": float(np.mean(b_val)),
            }

    return block_stats


def _get_neighbor_stats(
    block_stats: Dict[Tuple[int, int], Dict[str, float]],
    r: int,
    c: int,
) -> List[Dict[str, float]]:
    """Return stats of 8-connected neighbor blocks.

    Args:
        block_stats: Block statistics dictionary.
        r: Row index.
        c: Column index.

    Returns:
        List of neighbor stat dicts.
    """
    neighbors: List[Dict[str, float]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            key = (r + dr, c + dc)
            if key in block_stats:
                neighbors.append(block_stats[key])
    return neighbors


def _get_neighbor_sats(
    block_stats: Dict[Tuple[int, int], Dict[str, float]],
    r: int,
    c: int,
) -> List[float]:
    """Get saturation values of neighboring blocks.

    Args:
        block_stats: Block statistics dictionary.
        r: Row index.
        c: Column index.

    Returns:
        List of neighbor saturation values.
    """
    neighbor_sats: List[float] = []

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            key = (r + dr, c + dc)
            if key in block_stats:
                neighbor_sats.append(block_stats[key]["sat"])

    return neighbor_sats


def _merge_color_anomalies(
    anomalies: List[LEDAnomaly],
    block_size: int,
) -> List[LEDAnomaly]:
    """Merge adjacent color anomalies into larger regions.

    Adjacent blocks with the same anomaly are grouped into one
    larger bounding box to reduce noise in the output.

    Args:
        anomalies: List of color anomalies.
        block_size: Block size used.

    Returns:
        List of merged anomalies.
    """
    if not anomalies:
        return []

    merged: List[LEDAnomaly] = []
    used = [False] * len(anomalies)

    for i, a1 in enumerate(anomalies):
        if used[i]:
            continue
        group = [a1]
        used[i] = True

        for j, a2 in enumerate(anomalies):
            if used[j]:
                continue
            if (
                abs(a1.x - a2.x) <= block_size * 2
                and abs(a1.y - a2.y) <= block_size * 2
            ):
                group.append(a2)
                used[j] = True

        if len(group) >= 2:
            min_x = min(a.x for a in group)
            min_y = min(a.y for a in group)
            max_x = max(a.x + a.width for a in group)
            max_y = max(a.y + a.height for a in group)
            avg_sev = float(np.mean([a.severity for a in group]))
            merged.append(
                LEDAnomaly(
                    x=min_x,
                    y=min_y,
                    width=max_x - min_x,
                    height=max_y - min_y,
                    anomaly_type="color_error",
                    severity=avg_sev,
                    description=(
                        f"Color error cluster ({len(group)} blocks)"
                    ),
                )
            )
        else:
            merged.append(a1)

    return merged
