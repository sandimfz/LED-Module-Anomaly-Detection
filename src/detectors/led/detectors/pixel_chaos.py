"""
Pixel chaos detection module.

Detects glitchy or chaotic pixel patterns indicating module damage.
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


def detect_pixel_chaos(
    gray: np.ndarray,
    hsv: np.ndarray,
    panel: LEDPanelInfo,
    led_mask: np.ndarray,
) -> List[LEDAnomaly]:
    """Detect pixel chaos or glitch patterns.

    Uses high hue_std + moderate sat_std + low-mid brightness
    to identify damaged modules with random color noise.

    Args:
        gray: Grayscale LED region.
        hsv: HSV LED region.
        panel: LED panel information.
        led_mask: Binary mask of LED content.

    Returns:
        List of detected pixel chaos anomalies.
    """
    h, w = gray.shape
    block_size = 32

    rows = h // block_size
    cols = w // block_size

    if rows == 0 or cols == 0:
        return []

    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    block_stats = _compute_chaos_stats(
        gray, hue, sat, led_mask, block_size, rows, cols
    )

    if not block_stats:
        return []

    chaos_threshold = _compute_chaos_threshold(block_stats)
    panel_brightness = _compute_panel_brightness(block_stats)

    suspicious = _find_suspicious_blocks(
        block_stats, chaos_threshold, panel_brightness
    )

    confirmed_chaos = _confirm_chaos_clusters(suspicious)

    anomalies = _create_chaos_anomalies(
        confirmed_chaos, block_stats, panel, chaos_threshold
    )

    return merge_chaos_anomalies(anomalies, block_size)


def _compute_chaos_stats(
    gray: np.ndarray,
    hue: np.ndarray,
    sat: np.ndarray,
    led_mask: np.ndarray,
    block_size: int,
    rows: int,
    cols: int,
) -> Dict[Tuple[int, int], Dict]:
    """Compute statistics for chaos detection.

    Args:
        gray: Grayscale image.
        hue: Hue channel.
        sat: Saturation channel.
        led_mask: Binary mask.
        block_size: Block size.
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        Dictionary of block statistics.
    """
    block_stats: Dict[Tuple[int, int], Dict] = {}

    for r in range(rows):
        for c in range(cols):
            by, bx = r * block_size, c * block_size
            block_mask = led_mask[by:by + block_size, bx:bx + block_size]
            led_ratio = float(np.mean(block_mask))

            if led_ratio < 0.8:
                continue

            b_hue = hue[by:by + block_size, bx:bx + block_size]
            b_sat = sat[by:by + block_size, bx:bx + block_size]
            b_gray = gray[by:by + block_size, bx:bx + block_size]

            block_stats[(r, c)] = {
                "hue_std": float(np.std(b_hue)),
                "sat_std": float(np.std(b_sat)),
                "brightness": float(np.mean(b_gray)),
                "bx": bx,
                "by": by,
            }

    return block_stats


def _compute_chaos_threshold(
    block_stats: Dict[Tuple[int, int], Dict],
) -> float:
    """Compute chaos threshold using MAD.

    Args:
        block_stats: Block statistics.

    Returns:
        Chaos threshold value.
    """
    all_hue_stds = np.array([s["hue_std"] for s in block_stats.values()])
    median_hue_std = float(np.median(all_hue_stds))
    mad_hue_std = float(np.median(np.abs(all_hue_stds - median_hue_std)))
    return max(median_hue_std + 2.5 * (1.4826 * mad_hue_std), 30.0)


def _compute_panel_brightness(
    block_stats: Dict[Tuple[int, int], Dict],
) -> float:
    """Compute average panel brightness.

    Args:
        block_stats: Block statistics.

    Returns:
        Average brightness.
    """
    return float(
        np.mean([s["brightness"] for s in block_stats.values()])
    )


def _find_suspicious_blocks(
    block_stats: Dict[Tuple[int, int], Dict],
    chaos_threshold: float,
    panel_brightness: float,
) -> Set[Tuple[int, int]]:
    """Find blocks with suspicious chaos patterns.

    A genuine glitch block is:
    - hue_std significantly above adaptive threshold (chaos)
    - sat_std high enough to confirm colorful noise (not just dark area)
    - brightness NOT too low — dark areas naturally have high relative
      hue_std due to noise, so we require the block to be reasonably lit.
      Previously this was brightness < panel*0.85, which included dark
      background areas and caused false positives on normal content.

    Args:
        block_stats: Block statistics.
        chaos_threshold: Chaos threshold.
        panel_brightness: Panel average brightness.

    Returns:
        Set of suspicious block coordinates.
    """
    suspicious: Set[Tuple[int, int]] = set()

    for (r, c), stat in block_stats.items():
        if stat["hue_std"] <= chaos_threshold:
            continue
        # Must be reasonably bright — skip genuinely dark background blocks.
        # Dark blocks often have high hue variance from sensor noise,
        # not from actual glitch content.
        if stat["brightness"] < 60:
            continue
        # Also skip very bright blocks — they are likely clean highlights,
        # not chaos (inverted from before: was < panel*0.85, now we check
        # that brightness is in a mid range that suggests active but glitchy).
        if stat["brightness"] > panel_brightness * 1.4:
            continue
        if stat["sat_std"] < 20:
            continue
        suspicious.add((r, c))

    return suspicious


def _confirm_chaos_clusters(
    suspicious: Set[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    """Confirm chaos blocks that have enough neighbors.

    Args:
        suspicious: Set of suspicious block coordinates.

    Returns:
        Set of confirmed chaos block coordinates.
    """
    confirmed: Set[Tuple[int, int]] = set()

    for (r, c) in suspicious:
        neighbor_count = sum(
            1 for dr in range(-2, 3) for dc in range(-2, 3)
            if (dr, dc) != (0, 0) and (r + dr, c + dc) in suspicious
        )
        if neighbor_count >= 3:
            confirmed.add((r, c))

    return confirmed


def _create_chaos_anomalies(
    confirmed_chaos: Set[Tuple[int, int]],
    block_stats: Dict[Tuple[int, int], Dict],
    panel: LEDPanelInfo,
    chaos_threshold: float,
) -> List[LEDAnomaly]:
    """Create anomaly objects from confirmed chaos blocks.

    Args:
        confirmed_chaos: Set of confirmed chaos coordinates.
        block_stats: Block statistics.
        panel: LED panel information.
        chaos_threshold: Chaos threshold used.

    Returns:
        List of chaos anomalies.
    """
    anomalies: List[LEDAnomaly] = []
    block_size = 32

    for (r, c) in confirmed_chaos:
        stat = block_stats[(r, c)]
        bx = stat["bx"]
        by = stat["by"]
        severity = min(stat["hue_std"] / 80.0, 1.0)

        anomalies.append(
            LEDAnomaly(
                x=bx + panel.x,
                y=by + panel.y,
                width=block_size,
                height=block_size,
                anomaly_type="pixel_chaos",
                severity=severity,
                description=(
                    f"Pixel chaos di ({bx},{by}): "
                    f"hue_std={stat['hue_std']:.1f} "
                    f"(threshold={chaos_threshold:.1f})"
                ),
            )
        )

    return anomalies


def merge_chaos_anomalies(
    anomalies: List[LEDAnomaly],
    block_size: int,
) -> List[LEDAnomaly]:
    """Merge adjacent chaos anomalies into larger regions.

    Args:
        anomalies: List of pixel chaos anomalies.
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
                abs(a1.x - a2.x) < block_size * 3
                and abs(a1.y - a2.y) < block_size * 3
            ):
                group.append(a2)
                used[j] = True

        if len(group) >= 2:
            min_x = min(a.x for a in group)
            min_y = min(a.y for a in group)
            max_x = max(a.x + a.width for a in group)
            max_y = max(a.y + a.height for a in group)
            avg_severity = float(np.mean([a.severity for a in group]))

            merged.append(
                LEDAnomaly(
                    x=min_x,
                    y=min_y,
                    width=max_x - min_x,
                    height=max_y - min_y,
                    anomaly_type="pixel_chaos",
                    severity=avg_severity,
                    description=(
                        f"Pixel chaos cluster ({len(group)} blocks)"
                    ),
                )
            )

    return merged
