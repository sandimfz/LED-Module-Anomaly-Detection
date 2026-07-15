"""
LED Panel Analyzer.

Detector yang meniru cara manusia/AI menganalisis gambar LED:
1. Cari area LED secara otomatis (edge detection + contour)
2. Identifikasi konten normal di dalam LED
3. Deteksi anomali: blocking, dead pixels, line defects, color errors

Approach:
- Gunakan edge detection untuk menemukan boundaries LED
- Analisis texture dan konten di dalam LED area
- Bandingkan setiap region dengan sekitarnya untuk deteksi anomali
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.core.config import Config
from src.core.types import AnomalyLevel, DetectionResult, LocationConfig
from src.detectors.base import BaseDetector


def row_means_safe(y: int, gray: np.ndarray, led_mask: np.ndarray) -> float:
    """Hitung mean brightness baris y secara aman."""
    h, w = gray.shape
    if y < 0 or y >= h:
        return 999.0
    row = gray[y, :]
    mask = led_mask[y, :] > 0
    if np.sum(mask) == 0:
        return 999.0
    return float(np.mean(row[mask]))


def col_means_safe(x: int, gray: np.ndarray, led_mask: np.ndarray) -> float:
    """Hitung mean brightness kolom x secara aman."""
    h, w = gray.shape
    if x < 0 or x >= w:
        return 999.0
    col = gray[:, x]
    mask = led_mask[:, x] > 0
    if np.sum(mask) == 0:
        return 999.0
    return float(np.mean(col[mask]))


@dataclass
class LEDPanelInfo:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    mean_brightness: float
    mean_saturation: float


@dataclass
class LEDAnomaly:
    x: int
    y: int
    width: int
    height: int
    anomaly_type: str
    severity: float
    description: str


class LEDAnalyzer(BaseDetector):
    def __init__(
        self,
        config: LocationConfig,
        min_panel_area: int = 50000,
        edge_threshold: int = 50,
        blocking_threshold: float = 0.3,
    ) -> None:
        super().__init__(config)
        self.min_panel_area = min_panel_area
        self.edge_threshold = edge_threshold
        self.blocking_threshold = blocking_threshold

    def detect(
        self,
        image: np.ndarray,
        image_path: str,
    ) -> DetectionResult:
        led_panel = self._find_led_panel(image)

        if led_panel is None:
            return DetectionResult(
                location=self.config.name,
                image_path=image_path,
                anomaly_score=0.0,
                level=AnomalyLevel.NORMAL,
                message="Tidak dapat menemukan panel LED.",
            )

        led_region = image[
            led_panel.y : led_panel.y + led_panel.height,
            led_panel.x : led_panel.x + led_panel.width,
        ]

        anomalies = self._analyze_led_content(led_region, led_panel)
        anomalies = self._filter_anomalies_in_panel(anomalies, led_panel)
        score = self._calculate_score(anomalies, led_panel)

        annotated = self._annotate(image, led_panel, anomalies)
        status = self._determine_level(score)
        output_path = self.get_output_path(image_path, "led_analysis", status)
        self.save_annotated(annotated, output_path)

        return DetectionResult(
            location=self.config.name,
            image_path=image_path,
            anomaly_score=round(score, 4),
            level=AnomalyLevel(status),
            message=self._generate_message(score, anomalies, led_panel),
            flagged_cells=[],
            heatmap_path=output_path,
        )

    def _find_led_panel(self, image: np.ndarray) -> Optional[LEDPanelInfo]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_img, w_img = gray.shape

        value_channel = hsv[:, :, 2]
        sat_channel = hsv[:, :, 1]

        bright_mask = value_channel > 120
        sat_mask = sat_channel > 40
        led_mask = (bright_mask & sat_mask).astype(np.uint8)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(
            led_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_panel_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / h if h > 0 else 0
            if not (0.4 < aspect_ratio < 4.0):
                continue

            # FIX: pakai minAreaRect (rotated rectangle), bukan axis-aligned
            # bounding box, untuk hitung rectangularity. Axis-aligned bbox
            # gagal untuk kamera yang motret LED dari sudut miring (panel
            # jadi trapesium di gambar) -- bbox-nya jadi jauh lebih besar
            # dari luas panel sebenarnya, bikin rectangularity jatuh di
            # bawah threshold walau bentuknya sebenarnya rectangular (cuma
            # miring/skewed karena perspektif kamera).
            rot_rect = cv2.minAreaRect(contour)
            rot_w, rot_h = rot_rect[1]
            rot_area = rot_w * rot_h
            rectangularity = area / rot_area if rot_area > 0 else 0

            # Threshold diturunkan sedikit (0.5 -> 0.4) untuk akomodasi
            # panel yang difoto dari sudut sangat miring/ekstrem.
            if rectangularity < 0.4:
                continue

            roi_gray = gray[y : y + h, x : x + w]
            roi_sat = sat_channel[y : y + h, x : x + w]
            roi_val = value_channel[y : y + h, x : x + w]

            mean_brightness = np.mean(roi_gray)
            mean_saturation = np.mean(roi_sat)
            mean_value = np.mean(roi_val)

            if mean_brightness < 90 or mean_saturation < 35:
                continue

            size_score = area / (h_img * w_img)
            score = (
                rectangularity * 0.25
                + min(size_score * 2, 0.25)
                + (mean_brightness / 255.0) * 0.25
                + (mean_saturation / 255.0) * 0.25
            )

            candidates.append({
                "x": x, "y": y, "w": w, "h": h,
                "score": score,
                "brightness": mean_brightness,
                "saturation": mean_saturation,
                "area": area,
            })

        if not candidates:
            return None

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

    def _analyze_led_content(
        self,
        led_region: np.ndarray,
        panel: LEDPanelInfo,
    ) -> List[LEDAnomaly]:
        anomalies: List[LEDAnomaly] = []
        gray = cv2.cvtColor(led_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(led_region, cv2.COLOR_BGR2HSV)
        h_led, w_led = gray.shape

        led_content_mask = self._create_led_content_mask(gray, hsv, h_led, w_led)

        if np.sum(led_content_mask) == 0:
            return []

        blocking = self._detect_blocking(gray, panel, led_content_mask)
        anomalies.extend(blocking)

        flat_content = self._detect_flat_content(gray, panel, led_content_mask)
        anomalies.extend(flat_content)

        lines = self._detect_line_defects(gray, panel, led_content_mask)
        anomalies.extend(lines)

        dead_blocks = self._detect_dead_blocks_in_mask(gray, panel, led_content_mask)
        anomalies.extend(dead_blocks)

        color_errors = self._detect_color_errors_in_mask(hsv, panel, led_content_mask)
        anomalies.extend(color_errors)

        # Deteksi module glitch — pola baris horizontal dengan warna berbeda-beda
        line_pattern = self._detect_horizontal_line_pattern(gray, hsv, panel, led_content_mask)
        anomalies.extend(line_pattern)

        return anomalies

    def _create_led_content_mask(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        gradient_mag = cv2.GaussianBlur(gradient_mag, (5, 5), 0)

        value_channel = hsv[:, :, 2].astype(float)
        sat_channel = hsv[:, :, 1].astype(float)

        bright_score = np.clip(value_channel / 150.0, 0, 1)
        sat_score = np.clip(sat_channel / 80.0, 0, 1)

        grad_max = np.percentile(gradient_mag, 95) if np.max(gradient_mag) > 0 else 1
        texture_score = np.clip(gradient_mag / grad_max, 0, 1)

        combined = bright_score * 0.3 + sat_score * 0.3 + texture_score * 0.4

        led_mask = (combined > 0.4).astype(np.uint8)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

        led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_CLOSE, kernel_close)
        led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_OPEN, kernel_open)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        led_mask = cv2.dilate(led_mask, kernel_dilate, iterations=2)

        contours, _ = cv2.findContours(
            led_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            led_mask = np.zeros_like(led_mask)
            cv2.drawContours(led_mask, [largest], -1, 1, -1)

        return led_mask

    def _detect_blocking(
        self,
        gray: np.ndarray,
        panel: LEDPanelInfo,
        led_mask: np.ndarray,
    ) -> List[LEDAnomaly]:
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape

        led_pixels = gray[led_mask > 0]
        if len(led_pixels) == 0:
            return []

        led_mean = np.mean(led_pixels)
        led_std = np.std(led_pixels)

        threshold = max(led_mean - 2 * led_std, 30)

        dark_mask = (gray < threshold) & (led_mask > 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark_mask = cv2.morphologyEx(
            dark_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
        )
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_CLOSE, kernel
        )

        contours, _ = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)

            area_ratio = area / (h * w)

            if area_ratio < 0.001 or area_ratio > 0.5:
                continue

            is_blocking = False
            if cw > w * 0.3:
                is_blocking = True
                desc = "Horizontal blocking"
            elif ch > h * 0.3:
                is_blocking = True
                desc = "Vertical blocking"
            elif area_ratio > 0.01:
                is_blocking = True
                desc = "Large dark area"

            if is_blocking:
                severity = min(area_ratio * 10, 1.0)
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

    def _detect_flat_content(
        self,
        gray: np.ndarray,
        panel: LEDPanelInfo,
        led_mask: np.ndarray,
    ) -> List[LEDAnomaly]:
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape

        led_pixels = gray[led_mask > 0]
        if len(led_pixels) == 0:
            return []

        led_mean = float(np.mean(led_pixels))

        block_size = 64

        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block_mask = led_mask[by : by + block_size, bx : bx + block_size]
                led_ratio = np.mean(block_mask)

                if led_ratio < 0.8:
                    continue

                block = gray[by : by + block_size, bx : bx + block_size]
                block_var = float(np.var(block))
                block_mean = float(np.mean(block))

                is_blank_white = block_var < 5.0 and block_mean > 200
                is_blank_black = block_var < 5.0 and block_mean < 10
                is_abnormal_flat = (
                    block_var < 3.0
                    and block_mean > led_mean * 1.5
                )

                if is_blank_white or is_blank_black or is_abnormal_flat:
                    severity = 1.0 - min(block_var / 5.0, 1.0)
                    desc = "Blank white" if is_blank_white else (
                        "Blank black" if is_blank_black else "Abnormal flat"
                    )
                    anomalies.append(
                        LEDAnomaly(
                            x=bx + panel.x,
                            y=by + panel.y,
                            width=block_size,
                            height=block_size,
                            anomaly_type="flat_content",
                            severity=severity,
                            description=f"{desc} at ({bx},{by})",
                        )
                    )

        return self._merge_flat_anomalies(anomalies, panel)

    def _merge_flat_anomalies(
        self,
        anomalies: List[LEDAnomaly],
        panel: LEDPanelInfo,
    ) -> List[LEDAnomaly]:
        if not anomalies:
            return []

        merged: List[LEDAnomaly] = []
        used = [False] * len(anomalies)
        block_size = 64

        for i, a1 in enumerate(anomalies):
            if used[i]:
                continue

            group = [a1]
            used[i] = True

            for j, a2 in enumerate(anomalies):
                if used[j]:
                    continue
                if (
                    abs(a1.x - a2.x) < block_size * 2
                    and abs(a1.y - a2.y) < block_size * 2
                ):
                    group.append(a2)
                    used[j] = True

            if len(group) >= 4:
                min_x = min(a.x for a in group)
                min_y = min(a.y for a in group)
                max_x = max(a.x + a.width for a in group)
                max_y = max(a.y + a.height for a in group)
                avg_severity = np.mean([a.severity for a in group])

                merged.append(
                    LEDAnomaly(
                        x=min_x,
                        y=min_y,
                        width=max_x - min_x,
                        height=max_y - min_y,
                        anomaly_type="flat_content",
                        severity=avg_severity,
                        description=f"No content area ({len(group)} blocks)",
                    )
                )

        return merged

    def _detect_line_defects(
        self,
        gray: np.ndarray,
        panel: LEDPanelInfo,
        led_mask: np.ndarray,
    ) -> List[LEDAnomaly]:
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape

        led_pixels = gray[led_mask > 0]
        if len(led_pixels) == 0:
            return []

        led_mean = float(np.mean(led_pixels))
        led_std = float(np.std(led_pixels))

        threshold = max(led_mean - led_std * 2, 30)

        for y in range(2, h - 2):
            row = gray[y, :]
            row_led = led_mask[y, :] > 0

            dark_in_led = np.sum((row < threshold) & row_led)
            total_led_in_row = np.sum(row_led)

            if total_led_in_row > w * 0.3 and dark_in_led > total_led_in_row * 0.4:
                if row_means_safe(y, gray, led_mask) < row_means_safe(y - 1, gray, led_mask) \
                   and row_means_safe(y, gray, led_mask) < row_means_safe(y + 1, gray, led_mask):
                    dark_cols = np.where((row < threshold) & row_led)[0]
                    if len(dark_cols) > 0:
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

        for x in range(2, w - 2):
            col = gray[:, x]
            col_led = led_mask[:, x] > 0

            dark_in_led = np.sum((col < threshold) & col_led)
            total_led_in_col = np.sum(col_led)

            if total_led_in_col > h * 0.3 and dark_in_led > total_led_in_col * 0.4:
                if col_means_safe(x, gray, led_mask) < col_means_safe(x - 1, gray, led_mask) \
                   and col_means_safe(x, gray, led_mask) < col_means_safe(x + 1, gray, led_mask):
                    dark_rows = np.where((col < threshold) & col_led)[0]
                    if len(dark_rows) > 0:
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

    def _detect_dead_blocks(
        self,
        gray: np.ndarray,
        panel: LEDPanelInfo,
    ) -> List[LEDAnomaly]:
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape

        block_size = 32
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block = gray[by : by + block_size, bx : bx + block_size]
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

    def _detect_dead_blocks_in_mask(
        self,
        gray: np.ndarray,
        panel: LEDPanelInfo,
        led_mask: np.ndarray,
    ) -> List[LEDAnomaly]:
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape

        block_size = 32
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block_mask = led_mask[by : by + block_size, bx : bx + block_size]
                led_ratio = np.mean(block_mask)

                if led_ratio < 0.7:
                    continue

                block = gray[by : by + block_size, bx : bx + block_size]
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

    def _detect_color_errors(
        self,
        hsv: np.ndarray,
        panel: LEDPanelInfo,
    ) -> List[LEDAnomaly]:
        anomalies: List[LEDAnomaly] = []
        h, w = hsv.shape[:2]

        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]

        block_size = 64

        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block_hue = hue[by : by + block_size, bx : bx + block_size]
                block_sat = sat[by : by + block_size, bx : bx + block_size]
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

    def _detect_color_errors_in_mask(
        self,
        hsv: np.ndarray,
        panel: LEDPanelInfo,
        led_mask: np.ndarray,
    ) -> List[LEDAnomaly]:
        """Deteksi color error dengan membandingkan tiap blok ke TETANGGANYA
        (bukan ke rata-rata saturasi panel secara keseluruhan).

        FIX (bug lama): membandingkan ke rata-rata panel bikin banyak false
        positive, karena konten iklan NORMAL sering punya elemen saturasi
        rendah yang sah (teks putih, background hitam, jersey putih pemain
        bola, dll). Elemen itu low-saturation BUKAN karena rusak, tapi
        memang desainnya begitu.

        Sinyal yang lebih valid: 1 blok kecil yang TIBA-TIBA pudar padahal
        semua tetangganya di sekitarnya tetap penuh warna -- itu baru
        mencurigakan (potensi modul warna salah). Kalau area desaturated-nya
        luas dan konsisten dengan tetangga (misal logo/teks besar), itu
        wajar, bukan anomali.
        """
        anomalies: List[LEDAnomaly] = []
        h, w = hsv.shape[:2]

        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]

        block_size = 64
        rows = h // block_size
        cols = w // block_size

        # Pass 1: hitung statistik tiap blok grid
        block_stats = {}
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

        # Pass 2: bandingkan tiap blok ke tetangga (8-neighbor)
        for (r, c), stat in block_stats.items():
            neighbor_sats = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    key = (r + dr, c + dc)
                    if key in block_stats:
                        neighbor_sats.append(block_stats[key]["sat"])

            if not neighbor_sats:
                continue

            neighbor_sat_mean = float(np.mean(neighbor_sats))

            # Anomali kalau: tetangga cukup berwarna (bukan area monokrom
            # yang memang luas) TAPI blok ini jauh lebih pudar dari mereka
            is_anomaly = (
                neighbor_sat_mean > 60          # tetangga memang berwarna/vivid
                and stat["sat"] < neighbor_sat_mean * 0.35   # blok ini jauh lebih pudar
                and stat["sat"] < 30            # dan secara absolut memang pudar
            )

            if is_anomaly:
                by, bx = r * block_size, c * block_size
                severity = 1.0 - (stat["sat"] / neighbor_sat_mean)
                anomalies.append(
                    LEDAnomaly(
                        x=bx + panel.x,
                        y=by + panel.y,
                        width=block_size,
                        height=block_size,
                        anomaly_type="color_error",
                        severity=min(severity, 1.0),
                        description=f"Saturasi anomali di ({bx},{by}), tetangga={neighbor_sat_mean:.0f} vs blok={stat['sat']:.0f}",
                    )
                )

        return anomalies

    def _detect_pixel_chaos(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        panel: LEDPanelInfo,
        led_mask: np.ndarray,
    ) -> List[LEDAnomaly]:
        """Deteksi area piksel kacau/glitch (modul rusak dengan noise warna acak).

        Pendekatan: kombinasi hue_std tinggi + sat_std cukup + brightness rendah-sedang.
        Modul rusak biasanya menampilkan warna acak (hue_std tinggi) di area yang
        lebih redup — berbeda dari konten iklan berwarna cerah (teks, logo) yang
        brightness-nya tinggi.

        Catatan: tidak bisa membedakan glitch dari konten iklan berwarna cerah
        secara reliabel. Untuk kasus tersebut, gunakan Temporal Detector.

        Args:
            gray: Grayscale LED region.
            hsv: HSV LED region.
            panel: Info panel LED.
            led_mask: Mask area LED.

        Returns:
            List LEDAnomaly bertipe "pixel_chaos".
        """
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape
        block_size = 32

        rows = h // block_size
        cols = w // block_size

        if rows == 0 or cols == 0:
            return []

        hue = hsv[:, :, 0].astype(np.float32)
        sat = hsv[:, :, 1].astype(np.float32)

        # Pass 1: hitung statistik semua blok yang ada di led_mask
        block_stats: dict = {}
        for r in range(rows):
            for c in range(cols):
                by, bx = r * block_size, c * block_size
                block_mask = led_mask[by:by + block_size, bx:bx + block_size]
                led_ratio = float(np.mean(block_mask))

                if led_ratio < 0.5:
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

        if not block_stats:
            return []

        # Hitung median dan MAD hue_std di seluruh panel — robust terhadap outlier
        all_hue_stds = np.array([s["hue_std"] for s in block_stats.values()])
        median_hue_std = float(np.median(all_hue_stds))
        mad_hue_std = float(np.median(np.abs(all_hue_stds - median_hue_std)))
        chaos_threshold = max(median_hue_std + 2.5 * (1.4826 * mad_hue_std), 30.0)

        # Brightness rata-rata panel untuk filter kontekstual
        panel_brightness = float(
            np.mean([s["brightness"] for s in block_stats.values()])
        )

        # Pass 2: flag blok mencurigakan
        # Syarat: hue_std tinggi + sat_std cukup + brightness tidak terlalu tinggi
        # Filter brightness tinggi menghilangkan teks/logo terang (false positive)
        suspicious: set = set()
        for (r, c), stat in block_stats.items():
            if stat["hue_std"] <= chaos_threshold:
                continue
            # Brightness terlalu tinggi = kemungkinan teks/logo di area cerah
            if stat["brightness"] > panel_brightness * 0.85:
                continue
            # sat_std minimum — memastikan ada variasi warna nyata
            if stat["sat_std"] < 15:
                continue
            suspicious.add((r, c))

        # Pass 3: filter cluster — blok isolasi di tepi konten bukan glitch
        # Glitch nyata biasanya muncul sebagai cluster, bukan blok tunggal
        confirmed_chaos: set = set()
        for (r, c) in suspicious:
            neighbor_suspicious = sum(
                1 for dr in range(-2, 3) for dc in range(-2, 3)
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in suspicious
            )
            if neighbor_suspicious >= 2:
                confirmed_chaos.add((r, c))

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
                        f"(threshold={chaos_threshold:.1f}, "
                        f"median_panel={median_hue_std:.1f})"
                    ),
                )
            )

        # Merge blok yang berdekatan menjadi satu region
        return self._merge_chaos_anomalies(anomalies, block_size)

    def _merge_chaos_anomalies(
        self,
        anomalies: List[LEDAnomaly],
        block_size: int,
    ) -> List[LEDAnomaly]:
        """Merge blok chaos yang berdekatan menjadi satu region besar.

        Args:
            anomalies: List anomali pixel_chaos individual.
            block_size: Ukuran blok.

        Returns:
            List anomali yang sudah di-merge.
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
                # Merge jika dalam jarak 2 blok
                if (
                    abs(a1.x - a2.x) <= block_size * 2
                    and abs(a1.y - a2.y) <= block_size * 2
                ):
                    group.append(a2)
                    used[j] = True

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
                    description=f"Pixel chaos region ({len(group)} blocks)",
                )
            )

        return merged

    def _detect_horizontal_line_pattern(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        panel: LEDPanelInfo,
        led_mask: np.ndarray,
    ) -> List[LEDAnomaly]:
        """Deteksi modul glitch dengan pola baris horizontal berwarna acak.

        Modul LED yang error seringkali menampilkan data corrupt berupa baris-baris
        horizontal dengan warna berbeda-beda tiap baris (merah, biru, kuning, dll).
        Ini berbeda dari line defect biasa yang hanya 1-2 garis gelap.

        Sinyal kunci: area di mana brightness antar baris berubah drastis
        (row-to-row variance sangat tinggi), terlokalisir di satu region.

        Args:
            gray: Grayscale LED region.
            hsv: HSV LED region.
            panel: Info panel LED.
            led_mask: Mask area LED.

        Returns:
            List LEDAnomaly bertipe "module_glitch".
        """
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape
        
        if h < 64 or w < 64:
            return []

        # Hitung mean brightness tiap baris
        row_means = np.mean(gray, axis=1)
        
        # Hitung absolute diff antar baris berurutan
        row_diffs = np.abs(np.diff(row_means))
        
        # Sliding window: cari area 32-64 baris di mana row_diff sangat tinggi
        window_size = 32
        high_diff_regions = []
        
        for y in range(0, h - window_size, 8):
            window_diff = row_diffs[y:y+window_size]
            avg_diff = np.mean(window_diff)
            
            # Threshold: avg_diff > 3.0 menandakan baris sangat bervariasi
            if avg_diff > 3.0:
                high_diff_regions.append((y, y + window_size, avg_diff))
        
        # Untuk tiap region dengan high diff, cek apakah terlokalisir secara horizontal
        # (hanya sebagian kolom, bukan seluruh lebar panel)
        for y_start, y_end, avg_diff in high_diff_regions:
            # Bagi lebar panel jadi beberapa bagian, cek kolom mana yang punya diff tinggi
            num_sections = 8
            section_width = w // num_sections
            section_diffs = []
            
            for sx in range(num_sections):
                x_start = sx * section_width
                x_end = (sx + 1) * section_width
                section = gray[y_start:y_end, x_start:x_end]
                section_row_means = np.mean(section, axis=1)
                section_diff = np.mean(np.abs(np.diff(section_row_means)))
                section_diffs.append((sx, section_diff))
            
            # Cari section dengan diff tertinggi
            max_section_idx, max_section_diff = max(section_diffs, key=lambda x: x[1])
            
            # Jika diff-nya sangat tinggi (> 5.0) dan terlokalisir (tidak semua section tinggi)
            other_sections_avg = np.mean([d for i, d in section_diffs if i != max_section_idx])
            
            is_localized = max_section_diff > 5.0 and max_section_diff > other_sections_avg * 2.0
            
            if is_localized:
                x_anomaly = max_section_idx * section_width
                w_anomaly = section_width * 2  # span 2 section untuk coverage
                h_anomaly = y_end - y_start
                
                # Cek apakah region ini ada di dalam led_mask
                anomaly_mask = led_mask[y_start:y_end, x_anomaly:min(x_anomaly+w_anomaly, w)]
                if np.mean(anomaly_mask) < 0.3:
                    continue
                
                severity = min(max_section_diff / 10.0, 1.0)
                
                anomalies.append(
                    LEDAnomaly(
                        x=x_anomaly + panel.x,
                        y=y_start + panel.y,
                        width=min(w_anomaly, w - x_anomaly),
                        height=h_anomaly,
                        anomaly_type="module_glitch",
                        severity=severity,
                        description=(
                            f"Module glitch di ({x_anomaly},{y_start}): "
                            f"row_diff={max_section_diff:.1f}"
                        ),
                    )
                )
        
        return anomalies

    def _filter_anomalies_in_panel(
        self,
        anomalies: List[LEDAnomaly],
        panel: LEDPanelInfo,
    ) -> List[LEDAnomaly]:
        """Filter anomali — hanya yang SEPENUHNYA di dalam panel LED.

        Args:
            anomalies: List anomali.
            panel: Info panel LED.

        Returns:
            List anomali yang sudah difilter.
        """
        filtered: List[LEDAnomaly] = []
        panel_area = panel.width * panel.height

        # Panel boundaries
        px1 = panel.x
        py1 = panel.y
        px2 = panel.x + panel.width
        py2 = panel.y + panel.height

        for a in anomalies:
            # Seluruh anomali harus di dalam panel
            ax1 = a.x
            ay1 = a.y
            ax2 = a.x + a.width
            ay2 = a.y + a.height

            if ax1 < px1 or ay1 < py1 or ax2 > px2 or ay2 > py2:
                continue

            # Filter anomali terlalu kecil (noise)
            anomaly_area = a.width * a.height
            if anomaly_area < panel_area * 0.001:
                continue

            # Filter anomali terlalu besar (false positive)
            if anomaly_area > panel_area * 0.5:
                continue

            filtered.append(a)

        return filtered

    def _calculate_score(
        self,
        anomalies: List[LEDAnomaly],
        panel: LEDPanelInfo,
    ) -> float:
        if not anomalies:
            return 0.0

        total_area = panel.width * panel.height
        anomaly_area = sum(a.width * a.height for a in anomalies)

        avg_severity = np.mean([a.severity for a in anomalies])

        area_ratio = anomaly_area / total_area if total_area > 0 else 0
        score = (area_ratio * 0.5 + avg_severity * 0.5) * 2

        return min(score, 1.0)

    def _annotate(
        self,
        image: np.ndarray,
        panel: LEDPanelInfo,
        anomalies: List[LEDAnomaly],
    ) -> np.ndarray:
        annotated = image.copy()

        cv2.rectangle(
            annotated,
            (panel.x, panel.y),
            (panel.x + panel.width, panel.y + panel.height),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            f"LED Panel ({panel.width}x{panel.height})",
            (panel.x, panel.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )

        colors = {
            "blocking": (0, 0, 255),
            "line_defect": (0, 165, 255),
            "dead_pixel_block": (0, 0, 200),
            "color_error": (255, 0, 255),
            "pixel_chaos": (0, 255, 255),  # Cyan untuk glitch/chaos
            "flat_content": (255, 128, 0),
            "module_glitch": (0, 255, 255),  # Cyan untuk module glitch
        }

        for anomaly in anomalies:
            color = colors.get(anomaly.anomaly_type, (0, 0, 255))
            cv2.rectangle(
                annotated,
                (anomaly.x, anomaly.y),
                (anomaly.x + anomaly.width, anomaly.y + anomaly.height),
                color,
                2,
            )
            cv2.putText(
                annotated,
                anomaly.anomaly_type[:8],
                (anomaly.x, anomaly.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        return annotated

    def _generate_message(
        self,
        score: float,
        anomalies: List[LEDAnomaly],
        panel: LEDPanelInfo,
    ) -> str:
        if not anomalies:
            return (
                f"Panel LED terdeteksi ({panel.width}x{panel.height}). "
                f"Tidak ada anomali."
            )

        blocking = [a for a in anomalies if a.anomaly_type == "blocking"]
        flat = [a for a in anomalies if a.anomaly_type == "flat_content"]
        lines = [a for a in anomalies if a.anomaly_type == "line_defect"]
        dead = [a for a in anomalies if a.anomaly_type == "dead_pixel_block"]
        color = [a for a in anomalies if a.anomaly_type == "color_error"]
        chaos = [a for a in anomalies if a.anomaly_type == "pixel_chaos"]
        glitch = [a for a in anomalies if a.anomaly_type == "module_glitch"]

        parts = []
        if blocking:
            parts.append(f"{len(blocking)} blocking")
        if flat:
            parts.append(f"{len(flat)} no content areas")
        if lines:
            parts.append(f"{len(lines)} line defects")
        if dead:
            parts.append(f"{len(dead)} dead blocks")
        if color:
            parts.append(f"{len(color)} color errors")
        if chaos:
            parts.append(f"{len(chaos)} pixel chaos (modul rusak/glitch)")
        if glitch:
            parts.append(f"{len(glitch)} module glitch (baris warna acak)")

        return (
            f"Panel LED terdeteksi ({panel.width}x{panel.height}). "
            f"Anomali: {', '.join(parts)}. "
            f"Score: {score:.2f}"
        )