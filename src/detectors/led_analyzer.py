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


@dataclass
class LEDPanelInfo:
    """Informasi panel LED yang terdeteksi.

    Attributes:
        x: Koordinat x top-left.
        y: Koordinat y top-left.
        width: Lebar panel.
        height: Tinggi panel.
        confidence: Confidence score (0-1).
        mean_brightness: Rata-rata brightness panel.
        mean_saturation: Rata-rata saturasi panel.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float
    mean_brightness: float
    mean_saturation: float


@dataclass
class LEDAnomaly:
    """Anomali yang terdeteksi di LED.

    Attributes:
        x: Koordinat x.
        y: Koordinat y.
        width: Lebar anomali.
        height: Tinggi anomali.
        anomaly_type: Jenis anomali (blocking, dead_pixel, line_defect, color_error).
        severity: Tingkat keparahan (0-1).
        description: Deskripsi anomali.
    """

    x: int
    y: int
    width: int
    height: int
    anomaly_type: str
    severity: float
    description: str


class LEDAnalyzer(BaseDetector):
    """Analyzer yang meniru cara AI membaca gambar LED.

    Attributes:
        config: Konfigurasi lokasi.
        min_panel_area: Minimal area LED panel (pixel).
        edge_threshold: Threshold untuk Canny edge detection.
        blocking_threshold: Threshold untuk deteksi blocking.
    """

    def __init__(
        self,
        config: LocationConfig,
        min_panel_area: int = 50000,
        edge_threshold: int = 50,
        blocking_threshold: float = 0.3,
    ) -> None:
        """Initialize LED analyzer.

        Args:
            config: Konfigurasi lokasi.
            min_panel_area: Minimal area LED panel.
            edge_threshold: Threshold untuk edge detection.
            blocking_threshold: Threshold untuk blocking detection.
        """
        super().__init__(config)
        self.min_panel_area = min_panel_area
        self.edge_threshold = edge_threshold
        self.blocking_threshold = blocking_threshold

    def detect(
        self,
        image: np.ndarray,
        image_path: str,
    ) -> DetectionResult:
        """Analisis gambar LED seperti cara AI membaca.

        Args:
            image: Gambar BGR.
            image_path: Path gambar.

        Returns:
            DetectionResult.
        """
        # Step 1: Temukan area LED
        led_panel = self._find_led_panel(image)

        if led_panel is None:
            return DetectionResult(
                location=self.config.name,
                image_path=image_path,
                anomaly_score=0.0,
                level=AnomalyLevel.NORMAL,
                message="Tidak dapat menemukan panel LED.",
            )

        # Step 2: Ekstrak area LED
        led_region = image[
            led_panel.y : led_panel.y + led_panel.height,
            led_panel.x : led_panel.x + led_panel.width,
        ]

        # Step 3: Analisis konten LED
        anomalies = self._analyze_led_content(led_region, led_panel)

        # Step 4: Filter anomali — HANYA yang di area LED (terang + berwarna)
        anomalies = self._filter_anomalies_in_panel(anomalies, led_panel)

        # Step 5: Hitung skor
        score = self._calculate_score(anomalies, led_panel)

        # Step 5: Buat visualisasi
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
        """Temukan panel LED menggunakan color-based detection.

        LED panel memiliki karakteristik:
        1. Area terang dengan warna konsisten
        2. Bentuk rectangular
        3. Besar dan di posisi tengah/frame

        Args:
            image: Gambar BGR.

        Returns:
            LEDPanelInfo atau None jika tidak ditemukan.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_img, w_img = gray.shape

        # Step 1: Buat mask untuk area yang KEMUNGKINAN LED
        # LED harus terang (value tinggi) dan berwarna (saturation tinggi)
        # Threshold lebih tinggi untuk menghindari bezel gelap
        value_channel = hsv[:, :, 2]
        sat_channel = hsv[:, :, 1]

        # LED content: terang DAN berwarna
        bright_mask = value_channel > 120
        sat_mask = sat_channel > 40
        led_mask = (bright_mask & sat_mask).astype(np.uint8)

        # Step 2: Morphological cleanup
        # Close untuk menghubungkan area terputus
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_CLOSE, kernel_close)

        # Open untuk menghilangkan noise kecil
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_OPEN, kernel_open)

        # Step 3: Cari kontur
        contours, _ = cv2.findContours(
            led_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Step 4: Evaluate setiap kontur sebagai kandidat LED panel
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter area terlalu kecil
            if area < self.min_panel_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Aspect ratio harus wajar untuk LED panel
            aspect_ratio = w / h if h > 0 else 0
            if not (0.4 < aspect_ratio < 4.0):
                continue

            # Hitung rectangularity
            rect_area = w * h
            if rect_area == 0:
                continue
            rectangularity = area / rect_area

            # Harus cukup rectangular (>0.5)
            if rectangularity < 0.5:
                continue

            # Analisis isi bounding box
            roi_gray = gray[y : y + h, x : x + w]
            roi_sat = sat_channel[y : y + h, x : x + w]
            roi_val = value_channel[y : y + h, x : x + w]

            mean_brightness = np.mean(roi_gray)
            mean_saturation = np.mean(roi_sat)
            mean_value = np.mean(roi_val)

            # LED harus terang dan berwarna
            if mean_brightness < 90 or mean_saturation < 35:
                continue

            # Score: kombinasi ukuran, rectangularity, brightness, saturation
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

        # Step 5: Ambil kandidat TERBAIK (skor tertinggi)
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
        """Analisis konten di dalam area LED.

        Metode:
        1. Buat mask LED content menggunakan color analysis
        2. Analisis anomali HANYA di dalam mask
        3. Filter berdasarkan posisi aktual

        Args:
            led_region: Region gambar area LED.
            panel: Info panel LED.

        Returns:
            List anomali yang terdeteksi.
        """
        anomalies: List[LEDAnomaly] = []
        gray = cv2.cvtColor(led_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(led_region, cv2.COLOR_BGR2HSV)
        h_led, w_led = gray.shape

        # Step 1: Buat mask LED content
        led_content_mask = self._create_led_content_mask(gray, hsv, h_led, w_led)

        # Step 2: Jika tidak ada LED content, return empty
        if np.sum(led_content_mask) == 0:
            return []

        # Step 3: Analisis anomali
        # 1. Blocking
        blocking = self._detect_blocking(gray, panel, led_content_mask)
        anomalies.extend(blocking)

        # 2. Flat/No Content
        flat_content = self._detect_flat_content(gray, panel, led_content_mask)
        anomalies.extend(flat_content)

        # 3. Line Defects
        lines = self._detect_line_defects(gray, panel, led_content_mask)
        anomalies.extend(lines)

        # 4. Dead Pixel Blocks
        dead_blocks = self._detect_dead_blocks_in_mask(gray, panel, led_content_mask)
        anomalies.extend(dead_blocks)

        # 5. Color Errors
        color_errors = self._detect_color_errors_in_mask(hsv, panel, led_content_mask)
        anomalies.extend(color_errors)

        return anomalies

    def _create_led_content_mask(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Buat mask untuk area LED content.

        Pendekatan sederhana:
        1. Threshold brightness + saturation
        2. Morphological cleanup
        3. Ambil connected component terbesar

        Args:
            gray: Grayscale image.
            hsv: HSV image.
            h: Height.
            w: Width.

        Returns:
            Binary mask (1 = LED content, 0 = non-LED).
        """
        # Threshold untuk LED content
        value_channel = hsv[:, :, 2]
        sat_channel = hsv[:, :, 1]

        bright_mask = value_channel > 100
        sat_mask = sat_channel > 40

        led_mask = (bright_mask & sat_mask).astype(np.uint8)

        # Morphological cleanup
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_CLOSE, kernel_close)
        led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_OPEN, kernel_open)

        # Dilate untuk menghubungkan area terputus
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        led_mask = cv2.dilate(led_mask, kernel_dilate, iterations=2)

        # Ambil connected component TERBESAR
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
        """Deteksi blocking (area gelap yang memotong konten LED).

        Metode yang lebih akurat:
        1. Gunakan adaptive threshold untuk menemukan area gelap
        2. Filter berdasarkan ukuran dan posisi
        3. Hanya deteksi di dalam LED content mask

        Args:
            gray: Gambar grayscale area LED.
            panel: Info panel LED.
            led_mask: Mask area LED content (hanya analisis di sini).

        Returns:
            List blocking anomalies.
        """
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape

        # Hitung threshold berdasarkan local statistics
        # Gunakan mean brightness dari area LED sebagai referensi
        led_pixels = gray[led_mask > 0]
        if len(led_pixels) == 0:
            return []

        led_mean = np.mean(led_pixels)
        led_std = np.std(led_pixels)

        # Threshold: area gelap jika < mean - 2*std
        # Ini lebih adaptif daripada fixed percentage
        threshold = max(led_mean - 2 * led_std, 30)

        # Dark mask: area yang gelap DAN di dalam LED content
        dark_mask = (gray < threshold) & (led_mask > 0)

        # Cleanup
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
            if area < 200:  # Abaikan noise (naikkan threshold)
                continue

            x, y, cw, ch = cv2.boundingRect(contour)

            # Hitung area ratio
            area_ratio = area / (h * w)

            # Filter berdasarkan ukuran
            # Terlalu kecil = noise, terlalu besar = false positive
            if area_ratio < 0.001 or area_ratio > 0.5:
                continue

            # Cek apakah ini blocking (membentuk garis/rectangle panjang)
            is_blocking = False
            if cw > w * 0.3:  # Horizontal blocking
                is_blocking = True
                desc = "Horizontal blocking"
            elif ch > h * 0.3:  # Vertical blocking
                is_blocking = True
                desc = "Vertical blocking"
            elif area_ratio > 0.01:  # Large dark area (> 1% of LED)
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
        """Deteksi area flat/rata tanpa konten (no content area).

        Area yang memiliki variance sangat rendah menandakan
        tidak ada konten yang ditampilkan (blank/flat).

        Args:
            gray: Gambar grayscale area LED.
            panel: Info panel LED.
            led_mask: Mask area LED content.

        Returns:
            List flat content anomalies.
        """
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape

        # Buat mask yang HANYA mencakup area di dalam panel bounding box
        # Ini memastikan tidak ada deteksi di luar LED
        panel_mask = np.zeros_like(led_mask)
        # panel.x dan panel.y adalah koordinat relatif terhadap led_region
        panel_mask[0:h, 0:w] = 1  # Karena led_region sudah di-crop ke panel

        # Gabungkan dengan led_mask
        combined_mask = (led_mask > 0) & (panel_mask > 0)

        # Bagi menjadi blok-blok dan cek variance
        block_size = 64
        flat_threshold = 15.0  # Variance threshold untuk dianggap "flat"

        for by in range(0, h - block_size, block_size // 2):
            for bx in range(0, w - block_size, block_size // 2):
                # Cek apakah block ini di dalam area LED
                block_mask = combined_mask[by : by + block_size, bx : bx + block_size]
                led_ratio = np.mean(block_mask)

                # Hanya analisis jika > 70% block adalah LED content
                if led_ratio < 0.7:
                    continue

                block = gray[by : by + block_size, bx : bx + block_size]
                block_var = float(np.var(block))

                # Cek apakah block ini flat
                if block_var < flat_threshold:
                    block_mean = np.mean(block)
                    if block_mean > panel.mean_brightness * 0.3:
                        # Area terang tapi flat = no content
                        severity = 1.0 - (block_var / flat_threshold)
                        anomalies.append(
                            LEDAnomaly(
                                x=bx + panel.x,
                                y=by + panel.y,
                                width=block_size,
                                height=block_size,
                                anomaly_type="flat_content",
                                severity=severity,
                                description=f"No content at ({bx},{by})",
                            )
                        )

        # Merge overlapping flat areas
        return self._merge_flat_anomalies(anomalies, panel)

    def _merge_flat_anomalies(
        self,
        anomalies: List[LEDAnomaly],
        panel: LEDPanelInfo,
    ) -> List[LEDAnomaly]:
        """Merge flat anomalies yang overlap untuk mengurangi noise.

        Args:
            anomalies: List flat anomalies.
            panel: Info panel LED.

        Returns:
            List merged anomalies.
        """
        if not anomalies:
            return []

        # Group by proximity dan merge
        merged: List[LEDAnomaly] = []
        used = [False] * len(anomalies)
        block_size = 64  # Same as in _detect_flat_content

        for i, a1 in enumerate(anomalies):
            if used[i]:
                continue

            # Cari anomaly yang berdekatan
            group = [a1]
            used[i] = True

            for j, a2 in enumerate(anomalies):
                if used[j]:
                    continue
                # Cek apakah berdekatan (dalam 1 block size)
                if (
                    abs(a1.x - a2.x) < block_size * 2
                    and abs(a1.y - a2.y) < block_size * 2
                ):
                    group.append(a2)
                    used[j] = True

            # Merge group
            if len(group) >= 3:  # Minimal 3 blocks untuk dianggap anomali
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
        """Deteksi line defects (garis horizontal/vertical).

        Metode yang lebih akurat:
        1. Gunakan profiling untuk menemukan garis
        2. Filter berdasarkan intensitas dan panjang
        3. Hanya deteksi di dalam LED content mask

        Args:
            gray: Gambar grayscale area LED.
            panel: Info panel LED.
            led_mask: Mask area LED content.

        Returns:
            List line defect anomalies.
        """
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape

        # Hitung profil brightness HANYA di area LED
        masked_gray = gray.astype(float) * led_mask

        # Hitung row dan column profiles
        row_counts = np.sum(led_mask, axis=1)
        col_counts = np.sum(led_mask, axis=0)

        # Hindari division by zero
        row_counts[row_counts == 0] = 1
        col_counts[col_counts == 0] = 1

        row_means = np.sum(masked_gray, axis=1) / row_counts
        col_means = np.sum(masked_gray, axis=0) / col_counts

        # Hitung statistics dari area LED
        led_row_means = row_means[row_counts > w * 0.3]
        led_col_means = col_means[col_counts > h * 0.3]

        if len(led_row_means) == 0 or len(led_col_means) == 0:
            return []

        # Threshold: garis jika < mean - 1.5*std (lebih sensitif)
        row_threshold = np.mean(led_row_means) - np.std(led_row_means) * 1.5
        col_threshold = np.mean(led_col_means) - np.std(led_col_means) * 1.5

        # Deteksi garis horizontal
        for i in range(2, h - 2):
            # Cek apakah baris ini di area LED (minimal 30% width)
            if row_counts[i] < w * 0.3:
                continue

            # Cek apakah baris ini lebih gelap dari threshold
            if row_means[i] < row_threshold:
                # Cek apakah ini local minimum (garis, bukan area luas)
                if (
                    row_means[i] < row_means[i - 1]
                    and row_means[i] < row_means[i + 1]
                ):
                    # Hitung panjang garis horizontal
                    line_length = 0
                    for j in range(w):
                        if led_mask[i, j] > 0 and gray[i, j] < row_threshold:
                            line_length += 1

                    # Harus cukup panjang (> 30% width)
                    if line_length > w * 0.3:
                        severity = min(
                            (row_threshold - row_means[i]) / (row_threshold + 1),
                            1.0,
                        )
                        anomalies.append(
                            LEDAnomaly(
                                x=panel.x,
                                y=panel.y + i,
                                width=w,
                                height=1,
                                anomaly_type="line_defect",
                                severity=severity,
                                description=f"Horizontal line at row {i} (len={line_length})",
                            )
                        )

        # Deteksi garis vertical
        for i in range(2, w - 2):
            # Cek apakah kolom ini di area LED (minimal 30% height)
            if col_counts[i] < h * 0.3:
                continue

            # Cek apakah kolom ini lebih gelap dari threshold
            if col_means[i] < col_threshold:
                # Cek apakah ini local minimum
                if (
                    col_means[i] < col_means[i - 1]
                    and col_means[i] < col_means[i + 1]
                ):
                    # Hitung panjang garis vertical
                    line_length = 0
                    for j in range(h):
                        if led_mask[j, i] > 0 and gray[j, i] < col_threshold:
                            line_length += 1

                    # Harus cukup panjang (> 30% height)
                    if line_length > h * 0.3:
                        severity = min(
                            (col_threshold - col_means[i]) / (col_threshold + 1),
                            1.0,
                        )
                        anomalies.append(
                            LEDAnomaly(
                                x=panel.x + i,
                                y=panel.y,
                                width=1,
                                height=h,
                                anomaly_type="line_defect",
                                severity=severity,
                                description=f"Vertical line at col {i} (len={line_length})",
                            )
                        )

        return anomalies

    def _detect_dead_blocks(
        self,
        gray: np.ndarray,
        panel: LEDPanelInfo,
    ) -> List[LEDAnomaly]:
        """Deteksi dead pixel blocks.

        Args:
            gray: Gambar grayscale area LED.
            panel: Info panel LED.

        Returns:
            List dead block anomalies.
        """
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape

        # Grid analysis: bagi menjadi blok-blok kecil
        block_size = 32
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block = gray[by : by + block_size, bx : bx + block_size]
                block_mean = np.mean(block)

                # Bandingkan dengan rata-rata panel
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
        """Deteksi dead pixel blocks HANYA di dalam LED content mask.

        Args:
            gray: Gambar grayscale area LED.
            panel: Info panel LED.
            led_mask: Mask area LED content.

        Returns:
            List dead block anomalies.
        """
        anomalies: List[LEDAnomaly] = []
        h, w = gray.shape

        # Grid analysis: bagi menjadi blok-blok kecil
        block_size = 32
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                # Cek apakah block ini di dalam LED content
                block_mask = led_mask[by : by + block_size, bx : bx + block_size]
                led_ratio = np.mean(block_mask)

                # Hanya analisis jika > 70% block adalah LED content
                if led_ratio < 0.7:
                    continue

                block = gray[by : by + block_size, bx : bx + block_size]
                block_mean = np.mean(block)

                # Bandingkan dengan rata-rata panel
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
        """Deteksi color errors.

        Args:
            hsv: Gambar HSV area LED.
            panel: Info panel LED.

        Returns:
            List color error anomalies.
        """
        anomalies: List[LEDAnomaly] = []
        h, w = hsv.shape[:2]

        # Analisis hue distribution
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]

        # Grid analysis untuk warna
        block_size = 64

        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block_hue = hue[by : by + block_size, bx : bx + block_size]
                block_sat = sat[by : by + block_size, bx : bx + block_size]
                hue_mean = np.mean(block_hue)
                sat_mean = np.mean(block_sat)

                # Cek jika saturasi rendah (warna pudar/abu-abu)
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
        """Deteksi color errors HANYA di dalam LED content mask.

        Args:
            hsv: Gambar HSV area LED.
            panel: Info panel LED.
            led_mask: Mask area LED content.

        Returns:
            List color error anomalies.
        """
        anomalies: List[LEDAnomaly] = []
        h, w = hsv.shape[:2]

        # Analisis hue distribution
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]

        # Grid analysis untuk warna
        block_size = 64

        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                # Cek apakah block ini di dalam LED content
                block_mask = led_mask[by : by + block_size, bx : bx + block_size]
                led_ratio = np.mean(block_mask)

                # Hanya analisis jika > 70% block adalah LED content
                if led_ratio < 0.7:
                    continue

                block_hue = hue[by : by + block_size, bx : bx + block_size]
                block_sat = sat[by : by + block_size, bx : bx + block_size]
                hue_mean = np.mean(block_hue)
                sat_mean = np.mean(block_sat)

                # Cek jika saturasi rendah (warna pudar/abu-abu)
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

    def _filter_anomalies_in_panel(
        self,
        anomalies: List[LEDAnomaly],
        panel: LEDPanelInfo,
    ) -> List[LEDAnomaly]:
        """Filter anomali — hanya yang benar-benar di dalam LED content.

        Metode filtering:
        1. Pastikan anomali di dalam bounding box panel
        2. Hitung persentase overlap dengan area LED
        3. Hanya pertahankan anomali yang significant

        Args:
            anomalies: List anomali.
            panel: Info panel LED.

        Returns:
            List anomali yang sudah difilter.
        """
        filtered: List[LEDAnomaly] = []
        panel_area = panel.width * panel.height

        for a in anomalies:
            # 1. Pastikan anomali di dalam bounding box panel
            a_center_x = a.x + a.width // 2
            a_center_y = a.y + a.height // 2

            # Cek apakah center anomali di dalam panel
            if not (
                panel.x <= a_center_x <= panel.x + panel.width
                and panel.y <= a_center_y <= panel.y + panel.height
            ):
                continue

            # 2. Hitung area anomali
            anomaly_area = a.width * a.height

            # 3. Filter berdasarkan ukuran relatif terhadap panel
            # Anomali terlalu kecil (< 0.1% panel) = noise
            if anomaly_area < panel_area * 0.001:
                continue

            # 4. Filter anomali yang terlalu besar (> 80% panel)
            # Kemungkinan false positive
            if anomaly_area > panel_area * 0.8:
                continue

            filtered.append(a)

        return filtered

    def _calculate_score(
        self,
        anomalies: List[LEDAnomaly],
        panel: LEDPanelInfo,
    ) -> float:
        """Hitung skor anomali.

        Args:
            anomalies: List anomali.
            panel: Info panel LED.

        Returns:
            Skor 0.0 - 1.0.
        """
        if not anomalies:
            return 0.0

        # Hitung total area anomali
        total_area = panel.width * panel.height
        anomaly_area = sum(a.width * a.height for a in anomalies)

        # Hitung rata-rata severity
        avg_severity = np.mean([a.severity for a in anomalies])

        # Skor = kombinasi area dan severity
        area_ratio = anomaly_area / total_area if total_area > 0 else 0
        score = (area_ratio * 0.5 + avg_severity * 0.5) * 2

        return min(score, 1.0)

    def _annotate(
        self,
        image: np.ndarray,
        panel: LEDPanelInfo,
        anomalies: List[LEDAnomaly],
    ) -> np.ndarray:
        """Buat visualisasi hasil analisis.

        Args:
            image: Gambar asli.
            panel: Info panel LED.
            anomalies: List anomali.

        Returns:
            Gambar yang sudah dianotasi.
        """
        annotated = image.copy()

        # Gambar panel LED (hijau)
        cv2.rectangle(
            annotated,
            (panel.x, panel.y),
            (panel.x + panel.width, panel.y + panel.height),
            (0, 255, 0),
            2,
        )

        # Tandai panel info
        cv2.putText(
            annotated,
            f"LED Panel ({panel.width}x{panel.height})",
            (panel.x, panel.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )

        # Gambar anomali berdasarkan tipe
        colors = {
            "blocking": (0, 0, 255),  # Merah
            "line_defect": (0, 165, 255),  # Oranye
            "dead_pixel_block": (0, 0, 200),  # Merah gelap
            "color_error": (255, 0, 255),  # Magenta
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
        """Generate pesan deskriptif.

        Args:
            score: Skor anomali.
            anomalies: List anomali.
            panel: Info panel LED.

        Returns:
            Pesan deskriptif.
        """
        if not anomalies:
            return (
                f"Panel LED terdeteksi ({panel.width}x{panel.height}). "
                f"Tidak ada anomali."
            )

        # Hitung per tipe
        blocking = [a for a in anomalies if a.anomaly_type == "blocking"]
        flat = [a for a in anomalies if a.anomaly_type == "flat_content"]
        lines = [a for a in anomalies if a.anomaly_type == "line_defect"]
        dead = [a for a in anomalies if a.anomaly_type == "dead_pixel_block"]
        color = [a for a in anomalies if a.anomaly_type == "color_error"]

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

        return (
            f"Panel LED terdeteksi ({panel.width}x{panel.height}). "
            f"Anomali: {', '.join(parts)}. "
            f"Score: {score:.2f}"
        )
