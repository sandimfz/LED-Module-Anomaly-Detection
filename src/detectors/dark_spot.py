"""
Dark Spot Detector.

Mendeteksi area gelap kecil (dead pixel block / modul mati)
di dalam area LED yang aktif.

Approach:
1. Deteksi panel LED menggunakan saturasi (HSV)
2. Di dalam panel LED, cari area gelap yang tidak wajar
3. Hitung rasio dark spot terhadap panel area
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.core.config import Config
from src.core.types import AnomalyLevel, DetectionResult, LocationConfig
from src.detectors.base import BaseDetector


class DarkSpotDetector(BaseDetector):
    """Detector untuk dark spots di area LED.

    Attributes:
        config: Konfigurasi lokasi.
        led_sat_threshold: Threshold saturasi untuk deteksi area LED.
        led_bright_threshold: Threshold brightness untuk area LED.
        dark_ratio: Rasio brightness untuk dark spot (terhadap mean LED).
        min_spot_area: Minimal area dark spot (pixel).
    """

    def __init__(
        self,
        config: LocationConfig,
        led_sat_threshold: float = 50.0,
        led_bright_threshold: float = 80.0,
        dark_ratio: float = 0.25,
        min_spot_area: int = 50,
    ) -> None:
        """Initialize dark spot detector.

        Args:
            config: Konfigurasi lokasi.
            led_sat_threshold: Threshold saturasi untuk area LED.
            led_bright_threshold: Threshold brightness untuk area LED.
            dark_ratio: Rasio brightness untuk dark spot.
            min_spot_area: Minimal luas dark spot.
        """
        super().__init__(config)
        self.led_sat_threshold = led_sat_threshold
        self.led_bright_threshold = led_bright_threshold
        self.dark_ratio = dark_ratio
        self.min_spot_area = min_spot_area

    def detect(
        self,
        image: np.ndarray,
        image_path: str,
    ) -> DetectionResult:
        """Deteksi dark spots di area LED.

        Args:
            image: Gambar BGR.
            image_path: Path gambar.

        Returns:
            DetectionResult.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape

        # Step 1: Deteksi area LED menggunakan saturasi + brightness
        sat = hsv[:, :, 1]
        bright_mask = (gray > self.led_bright_threshold) & (
            sat > self.led_sat_threshold
        )

        # Step 2: Dilate untuk connect area LED yang terputus
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bright_mask_dilated = cv2.dilate(
            bright_mask.astype(np.uint8), kernel, iterations=2
        )

        # Step 3: Cari kontur terbesar (panel LED)
        contours, _ = cv2.findContours(
            bright_mask_dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return self._make_result(image_path, 0.0, [])

        largest = max(contours, key=cv2.contourArea)
        lx, ly, lw, lh = cv2.boundingRect(largest)

        # Step 4: Crop area LED
        led_panel = gray[ly : ly + lh, lx : lx + lw]
        led_mean = float(np.mean(led_panel))

        # Step 5: Cari dark spots di dalam LED panel
        dark_threshold = led_mean * self.dark_ratio
        dark_in_panel = led_panel < dark_threshold

        # Step 6: Cleanup morphological
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark_uint8 = dark_in_panel.astype(np.uint8) * 255
        dark_uint8 = cv2.morphologyEx(dark_uint8, cv2.MORPH_OPEN, kernel2)

        # Step 7: Cari contours dark spots
        contours2, _ = cv2.findContours(
            dark_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Step 8: Filter berdasarkan ukuran minimum
        dark_spots: List[Tuple[int, int, int, int]] = []
        total_dark_area = 0

        for contour in contours2:
            area = cv2.contourArea(contour)
            if area >= self.min_spot_area:
                sx, sy, sw, sh = cv2.boundingRect(contour)
                # Offset ke koordinat asli
                orig_x, orig_y = sx + lx, sy + ly
                dark_spots.append((orig_x, orig_y, sw, sh))
                total_dark_area += area

        # Step 9: Hitung skor
        led_area = float(lw * lh)
        if led_area > 0:
            score = min(total_dark_area / led_area * 10.0, 1.0)
        else:
            score = 0.0

        # Step 10: Buat visualisasi
        annotated = self._annotate(
            image, dark_spots, (lx, ly, lw, lh), led_mean, dark_threshold
        )
        status = self._determine_level(score)
        output_path = self.get_output_path(image_path, "darkspot", status)
        self.save_annotated(annotated, output_path)

        return self._make_result(
            image_path, score, dark_spots, output_path
        )

    def _annotate(
        self,
        image: np.ndarray,
        dark_spots: List[Tuple[int, int, int, int]],
        led_box: Tuple[int, int, int, int],
        led_mean: float,
        dark_threshold: float,
    ) -> np.ndarray:
        """Buat visualisasi hasil deteksi.

        Args:
            image: Gambar asli.
            dark_spots: List dark spots.
            led_box: Bounding box LED (x, y, w, h).
            led_mean: Rata-rata brightness LED.
            dark_threshold: Threshold untuk dark.

        Returns:
            Gambar yang sudah dianotasi.
        """
        annotated = image.copy()
        lx, ly, lw, lh = led_box

        # Gambar bounding box LED (hijau)
        cv2.rectangle(
            annotated, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2
        )

        # Tandai dark spots (merah)
        for x, y, w, h in dark_spots:
            cv2.rectangle(
                annotated, (x, y), (x + w, y + h), (0, 0, 255), 2
            )
            area = w * h
            cv2.putText(
                annotated,
                f"DARK {area}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # Info text
        cv2.putText(
            annotated,
            f"LED mean: {led_mean:.0f}, Dark thresh: {dark_threshold:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )

        return annotated

    def _make_result(
        self,
        image_path: str,
        score: float,
        dark_spots: List[Tuple[int, int, int, int]],
        heatmap_path: str = "",
    ) -> DetectionResult:
        """Buat DetectionResult.

        Args:
            image_path: Path gambar.
            score: Skor anomali.
            dark_spots: List dark spots.
            heatmap_path: Path heatmap output.

        Returns:
            DetectionResult.
        """
        level = self._determine_level(score)

        if not dark_spots:
            message = "Tidak ada dark spot terdeteksi di area LED."
        else:
            big_spots = [s for s in dark_spots if s[2] * s[3] > 500]
            message = (
                f"Terdeteksi {len(dark_spots)} dark spot "
                f"({len(big_spots)} besar) di area LED. "
                f"Kemungkinan modul mati/dead pixel block."
            )

        return DetectionResult(
            location=self.config.name,
            image_path=image_path,
            anomaly_score=round(score, 4),
            level=AnomalyLevel(level),
            message=message,
            flagged_cells=[],
            heatmap_path=heatmap_path if heatmap_path else None,
        )
