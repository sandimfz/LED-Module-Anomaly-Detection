"""
Grid Neighbor-Contrast Detector.

Mendeteksi anomali dengan membandingkan tiap cell dengan tetangganya.
Tidak memerlukan training, bisa digunakan langsung.

Jenis anomali yang terdeteksi:
- Dark: Lebih gelap dari tetangga (modul mati)
- Flat: Terlalu rata/flatt dari tetangga (stuck color)
- Color: Hue menyimpang dari tetangga (warna salah)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.config import Config
from src.core.types import (
    AnomalyLevel,
    CellResult,
    DetectionResult,
    LocationConfig,
)
from src.core.utils import (
    extract_grid_stats,
    get_neighbor_stats,
    hue_delta,
)
from src.detectors.base import BaseDetector


class GridDetector(BaseDetector):
    """Detector berdasarkan perbandingan dengan tetangga.

    Setiap cell dibandingkan dengan 8 tetangganya.
    Cell yang "menonjol" akan dianggap anomali.

    Attributes:
        config: Konfigurasi lokasi.
    """

    def __init__(self, config: LocationConfig) -> None:
        """Initialize grid detector.

        Args:
            config: Konfigurasi lokasi.
        """
        super().__init__(config)

    def detect(
        self,
        image: np.ndarray,
        image_path: str,
    ) -> DetectionResult:
        """Deteksi anomali menggunakan neighbor contrast.

        Args:
            image: Gambar BGR.
            image_path: Path gambar.

        Returns:
            DetectionResult.
        """
        grid = extract_grid_stats(
            image,
            self.config.grid_rows,
            self.config.grid_cols,
        )

        cell_results, flagged = self._analyze_grid(grid)

        score = self._calculate_anomaly_score(
            total_cells=self.config.grid_rows * self.config.grid_cols,
            flagged_cells=len(flagged),
        )

        # Generate annotated image
        annotated = self._annotate_image(image, cell_results)
        status = self._determine_level(score)
        output_path = self.get_output_path(image_path, "grid", status)
        self.save_annotated(annotated, output_path)

        return DetectionResult(
            location=self.config.name,
            image_path=image_path,
            anomaly_score=round(score, 4),
            level=AnomalyLevel(status),
            message=self._generate_message(score, flagged, cell_results),
            flagged_cells=flagged,
            heatmap_path=output_path,
        )

    def _analyze_grid(
        self,
        grid: List[List[Dict[str, float]]],
    ) -> Tuple[List[CellResult], List[Tuple[int, int]]]:
        """Analisis grid untuk deteksi anomali.

        Args:
            grid: Grid statistik.

        Returns:
            Tuple (list CellResult, list flagged coordinates).
        """
        rows = len(grid)
        cols = len(grid[0]) if grid else 0

        cell_results: List[CellResult] = []
        flagged: List[Tuple[int, int]] = []

        for row in range(rows):
            for col in range(cols):
                stat = grid[row][col]
                neighbor = get_neighbor_stats(grid, row, col, rows, cols)

                if neighbor is None:
                    continue

                is_dark = self._check_dark_anomaly(stat, neighbor)
                is_flat = self._check_flat_anomaly(stat, neighbor)
                is_color = self._check_color_anomaly(stat, neighbor)
                is_anomaly = is_dark or is_flat or is_color

                cell = CellResult(
                    row=row,
                    col=col,
                    x=int(stat["x"]),
                    y=int(stat["y"]),
                    w=int(stat["w"]),
                    h=int(stat["h"]),
                    brightness=round(stat["brightness"], 1),
                    std_dev=round(stat["std"], 1),
                    neighbor_brightness=round(neighbor["brightness"], 1),
                    neighbor_std=round(neighbor["std"], 1),
                    is_dark_anomaly=is_dark,
                    is_flat_anomaly=is_flat,
                    is_color_anomaly=is_color,
                    is_anomaly=is_anomaly,
                )
                cell_results.append(cell)

                if is_anomaly:
                    flagged.append((row, col))

        return cell_results, flagged

    def _check_dark_anomaly(
        self,
        stat: Dict[str, float],
        neighbor: Dict[str, float],
    ) -> bool:
        """Cek apakah cell gelap dari tetangga.

        Args:
            stat: Statistik cell.
            neighbor: Statistik tetangga.

        Returns:
            True jika dark anomaly.
        """
        return (
            neighbor["brightness"] >= self.config.dead_brightness_threshold
            and (
                neighbor["brightness"] - stat["brightness"]
                >= self.config.dark_delta_threshold
            )
        )

    def _check_flat_anomaly(
        self,
        stat: Dict[str, float],
        neighbor: Dict[str, float],
    ) -> bool:
        """Cek apakah cell terlalu rata/flatt.

        Args:
            stat: Statistik cell.
            neighbor: Statistik tetangga.

        Returns:
            True jika flat anomaly.
        """
        return (
            neighbor["std"] > 0
            and neighbor["brightness"] >= self.config.dead_brightness_threshold
            and (stat["std"] / neighbor["std"])
            <= self.config.flat_ratio_threshold
        )

    def _check_color_anomaly(
        self,
        stat: Dict[str, float],
        neighbor: Dict[str, float],
    ) -> bool:
        """Cek apakah warna cell menyimpang.

        Args:
            stat: Statistik cell.
            neighbor: Statistik tetangga.

        Returns:
            True jika color anomaly.
        """
        return (
            neighbor["brightness"] >= self.config.dead_brightness_threshold
            and stat["brightness"] >= self.config.dead_brightness_threshold
            and hue_delta(stat["hue"], neighbor["hue"])
            >= self.config.color_hue_delta_threshold
        )

    def _annotate_image(
        self,
        image: np.ndarray,
        cells: List[CellResult],
    ) -> np.ndarray:
        """Buat gambar annotated dengan bounding box.

        Args:
            image: Gambar asli.
            cells: List hasil analisis cell.

        Returns:
            Gambar yang sudah dianotasi.
        """
        annotated = image.copy()

        for cell in cells:
            if cell.is_anomaly:
                color = (0, 0, 255)  # Merah
                thickness = 3
            else:
                color = (60, 60, 60)  # Abu-abu gelap
                thickness = 1

            cv2.rectangle(
                annotated,
                (cell.x, cell.y),
                (cell.x + cell.w, cell.y + cell.h),
                color,
                thickness,
            )

            if cell.is_anomaly:
                tags = []
                if cell.is_dark_anomaly:
                    tags.append("D")
                if cell.is_flat_anomaly:
                    tags.append("F")
                if cell.is_color_anomaly:
                    tags.append("C")

                cv2.putText(
                    annotated,
                    "".join(tags),
                    (cell.x + 3, cell.y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        return annotated

    def _generate_message(
        self,
        score: float,
        flagged: List[Tuple[int, int]],
        cells: List[CellResult],
    ) -> str:
        """Generate pesan deskriptif.

        Args:
            score: Skor anomali.
            flagged: List cell terdeteksi.
            cells: Semua cell results.

        Returns:
            Pesan deskriptif.
        """
        if not flagged:
            return "Tidak ada anomali terdeteksi."

        dark = sum(1 for c in cells if c.is_dark_anomaly)
        flat = sum(1 for c in cells if c.is_flat_anomaly)
        color = sum(1 for c in cells if c.is_color_anomaly)

        parts = []
        if dark:
            parts.append(f"{dark} dark (gelap)")
        if flat:
            parts.append(f"{flat} flat (rata)")
        if color:
            parts.append(f"{color} color (warna)")

        return (
            f"Anomali terdeteksi: {', '.join(parts)}. "
            f"Total: {len(flagged)} cell. Score: {score:.2f}"
        )
