"""
Temporal Consistency Detector.

Mendeteksi modul LED yang tidak berubah seiring waktu.
Cocok untuk videotron dengan konten yang selalu berganti.

Logika:
- Modul SEHAT pasti BERUBAH karena iklan berganti
- Modul RUSAK akan TIDAK BERUBAH (stuck/mati)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.config import Config
from src.core.types import (
    AnomalyLevel,
    DetectionResult,
    LocationConfig,
    TemporalCellResult,
)
from src.core.utils import (
    calculate_activity_score,
    calculate_robust_z,
    load_frames,
)
from src.detectors.base import BaseDetector


class TemporalDetector(BaseDetector):
    """Detector berdasarkan konsistensi temporal.

    Mendeteksi modul yang tidak berubah seiring waktu
    dengan analisis activity score dan robust z-score.

    Attributes:
        config: Konfigurasi lokasi.
        frames: List frame yang sudah diload.
    """

    def __init__(
        self,
        config: LocationConfig,
        frames: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Initialize temporal detector.

        Args:
            config: Konfigurasi lokasi.
            frames: List frame (opsional, bisa load nanti).
        """
        super().__init__(config)
        self.frames = frames or []

    def load_from_folder(self, folder: str) -> None:
        """Load frame dari folder.

        Args:
            folder: Path ke folder screenshot berurutan.

        Raises:
            ValueError: Jika frame kurang dari min_frames.
        """
        self.frames = load_frames(folder)
        if len(self.frames) < self.config.min_frames:
            raise ValueError(
                f"Butuh minimal {self.config.min_frames} frame, "
                f"cum ada {len(self.frames)}"
            )

    def detect(
        self,
        image: np.ndarray,
        image_path: str,
    ) -> DetectionResult:
        """Deteksi anomali menggunakan multiple frames.

        Args:
            image: Frame terakhir (untuk visualisasi).
            image_path: Path gambar.

        Returns:
            DetectionResult.
        """
        if not self.frames:
            self.frames = [image]

        cell_results = self._analyze_temporal()
        flagged = [c for c in cell_results if c.is_frozen]

        score = self._calculate_anomaly_score(
            total_cells=self.config.grid_rows * self.config.grid_cols,
            flagged_cells=len(flagged),
        )

        # Generate heatmap
        heatmap = self._create_heatmap(cell_results)
        overlay = self._create_overlay(image, heatmap)

        # Save
        status = self._determine_level(score)
        output_path = self.get_output_path(image_path, "temporal", status)
        self.save_annotated(overlay, output_path)

        return DetectionResult(
            location=self.config.name,
            image_path=image_path,
            anomaly_score=round(score, 4),
            level=AnomalyLevel(status),
            message=self._generate_message(score, flagged),
            flagged_cells=[(c.row, c.col) for c in flagged],
            heatmap_path=output_path,
        )

    def _analyze_temporal(self) -> List[TemporalCellResult]:
        """Analisis temporal semua cell.

        Returns:
            List TemporalCellResult.
        """
        gray_frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self.frames
        ]
        h_total, w_total = gray_frames[0].shape[:2]
        cell_h = h_total // self.config.grid_rows
        cell_w = w_total // self.config.grid_cols

        raw_cells: List[Dict] = []
        activities: List[float] = []

        for row in range(self.config.grid_rows):
            for col in range(self.config.grid_cols):
                y0, y1 = row * cell_h, (row + 1) * cell_h
                x0, x1 = col * cell_w, (col + 1) * cell_w

                per_frame_means = [
                    float(np.mean(g[y0:y1, x0:x1]))
                    for g in gray_frames
                ]

                activity = calculate_activity_score(per_frame_means)
                mean_brightness = float(np.mean(per_frame_means))

                raw_cells.append({
                    "row": row,
                    "col": col,
                    "x": x0,
                    "y": y0,
                    "w": cell_w,
                    "h": cell_h,
                    "activity": activity,
                    "brightness": mean_brightness,
                })
                activities.append(activity)

        # Hitung robust z-score
        activities_arr = np.array(activities)
        median_activity = float(np.median(activities_arr))
        mad = float(
            np.median(np.abs(activities_arr - median_activity))
        ) or 1e-6

        cell_results: List[TemporalCellResult] = []
        for cell in raw_cells:
            robust_z = calculate_robust_z(
                cell["activity"],
                median_activity,
                mad,
            )
            is_frozen = robust_z <= self.config.frozen_threshold
            likely_dead = (
                is_frozen
                and cell["brightness"] <= self.config.dead_brightness_threshold
            )

            cell_results.append(TemporalCellResult(
                row=cell["row"],
                col=cell["col"],
                x=cell["x"],
                y=cell["y"],
                w=cell["w"],
                h=cell["h"],
                activity_score=round(cell["activity"], 3),
                mean_brightness=round(cell["brightness"], 1),
                robust_z=round(robust_z, 2),
                is_frozen=is_frozen,
                likely_dead_module=likely_dead,
            ))

        return cell_results

    def _create_heatmap(
        self,
        cells: List[TemporalCellResult],
    ) -> np.ndarray:
        """Buat heatmap dari cell results.

        Args:
            cells: List hasil analisis cell.

        Returns:
            Heatmap sebagai numpy array (0-1 float).
        """
        heatmap = np.zeros((
            self.config.grid_rows,
            self.config.grid_cols,
        ), dtype=np.float32)

        for cell in cells:
            if cell.likely_dead_module:
                heatmap[cell.row][cell.col] = 1.0
            elif cell.is_frozen:
                heatmap[cell.row][cell.col] = 0.6
            else:
                # Activity score di-map ke heatmap
                # Aktivitas rendah = area mencurigakan
                activity_norm = min(cell.activity_score / 5.0, 1.0)
                heatmap[cell.row][cell.col] = 1.0 - activity_norm

        return heatmap

    def _create_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
    ) -> np.ndarray:
        """Buat overlay gambar dengan heatmap.

        Args:
            image: Gambar asli.
            heatmap: Heatmap.

        Returns:
            Gambar dengan overlay.
        """
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    def _generate_message(
        self,
        score: float,
        flagged: List[TemporalCellResult],
    ) -> str:
        """Generate pesan deskriptif.

        Args:
            score: Skor anomali.
            flagged: List cell yang terdeteksi.

        Returns:
            Pesan deskriptif.
        """
        if not flagged:
            return "Tidak ada anomali terdeteksi. Semua modul aktif berubah."

        dead = [c for c in flagged if c.likely_dead_module]
        frozen = [c for c in flagged if not c.likely_dead_module]

        parts = []
        if dead:
            parts.append(f"{len(dead)} modul kemungkinan MATI")
        if frozen:
            parts.append(f"{len(frozen)} modul FROZEN (tidak berubah)")

        return f"Anomali terdeteksi: {', '.join(parts)}. " f"Score: {score:.2f}"
