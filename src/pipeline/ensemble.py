"""
Ensemble Pipeline.

Menggabungkan hasil dari multiple detectors untuk
keputusan yang lebih robust.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.config import Config
from src.core.types import (
    AnomalyLevel,
    DetectionResult,
    LocationConfig,
)
from src.detectors.dark_spot import DarkSpotDetector
from src.detectors.grid import GridDetector
from src.detectors.patchcore import PatchCoreDetector
from src.detectors.temporal import TemporalDetector


class EnsemblePipeline:
    """Pipeline yang menggabungkan multiple detectors.

    Strategi:
    - GridDetector: Cepat, butuh 1 frame
    - DarkSpotDetector: Deteksi modul mati kecil
    - TemporalDetector: Akurat untuk videotron, butuh multiple frames
    - PatchCoreDetector: Generalisasi ke jenis kerusakan baru

    Attributes:
        location: Nama lokasi.
        config: Konfigurasi lokasi.
        use_grid: Gunakan grid detector.
        use_dark_spot: Gunakan dark spot detector.
        use_temporal: Gunakan temporal detector.
        use_patchcore: Gunakan patchcore detector.
    """

    def __init__(
        self,
        location: str,
        use_grid: bool = True,
        use_dark_spot: bool = True,
        use_temporal: bool = True,
        use_patchcore: bool = False,
    ) -> None:
        """Initialize ensemble pipeline.

        Args:
            location: Nama lokasi.
            use_grid: Aktifkan grid detector.
            use_dark_spot: Aktifkan dark spot detector.
            use_temporal: Aktifkan temporal detector.
            use_patchcore: Aktifkan patchcore detector.
        """
        self.location = location
        self.config = Config.get_location_config(location)
        self.use_grid = use_grid
        self.use_dark_spot = use_dark_spot
        self.use_temporal = use_temporal
        self.use_patchcore = use_patchcore

        # Lazy init detectors
        self._grid: Optional[GridDetector] = None
        self._dark_spot: Optional[DarkSpotDetector] = None
        self._temporal: Optional[TemporalDetector] = None
        self._patchcore: Optional[PatchCoreDetector] = None

    @property
    def grid_detector(self) -> GridDetector:
        """Grid detector (lazy init)."""
        if self._grid is None:
            self._grid = GridDetector(self.config)
        return self._grid

    @property
    def dark_spot_detector(self) -> DarkSpotDetector:
        """Dark spot detector (lazy init)."""
        if self._dark_spot is None:
            self._dark_spot = DarkSpotDetector(self.config)
        return self._dark_spot

    @property
    def temporal_detector(self) -> TemporalDetector:
        """Temporal detector (lazy init)."""
        if self._temporal is None:
            self._temporal = TemporalDetector(self.config)
        return self._temporal

    @property
    def patchcore_detector(self) -> PatchCoreDetector:
        """PatchCore detector (lazy init)."""
        if self._patchcore is None:
            self._patchcore = PatchCoreDetector(self.config)
        return self._patchcore

    def analyze_single(
        self,
        image_path: str,
    ) -> Dict[str, DetectionResult]:
        """Analisis satu gambar dengan semua detector.

        Args:
            image_path: Path ke gambar.

        Returns:
            Dict dengan key nama detector, value DetectionResult.
        """
        from src.core.utils import load_image

        image = load_image(image_path)
        results: Dict[str, DetectionResult] = {}

        # LED Analyzer (paling akurat)
        if hasattr(self, "use_led_analyzer") and self.use_led_analyzer:
            results["led_analyzer"] = self._led_analyzer.detect(
                image, image_path
            )
        else:
            # Detektor standar
            if self.use_grid:
                results["grid"] = self.grid_detector.detect(
                    image, image_path
                )

            if self.use_dark_spot:
                results["dark_spot"] = self.dark_spot_detector.detect(
                    image, image_path
                )

        if self.use_patchcore:
            try:
                results["patchcore"] = self.patchcore_detector.detect(
                    image, image_path
                )
            except FileNotFoundError as e:
                results["patchcore"] = DetectionResult(
                    location=self.location,
                    image_path=image_path,
                    anomaly_score=0.0,
                    level=AnomalyLevel.NORMAL,
                    message=f"PatchCore tidak tersedia: {e}",
                )

        return results

    def analyze_video_frames(
        self,
        frames: List[np.ndarray],
        frame_paths: List[str],
    ) -> Dict[str, DetectionResult]:
        """Analisis multiple frames dari video/screenshot.

        Args:
            frames: List gambar BGR.
            frame_paths: List path gambar.

        Returns:
            Dict dengan key nama detector, value DetectionResult.
        """
        results: Dict[str, DetectionResult] = {}

        # Grid detector: analisis frame terakhir
        if self.use_grid and frames:
            results["grid"] = self.grid_detector.detect(
                frames[-1], frame_paths[-1]
            )

        # Temporal detector: analisis semua frames
        if self.use_temporal and len(frames) >= self.config.min_frames:
            self.temporal_detector.frames = frames
            results["temporal"] = self.temporal_detector.detect(
                frames[-1], frame_paths[-1]
            )

        # PatchCore: analisis frame terakhir
        if self.use_patchcore and frames:
            try:
                results["patchcore"] = self.patchcore_detector.detect(
                    frames[-1], frame_paths[-1]
                )
            except FileNotFoundError:
                pass

        return results

    def combine_results(
        self,
        results: Dict[str, DetectionResult],
    ) -> DetectionResult:
        """Gabungkan hasil dari multiple detectors.

        Strategi:
        - Weighted average dengan dark_spot sebagai detector utama
        - dark_spot=0.4, grid=0.3, temporal=0.2, patchcore=0.1

        Args:
            results: Dict hasil dari tiap detector.

        Returns:
            DetectionResult gabungan.
        """
        if not results:
            return DetectionResult(
                location=self.location,
                image_path="",
                anomaly_score=0.0,
                level=AnomalyLevel.NORMAL,
                message="Tidak ada hasil deteksi.",
            )

        # Weighted average - led_analyzer punya bobot tertinggi
        weights = {
            "led_analyzer": 0.5,
            "dark_spot": 0.3,
            "grid": 0.1,
            "temporal": 0.05,
            "patchcore": 0.05,
        }

        weighted_score = 0.0
        total_weight = 0.0
        all_flagged: List[Tuple[int, int]] = []
        messages: List[str] = []
        heatmap_path: Optional[str] = None
        max_score = 0.0
        has_critical = False

        for name, result in results.items():
            weight = weights.get(name, 0.3)
            weighted_score += result.anomaly_score * weight
            total_weight += weight
            all_flagged.extend(result.flagged_cells)
            messages.append(f"[{name.upper()}] {result.message}")
            if result.heatmap_path:
                heatmap_path = result.heatmap_path
            # Track max score and critical status
            if result.anomaly_score > max_score:
                max_score = result.anomaly_score
            if result.level == AnomalyLevel.CRITICAL:
                has_critical = True

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Jika ada CRITICAL dari detector manapun, boost score
        if has_critical:
            final_score = max(final_score, 0.8)
        # Jika max score tinggi, pastikan final score tidak terlalu rendah
        elif max_score > 0.7:
            final_score = max(final_score, max_score * 0.7)

        # Deduplicate flagged cells
        unique_flagged = list(set(all_flagged))

        # Determine level
        if has_critical or final_score >= 0.7:
            final_level = AnomalyLevel.CRITICAL
        elif final_score >= 0.4:
            final_level = AnomalyLevel.WARNING
        else:
            final_level = AnomalyLevel.NORMAL

        return DetectionResult(
            location=self.location,
            image_path=results[list(results.keys())[0]].image_path,
            anomaly_score=round(final_score, 4),
            level=final_level,
            message=" | ".join(messages),
            flagged_cells=unique_flagged,
            heatmap_path=heatmap_path,
        )

    def save_report(
        self,
        results: Dict[str, DetectionResult],
        combined: DetectionResult,
        output_path: Optional[str] = None,
    ) -> str:
        """Simpan laporan JSON.

        Args:
            results: Dict hasil per detector.
            combined: Hasil gabungan.
            output_path: Path output (opsional).

        Returns:
            Path file laporan.
        """
        if output_path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Config.get_report_path(self.location)
            output_path = str(report_dir / f"report_{timestamp}.json")

        report = {
            "location": self.location,
            "combined": {
                "anomaly_score": combined.anomaly_score,
                "level": combined.level.value,
                "message": combined.message,
                "flagged_cells": combined.flagged_cells,
            },
            "detectors": {},
        }

        for name, result in results.items():
            report["detectors"][name] = {
                "anomaly_score": result.anomaly_score,
                "level": result.level.value,
                "message": result.message,
                "flagged_cells": result.flagged_cells,
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return output_path
