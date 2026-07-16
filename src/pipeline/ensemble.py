"""
Ensemble Pipeline.

Menggabungkan hasil dari multiple detectors untuk
keputusan yang lebih robust.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
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

    def _crop_to_screen(self, image: np.ndarray) -> np.ndarray:
        """Crop gambar ke area screen LED menggunakan perspective transform.

        Menyimpan inverse matrix untuk mapping balik koordinat anomali
        ke gambar asli.

        Args:
            image: Gambar BGR asli.

        Returns:
            Gambar yang sudah di-crop ke area screen LED, atau gambar asli.
        """
        self._M_inv = None
        self._crop_params = None

        if self.config.screen_points is None:
            return image

        h, w = image.shape[:2]
        current_res = f"{w}x{h}"

        # Cari screen_points yang sesuai resolusi:
        # 1. screen_points_map (kalibrasi manual per resolusi)
        # 2. screen_points langsung (jika resolusi cocok)
        # 3. Linear scaling sebagai fallback
        pts_src = None
        if self.config.screen_points_map and current_res in self.config.screen_points_map:
            pts_src = np.array(self.config.screen_points_map[current_res], dtype="float32")
        elif self.config.screen_resolution == current_res:
            pts_src = np.array(self.config.screen_points, dtype="float32")
        elif self.config.screen_points:
            pts_src = np.array(self.config.screen_points, dtype="float32")
            cfg_w, cfg_h = map(int, self.config.screen_resolution.split("x"))
            pts_src[:, 0] *= w / cfg_w
            pts_src[:, 1] *= h / cfg_h
        tl, tr, br, bl = pts_src

        max_width = int(max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl),
        ))
        max_height = int(max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl),
        ))

        pts_dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        self._M_inv = cv2.getPerspectiveTransform(pts_dst, pts_src)
        self._crop_params = (max_width, max_height)
        return cv2.warpPerspective(image, M, (max_width, max_height))

    def _point_to_original(self, cx: int, cy: int) -> tuple:
        """Map titik dari cropped space ke original image space.

        Args:
            cx, cy: Koordinat di cropped image.

        Returns:
            (ox, oy) koordinat di original image.
        """
        if self._M_inv is None:
            return (cx, cy)
        pts = np.array([[[cx, cy]]], dtype=np.float32)
        orig = cv2.perspectiveTransform(pts, self._M_inv)
        return (int(orig[0][0][0]), int(orig[0][0][1]))

    def _rect_to_original(self, x: int, y: int, w: int, h: int) -> tuple:
        """Map bounding box dari cropped ke original space.

        Transform semua 4 sudut, lalu ambil bounding rect.

        Args:
            x, y, w, h: Bounding box di cropped space.

        Returns:
            (ox, oy, ow, oh) di original space.
        """
        if self._M_inv is None:
            return (x, y, w, h)
        corners = np.array([
            [[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]
        ], dtype=np.float32)
        orig_corners = cv2.perspectiveTransform(corners, self._M_inv)
        ox = int(min(c[0][0] for c in orig_corners))
        oy = int(min(c[0][1] for c in orig_corners))
        ox2 = int(max(c[0][0] for c in orig_corners))
        oy2 = int(max(c[0][1] for c in orig_corners))
        return (ox, oy, ox2 - ox, oy2 - oy)

    def _draw_screen_border(self, image: np.ndarray) -> None:
        """Gambar border hijau di sekeliling area screen LED.

        Args:
            image: Gambar original (dimodifikasi in-place).
        """
        if self._M_inv is None or self._crop_params is None:
            return
        crop_w, crop_h = self._crop_params
        corners_src = np.array([
            [[0, 0]], [[crop_w - 1, 0]],
            [[crop_w - 1, crop_h - 1]], [[0, crop_h - 1]]
        ], dtype=np.float32)
        orig = cv2.perspectiveTransform(corners_src, self._M_inv)
        pts = np.array([[int(p[0][0]), int(p[0][1])] for p in orig], dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    def analyze_single(
        self,
        image_path: str,
    ) -> Dict[str, DetectionResult]:
        """Analisis satu gambar dengan semua detector.

        Deteksi dilakukan di area crop (screen LED).
        Output annotasi digambar di gambar FULL (original),
        agar konteks sekitar tetap terlihat.

        Args:
            image_path: Path ke gambar.

        Returns:
            Dict dengan key nama detector, value DetectionResult.
        """
        from src.core.utils import load_image

        image = load_image(image_path)
        original = image.copy()
        image = self._crop_to_screen(image)
        results: Dict[str, DetectionResult] = {}

        # LED Analyzer (paling akurat) — jalan bareng grid & dark_spot
        if hasattr(self, "use_led_analyzer") and self.use_led_analyzer:
            results["led_analyzer"] = self._led_analyzer.detect(
                image, image_path
            )

        # Grid & DarkSpot jalan selalu
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

        # Re-annotate on full original image jika crop dilakukan
        if self._M_inv is not None:
            self._annotate_results_on_original(
                original, image, image_path, results
            )

        return results

    def _reannotate_led_analyzer(
        self,
        original: np.ndarray,
        cropped: np.ndarray,
        image_path: str,
        result: DetectionResult,
    ) -> None:
        """Re-annotate LED Analyzer results on full original image.

        Args:
            original: Full original image.
            cropped: Cropped screen image.
            image_path: Path gambar asli.
            result: DetectionResult dari LED Analyzer.
        """
        # Cari file annotasi yang sudah disimpan detector
        import re
        from pathlib import Path

        # Baca anomalies dari result message — sayangnya tidak available langsung.
        # Kita perlu akses internal LED Analyzer, tapi terlalu dalam.
        # Alternatif: baca annotated image yang sudah disimpan, warp balik.
        
        # Untuk LED Analyzer, output_nya di heatmap_path
        if not result.heatmap_path or not Path(result.heatmap_path).exists():
            return

        # Baca annotated cropped image
        annotated_cropped = cv2.imread(result.heatmap_path)
        if annotated_cropped is None:
            return

        # Warp balik ke original space
        h_orig, w_orig = original.shape[:2]
        full_annotated = cv2.warpPerspective(
            annotated_cropped, self._M_inv, (w_orig, h_orig),
            borderMode=cv2.BORDER_TRANSPARENT
        )

        # Composite: original bg + annotated overlay
        # Buat mask dari area non-hitam di full_annotated
        mask = np.any(full_annotated > 0, axis=2).astype(np.uint8) * 255
        if mask.sum() > 0:
            overlay = original.copy()
            # Copy annotated pixels where mask is active
            overlay[mask > 0] = full_annotated[mask > 0]
            # Where annotated has content, blend
            alpha = 0.7
            blended = cv2.addWeighted(original, 1 - alpha, full_annotated, alpha, 0)
            # But keep original where no annotation
            blended[mask == 0] = original[mask == 0]
            full_annotated = blended

        # Gambar border screen
        self._draw_screen_border(full_annotated)

        # Simpan (overwrite file yang cropped)
        cv2.imwrite(result.heatmap_path, full_annotated)

    def _reannotate_grid(
        self,
        original: np.ndarray,
        cropped: np.ndarray,
        image_path: str,
        result: DetectionResult,
    ) -> None:
        """Re-annotate Grid results on full original image.

        Args:
            original: Full original image.
            cropped: Cropped screen image.
            image_path: Path gambar asli.
            result: DetectionResult dari Grid detector.
        """
        if not result.heatmap_path or not Path(result.heatmap_path).exists():
            return

        annotated_cropped = cv2.imread(result.heatmap_path)
        if annotated_cropped is None:
            return

        h_orig, w_orig = original.shape[:2]
        full_annotated = cv2.warpPerspective(
            annotated_cropped, self._M_inv, (w_orig, h_orig),
            borderMode=cv2.BORDER_TRANSPARENT
        )

        # Composite
        mask = np.any(full_annotated > 0, axis=2).astype(np.uint8) * 255
        if mask.sum() > 0:
            alpha = 0.7
            blended = cv2.addWeighted(
                original, 1 - alpha, full_annotated, alpha, 0
            )
            blended[mask == 0] = original[mask == 0]
            full_annotated = blended

        self._draw_screen_border(full_annotated)
        cv2.imwrite(result.heatmap_path, full_annotated)

    def _reannotate_dark_spot(
        self,
        original: np.ndarray,
        cropped: np.ndarray,
        image_path: str,
        result: DetectionResult,
    ) -> None:
        """Re-annotate DarkSpot results on full original image."""
        if not result.heatmap_path or not Path(result.heatmap_path).exists():
            return

        annotated_cropped = cv2.imread(result.heatmap_path)
        if annotated_cropped is None:
            return

        h_orig, w_orig = original.shape[:2]
        full_annotated = cv2.warpPerspective(
            annotated_cropped, self._M_inv, (w_orig, h_orig),
            borderMode=cv2.BORDER_TRANSPARENT
        )

        mask = np.any(full_annotated > 0, axis=2).astype(np.uint8) * 255
        if mask.sum() > 0:
            alpha = 0.7
            blended = cv2.addWeighted(
                original, 1 - alpha, full_annotated, alpha, 0
            )
            blended[mask == 0] = original[mask == 0]
            full_annotated = blended

        self._draw_screen_border(full_annotated)
        cv2.imwrite(result.heatmap_path, full_annotated)

    def _annotate_results_on_original(
        self,
        original: np.ndarray,
        cropped: np.ndarray,
        image_path: str,
        results: Dict[str, DetectionResult],
    ) -> None:
        """Re-annotate semua detector results ke gambar original (full).

        Args:
            original: Gambar asli (full).
            cropped: Gambar yang sudah di-crop.
            image_path: Path gambar asli.
            results: Dict hasil deteksi.
        """
        from pathlib import Path

        for name in ["led_analyzer", "grid", "dark_spot"]:
            result = results.get(name)
            if result is None:
                continue
            if name == "led_analyzer":
                self._reannotate_led_analyzer(
                    original, cropped, image_path, result
                )
            elif name == "grid":
                self._reannotate_grid(
                    original, cropped, image_path, result
                )
            elif name == "dark_spot":
                self._reannotate_dark_spot(
                    original, cropped, image_path, result
                )

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

        # Crop semua frame ke area screen terlebih dahulu
        frames = [self._crop_to_screen(f) for f in frames]

        # LED Analyzer: analisis frame terakhir (paling akurat)
        if hasattr(self, "use_led_analyzer") and self.use_led_analyzer and frames:
            results["led_analyzer"] = self._led_analyzer.detect(
                frames[-1], frame_paths[-1]
            )

        # Grid detector: analisis frame terakhir
        if self.use_grid and frames:
            results["grid"] = self.grid_detector.detect(
                frames[-1], frame_paths[-1]
            )

        # DarkSpot detector: analisis frame terakhir
        if self.use_dark_spot and frames:
            results["dark_spot"] = self.dark_spot_detector.detect(
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

        # Weighted average — LED Analyzer dominan (0.6) karena paling akurat.
        # Grid & dark_spot sebagai pelengkap untuk catching hal yg terlewat.
        weights = {
            "led_analyzer": 0.6,
            "dark_spot": 0.15,
            "grid": 0.1,
            "temporal": 0.1,
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
            all_flagged.extend(result.flagged_cells)
            messages.append(f"[{name.upper()}] {result.message}")
            if result.heatmap_path:
                heatmap_path = result.heatmap_path
            # Track max score and critical status
            if result.anomaly_score > max_score:
                max_score = result.anomaly_score
            if result.level == AnomalyLevel.CRITICAL:
                has_critical = True

            # Skip detectors with score ~0 agar tidak mengencerkan hasil.
            # Detector yang tidak mendeteksi apapun tidak boleh
            # menurunkan kontribusi detector lain yang berhasil.
            if result.anomaly_score < 0.01:
                continue

            weighted_score += result.anomaly_score * weight
            total_weight += weight

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Max-score strategy: jika ada detector yang score tinggi,
        # pastikan final score tidak terlalu rendah.
        # Ini mencegah kasus di mana 1 detector detect tapi lainnya miss.
        if max_score > 0.5:
            final_score = max(final_score, max_score * 0.75)

        # Jika max dari LED Analyzer tinggi, pastikan final tidak terlalu rendah.
        la_score = results.get("led_analyzer", None)
        if la_score and la_score.anomaly_score > 0.35:
            final_score = max(final_score, la_score.anomaly_score * 0.80)

        # Determine level — turunkan threshold untuk lebih sensitif
        if final_score >= 0.55:
            final_level = AnomalyLevel.CRITICAL
        elif final_score >= 0.30:
            final_level = AnomalyLevel.WARNING
        else:
            final_level = AnomalyLevel.NORMAL

        # Deduplicate flagged cells
        unique_flagged = list(set(all_flagged))

        # Pastikan final level tidak lebih rendah dari level detector
        # dengan deteksi tertinggi.
        if has_critical and final_level != AnomalyLevel.CRITICAL:
            if final_score < 0.55:
                final_score = max(final_score, 0.55)
            final_level = AnomalyLevel.CRITICAL

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
