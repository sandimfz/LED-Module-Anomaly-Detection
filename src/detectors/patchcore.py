"""
PatchCore Anomaly Detector.

Wrapper untuk Anomalib PatchCore.
Memerlukan training terlebih dahulu.

Kelebihan:
- Bisa generalisasi ke jenis kerusakan baru
- Menghasilkan heatmap otomatis

Kekurangan:
- Perlu training per lokasi
- Threshold perlu kalibrasi dengan data bad
"""

import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.core.config import Config
from src.core.types import (
    AnomalyLevel,
    DetectionResult,
    LocationConfig,
)
from src.core.utils import create_heatmap_overlay
from src.detectors.base import BaseDetector

warnings.filterwarnings("ignore")


class PatchCoreDetector(BaseDetector):
    """Detector menggunakan Anomalib PatchCore.

    Attributes:
        config: Konfigurasi lokasi.
        model: Model PatchCore (lazy loaded).
        engine: Engine Anomalib (lazy loaded).
    """

    def __init__(self, config: LocationConfig) -> None:
        """Initialize PatchCore detector.

        Args:
            config: Konfigurasi lokasi.
        """
        super().__init__(config)
        self._model = None
        self._engine = None

    @property
    def is_loaded(self) -> bool:
        """Cek apakah model sudah diload."""
        return self._model is not None

    def load_model(self) -> None:
        """Load model dari checkpoint.

        Raises:
            FileNotFoundError: Jika checkpoint tidak ditemukan.
        """
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        search_dir = Config.get_model_path(self.config.name)
        checkpoint_path = self._find_checkpoint(search_dir)

        self._model = Patchcore.load_from_checkpoint(
            str(checkpoint_path),
            weights_only=False,
        )
        self._engine = Engine(accelerator="cpu", devices=1)

    def _find_checkpoint(self, search_dir: Path) -> Path:
        """Cari checkpoint terbaru.

        Args:
            search_dir: Directory pencarian.

        Returns:
            Path ke checkpoint.

        Raises:
            FileNotFoundError: Jika tidak ada checkpoint.
        """
        ckpts = sorted(
            search_dir.rglob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not ckpts:
            raise FileNotFoundError(
                f"Tidak ada checkpoint di {search_dir}. "
                "Jalankan train.py dulu."
            )
        return ckpts[-1]

    def detect(
        self,
        image: np.ndarray,
        image_path: str,
    ) -> DetectionResult:
        """Deteksi anomali menggunakan PatchCore.

        Args:
            image: Gambar BGR.
            image_path: Path gambar.

        Returns:
            DetectionResult.
        """
        if not self.is_loaded:
            self.load_model()

        from anomalib.data import PredictDataset

        dataset = PredictDataset(path=image_path)
        predictions = self._engine.predict(
            model=self._model,
            dataset=dataset,
        )

        pred = predictions[0]
        score = float(pred.pred_score)
        anomaly_map = pred.anomaly_map.squeeze().cpu().numpy()

        # Generate heatmap overlay
        overlay = create_heatmap_overlay(image, anomaly_map)
        status = self._determine_level(score)
        output_path = self.get_output_path(image_path, "patchcore", status)
        self.save_annotated(overlay, output_path)

        return DetectionResult(
            location=self.config.name,
            image_path=image_path,
            anomaly_score=round(score, 4),
            level=AnomalyLevel(status),
            message=self._generate_message(score),
            heatmap_path=output_path,
        )

    def _generate_message(self, score: float) -> str:
        """Generate pesan deskriptif.

        Args:
            score: Skor anomali.

        Returns:
            Pesan deskriptif.
        """
        if score < 0.4:
            return "PatchCore: Tidak ada anomali terdeteksi."
        elif score < 0.7:
            return f"PatchCore: Kemungkinan anomali. Score: {score:.2f}"
        return f"PatchCore: Anomali terdeteksi. Score: {score:.2f}"
