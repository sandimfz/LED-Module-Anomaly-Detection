"""
Base class untuk semua detectors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.core.config import Config
from src.core.types import DetectionResult, LocationConfig


class BaseDetector(ABC):
    """Base class untuk semua anomaly detector.

    Semua detector harus mengimplementasikan method detect().

    Attributes:
        config: Konfigurasi lokasi.
    """

    def __init__(self, config: LocationConfig) -> None:
        """Initialize detector.

        Args:
            config: Konfigurasi lokasi.
        """
        self.config = config

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        image_path: str,
    ) -> DetectionResult:
        """Deteksi anomali pada gambar.

        Args:
            image: Gambar BGR.
            image_path: Path gambar (untuk tracking).

        Returns:
            DetectionResult dengan skor dan level anomali.
        """
        ...

    def get_output_path(
        self,
        image_path: str,
        suffix: str,
        status: str,
    ) -> str:
        """Generate output path berdasarkan status.

        Args:
            image_path: Path gambar asli.
            suffix: Suffix untuk nama file (misal: "grid", "darkspot").
            status: Status deteksi (good, warning, critical).

        Returns:
            Path output lengkap.
        """
        filename = Path(image_path).stem
        output_dir = Config.get_output_path(self.config.name, status)
        return str(output_dir / f"{filename}_{suffix}.jpg")

    def save_annotated(
        self,
        image: np.ndarray,
        output_path: str,
    ) -> str:
        """Simpan gambar annotated/overlay.

        Args:
            image: Gambar yang sudah dianotasi.
            output_path: Path output.

        Returns:
            Path file yang tersimpan.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
        return output_path

    def _calculate_anomaly_score(
        self,
        total_cells: int,
        flagged_cells: int,
    ) -> float:
        """Hitung skor anomali berdasarkan jumlah cell terdeteksi.

        Args:
            total_cells: Total cell.
            flagged_cells: Jumlah cell terdeteksi anomali.

        Returns:
            Skor anomali 0.0 - 1.0.
        """
        if total_cells == 0:
            return 0.0
        ratio = flagged_cells / total_cells
        # Linear scale: 50% cells flagged → score 0.5
        # Was ratio * 3.0 which meant 33% flagged = score 1.0 (too aggressive)
        return min(ratio, 1.0)

    def _determine_level(self, score: float) -> str:
        """Tentukan level anomali berdasarkan skor.

        Args:
            score: Skor anomali (0.0 - 1.0).

        Returns:
            Level: "normal", "warning", atau "critical".
        """
        if score >= 0.7:
            return "critical"
        elif score >= 0.4:
            return "warning"
        return "normal"
