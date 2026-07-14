"""
Konfigurasi untuk LED Anomaly Detection System.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.core.types import LocationConfig


class Config:
    """Konfigurasi global sistem.

    Attributes:
        BASE_DIR: Directory root project.
        DATASET_ROOT: Directory dataset.
        MODELS_ROOT: Directory model checkpoints.
        OUTPUT_ROOT: Directory output results.
        LOCATIONS: Konfigurasi per lokasi.
    """

    BASE_DIR: Path = Path("/home/alen/apps/led")
    DATASET_ROOT: Path = BASE_DIR / "dataset"
    MODELS_ROOT: Path = BASE_DIR / "models"
    OUTPUT_ROOT: Path = BASE_DIR / "output"

    # Default configuration untuk lokasi baru
    DEFAULT_LOCATION: LocationConfig = LocationConfig(name="default")

    @classmethod
    def get_location_config(cls, location: str) -> LocationConfig:
        """Ambil konfigurasi untuk lokasi tertentu.

        Args:
            location: Nama lokasi.

        Returns:
            LocationConfig untuk lokasi tersebut.
        """
        configs: Dict[str, LocationConfig] = {
            "sigma_cirebon": LocationConfig(
                name="sigma_cirebon",
                grid_rows=12,
                grid_cols=16,
            ),
            "lengkong": LocationConfig(
                name="lengkong",
                grid_rows=12,
                grid_cols=16,
            ),
        }
        return configs.get(location, cls.DEFAULT_LOCATION)

    @classmethod
    def get_dataset_path(cls, location: str) -> Path:
        """Ambil path dataset untuk lokasi.

        Args:
            location: Nama lokasi.

        Returns:
            Path ke folder dataset lokasi.
        """
        return cls.DATASET_ROOT / location

    @classmethod
    def get_model_path(cls, location: str) -> Path:
        """Ambil path model untuk lokasi.

        Args:
            location: Nama lokasi.

        Returns:
            Path ke folder model lokasi.
        """
        return cls.MODELS_ROOT / location

    @classmethod
    def get_output_path(
        cls,
        location: str,
        status: Optional[str] = None,
    ) -> Path:
        """Ambil path output untuk lokasi.

        Struktur folder:
            output/<location>/good/      <- Gambar normal
            output/<location>/warning/   <- Gambar warning
            output/<location>/critical/  <- Gambar anomali
            output/<location>/reports/   <- JSON reports

        Args:
            location: Nama lokasi.
            status: Status deteksi (good, warning, critical).
                    Jika None, return path base lokasi.

        Returns:
            Path ke folder output.
        """
        if status:
            # Validasi status
            valid_statuses = ("good", "warning", "critical")
            if status not in valid_statuses:
                status = "good"
            path = cls.OUTPUT_ROOT / location / status
        else:
            path = cls.OUTPUT_ROOT / location

        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_report_path(cls, location: str) -> Path:
        """Ambil path untuk reports.

        Args:
            location: Nama lokasi.

        Returns:
            Path ke folder reports.
        """
        path = cls.OUTPUT_ROOT / location / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def ensure_directories(cls) -> None:
        """Buat semua direktori yang diperlukan."""
        cls.DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        cls.MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
