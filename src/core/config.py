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

    # Auto-detect project root dari lokasi file ini
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATASET_ROOT: Path = BASE_DIR / "dataset"
    MODELS_ROOT: Path = BASE_DIR / "models"
    OUTPUT_ROOT: Path = BASE_DIR / "output"

    # Default configuration untuk lokasi baru
    DEFAULT_LOCATION: LocationConfig = LocationConfig(name="default")

    # Cache session paths supaya satu runtime = satu folder
    _session_paths: Dict[str, Path] = {}

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
                # 4 titik corner LED screen untuk perspective transform
                # Format: TL, TR, BR, BL — untuk resolusi 1280x720
                screen_points=[
                    [327, 237],   # Top-Left
                    [426, 697],   # Top-Right
                    [997, 445],   # Bottom-Right
                    [1071, 172],  # Bottom-Left
                ],
                screen_resolution="1280x720",
            ),
            "paskal": LocationConfig(
                name="paskal",
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
    def get_session_path(cls, location: str) -> Path:
        """Ambil/buat session folder berdasarkan tanggal & waktu.

        Satu runtime = satu folder. Tidak dibuat baru setiap kali dipanggil.

        Struktur: output/<location>/<YYYY-MM-DD>/<HH-MM-SS>/

        Args:
            location: Nama lokasi.

        Returns:
            Path ke session folder.
        """
        if location not in cls._session_paths:
            now = datetime.now()
            date_folder = now.strftime("%Y-%m-%d")
            time_folder = now.strftime("%H-%M-%S")
            path = cls.OUTPUT_ROOT / location / date_folder / time_folder
            path.mkdir(parents=True, exist_ok=True)
            cls._session_paths[location] = path

        return cls._session_paths[location]

    @classmethod
    def get_output_path(
        cls,
        location: str,
        status: Optional[str] = None,
    ) -> Path:
        """Ambil path output untuk lokasi.

        Struktur folder:
            output/<location>/<date>/<time>/good/
            output/<location>/<date>/<time>/warning/
            output/<location>/<date>/<time>/critical/
            output/<location>/<date>/<time>/reports/

        Args:
            location: Nama lokasi.
            status: Status deteksi (good, warning, critical).
                    Jika None, return path base lokasi.

        Returns:
            Path ke folder output.
        """
        session = cls.get_session_path(location)

        if status:
            valid_statuses = ("good", "warning", "critical")
            if status not in valid_statuses:
                status = "good"
            path = session / status
        else:
            path = session

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
        session = cls.get_session_path(location)
        path = session / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def ensure_directories(cls) -> None:
        """Buat semua direktori yang diperlukan."""
        cls.DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        cls.MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
