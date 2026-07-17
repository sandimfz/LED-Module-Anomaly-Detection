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
    """

    # Auto-detect project root dari lokasi file ini
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATASET_ROOT: Path = BASE_DIR / "dataset"
    MODELS_ROOT: Path = BASE_DIR / "models"
    OUTPUT_ROOT: Path = BASE_DIR / "output"

    # Default configuration untuk lokasi baru
    DEFAULT_LOCATION: LocationConfig = LocationConfig(name="default")

    # Cache session path supaya satu runtime = satu folder
    _session_path: Optional[Path] = None

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
                module_rows=12,
                module_cols=16,
                # Format: TL, TR, BR, BL — untuk resolusi 1920x1080
                screen_points=[
                    [624, 102],    # Top-Left
                    [1832, 32],    # Top-Right
                    [1761, 544],   # Bottom-Right
                    [786, 1012],   # Bottom-Left
                ],
                screen_resolution="1920x1080",
            ),
            "lengkong": LocationConfig(
                name="lengkong",
                grid_rows=12,
                grid_cols=16,
                module_rows=12,
                module_cols=16,
                # 4 titik corner LED screen — dikalibrasi ulang 2026-07-17
                # Format: TL, TR, BR, BL — untuk resolusi 1280x720
                screen_points=[
                    [349, 258],   # Top-Left
                    [1046, 190],  # Top-Right
                    [988, 434],   # Bottom-Right
                    [441, 666],   # Bottom-Left
                ],
                screen_resolution="1280x720",
                # Screen points untuk resolusi berbeda (kalibrasi manual)
                screen_points_map={
                    "1920x1080": [
                        [490, 356],   # Top-Left
                        [1598, 256],  # Top-Right
                        [1499, 670],  # Bottom-Right
                        [634, 1047],  # Bottom-Left
                    ],
                },
            ),
            "paskal": LocationConfig(
                name="paskal",
                grid_rows=12,
                grid_cols=16,
                module_rows=12,
                module_cols=16,
                # Format: TL, TR, BR, BL — untuk resolusi 1280x720
                screen_points=[
                    [977, 1],      # Top-Left
                    [926, 719],    # Top-Right
                    [135, 391],    # Bottom-Right
                    [60, 27],      # Bottom-Left
                ],
                screen_resolution="1280x720",
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

        Struktur output/<DD-MM-YYYY>/<HH>/hasilnya/

        Args:
            location: Nama lokasi (tidak dipakai di path, hanya tracking).

        Returns:
            Path ke session folder hasilnya.
        """
        if cls._session_path is None:
            now = datetime.now()
            date_folder = now.strftime("%d-%m-%Y")
            hour_folder = now.strftime("%H")
            path = cls.OUTPUT_ROOT / date_folder / hour_folder
            path.mkdir(parents=True, exist_ok=True)
            cls._session_path = path

        return cls._session_path

    @classmethod
    def get_output_path(
        cls,
        location: str,
        status: Optional[str] = None,
    ) -> Path:
        """Ambil path output untuk lokasi.

        Struktur folder:
            output/<DD-MM-YYYY>/<HH>/good/   — untuk status normal
            output/<DD-MM-YYYY>/<HH>/bad/     — untuk warning / critical

        Args:
            location: Nama lokasi.
            status: Status deteksi (normal, warning, critical).
                    normal → good/, lainnya → bad/.

        Returns:
            Path ke folder output.
        """
        session = cls.get_session_path(location)
        if status:
            folder = "good" if status == "normal" else "bad"
            path = session / folder
        else:
            path = session
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_report_path(cls, location: str) -> Path:
        """Ambil path untuk reports.

        Report ada di folder jam, sama dengan gambar.

        Args:
            location: Nama lokasi.

        Returns:
            Path ke folder jam.
        """
        session = cls.get_session_path(location)
        session.mkdir(parents=True, exist_ok=True)
        return session

    @classmethod
    def ensure_directories(cls) -> None:
        """Buat semua direktori yang diperlukan."""
        cls.DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        cls.MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
