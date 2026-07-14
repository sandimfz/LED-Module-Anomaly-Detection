"""
Type definitions untuk LED Anomaly Detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class AnomalyLevel(Enum):
    """Level keparahan anomali."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class LocationConfig:
    """Konfigurasi untuk satu lokasi LED.

    Attributes:
        name: Nama lokasi (misal: sigma_cirebon, lengkong).
        grid_rows: Jumlah baris grid untuk analisis.
        grid_cols: Jumlah kolom grid untuk analisis.
        min_frames: Minimal jumlah frame untuk analisis temporal.
        frozen_threshold: Threshold z-score untuk mendeteksi frozen module.
        dead_brightness_threshold: Threshold brightness untuk modul mati.
        dark_delta_threshold: Selisih brightness minimal untuk dark anomaly.
        flat_ratio_threshold: Rasio std dev untuk flat anomaly.
        color_hue_delta_threshold: Selisih hue untuk color anomaly.
    """

    name: str
    grid_rows: int = 12
    grid_cols: int = 16
    min_frames: int = 3
    frozen_threshold: float = -2.5
    dead_brightness_threshold: float = 25.0
    dark_delta_threshold: float = 40.0
    flat_ratio_threshold: float = 0.35
    color_hue_delta_threshold: float = 35.0


@dataclass
class CellResult:
    """Hasil analisis untuk satu cell/grid.

    Attributes:
        row: Index baris.
        col: Index kolom.
        x: Koordinat x pixel.
        y: Koordinat y pixel.
        w: Lebar cell dalam pixel.
        h: Tinggi cell dalam pixel.
        brightness: Rata-rata brightness.
        std_dev: Standar deviasi brightness.
        neighbor_brightness: Rata-rata brightness tetangga.
        neighbor_std: Rata-rata std dev tetangga.
        is_dark_anomaly: True jika gelap dari tetangga.
        is_flat_anomaly: True jika terlalu rata/flatt.
        is_color_anomaly: True jika warna menyimpang.
        is_anomaly: True jika ada salah satu anomali.
    """

    row: int
    col: int
    x: int
    y: int
    w: int
    h: int
    brightness: float
    std_dev: float
    neighbor_brightness: float = 0.0
    neighbor_std: float = 0.0
    is_dark_anomaly: bool = False
    is_flat_anomaly: bool = False
    is_color_anomaly: bool = False
    is_anomaly: bool = False


@dataclass
class TemporalCellResult:
    """Hasil analisis temporal untuk satu cell.

    Attributes:
        row: Index baris.
        col: Index kolom.
        x: Koordinat x pixel.
        y: Koordinat y pixel.
        w: Lebar cell dalam pixel.
        h: Tinggi cell dalam pixel.
        activity_score: Skor aktivitas (makin tinggi = makin sering berubah).
        mean_brightness: Rata-rata brightness sepanjang waktu.
        robust_z: Z-score robust (MAD-based).
        is_frozen: True jika tidak berubah.
        likely_dead_module: True jika frozen + gelap total.
    """

    row: int
    col: int
    x: int
    y: int
    w: int
    h: int
    activity_score: float
    mean_brightness: float
    robust_z: float
    is_frozen: bool = False
    likely_dead_module: bool = False


@dataclass
class DetectionResult:
    """Hasil deteksi anomali untuk satu gambar.

    Attributes:
        location: Nama lokasi.
        image_path: Path gambar yang dianalisis.
        anomaly_score: Skor anomali (0.0 - 1.0).
        level: Level anomali (NORMAL, WARNING, CRITICAL).
        message: Pesan deskriptif.
        flagged_cells: Koordinat cell yang terdeteksi anomali.
        heatmap_path: Path heatmap output (jika ada).
    """

    location: str
    image_path: str
    anomaly_score: float
    level: AnomalyLevel
    message: str
    flagged_cells: List[Tuple[int, int]] = field(default_factory=list)
    heatmap_path: Optional[str] = None


@dataclass
class FrameStats:
    """Statistik untuk satu frame.

    Attributes:
        frame_index: Index frame dalam sequence.
        timestamp: Timestamp file (jika ada).
        mean_brightness: Rata-rata brightness frame.
        std_brightness: Standar deviasi brightness frame.
    """

    frame_index: int
    timestamp: Optional[str] = None
    mean_brightness: float = 0.0
    std_brightness: float = 0.0
