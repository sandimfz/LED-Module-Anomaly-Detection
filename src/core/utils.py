"""
Utility functions untuk LED Anomaly Detection.
"""

import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load gambar dari path.

    Args:
        path: Path ke file gambar.

    Returns:
        Gambar dalam format BGR (OpenCV default).

    Raises:
        FileNotFoundError: Jika gambar tidak bisa dibaca.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak bisa dibaca: {path}")
    return img


def load_frames(folder: str) -> List[np.ndarray]:
    """Load semua frame dari folder.

    Args:
        folder: Path ke folder yang berisi gambar.

    Returns:
        List gambar dalam format BGR.

    Raises:
        ValueError: Jika kurang dari 3 frame.
    """
    extensions = ("*.jpg", "*.jpeg", "*.png")
    paths: List[str] = []
    for ext in extensions:
        paths.extend(glob.glob(f"{folder}/{ext}"))
    paths.sort()

    if len(paths) < 3:
        raise ValueError(
            f"Butuh minimal 3 frame, cuma ada {len(paths)} di {folder}"
        )

    frames: List[np.ndarray] = []
    ref_shape: Optional[Tuple[int, int]] = None

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        if ref_shape is None:
            ref_shape = img.shape[:2]
        elif img.shape[:2] != ref_shape:
            img = cv2.resize(img, (ref_shape[1], ref_shape[0]))
        frames.append(img)

    return frames


def hue_delta(h1: float, h2: float) -> float:
    """Selisih hue yang benar secara sirkular.

    Di OpenCV HSV, hue range 0-180. Nilai 0 dan 180 itu deket.

    Args:
        h1: Hue pertama (0-180).
        h2: Hue kedua (0-180).

    Returns:
        Selisih hue (0-90).
    """
    d = abs(h1 - h2)
    return min(d, 180 - d)


def calculate_robust_z(
    value: float,
    median: float,
    mad: float,
) -> float:
    """Hitung z-score robust menggunakan MAD.

    Args:
        value: Nilai yang akan di-z-score.
        median: Median dari distribusi.
        mad: Median Absolute Deviation.

    Returns:
        Z-score robust.
    """
    return (value - median) / (1.4826 * mad) if mad > 1e-6 else 0.0


def calculate_activity_score(values: List[float]) -> float:
    """Hitung activity score dari sequence nilai.

    Activity score = rata-rata selisih absolut antar nilai berurutan.
    Makin tinggi = makin sering berubah.

    Args:
        values: Sequence nilai (brightness per frame).

    Returns:
        Activity score.
    """
    if len(values) < 2:
        return 0.0
    diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    return float(np.mean(diffs))


def extract_grid_stats(
    image: np.ndarray,
    grid_rows: int,
    grid_cols: int,
) -> List[List[Dict[str, float]]]:
    """Ekstrak statistik brightness, std, dan hue untuk tiap cell grid.

    Args:
        image: Gambar BGR.
        grid_rows: Jumlah baris grid.
        grid_cols: Jumlah kolom grid.

    Returns:
        List 2D berisi dict dengan keys: x, y, w, h, brightness, std, hue.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_total, w_total = gray.shape[:2]
    cell_h = h_total // grid_rows
    cell_w = w_total // grid_cols

    grid: List[List[Dict[str, float]]] = []

    for row in range(grid_rows):
        row_data: List[Dict[str, float]] = []
        for col in range(grid_cols):
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w

            gray_cell = gray[y0:y1, x0:x1]
            hue_cell = hsv[y0:y1, x0:x1, 0]

            row_data.append({
                "x": float(x0),
                "y": float(y0),
                "w": float(cell_w),
                "h": float(cell_h),
                "brightness": float(np.mean(gray_cell)),
                "std": float(np.std(gray_cell)),
                "hue": float(np.median(hue_cell)),
            })
        grid.append(row_data)

    return grid


def get_neighbor_stats(
    grid: List[List[Dict[str, float]]],
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> Optional[Dict[str, float]]:
    """Ambil rata-rata statistik dari 8 tetangga.

    Args:
        grid: Grid statistik.
        row: Index baris cell.
        col: Index kolom cell.
        rows: Total baris grid.
        cols: Total kolom grid.

    Returns:
        Dict dengan keys brightness, std, hue atau None jika tidak ada tetangga.
    """
    neighbor_brightness: List[float] = []
    neighbor_std: List[float] = []
    neighbor_hue: List[float] = []

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                stat = grid[r][c]
                neighbor_brightness.append(stat["brightness"])
                neighbor_std.append(stat["std"])
                neighbor_hue.append(stat["hue"])

    if not neighbor_brightness:
        return None

    return {
        "brightness": float(np.mean(neighbor_brightness)),
        "std": float(np.mean(neighbor_std)),
        "hue": float(np.mean(neighbor_hue)),
    }


def create_heatmap_overlay(
    original: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Buat heatmap overlay di atas gambar asli.

    Args:
        original: Gambar asli (BGR).
        anomaly_map: Peta anomali (0-1 float).
        alpha: Transparansi heatmap (0-1).

    Returns:
        Gambar dengan heatmap overlay.
    """
    heat = (anomaly_map * 255).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color = cv2.resize(
        heat_color,
        (original.shape[1], original.shape[0]),
    )
    overlay = cv2.addWeighted(original, 1 - alpha, heat_color, alpha, 0)
    return overlay
