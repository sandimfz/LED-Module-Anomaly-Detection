"""
Baseline Store for LED Anomaly Detection.

Stores rolling statistics per panel (per location) for adaptive thresholds.
Uses JSON file for simplicity. Updates only from frames classified as NORMAL.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.core.config import Config


class BaselineStore:
    """Rolling baseline statistics per location.

    Stores median/std brightness per panel to enable
    relative thresholds instead of absolute thresholds.
    """

    def __init__(self, location: str, window_size: int = 50):
        """Initialize baseline store.

        Args:
            location: Location name.
            window_size: Number of frames to keep in rolling window.
        """
        self.location = location
        self.window_size = window_size
        self._data: Dict = {"frames": [], "stats": {}}
        self._path = Config.MODELS_ROOT / location / "baseline.json"
        self._load()

    def _load(self) -> None:
        """Load baseline from disk."""
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {"frames": [], "stats": {}}

    def _save(self) -> None:
        """Save baseline to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def update(self, panel_brightness: float, level: str) -> None:
        """Update baseline with new frame data.

        Only updates if level is NORMAL (exclude anomalous frames).

        Args:
            panel_brightness: Mean brightness of the panel.
            level: Detection level ("normal", "warning", "critical").
        """
        if level != "normal":
            return  # Only learn from normal frames

        self._data["frames"].append(panel_brightness)

        # Keep only last N frames
        if len(self._data["frames"]) > self.window_size:
            self._data["frames"] = self._data["frames"][-self.window_size:]

        # Recompute stats
        if len(self._data["frames"]) >= 5:
            arr = np.array(self._data["frames"])
            self._data["stats"] = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "mad": float(np.median(np.abs(arr - np.median(arr)))),
                "n_frames": len(self._data["frames"]),
            }
        self._save()

    def get_stats(self) -> Optional[Dict]:
        """Get current baseline statistics.

        Returns:
            Dict with median, mean, std, mad, n_frames.
            None if not enough data (< 5 frames).
        """
        stats = self._data.get("stats", {})
        if stats.get("n_frames", 0) < 5:
            return None
        return stats

    def is_ready(self) -> bool:
        """Check if baseline has enough data."""
        return self._data.get("stats", {}).get("n_frames", 0) >= 5

    def get_brightness_threshold(self) -> Optional[float]:
        """Get adaptive brightness threshold based on baseline.

        Returns:
            Threshold value, or None if baseline not ready.
        """
        stats = self.get_stats()
        if stats is None:
            return None
        # Threshold: median - 2 * MAD (robust outlier detection)
        return stats["median"] - 2 * stats["mad"]
