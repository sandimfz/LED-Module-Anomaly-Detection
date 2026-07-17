"""
LED Analyzer module.

Orchestrates all detection algorithms for LED panel analysis.
"""

from typing import List, Optional

import cv2
import numpy as np

from src.core.types import AnomalyLevel, DetectionResult, LocationConfig
from src.detectors.base import BaseDetector
from src.detectors.led.annotation import annotate_image
from src.detectors.led.content_mask import (
    create_full_image_mask,
    create_led_content_mask,
    refine_led_mask,
)
from src.detectors.led.defect_scorer import DefectScorer
from src.detectors.led.detectors import (
    detect_blocking,
    detect_color_errors,
    detect_color_errors_in_mask,
    detect_dark_regions_by_local_contrast,
    detect_dead_blocks,
    detect_dead_blocks_in_mask,
    detect_flat_content,
    detect_horizontal_line_pattern,
    detect_line_defects,
    detect_module_errors,
    detect_pixel_chaos,
    detect_region_contrast_anomalies,
    detect_uniform_content,
)
from src.detectors.led.feature_extractor import FeatureExtractor
from src.detectors.led.module_grid import calibrate_grid
from src.detectors.led.panel_finder import find_led_panel
from src.detectors.led.scoring import calculate_score
from src.detectors.led.spatial_analyzer import SpatialAnalyzer
from src.detectors.led.types import LEDAnomaly, LEDPanelInfo


class LEDAnalyzer(BaseDetector):
    """Main LED analyzer that orchestrates detection algorithms.

    Attributes:
        min_panel_area: Minimum area for panel detection.
        edge_threshold: Edge detection threshold.
        blocking_threshold: Blocking detection threshold.
    """

    def __init__(
        self,
        config: LocationConfig,
        min_panel_area: int = 50000,
        edge_threshold: int = 50,
        blocking_threshold: float = 0.3,
    ) -> None:
        """Initialize LED analyzer.

        Args:
            config: Location configuration.
            min_panel_area: Minimum panel area threshold.
            edge_threshold: Edge detection threshold.
            blocking_threshold: Blocking detection threshold.
        """
        super().__init__(config)
        self.min_panel_area = min_panel_area
        self.edge_threshold = edge_threshold
        self.blocking_threshold = blocking_threshold

    def detect(
        self,
        image: np.ndarray,
        image_path: str,
    ) -> DetectionResult:
        """Detect anomalies in LED panel image.

        Works on full image without cropping. Detects LED panel
        area and runs all detectors within that area.

        Args:
            image: Input BGR image.
            image_path: Path to the image file.

        Returns:
            Detection result with anomalies and score.
        """
        h_img, w_img = image.shape[:2]
        current_res = f"{w_img}x{h_img}"

        led_panel, panel_mask = self._find_led_area(
            image, current_res
        )

        if led_panel is None:
            return self._create_no_panel_result(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray[panel_mask > 0]))
        dark_pixel_ratio = float(np.mean(gray[panel_mask > 0] < 60))

        anomalies = self._detect_anomalies(
            image, led_panel, panel_mask,
            mean_brightness, dark_pixel_ratio
        )

        score = calculate_score(anomalies, led_panel)
        annotated = annotate_image(image, led_panel, anomalies)
        status = self._determine_level(score)
        output_path = self.get_output_path(
            image_path, "led_analysis", status
        )
        self.save_annotated(annotated, output_path)

        return DetectionResult(
            location=self.config.name,
            image_path=image_path,
            anomaly_score=round(score, 4),
            level=AnomalyLevel(status),
            message=self._generate_message(score, anomalies, led_panel),
            flagged_cells=[],
            heatmap_path=output_path,
        )

    def _find_led_area(
        self,
        image: np.ndarray,
        current_res: str,
    ) -> tuple[Optional[LEDPanelInfo], Optional[np.ndarray]]:
        """Find LED area in image without cropping.

        Detects LED panel location and creates a mask for
        the LED content area.

        Args:
            image: Input BGR image.
            current_res: Current resolution string.

        Returns:
            Tuple of (led_panel_info, panel_mask) or (None, None).
        """
        if (
            self.config.screen_points is not None
            and self.config.screen_resolution == current_res
        ):
            led_panel = self._screen_points_to_panel(
                self.config.screen_points
            )
            panel_mask = create_full_image_mask(
                image.shape, led_panel, self.config.screen_points
            )
            return led_panel, panel_mask

        led_panel = find_led_panel(image, self.min_panel_area)
        if led_panel is None:
            return None, None

        panel_mask = create_full_image_mask(
            image.shape, led_panel
        )

        return led_panel, panel_mask

    def _screen_points_to_panel(
        self,
        screen_points: List[List[int]],
    ) -> LEDPanelInfo:
        """Convert screen points to LEDPanelInfo.

        Args:
            screen_points: 4 corner points [[x,y], ...].

        Returns:
            LED panel information.
        """
        pts = np.array(screen_points)
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)

        return LEDPanelInfo(
            x=int(x_min),
            y=int(y_min),
            width=int(x_max - x_min),
            height=int(y_max - y_min),
            confidence=1.0,
            mean_brightness=0.0,
            mean_saturation=0.0,
        )

    def _create_no_panel_result(
        self,
        image_path: str,
    ) -> DetectionResult:
        """Create result when no LED panel is found.

        Args:
            image_path: Path to the image file.

        Returns:
            Detection result with no panel message.
        """
        return DetectionResult(
            location=self.config.name,
            image_path=image_path,
            anomaly_score=0.0,
            level=AnomalyLevel.NORMAL,
            message="Tidak dapat menemukan panel LED.",
        )

    def _detect_anomalies(
        self,
        image: np.ndarray,
        led_panel: LEDPanelInfo,
        panel_mask: np.ndarray,
        mean_brightness: float,
        dark_pixel_ratio: float,
    ) -> List[LEDAnomaly]:
        """Detect all anomalies in LED area.

        Always runs sub-detectors regardless of brightness level.
        Only falls back to "LED off" if sub-detectors find nothing
        AND the panel is truly dark — this avoids falsely classifying
        dim-but-functional LEDs as dead.

        Args:
            image: Full input image.
            led_panel: LED panel information.
            panel_mask: Binary mask of LED area.
            mean_brightness: Mean brightness value.
            dark_pixel_ratio: Ratio of dark pixels.

        Returns:
            List of detected anomalies.
        """
        # Always run sub-detectors first — don't skip based on brightness.
        anomalies = self._analyze_led_content_full(
            image, led_panel, panel_mask
        )

        # If sub-detectors found real issues, return them regardless
        # of overall brightness.
        if anomalies:
            return anomalies

        # Only fall back to "off" / "mostly off" if the panel is
        # genuinely dark AND sub-detectors found nothing.
        if mean_brightness < 40:
            return [self._create_led_off_anomaly(led_panel)]

        if dark_pixel_ratio > 0.4:
            anomalies.append(
                self._create_led_mostly_off_anomaly(
                    led_panel, dark_pixel_ratio
                )
            )
            return anomalies

        return anomalies

    def _create_led_off_anomaly(
        self,
        panel: LEDPanelInfo,
    ) -> LEDAnomaly:
        """Create anomaly for completely off LED.

        Args:
            panel: LED panel information.

        Returns:
            LED off anomaly.
        """
        return LEDAnomaly(
            x=panel.x,
            y=panel.y,
            width=panel.width,
            height=panel.height,
            anomaly_type="led_off",
            severity=1.0,
            description=f"LED mati total (brightness={panel.mean_brightness:.0f})",
        )

    def _create_led_mostly_off_anomaly(
        self,
        panel: LEDPanelInfo,
        dark_ratio: float,
    ) -> LEDAnomaly:
        """Create anomaly for mostly off LED.

        Args:
            panel: LED panel information.
            dark_ratio: Ratio of dark pixels.

        Returns:
            LED mostly off anomaly.
        """
        return LEDAnomaly(
            x=panel.x,
            y=panel.y,
            width=panel.width,
            height=panel.height,
            anomaly_type="led_mostly_off",
            severity=min(dark_ratio * 1.5, 1.0),
            description=(
                f"Sebagian besar LED gelap "
                f"({dark_ratio*100:.0f}% area gelap)"
            ),
        )

    def _analyze_led_content_full(
        self,
        image: np.ndarray,
        panel: LEDPanelInfo,
        panel_mask: np.ndarray,
    ) -> List[LEDAnomaly]:
        """Analyze LED content for anomalies on full image.

        Runs all detectors on the full image using masks to
        identify LED content area.

        Args:
            image: Full input image.
            panel: LED panel information.
            panel_mask: Binary mask of LED area.

        Returns:
            List of detected anomalies.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        led_mask = refine_led_mask(gray, hsv, panel_mask)

        # Content classification for adaptive thresholds
        from src.detectors.led.content_classifier import ContentClassifier
        classifier = ContentClassifier()
        content_type, content_confidence = classifier.classify(gray, hsv, led_mask)
        thresholds = classifier.get_adaptive_thresholds(content_type)

        anomalies: List[LEDAnomaly] = []
        anomalies.extend(detect_blocking(gray, panel, led_mask, panel_mask))
        anomalies.extend(
            detect_flat_content(gray, panel, led_mask, panel_mask, hsv)
        )
        anomalies.extend(detect_line_defects(gray, panel, led_mask))
        anomalies.extend(
            detect_dead_blocks_in_mask(gray, panel, led_mask)
        )
        anomalies.extend(
            detect_dark_regions_by_local_contrast(gray, panel, led_mask)
        )
        anomalies.extend(
            detect_color_errors_in_mask(hsv, panel, led_mask)
        )
        # pixel_chaos disabled — produces too many false positives on
        # normal colourful content (night shots, vivid ads).
        # anomalies.extend(detect_pixel_chaos(...))
        anomalies.extend(
            detect_horizontal_line_pattern(gray, hsv, panel, led_mask)
        )
        anomalies.extend(
            detect_uniform_content(gray, hsv, panel, led_mask)
        )
        # NEW: Region-based contrast comparison (MAD-based)
        anomalies.extend(
            detect_region_contrast_anomalies(gray, panel, led_mask, panel_mask)
        )
        # NEW: Module error detection (glitchy/corrupted sections)
        anomalies.extend(
            detect_module_errors(gray, hsv, panel, led_mask)
        )

        return anomalies

    def _generate_message(
        self,
        score: float,
        anomalies: List[LEDAnomaly],
        panel: LEDPanelInfo,
    ) -> str:
        """Generate descriptive message for detection result.

        Args:
            score: Anomaly score.
            anomalies: List of detected anomalies.
            panel: LED panel information.

        Returns:
            Descriptive message string.
        """
        if not anomalies:
            return (
                f"Panel LED terdeteksi "
                f"({panel.width}x{panel.height}). "
                f"Tidak ada anomali."
            )

        parts = self._categorize_anomalies(anomalies)

        return (
            f"Panel LED terdeteksi "
            f"({panel.width}x{panel.height}). "
            f"Anomali: {', '.join(parts)}. "
            f"Score: {score:.2f}"
        )

    def _categorize_anomalies(
        self,
        anomalies: List[LEDAnomaly],
    ) -> List[str]:
        """Categorize anomalies by type.

        Args:
            anomalies: List of anomalies.

        Returns:
            List of categorized anomaly descriptions.
        """
        categories = {
            "led_off": [],
            "blocking": [],
            "flat_content": [],
            "line_defect": [],
            "dead": [],
            "color_error": [],
            "pixel_chaos": [],
            "module_glitch": [],
            "horizontal_pattern": [],
            "uniform_content": [],
            "dark_region": [],
        }

        for a in anomalies:
            if a.anomaly_type in ("led_off", "led_mostly_off"):
                categories["led_off"].append(a)
            elif a.anomaly_type == "blocking":
                categories["blocking"].append(a)
            elif a.anomaly_type == "flat_content":
                categories["flat_content"].append(a)
            elif a.anomaly_type == "line_defect":
                categories["line_defect"].append(a)
            elif a.anomaly_type in ("dead_pixel_block",):
                categories["dead"].append(a)
            elif a.anomaly_type == "color_error":
                categories["color_error"].append(a)
            elif a.anomaly_type == "pixel_chaos":
                categories["pixel_chaos"].append(a)
            elif a.anomaly_type in ("module_glitch", "horizontal_glitch", "vertical_glitch"):
                categories["module_glitch"].append(a)
            elif a.anomaly_type == "horizontal_pattern":
                categories["horizontal_pattern"].append(a)
            elif a.anomaly_type == "uniform_content":
                categories["uniform_content"].append(a)
            elif a.anomaly_type in ("dark_region", "dark_spot", "horizontal_dark_bar", "vertical_dark_bar"):
                categories["dark_region"].append(a)

        parts: List[str] = []
        if categories["led_off"]:
            desc = categories["led_off"][0].description
            parts.append(f"LED mati/sebagian besar gelap ({desc})")
        if categories["blocking"]:
            parts.append(f"{len(categories['blocking'])} blocking")
        if categories["dark_region"]:
            parts.append(f"{len(categories['dark_region'])} dark regions")
        if categories["flat_content"]:
            count = len(categories["flat_content"])
            parts.append(f"{count} no content areas")
        if categories["line_defect"]:
            count = len(categories["line_defect"])
            parts.append(f"{count} line defects")
        if categories["dead"]:
            parts.append(f"{len(categories['dead'])} dead blocks")
        if categories["color_error"]:
            count = len(categories["color_error"])
            parts.append(f"{count} color errors")
        if categories["pixel_chaos"]:
            count = len(categories["pixel_chaos"])
            parts.append(f"{count} pixel chaos (modul rusak/glitch)")
        if categories["module_glitch"]:
            count = len(categories["module_glitch"])
            parts.append(f"{count} module glitch (baris warna acak)")
        if categories["horizontal_pattern"]:
            count = len(categories["horizontal_pattern"])
            parts.append(f"{count} horizontal line patterns")
        if categories["uniform_content"]:
            count = len(categories["uniform_content"])
            parts.append(f"{count} uniform content (frozen/stuck?)")

        return parts
