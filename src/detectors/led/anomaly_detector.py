"""
ML-based anomaly detector using Isolation Forest.

Learns "normal" pattern per location and detects anomalies.
This complements rule-based detectors by catching subtle
patterns that fixed thresholds might miss.
"""

import os
import pickle
from typing import List, Optional

import cv2
import numpy as np

from src.core.types import AnomalyLevel, DetectionResult


class AnomalyDetector:
    """ML-based anomaly detector using Isolation Forest.

    Learns "normal" pattern per location and detects anomalies
    that rule-based detectors might miss.
    """

    def __init__(self, model_dir: str = "models"):
        """Initialize anomaly detector.

        Args:
            model_dir: Directory to store/load trained models.
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.location = None
        self._available = False

        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            self._available = True
        except ImportError:
            print("scikit-learn not installed. Anomaly detector disabled.")

    @property
    def is_available(self) -> bool:
        """Check if anomaly detector is available."""
        return self._available

    def load_model(self, location: str) -> bool:
        """Load trained model for location.

        Args:
            location: Location name.

        Returns:
            True if model loaded successfully.
        """
        if not self._available:
            return False

        model_path = os.path.join(self.model_dir, f"anomaly_detector_{location}.pkl")

        if not os.path.exists(model_path):
            return False

        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.location = location
            return True
        except Exception as e:
            print(f"Error loading anomaly model: {e}")
            return False

    def train(self, images: List[np.ndarray], location: str) -> bool:
        """Train anomaly detector on normal images.

        Args:
            images: List of normal LED images (cropped).
            location: Location name.

        Returns:
            True if training successful.
        """
        if not self._available:
            return False

        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            # Extract features from all images
            features = []
            for img in images:
                feat = self._extract_features(img)
                if feat is not None:
                    features.append(feat)

            if len(features) < 10:
                print(f"Not enough images for training: {len(features)}/10 minimum")
                return False

            features = np.array(features)

            # Scale features
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)

            # Train Isolation Forest
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.1,  # Expect ~10% anomalies
                random_state=42
            )
            self.model.fit(features_scaled)

            # Save model
            self.location = location
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, f"anomaly_detector_{location}.pkl")

            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'location': location
                }, f)

            print(f"Anomaly detector trained: {len(images)} images, model saved to {model_path}")
            return True

        except Exception as e:
            print(f"Training error: {e}")
            return False

    def predict(
        self,
        image: np.ndarray,
        image_path: str = "",
    ) -> DetectionResult:
        """Predict if image is normal or anomalous.

        Args:
            image: Cropped LED image.
            image_path: Path to the image file.

        Returns:
            DetectionResult with anomaly_score and level.
        """
        if not self._available or self.model is None:
            # No model loaded, return neutral result
            return DetectionResult(
                location=self.location or "unknown",
                image_path=image_path,
                anomaly_score=0.0,
                level=AnomalyLevel.NORMAL,
                message="ML anomaly detector: no model loaded"
            )

        # Extract features
        features = self._extract_features(image)
        if features is None:
            return DetectionResult(
                location=self.location or "unknown",
                image_path=image_path,
                anomaly_score=0.0,
                level=AnomalyLevel.NORMAL,
                message="ML anomaly detector: feature extraction failed"
            )

        features_scaled = self.scaler.transform([features])

        # Predict (-1 = anomaly, 1 = normal)
        prediction = self.model.predict(features_scaled)[0]
        score = -self.model.decision_function(features_scaled)[0]  # Convert to 0-1 range

        # Normalize score to 0-1
        normalized_score = min(max((score + 0.5) / 1.0, 0.0), 1.0)

        # Determine level
        if normalized_score > 0.6:
            level = AnomalyLevel.CRITICAL
        elif normalized_score > 0.3:
            level = AnomalyLevel.WARNING
        else:
            level = AnomalyLevel.NORMAL

        return DetectionResult(
            location=self.location or "unknown",
            image_path=image_path,
            anomaly_score=round(normalized_score, 4),
            level=level,
            message=f"ML anomaly score: {normalized_score:.4f}"
        )

    def _extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract feature vector from image.

        Args:
            image: Cropped LED image.

        Returns:
            Feature vector or None if extraction fails.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            features = []

            # Global features
            features.append(float(np.mean(gray)))  # mean_brightness
            features.append(float(np.std(gray)))   # std_brightness
            features.append(float(np.mean(hsv[:,:,1])))  # mean_saturation
            features.append(float(np.std(hsv[:,:,1])))   # std_saturation
            features.append(float(np.mean(hsv[:,:,0])))  # mean_hue
            features.append(float(np.std(hsv[:,:,0])))   # std_hue

            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            features.append(float(np.mean(edges > 0)))  # edge_density

            # Color histogram features (16 bins)
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_h = hist_h.flatten() / hist_h.sum()
            features.extend(hist_h.tolist())

            # Texture features (using Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features.append(float(np.mean(np.abs(laplacian))))  # texture_strength
            features.append(float(np.std(laplacian)))  # texture_variation

            # Block statistics (8x8 grid)
            h, w = gray.shape
            block_h, block_w = h // 8, w // 8
            block_means = []
            for r in range(8):
                for c in range(8):
                    y0, y1 = r * block_h, (r + 1) * block_h
                    x0, x1 = c * block_w, (c + 1) * block_w
                    block_means.append(float(np.mean(gray[y0:y1, x0:x1])))

            features.append(float(np.std(block_means)))  # block_variation
            features.append(float(np.mean(block_means)))  # block_mean

            return np.array(features, dtype=np.float32)

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
