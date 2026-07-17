#!/usr/bin/env python3
"""
Training script untuk Anomaly Detector.

Melatih Isolation Forest model menggunakan gambar normal
dari setiap lokasi. Model digunakan untuk detect anomali
yang mungkin terlewat oleh rule-based detectors.

Cara pakai:
    python train_anomaly_detector.py --location lengkong
    python train_anomaly_detector.py --location all
"""

import argparse
import os
import sys

import cv2
import numpy as np

from src.core.config import Config
from src.pipeline.ensemble import EnsemblePipeline


def load_normal_images(location: str, max_images: int = 100) -> list:
    """Load normal images from dataset folder.

    Args:
        location: Location name.
        max_images: Maximum number of images to load.

    Returns:
        List of cropped LED images.
    """
    good_dir = f"dataset/{location}/good"
    if not os.path.exists(good_dir):
        print(f"Directory not found: {good_dir}")
        return []

    images = []
    for f in os.listdir(good_dir)[:max_images]:
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(good_dir, f)
        try:
            # Load and crop like pipeline does
            pipeline = EnsemblePipeline(location)
            from src.core.utils import load_image
            image = load_image(img_path)
            cropped = pipeline._crop_to_screen(image)
            images.append(cropped)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

    return images


def train_location(location: str) -> bool:
    """Train anomaly detector for a location.

    Args:
        location: Location name.

    Returns:
        True if training successful.
    """
    print(f"\n{'='*50}")
    print(f"Training anomaly detector for: {location}")
    print(f"{'='*50}")

    # Load normal images
    print(f"Loading normal images from dataset/{location}/good/...")
    images = load_normal_images(location, max_images=50)
    print(f"Loaded {len(images)} images")

    if len(images) < 10:
        print(f"Error: Need at least 10 images, got {len(images)}")
        return False

    # Train anomaly detector
    from src.detectors.led.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()

    print("Training Isolation Forest model...")
    success = detector.train(images, location)

    if success:
        print(f"✓ Model trained and saved to models/anomaly_detector_{location}.pkl")
    else:
        print("✗ Training failed")

    return success


def main():
    parser = argparse.ArgumentParser(description="Train LED Anomaly Detector")
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="Location name (e.g., lengkong, paskal) or 'all' for all locations"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum number of images to use for training"
    )

    args = parser.parse_args()

    if args.location == "all":
        # Train for all locations
        locations = list(Config.LOCATIONS.keys())
        print(f"Training for all locations: {locations}")
    else:
        locations = [args.location]

    results = {}
    for location in locations:
        success = train_location(location)
        results[location] = success

    # Print summary
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")

    for location, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {location}: {status}")

    # Return success if all locations succeeded
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
