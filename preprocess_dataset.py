#!/usr/bin/env python3
"""
Preprocess dataset: crop semua gambar ke area LED screen.

Hanya untuk lokasi yang punya screen_points.
Menyimpan hasil crop sebagai pengganti file asli (backup dulu).

Cara pakai:
    python preprocess_dataset.py --location lengkong
    python preprocess_dataset.py --location lengkong --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import Config


def crop_image(image: np.ndarray, screen_points: list) -> np.ndarray:
    """Apply perspective crop to image.

    Args:
        image: Input BGR image.
        screen_points: 4 corner points [[x,y], ...] in TL, TR, BR, BL order.

    Returns:
        Cropped image.
    """
    pts_src = np.array(screen_points, dtype="float32")
    tl, tr, br, bl = pts_src

    max_width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl),
    ))
    max_height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl),
    ))

    pts_dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def main(location: str, dry_run: bool = False) -> None:
    """Preprocess dataset for a location.

    Args:
        location: Location name.
        dry_run: Print what would be done without doing it.
    """
    config = Config.get_location_config(location)

    if config.screen_points is None:
        print(f"Lokasi '{location}' tidak punya screen_points. Skipping.")
        return

    dataset_path = Config.get_dataset_path(location)
    print(f"Location: {location}")
    print(f"Dataset: {dataset_path}")
    print(f"Screen points: {config.screen_points}")
    print(f"Screen resolution: {config.screen_resolution}")
    print(f"Dry run: {dry_run}")
    print("-" * 50)

    # Process good/ and bad/ directories
    for subdir in ["good", "bad"]:
        dir_path = dataset_path / subdir
        if not dir_path.exists():
            print(f"Folder {dir_path} tidak ditemukan. Skipping.")
            continue

        image_paths = sorted(
            list(dir_path.glob("*.jpg"))
            + list(dir_path.glob("*.jpeg"))
            + list(dir_path.glob("*.png"))
        )

        if not image_paths:
            print(f"Tidak ada gambar di {dir_path}")
            continue

        print(f"\nMemproses {len(image_paths)} gambar dari {subdir}/:")

        # Create backup folder
        backup_dir = dir_path.parent / f"{subdir}_backup"
        if not dry_run:
            backup_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {img_path.name}", end="")

            image = cv2.imread(str(img_path))
            if image is None:
                print(" — GAGAL baca")
                continue

            h, w = image.shape[:2]
            current_res = f"{w}x{h}"

            # Skip jika resolusi tidak cocok
            if config.screen_resolution and current_res != config.screen_resolution:
                print(f" — Skip (res {current_res}, perlu {config.screen_resolution})")
                continue

            # Backup original
            if not dry_run:
                shutil.copy2(img_path, backup_dir / img_path.name)

            # Crop
            cropped = crop_image(image, config.screen_points)
            out_path = str(img_path)

            if not dry_run:
                cv2.imwrite(out_path, cropped)
                print(f" — OK ({cropped.shape[1]}x{cropped.shape[0]})")
            else:
                print(f" — akan di-crop ke {cropped.shape[1]}x{cropped.shape[0]}")

    if not dry_run:
        print("\n" + "=" * 50)
        print("Selesai! Original tersimpan di folder *_backup.")
        print("Jalankan ulang training dengan gambar yang sudah di-crop:")
        print(f"  python train.py --location {location}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset — crop to LED screen area"
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="Nama lokasi",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Cek dulu tanpa mengubah file",
    )
    args = parser.parse_args()
    main(args.location, args.dry_run)
