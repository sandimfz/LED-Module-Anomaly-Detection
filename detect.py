#!/usr/bin/env python3
"""
Detection Script untuk LED Anomaly Detection.

Mendukung:
- Analisis satu gambar (grid + patchcore)
- Analisis multiple frames (grid + temporal + patchcore)

Cara pakai:
    # Analisis satu gambar
    python detect.py --location lengkong --image path/to/image.jpg

    # Analisis folder frames
    python detect.py --location lengkong --frames path/to/folder/

    # Analisis dengan semua detektor
    python detect.py --location lengkong --image image.jpg --all
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import Config
from src.core.utils import load_frames, load_image
from src.pipeline.ensemble import EnsemblePipeline


def detect_single(
    pipeline: EnsemblePipeline,
    image_path: str,
    verbose: bool = True,
) -> None:
    """Deteksi anomali pada satu gambar.

    Args:
        pipeline: Ensemble pipeline.
        image_path: Path gambar.
        verbose: Tampilkan output detail.
    """
    print(f"\nAnalisis gambar: {image_path}")
    print("-" * 50)

    results = pipeline.analyze_single(image_path)
    combined = pipeline.combine_results(results)

    if verbose:
        for name, result in results.items():
            print(f"\n[{name.upper()}]")
            print(f"  Score: {result.anomaly_score}")
            print(f"  Level: {result.level.value}")
            print(f"  Message: {result.message}")

        print("\n" + "=" * 50)
        print("HASIL GABUNGAN:")
        print(f"  Score: {combined.anomaly_score}")
        print(f"  Level: {combined.level.value}")
        print(f"  Message: {combined.message}")

    if combined.heatmap_path:
        print(f"\nHeatmap: {combined.heatmap_path}")

    # Save report
    report_path = pipeline.save_report(results, combined)
    print(f"Report: {report_path}")


def detect_frames(
    pipeline: EnsemblePipeline,
    frames_folder: str,
    verbose: bool = True,
) -> None:
    """Deteksi anomali dari multiple frames.

    Args:
        pipeline: Ensemble pipeline.
        frames_folder: Path folder frames.
        verbose: Tampilkan output detail.
    """
    print(f"\nAnalisis frames dari: {frames_folder}")
    print("-" * 50)

    try:
        frames = load_frames(frames_folder)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    frame_paths = sorted(
        list(Path(frames_folder).glob("*.jpg"))
        + list(Path(frames_folder).glob("*.png"))
    )
    frame_paths = [str(p) for p in frame_paths][:len(frames)]

    print(f"Loaded {len(frames)} frames")

    results = pipeline.analyze_video_frames(frames, frame_paths)
    combined = pipeline.combine_results(results)

    if verbose:
        for name, result in results.items():
            print(f"\n[{name.upper()}]")
            print(f"  Score: {result.anomaly_score}")
            print(f"  Level: {result.level.value}")
            print(f"  Message: {result.message}")

        print("\n" + "=" * 50)
        print("HASIL GABUNGAN:")
        print(f"  Score: {combined.anomaly_score}")
        print(f"  Level: {combined.level.value}")
        print(f"  Message: {combined.message}")

    if combined.heatmap_path:
        print(f"\nHeatmap: {combined.heatmap_path}")

    # Save report
    report_path = pipeline.save_report(results, combined)
    print(f"Report: {report_path}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="LED Anomaly Detection"
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="Nama lokasi",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path gambar untuk analisis single frame",
    )
    parser.add_argument(
        "--frames",
        type=str,
        help="Path folder untuk analisis multiple frames",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Gunakan semua detektor termasuk PatchCore (butuh model terlatih)",
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Gunakan detektor basic saja (grid + dark_spot), tanpa LED Analyzer",
    )
    parser.add_argument(
        "--analyzer",
        action="store_true",
        help="[DEPRECATED] Sekarang default. Gunakan --basic untuk fallback.",
    )
    parser.add_argument(
        "--no-patchcore",
        action="store_true",
        help="Nonaktifkan patchcore",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Nonaktifkan temporal",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Output minimal",
    )

    args = parser.parse_args()

    if not args.image and not args.frames:
        parser.error("Harus specify --image atau --frames")

    # Setup pipeline
    from src.detectors.led_analyzer import LEDAnalyzer

    # Default: LED Analyzer aktif, grid + dark_spot juga aktif sebagai pelengkap.
    # --basic: nonaktifkan LED Analyzer, hanya grid + dark_spot.
    # --analyzer: deprecated no-op (LED Analyzer sudah default ON).
    use_basic = args.basic

    pipeline = EnsemblePipeline(
        location=args.location,
        use_grid=True,
        use_dark_spot=True,
        use_temporal=not args.no_temporal and args.frames is not None,
        use_patchcore=args.all and not args.no_patchcore,
    )

    # Tambah LED Analyzer (default ON, kecuali --basic)
    if not use_basic:
        config = Config.get_location_config(args.location)
        pipeline._led_analyzer = LEDAnalyzer(config)
        pipeline.use_led_analyzer = True

    if args.image:
        detect_single(pipeline, args.image, verbose=not args.quiet)
    elif args.frames:
        detect_frames(pipeline, args.frames, verbose=not args.quiet)


if __name__ == "__main__":
    main()
