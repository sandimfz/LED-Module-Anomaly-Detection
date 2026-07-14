#!/usr/bin/env python3
"""
Training Script untuk PatchCore.

Cara pakai:
    python train.py --location sigma_cirebon
    python train.py --location lengkong
"""

import argparse
import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

from src.core.config import Config


def main(location: str, dataset_path: str | None = None) -> None:
    """Training PatchCore untuk lokasi tertentu.

    Args:
        location: Nama lokasi.
        dataset_path: Custom dataset path (override default).
    """
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    if dataset_path:
        data_dir = Path(dataset_path)
    else:
        data_dir = Config.get_dataset_path(location)
    model_dir = Config.get_model_path(location)

    if not (data_dir / "good").exists():
        print(f"Error: Folder {data_dir}/good tidak ditemukan.")
        sys.exit(1)

    print(f"Training PatchCore untuk lokasi: {location}")
    print(f"Dataset: {data_dir}")
    print(f"Model output: {model_dir}")
    print("-" * 50)

    # Setup datamodule
    datamodule = Folder(
        name=f"led_module_{location}",
        root=str(data_dir),
        normal_dir="good",
        abnormal_dir="bad" if (data_dir / "bad").exists() else None,
        normal_split_ratio=0.2,
        val_split_mode="from_train",
    )

    # Setup model
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )

    # Setup engine
    engine = Engine(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        default_root_dir=str(model_dir),
    )

    # Training
    print("\n[1/2] Training (membangun memory bank)...")
    engine.fit(model=model, datamodule=datamodule)
    print("Training selesai!")

    # Evaluasi
    if datamodule.abnormal_dir is not None:
        print("\n[2/2] Evaluasi pada test set...")
        results = engine.test(model=model, datamodule=datamodule)
        print(f"Hasil: {results}")
    else:
        print("\n[2/2] Skipping evaluasi (tidak ada folder 'bad')")

    print("\n" + "=" * 50)
    print(f"Model tersimpan di: {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PatchCore untuk LED anomaly detection"
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="Nama lokasi (folder di led_dataset/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Custom dataset path (default: led_dataset/<location>)",
    )
    args = parser.parse_args()
    main(args.location, args.dataset)
