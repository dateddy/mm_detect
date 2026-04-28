#!/usr/bin/env python3
"""Diagnostic script to identify why images are zero tensors during training."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import torch
import yaml
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import AdDataset
from src.data.preprocessing import get_image_transforms
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def diagnose_image_loading(config_path: str, n_samples: int = 20) -> None:
    """
    Diagnose image loading issues by testing first n_samples from train split.

    Args:
        config_path: Path to config YAML file.
        n_samples: Number of samples to test (default: 20).
    """
    logger.info("=" * 80)
    logger.info("IMAGE LOADING DIAGNOSTIC")
    logger.info("=" * 80)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")

    # Setup paths
    processed_dir = Path(config["paths"]["processed_dir"])
    images_dir = Path(config["paths"]["images_dir"])

    logger.info(f"Images directory: {images_dir.resolve()}")
    logger.info(f"Processed directory: {processed_dir.resolve()}")

    # Check if images directory exists
    if not images_dir.exists():
        logger.error(f"❌ Images directory does not exist: {images_dir}")
        return

    # Count actual image files
    image_files = list(images_dir.glob("*.*"))
    logger.info(f"Found {len(image_files)} image files in {images_dir}")
    if len(image_files) == 0:
        logger.error(f"❌ NO IMAGE FILES FOUND in {images_dir}")
        logger.error("This is the root cause — the image directory is empty or path is wrong.")
        return

    # Show sample filenames
    logger.info(f"Sample image files (first 5):")
    for img_file in image_files[:5]:
        logger.info(f"  - {img_file.name}")

    # Load train CSV
    train_csv_path = processed_dir / "splits" / "train.csv"
    if not train_csv_path.exists():
        logger.error(f"❌ Train CSV not found: {train_csv_path}")
        return

    # Force 'id' column to be read as string to prevent scientific notation
    train_df = pd.read_csv(train_csv_path, dtype={"id": str})
    logger.info(f"Loaded train CSV with {len(train_df)} rows")

    # Show sample ad_ids
    logger.info(f"Sample IDs from CSV (first 5):")
    for ad_id in train_df["id"].head(5).values:
        logger.info(f"  - {ad_id}")

    # Try to detect file extension mismatch
    logger.info("=" * 80)
    logger.info("Checking for file extension mismatches...")
    logger.info("=" * 80)

    image_extensions: Set[str] = set()
    for img_file in image_files:
        ext = img_file.suffix.lower()
        image_extensions.add(ext)

    logger.info(f"File extensions found in {images_dir}: {sorted(image_extensions)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["encoders"]["text_encoder_name"])

    # Load transforms
    image_size = config["encoders"].get("image_size", 224)
    image_transform = get_image_transforms("val", image_size)

    # Create dataset
    try:
        dataset = AdDataset(
            df=train_df.head(n_samples),
            images_dir=str(images_dir),
            tokenizer=tokenizer,
            image_transform=image_transform,  # Use val transform (no augmentation)
            metadata_cols=config.get("metadata_features", []),
            split="train",
            offline_embeddings_dir=None,  # Force online mode to test image loading
        )
        logger.info(f"Created AdDataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"❌ Failed to create dataset: {e}")
        return

    # Test first n_samples
    logger.info("=" * 80)
    logger.info(f"Testing first {min(n_samples, len(dataset))} samples...")
    logger.info("=" * 80)

    zero_count = 0
    missing_count = 0
    extension_mismatch_count = 0

    for i in range(min(n_samples, len(dataset))):
        try:
            sample = dataset[i]
            sample_id = sample["sample_id"]
            pixel_values = sample["pixel_values"]
            missing_image = sample["missing_image"]

            # Check pixel stats
            pv_sum = pixel_values.sum().item()
            pv_mean = pixel_values.mean().item()
            pv_std = pixel_values.std().item()

            # Try to find what files exist
            found_ext = None
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                candidate = images_dir / f"{sample_id}{ext}"
                if candidate.exists():
                    found_ext = ext
                    break

            logger.info(
                f"[Sample {i}] "
                f"id={sample_id} | "
                f"shape={pixel_values.shape} | "
                f"sum={pv_sum:.6f} | "
                f"mean={pv_mean:.6f} | "
                f"std={pv_std:.6f} | "
                f"missing={missing_image} | "
                f"found_ext={found_ext}"
            )

            if pv_sum == 0:
                zero_count += 1
            if missing_image:
                missing_count += 1
            if missing_image and found_ext is not None:
                extension_mismatch_count += 1

        except Exception as e:
            logger.error(f"[Sample {i}] ERROR: {e}")

    # Print diagnosis
    logger.info("=" * 80)
    logger.info("DIAGNOSIS")
    logger.info("=" * 80)

    if zero_count == n_samples:
        logger.error(
            f"❌ ALL {n_samples} samples have pixel_values.sum() == 0"
        )
        if missing_count == n_samples:
            logger.error(
                "  → ALL samples marked as missing_image=True"
            )
            if extension_mismatch_count > 0:
                logger.error(
                    f"  → But {extension_mismatch_count} files exist on disk with correct id"
                )
                logger.error(
                    "  → ROOT CAUSE: File extension mismatch"
                )
                logger.error(
                    f"     - CSV expects: {sample_id}.png/.jpg/.jpeg"
                )
                logger.error(
                    f"     - Actual files: {sorted(image_extensions)}"
                )
            else:
                logger.error(
                    "  → And NO files found on disk with matching ids"
                )
                logger.error(
                    "  → ROOT CAUSE: Images directory is wrong or files not copied"
                )
        else:
            logger.error(
                f"  → {missing_count}/{n_samples} marked as missing, but pixel_values still zero"
            )
            logger.error(
                "  → ROOT CAUSE: Likely bug in _load_image() fallback or transforms"
            )
    elif zero_count > 0:
        logger.warning(
            f"⚠ {zero_count}/{n_samples} samples have pixel_values.sum() == 0"
        )
        logger.warning(
            "  This is expected when images are missing"
        )
    else:
        logger.info(
            f"✓ All {n_samples} samples have non-zero pixel_values"
        )
        logger.info(
            "  If training still shows zero norms, the bug is downstream:"
        )
        logger.info(
            "  - collate_fn might be zeroing tensors"
        )
        logger.info(
            "  - Device transfer might be broken"
        )
        logger.info(
            "  - Model forward() might be zeroing embeddings"
        )

    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose image loading issues"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config YAML file (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of samples to test (default: 20)",
    )
    args = parser.parse_args()

    diagnose_image_loading(args.config, args.n_samples)
