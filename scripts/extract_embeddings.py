#!/usr/bin/env python3
"""Extract and save embeddings from PhoBERT and ViT models."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collate import collate_fn
from src.data.dataset import AdDataset
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def extract_embeddings(config: dict, split: str, batch_size: int = 32):
    """
    Extract and save embeddings for a given split.

    Args:
        config: Configuration dictionary.
        split: Data split ("train", "val", or "test").
        batch_size: Batch size for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load CSV for split - resolve relative paths from project root
    processed_dir = Path(config["paths"]["processed_dir"])
    if not processed_dir.is_absolute():
        project_root = Path(__file__).parent.parent
        processed_dir = project_root / processed_dir
    
    csv_path = processed_dir / "splits" / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {split} split")

    # Create dataset in online mode (raw inputs, not pre-extracted embeddings)
    images_dir = Path(config["paths"]["images_dir"])
    if not images_dir.is_absolute():
        project_root = Path(__file__).parent.parent
        images_dir = project_root / images_dir
    
    # Initialize tokenizer and image transform
    from transformers import AutoTokenizer
    import torchvision.transforms as transforms
    
    tokenizer = AutoTokenizer.from_pretrained(config["encoders"]["text_encoder_name"])
    image_size = config["encoders"].get("image_size", 224)
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AdDataset(
        df=df,
        images_dir=str(images_dir),
        tokenizer=tokenizer,
        image_transform=image_transform,
        metadata_cols=config.get("metadata_features", []),
        split=split,
        offline_embeddings_dir=None,  # Force online mode
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Initialize encoders
    text_encoder = TextEncoder(
        model_name=config["encoders"]["text_encoder_name"],
        freeze=True,
    ).to(device).eval()

    image_encoder = ImageEncoder(
        model_name=config["encoders"]["image_encoder_name"],
        freeze=True,
    ).to(device).eval()

    logger.info(f"Text encoder: {config['encoders']['text_encoder_name']}")
    logger.info(f"Image encoder: {config['encoders']['image_encoder_name']}")

    # Collect embeddings
    all_text_emb = []
    all_image_emb = []
    all_sample_ids = []

    logger.info(f"Extracting embeddings for {split} split...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {split}"):
            # Check if batch is in online mode (has input_ids, pixel_values)
            if "input_ids" in batch:
                # Online mode
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)

                # Forward passes
                text_emb = text_encoder(input_ids, attention_mask)  # (B, 768)
                image_emb = image_encoder(pixel_values)  # (B, 768)
            else:
                # Offline mode (pre-extracted)
                text_emb = batch["text_emb"].to(device)  # (B, 768)
                image_emb = batch["image_emb"].to(device)  # (B, 768)

            all_text_emb.append(text_emb.cpu().numpy())
            all_image_emb.append(image_emb.cpu().numpy())
            all_sample_ids.extend(batch["sample_id"])

    # Concatenate all embeddings
    text_emb_np = np.concatenate(all_text_emb, axis=0)  # (N, 768)
    image_emb_np = np.concatenate(all_image_emb, axis=0)  # (N, 768)
    sample_ids_np = np.array(all_sample_ids)

    logger.info(f"Text embeddings shape: {text_emb_np.shape}")
    logger.info(f"Image embeddings shape: {image_emb_np.shape}")
    logger.info(f"Sample IDs shape: {sample_ids_np.shape}")

    # Save embeddings - resolve relative paths from project root
    embeddings_dir = Path(config["paths"]["embeddings_dir"])
    if not embeddings_dir.is_absolute():
        project_root = Path(__file__).parent.parent
        embeddings_dir = project_root / embeddings_dir
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    text_emb_path = embeddings_dir / f"phobert_{split}.npy"
    image_emb_path = embeddings_dir / f"vit_{split}.npy"
    ids_path = embeddings_dir / f"ids_{split}.npy"

    np.save(text_emb_path, text_emb_np)
    np.save(image_emb_path, image_emb_np)
    np.save(ids_path, sample_ids_np)

    logger.info(f"Saved text embeddings to {text_emb_path}")
    logger.info(f"Saved image embeddings to {image_emb_path}")
    logger.info(f"Saved sample IDs to {ids_path}")


def main():
    """Main entry point for embedding extraction."""
    parser = argparse.ArgumentParser(
        description="Extract and save embeddings from pre-trained models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file (YAML or JSON)",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Data splits to extract embeddings for (can specify multiple)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    # If path is relative, resolve it relative to project root
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix == ".json":
            config = json.load(f)
        else:
            config = yaml.safe_load(f)

    # Setup logging
    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    get_logger("root", log_file=str(log_dir / "embedding_extraction.log"))

    # Set seed
    set_seed(config["data"].get("random_seed", 42))

    logger.info("=" * 70)
    logger.info("Embedding Extraction")
    logger.info("=" * 70)
    logger.info(f"Config: {config_path}")
    logger.info(f"Splits: {args.split}")
    logger.info(f"Batch size: {args.batch_size}")

    # Extract embeddings for each split
    for split in args.split:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'=' * 70}")
        extract_embeddings(config, split, batch_size=args.batch_size)

    logger.info("\n" + "=" * 70)
    logger.info("Embedding extraction complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

