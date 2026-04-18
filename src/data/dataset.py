# src/data/dataset.py
"""Dataset class for multimodal ad misinformation detection."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class AdDataset(Dataset):
    """
    PyTorch Dataset for multimodal ad misinformation detection.

    Supports two modes:
    1. Pre-extracted embeddings (offline): Loads pre-extracted text/image embeddings from .npy files.
    2. Online processing: Tokenizes text on-the-fly and loads/transforms images.

    Attributes:
        df: DataFrame with columns ['ad_id', 'ad_creativity_body', 'metadata_features', 'misinformation', ...]
        images_dir: Directory containing ad images
        tokenizer: HuggingFace tokenizer for text encoding
        image_transform: torchvision transform for image preprocessing
        metadata_cols: List of metadata feature column names
        split: Dataset split name ('train', 'val', 'test')
        offline_embeddings_dir: Optional directory containing pre-extracted embeddings
        use_offline_embeddings: Whether to use pre-extracted embeddings (determined by file existence)
        text_embeddings: Pre-loaded text embeddings (if offline mode)
        image_embeddings: Pre-loaded image embeddings (if offline mode)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        tokenizer: PreTrainedTokenizer,
        image_transform,
        metadata_cols: List[str],
        split: str,
        offline_embeddings_dir: Optional[str] = None,
    ):
        """
        Initialize AdDataset.

        Args:
            df: DataFrame with ad data and labels.
            images_dir: Directory containing ad images (subdirectory or flat).
            tokenizer: HuggingFace tokenizer for text encoding.
            image_transform: torchvision.transforms.Compose for image preprocessing.
            metadata_cols: List of metadata feature column names (e.g., ["ads_per_page", ...]).
            split: Dataset split identifier ('train', 'val', 'test').
            offline_embeddings_dir: Optional directory containing pre-extracted embeddings
                                   in format expected by this split.
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.metadata_cols = metadata_cols
        self.split = split
        self.use_offline_embeddings = False
        self.text_embeddings = None
        self.image_embeddings = None

        # Attempt to load pre-extracted embeddings if provided
        if offline_embeddings_dir:
            self._try_load_offline_embeddings(offline_embeddings_dir)

        logger.info(
            f"Initialized AdDataset (split={split}, n={len(self.df)}, "
            f"mode={'offline' if self.use_offline_embeddings else 'online'})"
        )

    def _try_load_offline_embeddings(self, offline_embeddings_dir: str) -> None:
        """
        Attempt to load pre-extracted embeddings from directory.

        Expected naming convention:
        - {split}_text_embeddings.npy (shape N x 768)
        - {split}_image_embeddings.npy (shape N x 768)

        Args:
            offline_embeddings_dir: Directory containing embedding files.
        """
        emb_dir = Path(offline_embeddings_dir)

        text_emb_path = emb_dir / f"{self.split}_text_embeddings.npy"
        image_emb_path = emb_dir / f"{self.split}_image_embeddings.npy"

        try:
            if text_emb_path.exists() and image_emb_path.exists():
                self.text_embeddings = np.load(text_emb_path)
                self.image_embeddings = np.load(image_emb_path)

                # Validate shapes
                if len(self.text_embeddings) != len(self.df):
                    logger.warning(
                        f"Text embeddings length {len(self.text_embeddings)} "
                        f"does not match df length {len(self.df)}"
                    )
                    self.text_embeddings = None
                    self.image_embeddings = None
                    return

                if len(self.image_embeddings) != len(self.df):
                    logger.warning(
                        f"Image embeddings length {len(self.image_embeddings)} "
                        f"does not match df length {len(self.df)}"
                    )
                    self.text_embeddings = None
                    self.image_embeddings = None
                    return

                self.use_offline_embeddings = True
                logger.info(
                    f"Loaded offline embeddings: text {self.text_embeddings.shape}, "
                    f"image {self.image_embeddings.shape}"
                )
            else:
                logger.debug(
                    f"Embedding files not found in {emb_dir} "
                    f"(looking for {self.split}_*.npy). Using online mode."
                )
        except Exception as e:
            logger.warning(f"Failed to load offline embeddings: {e}. Using online mode.")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
            - 'input_ids': Token IDs (online mode) or 'text_emb' (offline mode)
            - 'attention_mask': Attention mask (online mode) or omitted (offline mode)
            - 'pixel_values': Image tensor (online mode) or 'image_emb' (offline mode)
            - 'metadata': Metadata feature tensor, shape (9,)
            - 'label': Binary misinformation label, scalar
            - 'missing_image': Boolean flag indicating if image was missing
            - 'sample_id': Sample identifier string
        """
        row = self.df.iloc[idx]
        sample_id = str(row.get("ad_id", idx))
        missing_image = False

        if self.use_offline_embeddings:
            # Offline embeddings mode
            text_emb = torch.tensor(
                self.text_embeddings[idx], dtype=torch.float32
            )
            image_emb = torch.tensor(
                self.image_embeddings[idx], dtype=torch.float32
            )

            return {
                "text_emb": text_emb,
                "image_emb": image_emb,
                "metadata": self._get_metadata(idx),
                "label": torch.tensor(float(row["misinformation"]), dtype=torch.float32),
                "missing_image": False,
                "sample_id": sample_id,
            }
        else:
            # Online mode: tokenize text and load image
            # Get text
            text = str(row.get("ad_creative_body", ""))
            encoding = self.tokenizer(
                text,
                max_length=256,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

            # Get image
            pixel_values = self._load_image(idx)
            missing_image = pixel_values is None

            if pixel_values is None:
                # Return zero tensor if image missing
                # Assuming image_size=224 and 3 channels
                pixel_values = torch.zeros((3, 224, 224), dtype=torch.float32)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "metadata": self._get_metadata(idx),
                "label": torch.tensor(
                    float(row["misinformation"]), dtype=torch.float32
                ),
                "missing_image": missing_image,
                "sample_id": sample_id,
            }

    def _load_image(self, idx: int) -> Optional[torch.Tensor]:
        """
        Load and transform image for sample.

        Args:
            idx: Sample index.

        Returns:
            Transformed image tensor (3, 224, 224) or None if image not found.
        """
        row = self.df.iloc[idx]
        ad_id = str(row.get("ad_id", idx))

        # Try multiple possible image path patterns
        possible_paths = [
            self.images_dir / f"{ad_id}.png",
            self.images_dir / f"{ad_id}.jpg",
            self.images_dir / f"{ad_id}.jpeg",
        ]

        for image_path in possible_paths:
            if image_path.exists():
                try:
                    image = Image.open(image_path).convert("RGB")
                    pixel_values = self.image_transform(image)
                    return pixel_values
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
                    return None

        logger.debug(f"Image not found for ad_id={ad_id}")
        return None

    def _get_metadata(self, idx: int) -> torch.Tensor:
        """
        Get metadata features for sample.

        Args:
            idx: Sample index.

        Returns:
            Metadata tensor, shape (len(metadata_cols),).
        """
        row = self.df.iloc[idx]

        # Extract metadata features, fill missing with 0
        metadata_values = []
        for col in self.metadata_cols:
            value = row.get(col, 0.0)
            # Handle NaN
            if pd.isna(value):
                value = 0.0
            metadata_values.append(float(value))

        return torch.tensor(metadata_values, dtype=torch.float32)


def create_datasets(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    images_dir: str,
    tokenizer: PreTrainedTokenizer,
    image_transforms: Dict[str, object],
    metadata_cols: List[str],
    offline_embeddings_dir: Optional[str] = None,
) -> tuple:
    """
    Create train, val, and test AdDataset instances.

    Args:
        train_csv: Path to training split CSV.
        val_csv: Path to validation split CSV.
        test_csv: Path to test split CSV.
        images_dir: Directory containing ad images.
        tokenizer: HuggingFace tokenizer.
        image_transforms: Dict mapping split names to image transforms.
        metadata_cols: List of metadata feature column names.
        offline_embeddings_dir: Optional directory with pre-extracted embeddings.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_dataset = AdDataset(
        df=train_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        image_transform=image_transforms.get("train"),
        metadata_cols=metadata_cols,
        split="train",
        offline_embeddings_dir=offline_embeddings_dir,
    )

    val_dataset = AdDataset(
        df=val_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        image_transform=image_transforms.get("val"),
        metadata_cols=metadata_cols,
        split="val",
        offline_embeddings_dir=offline_embeddings_dir,
    )

    test_dataset = AdDataset(
        df=test_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        image_transform=image_transforms.get("test"),
        metadata_cols=metadata_cols,
        split="test",
        offline_embeddings_dir=offline_embeddings_dir,
    )

    return train_dataset, val_dataset, test_dataset
