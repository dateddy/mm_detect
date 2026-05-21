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
        ablation_mode: str = "full",
        text_cols: Optional[List[str]] = None,
        max_text_len: int = 256,
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
            offline_embeddings_dir: Optional directory containing pre-extracted embeddings.
            ablation_mode: Model ablation mode. Used to skip loading unused modalities (memory savings).
            text_cols: Text columns to concatenate for online tokenization.
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.metadata_cols = metadata_cols
        self.split = split
        self.text_cols = text_cols or ["ad_creative_bodies", "ad_creative_link_titles"]
        self.max_text_len = max_text_len
        self.use_offline_embeddings = False
        self.text_embeddings = None
        self.image_embeddings = None
        
        # === NEW: PROMPT 4 — Skip loading unused modalities based on ablation_mode ===
        self.load_text = ablation_mode in {
            "full", "full_no_contrastive", "full_no_modality_dropout",
            "full_no_dropout", "full_no_metadata_in_fusion",
            "full_no_attention", "full_no_gating",
            "text_only", "text_image", "text_metadata",
        }
        self.load_image = ablation_mode in {
            "full", "full_no_contrastive", "full_no_modality_dropout",
            "full_no_dropout", "full_no_metadata_in_fusion",
            "full_no_attention", "full_no_gating",
            "image_only", "text_image", "image_metadata",
        }
        self.load_metadata = ablation_mode in {
            "full", "full_no_contrastive", "full_no_modality_dropout",
            "full_no_dropout", "full_no_metadata_in_fusion",
            "full_no_attention", "full_no_gating",
            "metadata_only", "text_metadata", "image_metadata",
        }
        
        # Warning counter for missing images (log first 10 per epoch)
        self._missing_warning_count = 0

        # Assert that images directory exists and contains files
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory does not exist: {self.images_dir.resolve()}"
            )
        
        image_files = list(self.images_dir.glob("*.*"))
        if len(image_files) == 0:
            raise FileNotFoundError(
                f"No image files found in {self.images_dir.resolve()}. "
                f"Check the path in configs/base.yaml → paths.images_dir"
            )

        logger.info(
            f"AdDataset[{split}]: images_dir={self.images_dir.resolve()} "
            f"contains {len(image_files)} image files"
        )

        # Attempt to load pre-extracted embeddings if provided
        if offline_embeddings_dir:
            self._try_load_offline_embeddings(offline_embeddings_dir)

        logger.info(
            f"Initialized AdDataset (split={split}, n={len(self.df)}, "
            f"ablation={ablation_mode}, load_text={self.load_text}, "
            f"load_image={self.load_image}, load_metadata={self.load_metadata}, "
            f"mode={'offline' if self.use_offline_embeddings else 'online'})"
        )

    def _try_load_offline_embeddings(self, offline_embeddings_dir: str) -> None:
        """
        Attempt to load pre-extracted embeddings from directory.

        Tries multiple naming conventions:
        1. Standard: {split}_text_embeddings.npy, {split}_image_embeddings.npy
        2. Alternative: phobert_{split}.npy, vit_{split}.npy

        Args:
            offline_embeddings_dir: Directory containing embedding files.
        """
        emb_dir = Path(offline_embeddings_dir)

        # Try standard naming first
        text_emb_path = emb_dir / f"{self.split}_text_embeddings.npy"
        image_emb_path = emb_dir / f"{self.split}_image_embeddings.npy"

        # If standard naming not found, try alternative naming
        if not text_emb_path.exists() or not image_emb_path.exists():
            text_emb_path = emb_dir / f"phobert_{self.split}.npy"
            image_emb_path = emb_dir / f"vit_{self.split}.npy"

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

                if self.load_text and float(np.mean(np.std(self.text_embeddings, axis=0))) < 1e-6:
                    logger.warning(
                        f"Text embeddings in {text_emb_path} are effectively constant across rows. "
                        "Ignoring cached embeddings and using online tokenization."
                    )
                    self.text_embeddings = None
                    self.image_embeddings = None
                    return

                if self.load_image and float(np.mean(np.std(self.image_embeddings, axis=0))) < 1e-6:
                    logger.warning(
                        f"Image embeddings in {image_emb_path} are effectively constant across rows. "
                        "Ignoring cached embeddings and using online image encoding."
                    )
                    self.text_embeddings = None
                    self.image_embeddings = None
                    return

                self.use_offline_embeddings = True
                logger.info(
                    f"Loaded offline embeddings from {text_emb_path.name}, {image_emb_path.name}: "
                    f"text {self.text_embeddings.shape}, image {self.image_embeddings.shape}"
                )
            else:
                logger.debug(
                    f"Embedding files not found in {emb_dir}. Using online mode. "
                    f"(looked for {self.split}_*.npy and phobert/vit_{self.split}.npy)"
                )
        except Exception as e:
            logger.warning(f"Failed to load offline embeddings: {e}. Using online mode.")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, bool]]:
        """
        Get a single sample.
        
        Skips loading modalities that are not needed for the current ablation_mode (memory savings).

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
            - 'input_ids': Token IDs (online mode, if load_text=True) or omitted
            - 'attention_mask': Attention mask (online mode, if load_text=True) or omitted
            - 'pixel_values': Image tensor (online mode, if load_image=True) or omitted
            - 'metadata': Metadata feature tensor, shape (N_features,) (if load_metadata=True) or omitted
            - 'label': Binary misinformation label, scalar
            - 'missing_image': Boolean flag indicating if image was missing (if load_image=True) or omitted
            - 'sample_id': Sample identifier string
        """
        row = self.df.iloc[idx]
        sample_id = str(row.get("id", idx))

        if self.use_offline_embeddings:
            # Offline embeddings mode
            sample = {
                "label": torch.tensor(float(row["misinformation"]), dtype=torch.float32),
                "sample_id": sample_id,
            }
            
            if self.load_text:
                text_emb = torch.tensor(
                    self.text_embeddings[idx], dtype=torch.float32
                )
                sample["text_emb"] = text_emb
            
            if self.load_image:
                image_emb = torch.tensor(
                    self.image_embeddings[idx], dtype=torch.float32
                )
                sample["image_emb"] = image_emb
            
            if self.load_metadata:
                sample["metadata"] = self._get_metadata(idx)
            
            sample["missing_image"] = False
            return sample
            
        else:
            # Online mode: tokenize text and load image (if needed)
            sample = {
                "label": torch.tensor(
                    float(row["misinformation"]), dtype=torch.float32
                ),
                "sample_id": sample_id,
            }
            
            if self.load_text:
                # Get text
                text_parts = []
                for col in self.text_cols:
                    value = row.get(col, "")
                    if pd.notna(value):
                        text_parts.append(str(value))
                text = " ".join(part for part in text_parts if part.strip())
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_text_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                sample["input_ids"] = encoding["input_ids"].squeeze(0)
                sample["attention_mask"] = encoding["attention_mask"].squeeze(0)
            
            if self.load_image:
                # Get image
                pixel_values = self._load_image(idx)
                missing_image = pixel_values is None

                if pixel_values is None:
                    # Return zero tensor if image missing
                    pixel_values = torch.zeros((3, 224, 224), dtype=torch.float32)
                
                sample["pixel_values"] = pixel_values
                sample["missing_image"] = missing_image
            else:
                # Image not needed for this ablation
                sample["missing_image"] = False
            
            if self.load_metadata:
                sample["metadata"] = self._get_metadata(idx)
            
            return sample

    def _load_image(self, idx: int) -> Optional[torch.Tensor]:
        """
        Load and transform image for sample.

        Tries multiple file extensions (.png, .jpg, .jpeg, .webp) before giving up.
        Logs warnings (first 10 per epoch) when images are missing.

        Args:
            idx: Sample index.

        Returns:
            Transformed image tensor (3, 224, 224) or None if image not found after
            trying all extensions.
        """
        row = self.df.iloc[idx]
        # Use 'id' column from the CSV (not 'ad_id')
        sample_id = str(row.get("id", idx))

        # Try multiple file extensions
        extensions = [".png", ".jpg", ".jpeg", ".webp"]
        image_path = None

        for ext in extensions:
            candidate = self.images_dir / f"{sample_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            # Image not found after trying all extensions
            if self._missing_warning_count < 10:
                logger.warning(
                    f"Image not found for sample {sample_id} "
                    f"(tried .png/.jpg/.jpeg/.webp in {self.images_dir.resolve()}). "
                    f"Substituting zero tensor. (Warning {self._missing_warning_count + 1}/10)"
                )
                self._missing_warning_count += 1
            return None

        # Image file found, try to load it
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_transform(image)
            return pixel_values
        except Exception as e:
            logger.warning(
                f"Failed to load image {image_path}: {e}. "
                f"Substituting zero tensor."
            )
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
    ablation_mode: str = "full",
    text_cols: Optional[List[str]] = None,
    max_text_len: int = 256,
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
        text_cols: Text columns to concatenate for online tokenization.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    # Force 'id' column to be read as string to prevent scientific notation issues
    train_df = pd.read_csv(train_csv, dtype={"id": str})
    val_df = pd.read_csv(val_csv, dtype={"id": str})
    test_df = pd.read_csv(test_csv, dtype={"id": str})

    train_dataset = AdDataset(
        df=train_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        image_transform=image_transforms.get("train"),
        metadata_cols=metadata_cols,
        split="train",
        offline_embeddings_dir=offline_embeddings_dir,
        ablation_mode=ablation_mode,
        text_cols=text_cols,
        max_text_len=max_text_len,
    )

    val_dataset = AdDataset(
        df=val_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        image_transform=image_transforms.get("val"),
        metadata_cols=metadata_cols,
        split="val",
        offline_embeddings_dir=offline_embeddings_dir,
        ablation_mode=ablation_mode,
        text_cols=text_cols,
        max_text_len=max_text_len,
    )

    test_dataset = AdDataset(
        df=test_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        image_transform=image_transforms.get("test"),
        metadata_cols=metadata_cols,
        split="test",
        offline_embeddings_dir=offline_embeddings_dir,
        ablation_mode=ablation_mode,
        text_cols=text_cols,
        max_text_len=max_text_len,
    )

    return train_dataset, val_dataset, test_dataset
