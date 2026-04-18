# src/data/collate.py
"""Collate functions for batching multimodal samples."""

import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate samples from AdDataset into a single batch.

    Handles both modes transparently:
    - Offline embeddings mode: Batches pre-extracted text_emb and image_emb
    - Online mode: Batches input_ids, attention_mask, and pixel_values

    All tensors are stacked along batch dimension (dim=0).

    Args:
        batch: List of dictionaries from AdDataset.__getitem__()

    Returns:
        Dictionary with batched tensors and metadata:
        - 'input_ids' or 'text_emb': (batch_size, ...) tensor
        - 'attention_mask' or omitted in offline mode
        - 'pixel_values' or 'image_emb': (batch_size, ...) tensor
        - 'metadata': (batch_size, num_features) tensor
        - 'label': (batch_size,) tensor
        - 'missing_image': (batch_size,) boolean array
        - 'sample_id': List of sample IDs (not stacked)
    """
    if not batch:
        raise ValueError("Batch cannot be empty")

    # Detect mode from first sample
    is_offline = "text_emb" in batch[0]

    batched = {}

    if is_offline:
        # Offline embeddings mode: Stack text_emb and image_emb
        batched["text_emb"] = torch.stack(
            [sample["text_emb"] for sample in batch], dim=0
        )
        batched["image_emb"] = torch.stack(
            [sample["image_emb"] for sample in batch], dim=0
        )
    else:
        # Online mode: Stack input_ids, attention_mask, pixel_values
        batched["input_ids"] = torch.stack(
            [sample["input_ids"] for sample in batch], dim=0
        )
        batched["attention_mask"] = torch.stack(
            [sample["attention_mask"] for sample in batch], dim=0
        )
        batched["pixel_values"] = torch.stack(
            [sample["pixel_values"] for sample in batch], dim=0
        )

    # Stack metadata features
    batched["metadata"] = torch.stack(
        [sample["metadata"] for sample in batch], dim=0
    )

    # Stack labels
    batched["label"] = torch.stack([sample["label"] for sample in batch], dim=0)

    # Store missing_image flags (not stacked)
    batched["missing_image"] = [sample["missing_image"] for sample in batch]

    # Collect sample IDs (not stacked)
    batched["sample_id"] = [sample["sample_id"] for sample in batch]

    return batched
