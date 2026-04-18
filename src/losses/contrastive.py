# src/losses/contrastive.py
"""Contrastive loss for multimodal alignment."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for aligning text and image embeddings.

    Encourages text and image embeddings from the same ad to be similar (positive pairs)
    while being dissimilar to embeddings from different ads (negative pairs).

    The loss is symmetric: computes both text→image and image→text directions,
    then averages.

    Attributes:
        temperature: Temperature parameter for scaled similarity (default: 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCELoss.

        Args:
            temperature: Temperature for scaled similarity (default: 0.07).
                        Larger values soften the softmax; smaller values sharpen it.
        """
        super().__init__()
        self.temperature = temperature
        logger.info(f"Initialized InfoNCELoss (temperature={temperature})")

    def forward(
        self, text_emb: torch.Tensor, image_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss between text and image embeddings.

        Args:
            text_emb: Text embeddings, shape (batch_size, embedding_dim).
            image_emb: Image embeddings, shape (batch_size, embedding_dim).

        Returns:
            Scalar loss tensor.
        """
        batch_size = text_emb.shape[0]

        # Handle batch size of 1 (no contrastive pairs possible)
        if batch_size == 1:
            logger.debug("Batch size is 1, returning zero loss for contrastive term")
            return torch.tensor(0.0, device=text_emb.device, dtype=text_emb.dtype)

        # L2 normalize embeddings
        text_emb = F.normalize(text_emb, p=2, dim=-1)  # (B, D)
        image_emb = F.normalize(image_emb, p=2, dim=-1)  # (B, D)

        # Compute cosine similarity matrix
        # Shape: (B, B) where similarity[i, j] = text_i · image_j
        similarity = torch.matmul(text_emb, image_emb.t()) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=text_emb.device)

        # Direction 1: Text queries Image embeddings
        # Loss over each row (text perspective)
        loss_text_to_image = F.cross_entropy(similarity, labels)

        # Direction 2: Image queries Text embeddings
        # Loss over each column (image perspective)
        # Equivalent to transposing the similarity matrix
        loss_image_to_text = F.cross_entropy(similarity.t(), labels)

        # Average both directions
        loss = (loss_text_to_image + loss_image_to_text) / 2.0

        return loss
