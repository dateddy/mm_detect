# src/models/projection.py
"""Modality projection and dropout layers."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModalityProjection(nn.Module):
    """
    Linear projection layer with layer normalization for modality embeddings.

    Projects embeddings from their native dimension to a common projection dimension
    and applies layer normalization for stability.

    Architecture:
    - Linear(in_dim, out_dim)
    - LayerNorm(out_dim)

    Attributes:
        in_dim: Input embedding dimension (e.g., 768)
        out_dim: Output projection dimension (default: 256)
    """

    def __init__(self, in_dim: int, out_dim: int = 256):
        """
        Initialize ModalityProjection.

        Args:
            in_dim: Input embedding dimension.
            out_dim: Output projection dimension (default: 256).
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

        logger.debug(f"Initialized ModalityProjection ({in_dim} → {out_dim})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project and normalize embeddings.

        Args:
            x: Input embeddings, shape (batch_size, in_dim).

        Returns:
            Projected embeddings, shape (batch_size, out_dim).
        """
        x = self.linear(x)
        x = self.layer_norm(x)
        return x


class ModalityDropout(nn.Module):
    """
    Per-modality dropout that zeroes entire embedding vectors.

    During training, randomly masks complete embeddings along the batch dimension
    with probability p. Unlike standard dropout which zeros individual dimensions,
    this zeros entire (B, embed_dim) vectors for missing or unreliable modalities.

    Handles missing modalities gracefully: if an embedding is already all-zeros
    (indicating a missing modality), it is not re-zeroed.

    Attributes:
        p: Dropout probability (default: 0.15).
    """

    def __init__(self, p: float = 0.15):
        """
        Initialize ModalityDropout.

        Args:
            p: Dropout probability for zeroing modalities (default: 0.15).
        """
        super().__init__()
        self.p = p
        logger.debug(f"Initialized ModalityDropout (p={p})")

    def forward(self, *embeddings: torch.Tensor) -> tuple:
        """
        Apply dropout masks to embeddings.

        During training: independently zero each embedding vector with probability p.
        During eval: return embeddings unchanged.

        Args:
            *embeddings: Variable number of embedding tensors, each shape (batch_size, embed_dim).

        Returns:
            Tuple of masked embeddings, same shapes as inputs.
        """
        if not self.training or self.p == 0.0:
            return embeddings

        masked_embeddings = []

        for emb in embeddings:
            if emb is None:
                masked_embeddings.append(emb)
                continue

            batch_size = emb.shape[0]
            device = emb.device

            # Create Bernoulli mask of shape (B, 1)
            # 1 = keep, 0 = drop
            mask = torch.bernoulli(
                torch.full((batch_size, 1), 1.0 - self.p, device=device)
            )

            # Check if embedding is already all-zeros (missing modality)
            # If so, don't re-zero it
            is_missing = (emb == 0).all(dim=1, keepdim=True).float()

            # Apply mask only to non-missing embeddings
            effective_mask = mask * (1.0 - is_missing) + is_missing

            # Apply mask: (B, 1) broadcasts over (B, D)
            masked_emb = emb * effective_mask

            masked_embeddings.append(masked_emb)

        return tuple(masked_embeddings)
