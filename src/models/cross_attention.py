# src/models/cross_attention.py
"""Dual cross-attention module for multimodal fusion."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DualCrossAttention(nn.Module):
    """
    Bidirectional cross-attention module for multimodal fusion.

    Performs two attention branches:
    - Text attends to Image + Metadata
    - Image attends to Text + Metadata

    Each modality serves as query while the other modalities are keys and values.

    Attributes:
        embed_dim: Embedding dimension (default: 256).
        num_heads: Number of attention heads (default: 8).
        dropout: Dropout probability for attention (default: 0.1).
    """

    def __init__(
        self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1
    ):
        """
        Initialize DualCrossAttention.

        Args:
            embed_dim: Embedding dimension for all modalities (default: 256).
            num_heads: Number of attention heads (default: 8).
            dropout: Dropout probability for attention weights (default: 0.1).
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Branch A: Text queries Image + Metadata
        self.attn_text_to_image = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Branch B: Image queries Text + Metadata
        self.attn_image_to_text = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        logger.info(
            f"Initialized DualCrossAttention "
            f"(embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout})"
        )

    def forward(
        self,
        t_proj: torch.Tensor,
        i_proj: torch.Tensor,
        m_proj: torch.Tensor | None = None,
    ) -> tuple:
        """
        Apply bidirectional cross-attention.

        Args:
            t_proj: Text embeddings, shape (batch_size, 256).
            i_proj: Image embeddings, shape (batch_size, 256).
            m_proj: Optional metadata embeddings, shape (batch_size, 256).

        Returns:
            Tuple of (t_prime, i_prime) where:
            - t_prime: Text after attending to Image(+Metadata), shape (batch_size, 256)
            - i_prime: Image after attending to Text(+Metadata), shape (batch_size, 256)
        """
        # Branch A: Text → Image + Metadata
        #   Query: Text (B, 1, 256)
        #   Key/Value: [Image, Metadata] (B, 2, 256)
        q_text = t_proj.unsqueeze(1)  # (B, 1, 256)
        if m_proj is None:
            kv_image_metadata = i_proj.unsqueeze(1)  # (B, 1, 256)
        else:
            kv_image_metadata = torch.stack([i_proj, m_proj], dim=1)  # (B, 2, 256)

        t_prime, _ = self.attn_text_to_image(
            q_text, kv_image_metadata, kv_image_metadata
        )
        t_prime = t_prime.squeeze(1)  # (B, 256)

        # Branch B: Image → Text + Metadata
        #   Query: Image (B, 1, 256)
        #   Key/Value: [Text, Metadata] (B, 2, 256)
        q_image = i_proj.unsqueeze(1)  # (B, 1, 256)
        if m_proj is None:
            kv_text_metadata = t_proj.unsqueeze(1)  # (B, 1, 256)
        else:
            kv_text_metadata = torch.stack([t_proj, m_proj], dim=1)  # (B, 2, 256)

        i_prime, _ = self.attn_image_to_text(
            q_image, kv_text_metadata, kv_text_metadata
        )
        i_prime = i_prime.squeeze(1)  # (B, 256)

        return t_prime, i_prime
