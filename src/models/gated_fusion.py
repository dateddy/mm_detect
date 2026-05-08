# src/models/gated_fusion.py
"""Gated fusion module for combining multimodal embeddings."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GatedFusion(nn.Module):
    """
    Gated fusion layer for combining multimodal embeddings.

    Learns sample-specific per-modality weights to combine text, image, and metadata
    embeddings. Uses a gating mechanism to compute attention-like weights.

    Architecture:
    1. Concatenate all attended embeddings: (B, 768)
    2. Gate MLP predicts per-modality weights: (B, 768)
    3. Split weights into three (B, 256) components
    4. Weighted combination with residual connection
    5. Layer normalization

    Attributes:
        dim: Embedding dimension per modality (default: 256).
    """

    def __init__(self, dim: int = 256):
        """
        Initialize GatedFusion.

        Args:
            dim: Embedding dimension per modality (default: 256).
        """
        super().__init__()

        self.dim = dim

        # Gate layer: (dim*3) → (dim*3)
        # Takes concatenated embeddings and outputs gating weights
        self.gate_layer = nn.Linear(dim * 3, dim * 3)
        self.sigmoid = nn.Sigmoid()

        # Layer normalization for output
        self.layer_norm = nn.LayerNorm(dim)

        logger.info(f"Initialized GatedFusion (dim={dim})")

    def forward(
        self,
        t_prime: torch.Tensor,
        i_prime: torch.Tensor,
        m_proj: torch.Tensor | None,
        t_proj: torch.Tensor,
        i_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse attended embeddings with gating and residual connection.

        Args:
            t_prime: Text embeddings after cross-attention, shape (batch_size, 256).
            i_prime: Image embeddings after cross-attention, shape (batch_size, 256).
            m_proj: Optional metadata embeddings (projected), shape (batch_size, 256).
            t_proj: Original text embeddings (for residual), shape (batch_size, 256).
            i_proj: Original image embeddings (for residual), shape (batch_size, 256).

        Returns:
            Fused embeddings, shape (batch_size, 256).
        """
        if m_proj is None:
            # Parameter-controlled metadata ablation: keep the 3-way gate,
            # but feed zero metadata so no metadata signal reaches fusion.
            m_proj = torch.zeros_like(t_prime)

        # Step 1: Concatenate attended embeddings
        gate_input = torch.cat([t_prime, i_prime, m_proj], dim=-1)  # (B, 768)

        # Step 2: Compute gating weights
        gate_logits = self.gate_layer(gate_input)  # (B, 768)
        gate_weights = self.sigmoid(gate_logits)  # (B, 768)

        # Step 3: Split gate weights into three components
        g1, g2, g3 = gate_weights.chunk(3, dim=-1)  # Each (B, 256)

        # Step 4: Weighted combination of attended modalities
        fused = g1 * t_prime + g2 * i_prime + g3 * m_proj  # (B, 256)

        # Step 5: Add residual connections from original embeddings
        residual = fused + t_proj + i_proj + m_proj  # (B, 256)

        # Step 6: Layer normalization
        out = self.layer_norm(residual)  # (B, 256)

        return out
