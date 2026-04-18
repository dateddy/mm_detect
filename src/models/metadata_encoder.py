# src/models/metadata_encoder.py
"""Metadata encoder for behavioral features."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MetadataEncoder(nn.Module):
    """
    MLP encoder for behavioral metadata features.

    Transforms 9-dimensional behavioral features into a 256-dimensional embedding
    using three fully-connected layers with BatchNorm and GELU activations.

    Architecture:
    - Linear(input_dim, hidden_dim) → BatchNorm1d → GELU
    - Linear(hidden_dim, hidden_dim) → BatchNorm1d → GELU
    - Linear(hidden_dim, hidden_dim)

    Output is not activated, allowing the fusion module to apply its own transformations.

    Attributes:
        input_dim: Input feature dimension (default: 9 behavioral features)
        hidden_dim: Hidden and output dimension (default: 256)
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 256):
        """
        Initialize MetadataEncoder.

        Args:
            input_dim: Number of input behavioral features (default: 9).
            hidden_dim: Dimension of hidden layers and output (default: 256).
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # First layer: project from input_dim to hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gelu1 = nn.GELU()

        # Second layer: hidden_dim to hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gelu2 = nn.GELU()

        # Third layer: hidden_dim to hidden_dim (output)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        logger.info(
            f"Initialized MetadataEncoder "
            f"(input_dim={input_dim}, hidden_dim={hidden_dim})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode metadata features.

        Args:
            x: Metadata features, shape (batch_size, input_dim).

        Returns:
            Encoded embeddings, shape (batch_size, hidden_dim).
        """
        # First block: Linear → BatchNorm → GELU
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu1(x)

        # Second block: Linear → BatchNorm → GELU
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.gelu2(x)

        # Third block: Linear only (no activation)
        x = self.fc3(x)

        return x
