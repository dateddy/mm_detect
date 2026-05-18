# src/models/metadata_encoder.py
"""Metadata encoder for behavioral features."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MetadataEncoder(nn.Module):
    """
    MLP encoder for behavioral metadata features with gradual expansion.

    Transforms tabular metadata features into a 256-dimensional embedding
    using progressive layer expansion with BatchNorm, GELU, and Dropout.

    Gradual expansion design (16→64→128→256) with expansion ratios:
    - 1st layer: 4.0× (16→64)
    - 2nd layer: 2.0× (64→128)
    - 3rd layer: 2.0× (128→256)
    Much more reasonable than immediate 16.0× jump (16→256).

    Includes residual connection from input to output for improved gradient flow.

    Research: TabNet, NODE, MLP-Mixer show gradual expansion works better for
    tabular data than large immediate expansions.

    Attributes:
        input_dim: Input feature dimension (default: 16 metadata features)
        output_dim: Output embedding dimension (default: 256)
        hidden_dims: List of hidden layer dimensions (default: [64, 128])
    """

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 256,
        hidden_dims: list | None = None,
        dropout: float = 0.1,
    ):
        """
        Initialize MetadataEncoder with gradual expansion.

        Args:
            input_dim: Number of input metadata features (default: 16).
            output_dim: Dimension of output embedding (default: 256).
            hidden_dims: List of hidden layer dimensions (default: [64, 128]).
                        If None, uses [64, 128] for gradual 16→64→128→256 expansion.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout

        # Build sequential network with gradual expansion
        layers = []

        # First layer: input_dim → first hidden dimension
        layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),  # Light dropout on tabular features
        ])

        # Intermediate layers: hidden → hidden
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        # Final layer: last hidden → output_dim (no dropout on projection layer)
        layers.extend([
            nn.Linear(hidden_dims[-1], output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        ])

        self.net = nn.Sequential(*layers)

        # Residual path: direct linear from input to output for gradient flow
        # Analogous to residual connections in main architecture
        self.residual_proj = nn.Linear(input_dim, output_dim)

        logger.info(
            f"Initialized MetadataEncoder with gradual expansion "
            f"(input={input_dim} → {' → '.join(map(str, hidden_dims))} → output={output_dim}) "
            f"| Expansion ratios: {[f'{h/prev:.1f}×' for prev, h in zip([input_dim] + hidden_dims[:-1], hidden_dims + [output_dim])]}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode metadata features with residual connection.

        Args:
            x: Metadata features, shape (batch_size, input_dim).

        Returns:
            Encoded embeddings with residual, shape (batch_size, output_dim).
        """
        # Main path through gradual expansion network
        main = self.net(x)

        # Residual path: direct projection from input
        residual = self.residual_proj(x)

        # Combine: main path + residual connection
        return main + residual
