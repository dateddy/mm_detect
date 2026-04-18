# src/models/classification_head.py
"""Classification head for binary misinformation prediction."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """
    Binary classification head with dropout regularization.

    Transforms fused multimodal embeddings into binary logits for misinformation prediction.

    Architecture:
    - Linear(in_dim, 128) → GELU → Dropout(dropout)
    - Linear(128, 64) → GELU → Dropout(dropout)
    - Linear(64, 1)

    No final sigmoid — BCEWithLogitsLoss applies sigmoid internally for numerical stability.

    Attributes:
        in_dim: Input dimension (default: 256 from fusion block).
        dropout: Dropout probability (default: 0.3).
    """

    def __init__(self, in_dim: int = 256, dropout: float = 0.3):
        """
        Initialize ClassificationHead.

        Args:
            in_dim: Input dimension (default: 256).
            dropout: Dropout probability (default: 0.3).
        """
        super().__init__()

        self.in_dim = in_dim
        self.dropout_p = dropout

        # First layer: in_dim → 128
        self.fc1 = nn.Linear(in_dim, 128)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # Second layer: 128 → 64
        self.fc2 = nn.Linear(128, 64)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        # Output layer: 64 → 1
        self.fc3 = nn.Linear(64, 1)

        logger.info(
            f"Initialized ClassificationHead "
            f"(in_dim={in_dim}, dropout={dropout})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute logits.

        Args:
            x: Fused embeddings, shape (batch_size, in_dim).

        Returns:
            Raw logits (no sigmoid), shape (batch_size, 1).
        """
        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)

        logits = self.fc3(x)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted probabilities for inference.

        Args:
            x: Fused embeddings, shape (batch_size, in_dim).

        Returns:
            Predicted probabilities (sigmoid applied), shape (batch_size,).
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs
