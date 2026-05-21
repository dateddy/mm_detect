# src/models/classification_head.py
"""Classification head for binary misinformation prediction."""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_prior_pos_rate(config: dict | None, default: float = 0.609) -> float:
    """Read and clamp the configured positive class prior."""
    config = config or {}
    pos_rate = config.get("data", {}).get("estimated_pos_rate", default)
    return min(max(float(pos_rate), 1e-4), 1.0 - 1e-4)


def initialize_classifier_prior_bias(
    classifier: nn.Module,
    config: dict | None = None,
    default_pos_rate: float = 0.609,
) -> float | None:
    """
    Initialize the final binary classifier bias to log(p / (1 - p)).

    Works for both ClassificationHead and nn.Sequential classifiers by finding
    the last Linear layer with one output feature.
    """
    pos_rate = get_prior_pos_rate(config, default=default_pos_rate)
    prior_bias = math.log(pos_rate / (1.0 - pos_rate))

    final_linear = None
    for module in classifier.modules():
        if isinstance(module, nn.Linear) and module.out_features == 1:
            final_linear = module

    if final_linear is None or final_linear.bias is None:
        logger.warning("Could not initialize classifier prior bias: final Linear(?, 1) not found")
        return None

    with torch.no_grad():
        final_linear.bias.fill_(prior_bias)

    logger.info(
        f"Classifier final bias initialized to {prior_bias:.4f} "
        f"(matches prior={pos_rate:.4f})"
    )
    return prior_bias


def build_auxiliary_head(
    in_dim: int,
    dropout: float,
    config: dict | None = None,
    default_pos_rate: float = 0.609,
) -> nn.Sequential:
    """Build a shallow auxiliary classifier with prior-calibrated output bias."""
    head = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(64, 1),
    )
    initialize_classifier_prior_bias(head, config, default_pos_rate=default_pos_rate)
    return head


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

    def __init__(
        self,
        in_dim: int = 256,
        dropout: float = 0.3,
        config: dict | None = None,
        prior_pos_rate: float | None = None,
    ):
        """
        Initialize ClassificationHead.

        Args:
            in_dim: Input dimension (default: 256).
            dropout: Dropout probability (default: 0.3).
            config: Optional config containing data.estimated_pos_rate.
            prior_pos_rate: Optional explicit positive class prior.
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

        if prior_pos_rate is not None:
            config = {"data": {"estimated_pos_rate": prior_pos_rate}}
        initialize_classifier_prior_bias(self, config)

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
