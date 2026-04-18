# src/losses/combined_loss.py
"""Combined classification and contrastive loss."""

import logging

import torch
import torch.nn as nn

from src.losses.contrastive import InfoNCELoss

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined loss with classification and contrastive components.

    Combines Binary Cross-Entropy loss for the classification task with InfoNCE
    contrastive loss for aligning text and image embeddings.

    Total loss = L_cls + contrastive_lambda * L_con

    Attributes:
        bce_loss: BCEWithLogitsLoss with class weighting.
        contrastive_loss: InfoNCELoss module.
        contrastive_lambda: Weight for contrastive component (default: 0.1).
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        contrastive_lambda: float = 0.1,
        temperature: float = 0.07,
    ):
        """
        Initialize CombinedLoss.

        Args:
            class_weights: Class weights tensor of shape (2,).
                          Typically from compute_class_weights() in preprocessing.
            contrastive_lambda: Weight for contrastive loss component (default: 0.1).
            temperature: Temperature for InfoNCE loss (default: 0.07).
        """
        super().__init__()

        self.contrastive_lambda = contrastive_lambda

        # Extract positive class weight for BCEWithLogitsLoss
        # class_weights shape: (2,) with [weight_class_0, weight_class_1]
        # pos_weight is the weight for positive class (class 1)
        pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else torch.tensor(1.0)

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.contrastive_loss = InfoNCELoss(temperature=temperature)

        logger.info(
            f"Initialized CombinedLoss "
            f"(lambda={contrastive_lambda}, temperature={temperature}, "
            f"pos_weight={pos_weight:.4f})"
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            logits: Classification logits, shape (batch_size, 1).
            labels: Binary labels, shape (batch_size,) or (batch_size, 1).
            text_emb: Text embeddings (NOT normalized), shape (batch_size, 768).
            image_emb: Image embeddings (NOT normalized), shape (batch_size, 768).

        Returns:
            Dictionary with keys:
            - 'loss': Total loss (scalar tensor)
            - 'cls_loss': Classification loss component (scalar tensor)
            - 'con_loss': Contrastive loss component (scalar tensor)
        """
        # Ensure labels have correct shape for BCEWithLogitsLoss
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        # Classification loss
        cls_loss = self.bce_loss(logits, labels)

        # Contrastive loss
        con_loss = self.contrastive_loss(text_emb, image_emb)

        # Combined loss
        total_loss = cls_loss + self.contrastive_lambda * con_loss

        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "con_loss": con_loss.detach(),
        }
