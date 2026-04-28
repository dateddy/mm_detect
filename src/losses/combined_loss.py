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
        bce: BCEWithLogitsLoss with class weighting.
        contrastive: InfoNCELoss module.
        lambda_con: Weight for contrastive component (default: 0.1).
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        contrastive_lambda: float = 0.1,
        temperature: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize CombinedLoss.

        Args:
            class_weights: Class weights tensor of shape (2,).
                          Typically from compute_class_weights() in preprocessing.
            contrastive_lambda: Weight for contrastive loss component (default: 0.1).
            temperature: Temperature for InfoNCE loss (default: 0.1).
            label_smoothing: Label smoothing for classification (default: 0.0).
        """
        super().__init__()

        # pos_weight is the ratio: weight_positive / weight_negative
        pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else torch.tensor(1.0)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.contrastive = InfoNCELoss(temperature=temperature)
        self.lambda_con = contrastive_lambda
        self.label_smoothing = label_smoothing

        logger.info(
            f"Initialized CombinedLoss "
            f"(lambda={contrastive_lambda}, temperature={temperature}, "
            f"pos_weight={pos_weight:.4f}, label_smoothing={label_smoothing})"
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            logits: Classification logits, shape (batch_size, 1).
            labels: Binary labels, shape (batch_size,) or (batch_size, 1).
            text_emb: Text embeddings (projected), shape (batch_size, 256).
                      MUST have requires_grad=True.
            image_emb: Image embeddings (projected), shape (batch_size, 256).
                      MUST have requires_grad=True.
            valid_mask: Boolean mask indicating which samples should use contrastive loss.
                       Shape (batch_size,). True = valid, False = skip for contrastive loss.
                       Used to exclude samples where image was dropped by ModalityDropout.

        Returns:
            Dictionary with keys:
            - 'loss': Total loss (scalar tensor)
            - 'cls_loss': Classification loss component (scalar tensor, detached for logging)
            - 'con_loss': Contrastive loss component (scalar tensor, detached for logging)
        """
        # Ensure labels have correct shape for BCEWithLogitsLoss
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        # Classification loss with optional label smoothing
        targets = labels.float()
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        cls_loss = self.bce(logits.squeeze(-1), targets.squeeze(-1))

        # Contrastive loss — only on valid (non-dropped) samples
        con_loss = self.contrastive(text_emb, image_emb, valid_mask)

        total = cls_loss + self.lambda_con * con_loss

        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "con_loss": con_loss.detach(),
        }
