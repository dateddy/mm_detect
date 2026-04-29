# src/losses/combined_loss.py
"""Combined classification and contrastive loss."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.contrastive import InfoNCELoss

logger = logging.getLogger(__name__)


# ============================================================================
# Focal Loss Classes
# ============================================================================

class FocalLossWithLogits(nn.Module):
    """
    Symmetric Focal Loss for binary classification with logits input.

    Reference: Lin et al. 2017, "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

    Formulation:
        L = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        p_t = sigmoid(logit) if y=1, else 1-sigmoid(logit)
        α_t = α if y=1, else 1-α  (class balance factor)

    Args:
        alpha: Weight for positive class in [0, 1]. 0.5 = no class weighting.
               Set higher (e.g., 0.75) to emphasize positive class (recall focus).
        gamma: Focusing parameter. 0 = standard BCE, 2 = paper default.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"
        assert gamma >= 0.0, f"gamma must be >= 0, got {gamma}"
        assert reduction in ("mean", "sum", "none"), f"Unknown reduction: {reduction}"
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B] or [B, 1] raw logits (no sigmoid applied)
            targets: [B] or [B, 1] binary labels {0, 1}, can be float for label smoothing

        Returns:
            scalar loss (or [B] if reduction='none')
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Numerically stable computation using log-sigmoid identity:
        # log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x))
        # log(1 - sigmoid(x)) = -softplus(x) = -log(1 + exp(x))
        log_p = F.logsigmoid(logits)  # log(p) for positive class
        log_one_minus_p = F.logsigmoid(-logits)  # log(1-p) for negative class

        # p_t = p if y=1 else 1-p; equivalently:
        # log(p_t) = y*log(p) + (1-y)*log(1-p)
        log_pt = targets * log_p + (1 - targets) * log_one_minus_p
        pt = log_pt.exp().clamp(min=1e-7, max=1.0 - 1e-7)

        # α_t = α if y=1 else 1-α
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight: (1 - pt)^γ
        focal_weight = (1 - pt) ** self.gamma

        loss = -alpha_t * focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricFocalLossWithLogits(nn.Module):
    """
    Asymmetric Focal Loss with separate focusing parameters for positive and
    negative classes. Useful when FN and FP have asymmetric operational costs.

    Reference: Ben-Baruch et al. 2020, "Asymmetric Loss for Multi-Label Classification"

    Formulation:
        L = -α * (1-p)^γ_pos * log(p)         if y=1 (miss penalty)
        L = -(1-α) * p^γ_neg * log(1-p)       if y=0 (false alarm penalty)

    Recommended: γ_pos < γ_neg to make positive class easier to predict
                 (lower γ = less down-weighting = more gradient when wrong).
    For misinformation: γ_pos=1.0, γ_neg=4.0 emphasizes catching positives.

    Args:
        alpha: Class balance factor in [0, 1].
        gamma_pos: Focusing parameter for positive class (FN penalty). Lower = stronger.
        gamma_neg: Focusing parameter for negative class (FP penalty). Higher = down-weight easy negatives.
        clip: Probability shift for very confident negatives (helps stability).
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"
        assert gamma_pos >= 0.0, f"gamma_pos must be >= 0, got {gamma_pos}"
        assert gamma_neg >= 0.0, f"gamma_neg must be >= 0, got {gamma_neg}"
        assert 0.0 <= clip <= 0.5, f"clip must be in [0, 0.5], got {clip}"
        assert reduction in ("mean", "sum", "none"), f"Unknown reduction: {reduction}"
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B] or [B, 1] raw logits
            targets: [B] or [B, 1] binary labels {0, 1}

        Returns:
            scalar loss (or [B] if reduction='none')
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Numerically stable
        p = torch.sigmoid(logits)
        log_p = F.logsigmoid(logits)
        log_one_minus_p = F.logsigmoid(-logits)

        # Loss for positive class (FN penalty): -α * (1-p)^γ_pos * log(p)
        focal_weight_pos = (1 - p).clamp(min=1e-7) ** self.gamma_pos
        loss_pos = -self.alpha * focal_weight_pos * log_p

        # Loss for negative class (FP penalty): -(1-α) * p_clipped^γ_neg * log(1-p)
        # Clip prevents very confident correct negatives from contributing nothing
        p_clipped = (p - self.clip).clamp(min=0.0, max=1.0)
        focal_weight_neg = p_clipped.clamp(min=1e-7) ** self.gamma_neg
        loss_neg = -(1 - self.alpha) * focal_weight_neg * log_one_minus_p

        # Combine via target mask
        loss = targets * loss_pos + (1 - targets) * loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================================# ============================================================================
# CombinedLoss: Main loss combining classification and contrastive
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combined loss with classification and contrastive components.

    Supports multiple classification losses: BCE, Focal, Asymmetric Focal.
    Always includes learnable contrastive loss (InfoNCE with learned temperature).

    Total loss = L_cls + contrastive_lambda * L_con

    Attributes:
        cls_loss: Classification loss (BCEWithLogitsLoss, FocalLossWithLogits, or AsymmetricFocalLossWithLogits)
        contrastive: InfoNCELoss module (always learnable temperature)
        lambda_con: Weight for contrastive component
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        contrastive_lambda: float = 0.1,
        contrastive_temperature_init: float = 0.07,
        label_smoothing: float = 0.0,
        cls_loss_type: str = "bce",
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        focal_gamma_pos: float = 1.0,
        focal_gamma_neg: float = 4.0,
        focal_clip: float = 0.05,
    ):
        """
        Initialize CombinedLoss.

        Args:
            class_weights: Class weights tensor of shape (2,).
            contrastive_lambda: Weight for contrastive loss component (default: 0.1).
            contrastive_temperature_init: Initial temperature for InfoNCE loss (default: 0.07).
            label_smoothing: Label smoothing for classification (only used with BCE).
            cls_loss_type: Classification loss type. One of "bce", "focal", "asymmetric_focal".
            focal_alpha: Class balance factor for Focal losses (default: 0.5 = neutral).
            focal_gamma: Focusing parameter for FocalLossWithLogits (default: 2.0).
            focal_gamma_pos: Focusing parameter for positives in AsymmetricFocalLossWithLogits.
            focal_gamma_neg: Focusing parameter for negatives in AsymmetricFocalLossWithLogits.
            focal_clip: Probability clipping for AsymmetricFocalLossWithLogits.
        """
        super().__init__()

        # Contrastive loss (always learnable temperature)
        self.contrastive = InfoNCELoss(init_temperature=contrastive_temperature_init)
        self.lambda_con = contrastive_lambda
        self.label_smoothing = label_smoothing
        self.cls_loss_type = cls_loss_type

        # Classification loss — selectable via config
        if cls_loss_type == "bce":
            pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else torch.tensor(1.0)
            self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info(f"Using BCE loss with pos_weight={pos_weight:.4f}")

        elif cls_loss_type == "focal":
            self.cls_loss = FocalLossWithLogits(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction="mean",
            )
            logger.info(
                f"Using Focal loss (gamma={focal_gamma}, alpha={focal_alpha})"
            )

        elif cls_loss_type == "asymmetric_focal":
            self.cls_loss = AsymmetricFocalLossWithLogits(
                alpha=focal_alpha,
                gamma_pos=focal_gamma_pos,
                gamma_neg=focal_gamma_neg,
                clip=focal_clip,
                reduction="mean",
            )
            logger.info(
                f"Using Asymmetric Focal loss "
                f"(gamma_pos={focal_gamma_pos}, gamma_neg={focal_gamma_neg}, "
                f"alpha={focal_alpha}, clip={focal_clip})"
            )

        else:
            raise ValueError(
                f"Unknown cls_loss_type: {cls_loss_type}. "
                f"Must be one of: 'bce', 'focal', 'asymmetric_focal'"
            )

        logger.info(
            f"Initialized CombinedLoss "
            f"(cls_loss_type={cls_loss_type}, lambda={contrastive_lambda}, "
            f"init_temperature={contrastive_temperature_init}, "
            f"label_smoothing={label_smoothing})"
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
            - 'temperature': Current temperature value (scalar tensor, detached for logging)
        """
        # Ensure labels have correct shape for loss functions
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        # Classification loss
        targets = labels.float()

        # Label smoothing only for BCE (Focal handles confidence internally)
        if self.label_smoothing > 0 and self.cls_loss_type == "bce":
            targets_smoothed = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            cls_loss = self.cls_loss(logits.squeeze(-1), targets_smoothed.squeeze(-1))
        else:
            cls_loss = self.cls_loss(logits.squeeze(-1), targets.squeeze(-1))

        # Contrastive loss — only on valid (non-dropped) samples
        con_loss = self.contrastive(text_emb, image_emb, valid_mask)

        total = cls_loss + self.lambda_con * con_loss

        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "con_loss": con_loss.detach(),
            "temperature": self.contrastive.temperature.detach(),
        }
