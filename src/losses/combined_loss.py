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
    ADAPTIVE: Automatically detects which components to enable based on model architecture.

    Supports multiple classification losses: BCE, Focal, Asymmetric Focal.
    Contrastive loss is enabled only if the model output contains both 't_proj' and 'i_proj'.

    Total loss = L_cls + contrastive_lambda * L_con

    Attributes:
        cls_loss: Classification loss (BCEWithLogitsLoss, FocalLossWithLogits, or AsymmetricFocalLossWithLogits)
        contrastive: InfoNCELoss module (learnable temperature)
        lambda_con: Weight for contrastive component
        ablation_mode: Model ablation mode (detected or configured)
    """

    def __init__(
        self,
        class_weights: torch.Tensor = None,
        contrastive_lambda: float = 0.1,
        contrastive_temperature_init: float = 0.07,
        label_smoothing: float = 0.0,
        cls_loss_type: str = "bce",
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        focal_gamma_pos: float = 1.0,
        focal_gamma_neg: float = 4.0,
        focal_clip: float = 0.05,
        ablation_mode: str = "full",
        pos_weight: float = 1.0,
        aux_lambda: float = 0.1,
    ):
        """
        Initialize CombinedLoss.

        Args:
            class_weights: Class weights tensor (legacy, prefer pos_weight).
            contrastive_lambda: Weight for contrastive loss component (default: 0.1).
            contrastive_temperature_init: Initial temperature for InfoNCE loss (default: 0.07).
            label_smoothing: Label smoothing for classification (only used with BCE).
            cls_loss_type: Classification loss type. One of "bce", "focal", "asymmetric_focal".
            focal_alpha: Class balance factor for Focal losses (default: 0.5 = neutral).
            focal_gamma: Focusing parameter for FocalLossWithLogits (default: 2.0).
            focal_gamma_pos: Focusing parameter for positives in AsymmetricFocalLossWithLogits.
            focal_gamma_neg: Focusing parameter for negatives in AsymmetricFocalLossWithLogits.
            focal_clip: Probability clipping for AsymmetricFocalLossWithLogits.
            ablation_mode: Model ablation mode (e.g., "text_only", "full", "text_image").
            pos_weight: Positive class weight for BCE (default: 1.0).
            aux_lambda: Weight for auxiliary modality classification losses.
        """
        super().__init__()

        self.ablation_mode = ablation_mode
        self.label_smoothing = label_smoothing
        self.cls_loss_type = cls_loss_type
        self.contrastive_lambda_config = contrastive_lambda
        self.aux_lambda = 0.0 if ablation_mode in {
            "text_only",
            "image_only",
            "metadata_only",
        } else float(aux_lambda)
        self.aux_pos_weight = float(pos_weight)
        
        # === Determine if contrastive loss applies based on ablation mode ===
        self.contrastive_applicable = self._is_contrastive_applicable(ablation_mode)
        
        # === Classification loss — selectable via config ===
        if cls_loss_type == "bce":
            if class_weights is not None and len(class_weights) > 1:
                pos_weight = class_weights[1] / class_weights[0]
            self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            logger.info(f"Using BCE loss with pos_weight={pos_weight:.4f}")

        elif cls_loss_type == "focal":
            self.cls_loss = FocalLossWithLogits(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction="mean",
            )
            logger.info(f"Using Focal loss (gamma={focal_gamma}, alpha={focal_alpha})")

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

        # === Contrastive loss (only if applicable AND configured) ===
        self._has_contrastive = False
        self.contrastive = None
        self.contrastive_loss = None
        
        if ablation_mode == "full_no_contrastive":
            self.lambda_con = 0.0
            logger.info(
                "[CombinedLoss] ablation='full_no_contrastive': "
                "InfoNCE module NOT instantiated, contrastive disabled"
            )
        elif self.contrastive_applicable and contrastive_lambda > 0:
            self.contrastive = InfoNCELoss(init_temperature=contrastive_temperature_init)
            self.contrastive_loss = self.contrastive
            self._has_contrastive = True
            self.lambda_con = contrastive_lambda
        else:
            # Auto-zero contrastive lambda if not applicable
            self.lambda_con = 0.0
            if not self.contrastive_applicable and contrastive_lambda > 0:
                logger.warning(
                    f"[CombinedLoss] ablation_mode='{ablation_mode}' does NOT support contrastive loss "
                    f"(needs both text and image encoders). Forcing contrastive_lambda=0 "
                    f"(was {contrastive_lambda})"
                )
        
        logger.info(
            f"[CombinedLoss] mode={ablation_mode} | cls={cls_loss_type} | "
            f"contrastive={'enabled' if self._has_contrastive else 'DISABLED'} | "
            f"lambda_con={self.lambda_con:.4f} | aux_lambda={self.aux_lambda:.4f}"
        )

    @staticmethod
    def _is_contrastive_applicable(mode: str) -> bool:
        """
        Check if contrastive loss can apply for a given ablation mode.
        Contrastive needs BOTH text AND image projections.
        """
        modes_with_t_and_i = {
            "full",
            "full_no_contrastive",  # has T+I architecturally, but lambda might be forced to 0
            "full_no_modality_dropout",
            "full_no_dropout",
            "full_no_metadata_in_fusion",
            "full_no_attention",
            "full_no_gating",
            "text_image",  # T+I bimodal
        }
        return mode in modes_with_t_and_i
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        text_emb: torch.Tensor | None = None,
        image_emb: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        is_multimodal: bool = True,
        output_dict: dict = None,  # NEW: inspect model output dict directly
    ) -> dict:
        """
        Compute combined loss.
        
        ADAPTIVE: Automatically skips contrastive if projections are missing or model is unimodal.

        Args:
            logits: Classification logits, shape (batch_size, 1).
            labels: Binary labels, shape (batch_size,) or (batch_size, 1).
            text_emb: Text embeddings (projected), shape (batch_size, 256). Can be None in unimodal.
            image_emb: Image embeddings (projected), shape (batch_size, 256). Can be None in unimodal.
            valid_mask: Boolean mask (batch_size,) for samples with both modalities. True = both present.
            is_multimodal: Whether model is multimodal. Can be override via output_dict.
            output_dict: Optional model output dict. If provided, text_emb/image_emb extracted from it.

        Returns:
            Dictionary with keys:
            - 'loss': Total loss (scalar tensor)
            - 'total': Alias of 'loss' for compatibility with smoke tests
            - 'cls_loss': Classification loss component (detached for logging)
            - 'cls': Alias of 'cls_loss' for compatibility with smoke tests
            - 'con_loss': Contrastive loss component (detached) or None if N/A
            - 'con': Alias of 'con_loss' for compatibility with smoke tests
            - 'temperature': Current temperature value (detached) or None if N/A
        """
        # === STEP 1: Handle output_dict extraction (NEW ADAPTIVE LOGIC) ===
        if output_dict is not None:
            # Extract projections from output dict (if present)
            text_emb = output_dict.get("t_proj", text_emb)
            image_emb = output_dict.get("i_proj", image_emb)
            is_multimodal = output_dict.get("is_multimodal", is_multimodal)
            valid_mask = output_dict.get("image_valid", valid_mask)
        
        # === STEP 2: Ensure labels have correct shape for loss functions ===
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        # === STEP 3: Classification loss ===
        targets = labels.float()

        # Label smoothing only for BCE (Focal handles confidence internally)
        if self.label_smoothing > 0 and self.cls_loss_type == "bce":
            targets_smoothed = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            cls_loss = self.cls_loss(logits.squeeze(-1), targets_smoothed.squeeze(-1))
        else:
            cls_loss = self.cls_loss(logits.squeeze(-1), targets.squeeze(-1))

        # === STEP 4: ADAPTIVE Contrastive Loss ===
        con_loss = None
        temperature = None
        
        # Contrastive applies only if:
        # 1. Model is multimodal (or has both T and I architecturally)
        # 2. Contrastive lambda > 0
        # 3. Both t_proj and i_proj are present in the forward output
        if (self._has_contrastive 
                and self.lambda_con > 0 
                and text_emb is not None 
                and image_emb is not None
                and is_multimodal):
            con_loss = self.contrastive(text_emb, image_emb, valid_mask)
            temperature = self.contrastive.temperature.detach()
        else:
            # Unimodal or missing projections: skip contrastive
            con_loss = None

        if self.ablation_mode == "full_no_contrastive":
            con_loss = torch.zeros((), device=logits.device, dtype=cls_loss.dtype)
            temperature = None

        aux_text_loss = torch.zeros((), device=logits.device, dtype=cls_loss.dtype)
        aux_image_loss = torch.zeros((), device=logits.device, dtype=cls_loss.dtype)
        aux_meta_loss = torch.zeros((), device=logits.device, dtype=cls_loss.dtype)

        if self.aux_lambda > 0 and output_dict is not None:
            aux_pos_weight = torch.tensor(
                self.aux_pos_weight,
                device=logits.device,
                dtype=logits.dtype,
            )
            aux_targets = targets.squeeze(-1)
            if output_dict.get("aux_text_logits") is not None:
                aux_text_loss = F.binary_cross_entropy_with_logits(
                    output_dict["aux_text_logits"].squeeze(-1),
                    aux_targets,
                    pos_weight=aux_pos_weight,
                )
            if output_dict.get("aux_image_logits") is not None:
                aux_image_loss = F.binary_cross_entropy_with_logits(
                    output_dict["aux_image_logits"].squeeze(-1),
                    aux_targets,
                    pos_weight=aux_pos_weight,
                )
            if output_dict.get("aux_meta_logits") is not None:
                aux_meta_loss = F.binary_cross_entropy_with_logits(
                    output_dict["aux_meta_logits"].squeeze(-1),
                    aux_targets,
                    pos_weight=aux_pos_weight,
                )

        aux_total = aux_text_loss + aux_image_loss + aux_meta_loss
        con_total = con_loss if con_loss is not None else torch.zeros(
            (), device=logits.device, dtype=cls_loss.dtype
        )
        total = cls_loss + self.lambda_con * con_total + self.aux_lambda * aux_total

        return {
            "loss": total,
            "total": total,
            "cls_loss": cls_loss.detach(),
            "cls": cls_loss.detach(),
            "con_loss": con_total.detach(),
            "con": con_total.detach(),
            "aux_text": aux_text_loss.detach(),
            "aux_image": aux_image_loss.detach(),
            "aux_meta": aux_meta_loss.detach(),
            "aux_total": aux_total.detach(),
            "temperature": temperature,
        }
