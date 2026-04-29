# src/losses/contrastive.py
"""Contrastive loss for multimodal alignment with learnable temperature."""

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with learnable temperature (CLIP-style).

    The temperature τ is parameterized as log(1/τ) for numerical stability.
    Initialized to log(1/0.07) ≈ 2.659 following CLIP's published config.
    
    The temperature is clamped to [log(1/100), log(1/0.01)] to prevent 
    runaway scaling during training.

    Encourages text and image embeddings from the same ad to be similar (positive pairs)
    while being dissimilar to embeddings from different ads (negative pairs).

    The loss is symmetric: computes both text→image and image→text directions,
    then averages.

    Attributes:
        logit_scale: Learnable parameter for log(1/τ), initialized from init_temperature.
        logit_scale_max: Upper bound on logit_scale for numerical stability.
    """

    def __init__(self, init_temperature: float = 0.07):
        """
        Initialize InfoNCELoss with learnable temperature.

        Args:
            init_temperature: Initial temperature τ (default: 0.07).
                             Converted to log(1/τ) for parameterization.
                             Larger values soften the softmax; smaller values sharpen it.
        """
        super().__init__()
        
        # Parameterize as log(1/τ) for numerical stability
        # init_temperature=0.07 → logit_scale = log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1.0 / init_temperature)
        )
        
        # Hard upper bound: log(100) ≈ 4.605 corresponds to τ=0.01
        # Prevents runaway scaling during training
        self.logit_scale_max = np.log(100.0)
        
        logger.info(
            f"Initialized InfoNCELoss with learnable temperature "
            f"(init_temperature={init_temperature}, init_logit_scale={self.logit_scale.item():.4f})"
        )

    @property
    def temperature(self) -> torch.Tensor:
        """
        Read-only view of the current effective temperature τ = 1 / exp(logit_scale).
        
        Returns:
            scalar tensor with current temperature value
        """
        return 1.0 / self.logit_scale.exp()

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss between text and image embeddings.

        Parameters
        ----------
        text_emb : torch.Tensor
            Text embeddings (projected), shape (batch_size, embedding_dim).
            Must have requires_grad=True for loss to be trainable.
        image_emb : torch.Tensor
            Image embeddings (projected), shape (batch_size, embedding_dim).
            Must have requires_grad=True for loss to be trainable.
        valid_mask : torch.Tensor | None
            Boolean mask of shape (batch_size,) where True indicates valid samples.
            Used to exclude samples where image modality was dropped by ModalityDropout.
            If None, all samples are treated as valid.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor with requires_grad=True for training.
        """
        # 1. Clamp logit_scale BEFORE using it (CLIP does this every forward pass)
        # This prevents runaway scaling during training
        with torch.no_grad():
            self.logit_scale.clamp_(max=self.logit_scale_max)
        
        # 2. Filter to valid samples only (exclude modality-dropout zeroed images)
        if valid_mask is not None:
            valid = valid_mask.bool()
            text_emb = text_emb[valid]
            image_emb = image_emb[valid]

        B = text_emb.shape[0]

        # 3. Handle degenerate batch
        if B <= 1:
            logger.debug("Batch size is 1, returning zero loss for contrastive term")
            return torch.tensor(
                0.0, device=text_emb.device, dtype=text_emb.dtype, requires_grad=True
            )

        # 4. Verify gradients are attached — fail loudly if not
        if not text_emb.requires_grad and not image_emb.requires_grad:
            raise RuntimeError(
                "InfoNCELoss received inputs with no gradient. "
                "Check for .detach() calls on t_proj/i_proj in full_model.py "
                "or trainer.py before the loss computation."
            )

        # 5. L2 normalize — MANDATORY before cosine similarity
        text_norm = F.normalize(text_emb, dim=-1, eps=1e-8)
        image_norm = F.normalize(image_emb, dim=-1, eps=1e-8)

        # 6. Verify normalization succeeded (no NaN from zero vectors)
        if torch.isnan(text_norm).any() or torch.isnan(image_norm).any():
            raise RuntimeError(
                "NaN after L2 normalization in InfoNCELoss. "
                "Embeddings contain zero vectors. Check projection layer "
                "initialization and verify encoders produce non-zero output."
            )

        # 7. Full similarity matrix (B, B), scaled by learnable temperature
        # logit_scale.exp() = 1/τ, so similarity is divided by τ
        logit_scale = self.logit_scale.exp()
        sim = torch.matmul(text_norm, image_norm.T) * logit_scale

        # 8. Symmetric cross-entropy
        labels = torch.arange(B, device=sim.device)
        loss_t2i = F.cross_entropy(sim, labels)
        loss_i2t = F.cross_entropy(sim.T, labels)

        loss = (loss_t2i + loss_i2t) / 2.0

        # Verify loss is valid
        assert not torch.isnan(loss), "Loss computation resulted in NaN"
        assert loss.requires_grad, "Loss has no gradient for backpropagation"

        return loss


        # 3. Verify gradients are attached — fail loudly if not
        if not text_emb.requires_grad and not image_emb.requires_grad:
            raise RuntimeError(
                "InfoNCELoss received inputs with no gradient. "
                "Check for .detach() calls on t_proj/i_proj in full_model.py "
                "or trainer.py before the loss computation."
            )

        # 4. L2 normalize — MANDATORY before cosine similarity
        text_norm = F.normalize(text_emb, dim=-1, eps=1e-8)
        image_norm = F.normalize(image_emb, dim=-1, eps=1e-8)

        # 5. Verify normalization succeeded (no NaN from zero vectors)
        if torch.isnan(text_norm).any() or torch.isnan(image_norm).any():
            raise RuntimeError(
                "NaN after L2 normalization in InfoNCELoss. "
                "Embeddings contain zero vectors. Check projection layer "
                "initialization and verify encoders produce non-zero output."
            )

        # 6. Full similarity matrix (B, B), scaled by temperature
        sim = torch.matmul(text_norm, image_norm.T) / self.temperature

        # 7. Symmetric cross-entropy
        labels = torch.arange(B, device=sim.device)
        loss_t2i = F.cross_entropy(sim, labels)
        loss_i2t = F.cross_entropy(sim.T, labels)

        loss = (loss_t2i + loss_i2t) / 2.0

        # Verify loss is valid
        assert not torch.isnan(loss), "Loss computation resulted in NaN"
        assert loss.requires_grad, "Loss has no gradient for backpropagation"

        return loss
