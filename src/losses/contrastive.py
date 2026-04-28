# src/losses/contrastive.py
"""Contrastive loss for multimodal alignment."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for aligning text and image embeddings.

    Encourages text and image embeddings from the same ad to be similar (positive pairs)
    while being dissimilar to embeddings from different ads (negative pairs).

    The loss is symmetric: computes both text→image and image→text directions,
    then averages.

    Attributes:
        temperature: Temperature parameter for scaled similarity (default: 0.1).
    """

    def __init__(self, temperature: float = 0.1):
        """
        Initialize InfoNCELoss.

        Args:
            temperature: Temperature for scaled similarity (default: 0.1).
                        Larger values soften the softmax; smaller values sharpen it.
                        Use 0.07 for sharp loss; 0.1+ for stable early training.
        """
        super().__init__()
        self.temperature = temperature
        logger.info(f"Initialized InfoNCELoss (temperature={temperature})")

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
        # 1. Filter to valid samples only (exclude modality-dropout zeroed images)
        if valid_mask is not None:
            valid = valid_mask.bool()
            text_emb = text_emb[valid]
            image_emb = image_emb[valid]

        B = text_emb.shape[0]

        # 2. Handle degenerate batch
        if B <= 1:
            logger.debug("Batch size is 1, returning zero loss for contrastive term")
            return torch.tensor(
                0.0, device=text_emb.device, dtype=text_emb.dtype, requires_grad=True
            )

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
