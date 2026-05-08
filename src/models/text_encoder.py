# src/models/text_encoder.py
"""PhoBERT text encoder for Vietnamese ad text."""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """
    PhoBERT-based text encoder for Vietnamese ad text.

    Loads a pretrained BERT-family model, extracts [CLS] token representation,
    and provides freezing/unfreezing capabilities for fine-tuning.

    Attributes:
        model: Pretrained AutoModel instance
        config: Model configuration
    """

    def __init__(
        self, model_name: str = "vinai/phobert-base-v2", freeze: bool = True
    ):
        """
        Initialize TextEncoder.

        Args:
            model_name: HuggingFace model identifier (default: PhoBERT base).
            freeze: Whether to freeze all parameters at initialization.
        """
        super().__init__()

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
        )
        self.config = self.model.config

        if freeze:
            self.freeze_all()

        logger.info(
            f"Initialized TextEncoder ({model_name}), "
            f"output_dim={self.config.hidden_size}, pooler=disabled"
        )

    def freeze_all(self) -> None:
        """Freeze all parameters to prevent gradient updates."""
        for param in self.model.parameters():
            param.requires_grad = False

        logger.debug("Froze all TextEncoder parameters")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters to allow gradient updates."""
        for param in self.model.parameters():
            param.requires_grad = True

        logger.debug("Unfroze all TextEncoder parameters")

    def unfreeze_top_k(self, k: int) -> None:
        """
        Unfreeze the last k transformer encoder blocks.

        For BERT-family models, accesses self.model.encoder.layer[-k:].

        Args:
            k: Number of top blocks to unfreeze.
        """
        if not hasattr(self.model, "encoder"):
            logger.warning(
                f"Model {self.model_name} does not have 'encoder' attribute, "
                f"cannot unfreeze top-k blocks"
            )
            return

        if not hasattr(self.model.encoder, "layer"):
            logger.warning(
                f"Model encoder does not have 'layer' attribute, "
                f"cannot unfreeze top-k blocks"
            )
            return

        num_layers = len(self.model.encoder.layer)
        if k > num_layers:
            logger.warning(
                f"k={k} exceeds total layers {num_layers}, unfreezing all"
            )
            k = num_layers

        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze top-k encoder blocks
        for layer in self.model.encoder.layer[-k:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Always unfreeze layer norm if it exists.
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "LayerNorm"):
            for param in self.model.encoder.LayerNorm.parameters():
                param.requires_grad = True

        logger.info(
            f"Unfroze top-{k} encoder blocks in TextEncoder "
            f"(total layers: {num_layers})"
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to extract text embeddings.

        Args:
            input_ids: Tokenized input IDs, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).

        Returns:
            [CLS] token embeddings, shape (batch_size, hidden_size).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Extract [CLS] token (first token of last hidden state)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Get total number of parameters.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())

    @property
    def output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.config.hidden_size
