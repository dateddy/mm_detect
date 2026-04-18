# src/models/image_encoder.py
"""Vision Transformer image encoder using timm."""

import logging

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """
    Vision Transformer image encoder using timm library.

    Loads a pretrained ViT model, extracts [CLS] token representation,
    and provides freezing/unfreezing capabilities for fine-tuning.

    Attributes:
        model: Pretrained timm model instance
        model_name: Model identifier
        output_dim: Output embedding dimension (768 for ViT-B/16)
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        """
        Initialize ImageEncoder.

        Args:
            model_name: timm model identifier (default: ViT-B/16).
            pretrained: Whether to load pretrained weights.
            freeze: Whether to freeze all parameters at initialization.
        """
        super().__init__()

        self.model_name = model_name
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.output_dim = self.model.embed_dim

        if freeze:
            self.freeze_all()

        logger.info(
            f"Initialized ImageEncoder ({model_name}), "
            f"output_dim={self.output_dim}, pretrained={pretrained}"
        )

    def freeze_all(self) -> None:
        """Freeze all parameters to prevent gradient updates."""
        for param in self.model.parameters():
            param.requires_grad = False

        logger.debug("Froze all ImageEncoder parameters")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters to allow gradient updates."""
        for param in self.model.parameters():
            param.requires_grad = True

        logger.debug("Unfroze all ImageEncoder parameters")

    def unfreeze_top_k(self, k: int) -> None:
        """
        Unfreeze the last k Vision Transformer blocks.

        For ViT models, accesses self.model.blocks[-k:].

        Args:
            k: Number of top blocks to unfreeze.
        """
        if not hasattr(self.model, "blocks"):
            logger.warning(
                f"Model {self.model_name} does not have 'blocks' attribute, "
                f"cannot unfreeze top-k blocks"
            )
            return

        num_blocks = len(self.model.blocks)
        if k > num_blocks:
            logger.warning(
                f"k={k} exceeds total blocks {num_blocks}, unfreezing all"
            )
            k = num_blocks

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze top-k blocks
        for block in self.model.blocks[-k:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze norm layer if it exists
        if hasattr(self.model, "norm"):
            for param in self.model.norm.parameters():
                param.requires_grad = True

        # Unfreeze patch embedding
        if hasattr(self.model, "patch_embed"):
            for param in self.model.patch_embed.parameters():
                param.requires_grad = True

        logger.info(
            f"Unfroze top-{k} blocks in ImageEncoder "
            f"(total blocks: {num_blocks})"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract image embeddings.

        Args:
            pixel_values: Image tensors, shape (batch_size, 3, height, width).
                         Expected range: [0, 1] after normalization.

        Returns:
            [CLS] token embeddings, shape (batch_size, output_dim).
        """
        # Use forward_features to get the sequence output (without classification head)
        x = self.model.forward_features(pixel_values)

        # Extract [CLS] token (first token)
        cls_embedding = x[:, 0, :]

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
