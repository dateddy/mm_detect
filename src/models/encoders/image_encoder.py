"""Image encoder using ViT (Vision Transformer) from timm"""

import torch
import torch.nn as nn
import timm
from PIL import Image
from typing import Union, Optional


class ImageEncoder(nn.Module):
    """
    Image encoder using Vision Transformer (ViT-B/16).
    
    Extracts [CLS] token embeddings (768-dimensional) from images using
    ImageNet-21k pretrained ViT model from timm library.
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        image_size: int = 224
    ):
        """
        Initialize image encoder.
        
        Args:
            model_name: timm model identifier (default: vit_base_patch16_224)
            pretrained: Use ImageNet-21k pretrained weights if True
            freeze_encoder: Freeze all encoder parameters if True
            image_size: Input image size (224 for ViT-B/16)
        """
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        
        # Load pretrained ViT from timm
        pretrained_str = "timm" if pretrained else ""
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained
        )
        
        # Get data config for preprocessing
        self.data_config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        images: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: Image tensor of shape (batch_size, 3, 224, 224)
                   Values should be normalized to [0, 1] or [-1, 1]
            device: Device to use for computation
            
        Returns:
            Embeddings (batch_size, 768) - [CLS] token representations
        """
        images = images.to(device)
        
        # Forward pass through ViT
        with torch.no_grad():
            # timm models return logits by default, we need the features
            # Use forward_features to get hidden representation
            embeddings = self.model.forward_features(images)  # (batch_size, num_patches, 768)
        
        # Extract [CLS] token (first token)
        cls_embedding = embeddings[:, 0, :]  # (batch_size, 768)
        
        return cls_embedding
    
    def freeze(self) -> None:
        """Freeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def unfreeze_top_k_layers(self, k: int = 4) -> None:
        """
        Unfreeze top k transformer blocks.
        
        Args:
            k: Number of top layers to unfreeze from the end
        """
        # Freeze all
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze top k blocks in transformer
        if hasattr(self.model, 'blocks'):
            num_blocks = len(self.model.blocks)
            for block_idx in range(max(0, num_blocks - k), num_blocks):
                for param in self.model.blocks[block_idx].parameters():
                    param.requires_grad = True
        
        # Always unfreeze layer norm and cls token
        for param in self.model.cls_token.parameters():
            param.requires_grad = True
        if hasattr(self.model, 'norm'):
            for param in self.model.norm.parameters():
                param.requires_grad = True
