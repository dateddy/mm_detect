"""Image encoder using Vision Transformer (ViT)"""

import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
from typing import Union
from PIL import Image


class ImageEncoder(nn.Module):
    """
    Image encoder using Vision Transformer.
    
    Extracts 768-dimensional embeddings from images using the
    pretrained ViT model.
    """
    
    def __init__(
        self,
        model_name: str = 'google/vit-base-patch16-224',
        freeze_embeddings: bool = False
    ):
        """
        Initialize image encoder.
        
        Args:
            model_name: Pretrained model identifier
            freeze_embeddings: Freeze pretrained weights if True
        """
        super().__init__()
        self.model_name = model_name
        
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        
        if freeze_embeddings:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        images: Union[list, torch.Tensor],
        device: torch.device
    ) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: List of PIL Images or tensor of shape (N, 3, 224, 224)
            device: Device to use
            
        Returns:
            Embeddings (batch_size, 768)
        """
        # If list of PIL images, process with feature extractor
        if isinstance(images, list):
            image_tensor = self.feature_extractor(
                images=images,
                return_tensors='pt'
            )['pixel_values'].to(device)
        else:
            image_tensor = images.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Extract [CLS] token (pooled representation)
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    @property
    def output_dim(self) -> int:
        return 768
