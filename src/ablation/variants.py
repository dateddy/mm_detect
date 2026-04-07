"""Model variants for ablation studies"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class TextOnlyModel(nn.Module):
    """Text-only variant for ablation."""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_embedding, image_embedding=None, metadata_embedding=None):
        return self.classifier(text_embedding)


class ImageOnlyModel(nn.Module):
    """Image-only variant for ablation."""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_embedding=None, image_embedding=None, metadata_embedding=None):
        return self.classifier(image_embedding)


class MetadataOnlyModel(nn.Module):
    """Metadata-only variant for ablation."""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_embedding=None, image_embedding=None, metadata_embedding=None):
        return self.classifier(metadata_embedding)


class TextImageModel(nn.Module):
    """Text + Image variant (no metadata) for ablation."""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_embedding, image_embedding=None, metadata_embedding=None):
        combined = torch.cat([text_embedding, image_embedding], dim=1)
        return self.classifier(combined)


class FullMultimodalModel(nn.Module):
    """Full multimodal model (baseline for ablation)."""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_embedding, image_embedding=None, metadata_embedding=None):
        combined = torch.cat([text_embedding, image_embedding, metadata_embedding], dim=1)
        return self.classifier(combined)


def get_model_variant(
    variant_name: str,
    config: Dict = None
) -> nn.Module:
    """
    Get a model variant for ablation study.
    
    Args:
        variant_name: Name of variant
        config: Configuration dict (optional)
        
    Returns:
        Model instance
    """
    config = config or {}
    embedding_dim = config.get('embedding_dim', 768)
    hidden_dim = config.get('hidden_dim', 256)
    
    variants = {
        'text_only': TextOnlyModel,
        'image_only': ImageOnlyModel,
        'metadata_only': MetadataOnlyModel,
        'text_image': TextImageModel,
        'full_multimodal': FullMultimodalModel,
    }
    
    if variant_name not in variants:
        raise ValueError(f'Unknown variant: {variant_name}')
    
    model_class = variants[variant_name]
    return model_class(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
