"""Dataset class for loading and preprocessing multimodal data"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal data with cached embeddings.
    
    Loads pre-extracted embeddings (text, image) and metadata features.
    """
    
    def __init__(
        self,
        indices: np.ndarray,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        metadata_features: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Initialize dataset.
        
        Args:
            indices: Array of sample indices
            text_embeddings: Pre-extracted text embeddings (N, 768)
            image_embeddings: Pre-extracted image embeddings (N, 768)
            metadata_features: Metadata features (N, D)
            labels: Target labels (N,)
        """
        self.indices = indices
        self.text = text_embeddings
        self.image = image_embeddings
        self.metadata = metadata_features
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with 'text', 'image', 'metadata', 'label'
        """
        i = self.indices[idx]
        return {
            'text': self.text[i],
            'image': self.image[i],
            'metadata': self.metadata[i],
            'label': self.labels[i]
        }


class TextOnlyDataset(Dataset):
    """Dataset for text-only ablation study."""
    
    def __init__(self, indices: np.ndarray, text_embeddings: torch.Tensor, labels: torch.Tensor):
        self.indices = indices
        self.text = text_embeddings
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        return {'text': self.text[i], 'label': self.labels[i]}


class ImageOnlyDataset(Dataset):
    """Dataset for image-only ablation study."""
    
    def __init__(self, indices: np.ndarray, image_embeddings: torch.Tensor, labels: torch.Tensor):
        self.indices = indices
        self.image = image_embeddings
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        return {'image': self.image[i], 'label': self.labels[i]}


class MetadataOnlyDataset(Dataset):
    """Dataset for metadata-only ablation study."""
    
    def __init__(self, indices: np.ndarray, metadata_features: torch.Tensor, labels: torch.Tensor):
        self.indices = indices
        self.metadata = metadata_features
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        return {'metadata': self.metadata[i], 'label': self.labels[i]}
