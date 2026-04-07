"""Metadata encoder for categorical and numerical features"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class MetadataEncoder(nn.Module):
    """
    Metadata encoder for categorical and numerical features.
    
    Combines embeddings from categorical features with normalized
    numerical features into a single representation.
    """
    
    def __init__(
        self,
        categorical_features: Dict[str, int],
        embedding_dim: int = 16,
        numerical_features: int = 5,
        output_dim: int = 768,
        hidden_dim: int = 256
    ):
        """
        Initialize metadata encoder.
        
        Args:
            categorical_features: Dict mapping feature names to vocab sizes
            embedding_dim: Embedding dimension for categorical features
            numerical_features: Number of numerical features
            output_dim: Output embedding dimension (768 to match text/image)
            hidden_dim: Hidden dimension for MLP
        """
        super().__init__()
        
        self.categorical_features = categorical_features
        self.embedding_dim = embedding_dim
        self.numerical_features = numerical_features
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        for feat_name, vocab_size in categorical_features.items():
            self.embeddings[feat_name] = nn.Embedding(vocab_size, embedding_dim)
        
        # Calculate input dimension
        cat_dim = len(categorical_features) * embedding_dim
        input_dim = cat_dim + numerical_features
        
        # MLP projection to output dimension
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        categorical_data: Dict[str, torch.Tensor],
        numerical_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode metadata to embeddings.
        
        Args:
            categorical_data: Dict mapping feature names to LongTensors
            numerical_data: Tensor of shape (batch_size, num_numerical_features)
            
        Returns:
            Embeddings (batch_size, 768)
        """
        # Embed categorical features
        embedded_features = []
        for feat_name in self.categorical_features.keys():
            if feat_name in categorical_data:
                embedded = self.embeddings[feat_name](categorical_data[feat_name])
                embedded_features.append(embedded)
        
        # Concatenate embeddings and numerical features
        combined = torch.cat(embedded_features + [numerical_data], dim=1)
        
        # Project to output dimension
        output = self.mlp(combined)
        
        return output
    
    @property
    def output_dim(self) -> int:
        return 768
