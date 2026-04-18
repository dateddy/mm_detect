"""Metadata encoder for 9 behavioral features"""

import torch
import torch.nn as nn
from typing import Optional


class MetadataEncoder(nn.Module):
    """
    Metadata encoder for 9 engineered behavioral features.
    
    Architecture: 9 → 256 → 256 → 256 with BatchNorm and GELU activation.
    Output: 256-dimensional embedding.
    
    Features (all numeric after preprocessing):
    - ads_per_page (RobustScaled)
    - platform_count (RobustScaled)
    - FB_only_flag (binary 0/1)
    - all_targeted (binary 0/1)
    - burstiness (RobustScaled)
    - avg_ad_duration (RobustScaled)
    - launch_delay (RobustScaled)
    - num_countries (RobustScaled)
    - language_location_mismatch (binary 0/1)
    """
    
    def __init__(
        self,
        num_features: int = 9,
        hidden_dims: Optional[list[int]] = None,
        output_dim: int = 256,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
        activation: str = "gelu"
    ):
        """
        Initialize metadata encoder.
        
        Args:
            num_features: Number of input features (9)
            hidden_dims: Hidden layer dimensions, default [256, 256]
            output_dim: Output dimension (256)
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ("gelu", "relu", etc.)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.num_features = num_features
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP: 9 → 256 → 256 → 256
        layers = []
        
        # First layer: 9 → first hidden dim
        layers.append(nn.Linear(num_features, hidden_dims[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(act_fn)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Middle layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Final output layer: last hidden dim → output dim
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Encode metadata to embeddings.
        
        Args:
            metadata: Metadata tensor (batch_size, 9)
            
        Returns:
            Embeddings (batch_size, 256)
        """
        return self.mlp(metadata)
