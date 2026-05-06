"""
Reusable model components shared across ablation models.
"""
import torch
import torch.nn as nn
from typing import List


class MetadataMLP(nn.Module):
    """
    Shared metadata encoder used by full model, metadata-only, T+M, I+M ablations.
    
    Architecture: input_dim → 64 → 128 → 256 with BatchNorm, GELU, Dropout, residual.
    
    Args:
        input_dim: Number of input metadata features (default: 17)
        output_dim: Dimension of output embedding (default: 256)
        hidden_dims: List of hidden layer dimensions (default: [64, 128, 256])
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(
        self,
        input_dim: int = 17,
        output_dim: int = 256,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Residual projection for direct path (preserves binary feature signals)
        if in_dim != output_dim:
            self.residual = nn.Linear(input_dim, output_dim)
        else:
            self.residual = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, input_dim] metadata features
            
        Returns:
            [B, output_dim] metadata embedding
        """
        out = self.mlp(x)
        
        if self.residual is not None:
            out = out + self.residual(x)
        
        return out
