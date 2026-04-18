"""Shared projection layers for each modality"""

import torch
import torch.nn as nn


class ModalityProjection(nn.Module):
    """
    Projects each modality embedding to a shared dimension with projection and layer norm.
    
    Architecture: Linear(input_dim → hidden_dim) + LayerNorm(hidden_dim)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_layer_norm: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize modality projection.
        
        Args:
            input_dim: Input embedding dimension (768 for text/image, 256 for metadata)
            output_dim: Output projection dimension (256)
            use_layer_norm: Apply LayerNorm after projection
            dropout: Dropout rate applied after projection
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, output_dim)
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = nn.Identity()
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to shared space.
        
        Args:
            x: Input embeddings (batch_size, input_dim)
            
        Returns:
            Projected embeddings (batch_size, output_dim)
        """
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class ModalityDropout(nn.Module):
    """
    Randomly zeros entire modality embeddings during training.
    
    Encourages graceful degradation when modalities are missing at inference.
    Each modality is independently zeroed with probability p during training.
    Disabled during inference (eval mode).
    """
    
    def __init__(self, dropout_rate: float = 0.15):
        """
        Initialize modality dropout.
        
        Args:
            dropout_rate: Probability of zeroing each modality (default: 0.15)
        """
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply modality dropout.
        
        Args:
            x: Embedding tensor (batch_size, dim)
            
        Returns:
            Embedding with possible zero-ing (batch_size, dim)
        """
        if self.training and self.dropout_rate > 0:
            # Probability of keeping the modality
            keep_prob = 1.0 - self.dropout_rate
            
            # Create binary mask for this batch
            # All samples in batch get same dropout decision
            mask = torch.bernoulli(
                torch.tensor([keep_prob] * x.shape[0])
            ).to(x.device)
            
            # Apply mask
            x = x * mask.unsqueeze(-1)
        
        return x
