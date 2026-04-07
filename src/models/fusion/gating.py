"""Gating mechanism for dynamic modality weighting"""

import torch
import torch.nn as nn
from typing import Tuple


class GatingFusion(nn.Module):
    """
    Gating-based fusion mechanism for multimodal inputs.
    
    Learns dynamic weights for each modality based on their combined
    representation, allowing the model to adjust fusion weights per sample.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        num_modalities: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize gating fusion.
        
        Args:
            embedding_dim: Embedding dimension
            num_modalities: Number of modalities (default: text, image, metadata)
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        
        # MLP for gate computation
        self.gate_mlp = nn.Sequential(
            nn.Linear(embedding_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid()
        )
    
    def forward(self, *modality_embeddings) -> torch.Tensor:
        """
        Fuse modalities using learned gates.
        
        Args:
            *modality_embeddings: Variable number of embeddings (N, 768)
            
        Returns:
            Fused embedding (batch_size, 768)
        """
        # Concatenate all modalities
        combined = torch.cat(modality_embeddings, dim=-1)  # (batch, 768*3)
        
        # Compute gates
        gates = self.gate_mlp(combined)  # (batch, num_modalities)
        
        # Stack modalities for broadcasting
        stacked = torch.stack(modality_embeddings, dim=1)  # (batch, num_modalities, 768)
        
        # Apply gates
        gates = gates.unsqueeze(-1)  # (batch, num_modalities, 1)
        weighted = stacked * gates  # (batch, num_modalities, 768)
        
        # Sum weighted modalities
        fused = weighted.sum(dim=1)  # (batch, 768)
        
        return fused
