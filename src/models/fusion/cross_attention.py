"""Cross-attention mechanism for multimodal interaction"""

import torch
import torch.nn as nn
from typing import Tuple


class CrossAttention(nn.Module):
    """
    Cross-attention layer for multimodal fusion.
    
    Allows each modality to attend to other modalities.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention.
        
        Args:
            embedding_dim: Embedding dimension (768)
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head attention for cross-modal interaction
        self.cross_attn = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        residual: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply cross-attention.
        
        Args:
            query: Query embeddings (batch_size, 1, embedding_dim)
            key: Key embeddings (batch_size, 1, embedding_dim)
            value: Value embeddings (batch_size, 1, embedding_dim)
            residual: Optional residual connection
            
        Returns:
            Attended embeddings (batch_size, 1, embedding_dim)
        """
        # Cross-attention
        attn_out, _ = self.cross_attn(query, key, value)
        attn_out = self.dropout(attn_out)
        
        # Residual + layer norm
        if residual is None:
            residual = query
        attn_out = self.ln1(attn_out + residual)
        
        # Feed-forward
        ff_out = self.ff(attn_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.ln2(ff_out + attn_out)
        
        return ff_out
