"""Cross-attention mechanism for bidirectional modality interaction"""

import torch
import torch.nn as nn
from typing import Optional


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention fusion for multimodal learning.
    
    Implements dual cross-attention branches:
    - Branch A: Text attends to Image + Metadata
    - Branch B: Image attends to Text + Metadata
    
    Each branch uses MultiheadAttention with:
    - Q: (B, 1, 256) from one modality
    - K,V: (B, 2, 256) from concatenation of other two modalities
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize bidirectional cross-attention.
        
        Args:
            embed_dim: Embedding dimension (256)
            num_heads: Number of attention heads (8)
            dropout: Attention dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Text-to-Image+Metadata attention
        # Q: text, K,V: concat([image, metadata])
        self.cross_attn_tim = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Image-to-Text+Metadata attention
        # Q: image, K,V: concat([text, metadata])
        self.cross_attn_itm = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm_t = nn.LayerNorm(embed_dim)
        self.norm_i = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        text_proj: torch.Tensor,
        image_proj: torch.Tensor,
        metadata_proj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply bidirectional cross-attention fusion.
        
        Args:
            text_proj: Text projection (B, 256)
            image_proj: Image projection (B, 256)
            metadata_proj: Metadata projection (B, 256)
            
        Returns:
            Tuple of (text_attended, image_attended) each (B, 256)
        """
        # Add sequence dimension for attention
        text_seq = text_proj.unsqueeze(1)  # (B, 1, 256)
        image_seq = image_proj.unsqueeze(1)  # (B, 1, 256)
        metadata_seq = metadata_proj.unsqueeze(1)  # (B, 1, 256)
        
        # Branch A: text attends to image + metadata
        kv_im = torch.cat([image_seq, metadata_seq], dim=1)  # (B, 2, 256)
        text_attended, _ = self.cross_attn_tim(text_seq, kv_im, kv_im)
        text_attended = text_attended.squeeze(1)  # (B, 256)
        text_attended = self.norm_t(text_attended)
        
        # Branch B: image attends to text + metadata
        kv_tm = torch.cat([text_seq, metadata_seq], dim=1)  # (B, 2, 256)
        image_attended, _ = self.cross_attn_itm(image_seq, kv_tm, kv_tm)
        image_attended = image_attended.squeeze(1)  # (B, 256)
        image_attended = self.norm_i(image_attended)
        
        return text_attended, image_attended


class CrossAttention(nn.Module):
    """
    General cross-attention layer (deprecated - use BidirectionalCrossAttention).
    
    Kept for backwards compatibility.
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
        residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention.
        
        Args:
            query: Query embeddings (batch_size, seq_len, embedding_dim)
            key: Key embeddings (batch_size, seq_len, embedding_dim)
            value: Value embeddings (batch_size, seq_len, embedding_dim)
            residual: Optional residual connection
            
        Returns:
            Attended embeddings (batch_size, seq_len, embedding_dim)
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
        
        # Residual + layer norm
        output = self.ln2(ff_out + attn_out)
        
        return output
            residual = query
        attn_out = self.ln1(attn_out + residual)
        
        # Feed-forward
        ff_out = self.ff(attn_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.ln2(ff_out + attn_out)
        
        return ff_out
