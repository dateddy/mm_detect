"""Fusion block combining cross-attention and gating"""

import torch
import torch.nn as nn
from .cross_attention import CrossAttention
from .gating import GatingFusion


class MultimodalFusionBlock(nn.Module):
    """
    Complete fusion block orchestrating cross-attention and gating.
    
    Combines:
    1. Cross-attention between modality pairs
    2. Residual connections
    3. Layer normalization
    4. Gated fusion
    5. Final projection
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        """
        Initialize fusion block.
        
        Args:
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout rate
            use_gating: Whether to use gating fusion
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_gating = use_gating
        
        # Cross-attention layers
        self.cross_attn_ti = CrossAttention(embedding_dim, num_heads, ff_dim, dropout)  # text-image
        self.cross_attn_tm = CrossAttention(embedding_dim, num_heads, ff_dim, dropout)  # text-metadata
        self.cross_attn_im = CrossAttention(embedding_dim, num_heads, ff_dim, dropout)  # image-metadata
        
        # Gating fusion
        if use_gating:
            self.gating_fusion = GatingFusion(embedding_dim, 3, ff_dim, dropout)
        
        # Final layer norm
        self.final_ln = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        metadata_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multimodal embeddings.
        
        Args:
            text_embedding: Text embeddings (batch_size, 768)
            image_embedding: Image embeddings (batch_size, 768)
            metadata_embedding: Metadata embeddings (batch_size, 768)
            
        Returns:
            Fused embedding (batch_size, 768)
        """
        # Reshape for attention (add sequence dimension)
        text_seq = text_embedding.unsqueeze(1)  # (batch, 1, 768)
        image_seq = image_embedding.unsqueeze(1)
        metadata_seq = metadata_embedding.unsqueeze(1)
        
        # Cross-attention
        text_attended = self.cross_attn_ti(text_seq, image_seq, image_seq, text_seq).squeeze(1)
        image_attended = self.cross_attn_im(image_seq, metadata_seq, metadata_seq, image_seq).squeeze(1)
        metadata_attended = self.cross_attn_tm(metadata_seq, text_seq, text_seq, metadata_seq).squeeze(1)
        
        # Gating fusion
        if self.use_gating:
            fused = self.gating_fusion(text_attended, image_attended, metadata_attended)
        else:
            # Simple average if no gating
            fused = (text_attended + image_attended + metadata_attended) / 3
        
        # Final layer norm
        fused = self.final_ln(fused)
        
        return fused
