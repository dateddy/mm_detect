"""Main multimodal model for misinformation classification"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .fusion.fusion_block import MultimodalFusionBlock


class MultimodalModel(nn.Module):
    """
    Multimodal misinformation detection model.
    
    Combines text (PhoBERT), image (ViT), and metadata features
    through cross-attention and gated fusion.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 8,
        num_classes: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_gating: bool = True,
        use_fusion: bool = True
    ):
        """
        Initialize multimodal model.
        
        Args:
            embedding_dim: Embedding dimension (768)
            num_heads: Number of attention heads
            num_classes: Number of output classes (1 for binary)
            hidden_dim: Hidden dimensions for MLPs
            dropout: Dropout rate
            use_gating: Use gating fusion mechanism
            use_fusion: Use fusion block (vs simple concatenation)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_fusion = use_fusion
        
        # Fusion block
        if use_fusion:
            self.fusion = MultimodalFusionBlock(
                embedding_dim,
                num_heads,
                hidden_dim,
                dropout,
                use_gating
            )
            fusion_output_dim = embedding_dim
        else:
            # Simple concatenation
            fusion_output_dim = embedding_dim * 3
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        metadata_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text_embedding: Text embeddings (batch_size, 768)
            image_embedding: Image embeddings (batch_size, 768)
            metadata_embedding: Metadata embeddings (batch_size, 768)
            
        Returns:
            Logits (batch_size, 1)
        """
        # Fusion
        if self.use_fusion:
            fused = self.fusion(text_embedding, image_embedding, metadata_embedding)
        else:
            # Concatenate all modalities
            fused = torch.cat([text_embedding, image_embedding, metadata_embedding], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
