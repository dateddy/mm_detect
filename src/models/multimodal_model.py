"""Main multimodal model for misinformation classification

Implements the complete architecture as specified:
- Per-modality projection (768→256 for text/image, 256→256 for metadata)
- Modality dropout (p=0.15 during training)
- Dual cross-attention (text→image+metadata, image→text+metadata)
- Nonlinear gated fusion
- Residual connection
- Classification head (256→128→64→1 with GELU+Dropout)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .fusion.cross_attention import BidirectionalCrossAttention
from .fusion.gating import GatingFusion
from .fusion.projection import ModalityProjection, ModalityDropout


class MultimodalFusionBlock(nn.Module):
    """
    Complete fusion pipeline:
    1. Per-modality projection (Linear + LayerNorm)
    2. Modality dropout
    3. Dual cross-attention
    4. Nonlinear gated fusion
    5. Residual + layer norm
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout_rate: float = 0.15
    ):
        """
        Initialize fusion block.
        
        Args:
            hidden_dim: Projection output dimension (256)
            num_heads: Number of attention heads (8)
            dropout: Attention dropout
            modality_dropout_rate: Probability of zeroing each modality (0.15)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Per-modality projections: 768→256 (text/image), 256→256 (metadata)
        self.text_proj = ModalityProjection(768, hidden_dim, use_layer_norm=True)
        self.image_proj = ModalityProjection(768, hidden_dim, use_layer_norm=True)
        self.metadata_proj = ModalityProjection(256, hidden_dim, use_layer_norm=True)
        
        # Modality dropout
        self.modality_dropout = ModalityDropout(dropout_rate=modality_dropout_rate)
        
        # Dual cross-attention
        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Nonlinear gated fusion
        self.gating_fusion = GatingFusion(
            input_dim=3 * hidden_dim,  # 768 = 3 * 256
            num_modalities=3,
            hidden_dim=3 * hidden_dim,
            dropout=dropout
        )
        
        # Final layer norm after residual
        self.final_ln = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        metadata_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multimodal embeddings through complete pipeline.
        
        Args:
            text_emb: Text embeddings from PhoBERT (B, 768)
            image_emb: Image embeddings from ViT (B, 768)
            metadata_emb: Metadata embeddings from MLP encoder (B, 256)
            
        Returns:
            Fused embedding (B, 256)
        """
        # Step 1: Per-modality projection
        text_proj = self.text_proj(text_emb)      # (B, 256)
        image_proj = self.image_proj(image_emb)    # (B, 256)
        metadata_proj = self.metadata_proj(metadata_emb)  # (B, 256)
        
        # Step 2: Modality dropout (training only)
        text_proj = self.modality_dropout(text_proj)
        image_proj = self.modality_dropout(image_proj)
        metadata_proj = self.modality_dropout(metadata_proj)
        
        # Step 3: Dual cross-attention
        # Branch A: text attends to image+metadata → text'
        # Branch B: image attends to text+metadata → image'
        text_attended, image_attended = self.cross_attention(
            text_proj, image_proj, metadata_proj
        )
        
        # Step 4: Nonlinear gated fusion
        # gate_input = concat([T', I', M]) → (B, 768)
        # gates = sigmoid(Linear(768, 768)) → (B, 3, 256) after chunking
        # fused = g1*T' + g2*I' + g3*M
        fused = self.gating_fusion(text_attended, image_attended, metadata_proj)
        
        # Step 5: Residual connection and layer norm
        # out = LayerNorm(fused + text_proj + image_proj + metadata_proj)
        residual_sum = text_proj + image_proj + metadata_proj
        output = self.final_ln(fused + residual_sum)
        
        return output


class ClassificationHead(nn.Module):
    """
    Classification head:
    256 → 128 → GELU → Dropout(0.3)
    → 64 → GELU → Dropout(0.3)
    → 1
    
    Output: logits (no sigmoid; use BCEWithLogitsLoss)
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize classification head.
        
        Args:
            input_dim: Input dimension (256)
            hidden_dim: First hidden layer dimension (128)
            dropout: Dropout rate (0.3)
        """
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 128 // 2 = 64
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output logit
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Fused embedding (B, 256)
            
        Returns:
            Logits (B, 1) - no sigmoid applied
        """
        return self.head(x)


class MultimodalModel(nn.Module):
    """
    Complete multimodal misinformation detection model.
    
    Pipeline:
    1. Encode text (PhoBERT) → 768d
    2. Encode image (ViT) → 768d
    3. Encode metadata (MLP) → 256d
    4. Fusion block (projection + dropout + cross-attn + gating)
    5. Classification head → logit
    
    Loss:
    - BCEWithLogitsLoss (with class weights)
    - + 0.1 * InfoNCE(text_proj, image_proj, temp=0.07)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        modality_dropout_rate: float = 0.15,
        head_dropout: float = 0.3
    ):
        """
        Initialize multimodal model.
        
        Args:
            hidden_dim: Projection / fusion hidden dimension (256)
            num_heads: Number of attention heads (8)
            attention_dropout: Dropout in attention (0.1)
            modality_dropout_rate: Modality dropout rate (0.15)
            head_dropout: Classification head dropout (0.3)
        """
        super().__init__()
        
        # Fusion block
        self.fusion = MultimodalFusionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            modality_dropout_rate=modality_dropout_rate
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=hidden_dim,
            hidden_dim=128,
            dropout=head_dropout
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
            text_embedding: Text embeddings from encoder (B, 768)
            image_embedding: Image embeddings from encoder (B, 768)
            metadata_embedding: Metadata embeddings from encoder (B, 256)
            
        Returns:
            Logits (B, 1) - no sigmoid
        """
        # Fusion with projection, dropout, cross-attention, gating
        fused = self.fusion(text_embedding, image_embedding, metadata_embedding)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
