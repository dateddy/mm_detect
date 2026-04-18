"""Gating mechanism for nonlinear multimodal fusion"""

import torch
import torch.nn as nn


class GatingFusion(nn.Module):
    """
    Nonlinear gated fusion for multimodal inputs.
    
    Architecture:
    1. Concatenate all modality projections: (B, 768)
    2. Pass through MLP to get per-modality gates: (B, 3)
    3. Element-wise multiply each modality by its gate
    4. Sum weighted modalities
    
    This allows the model to learn dynamic, sample-specific fusion weights.
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # 3 * 256
        num_modalities: int = 3,
        hidden_dim: int = 768,
        dropout: float = 0.1
    ):
        """
        Initialize gating fusion.
        
        Args:
            input_dim: Concatenated dimension (3 * 256 = 768)
            num_modalities: Number of modalities (3)
            hidden_dim: MLP hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_modalities = num_modalities
        
        # MLP to compute gating weights
        # Input: concat([T', I', M]) → (B, 768)
        # Output: gate weights for each modality → (B, 3)
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(
        self,
        text_proj: torch.Tensor,
        image_proj: torch.Tensor,
        metadata_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse modalities using learned gates.
        
        Args:
            text_proj: Text projection (B, 256)
            image_proj: Image projection (B, 256)
            metadata_proj: Metadata projection (B, 256)
            
        Returns:
            Gated fusion output (B, 256)
        """
        # Concatenate all modalities
        concatenated = torch.cat(
            [text_proj, image_proj, metadata_proj],
            dim=-1
        )  # (B, 768)
        
        # Compute per-modality gates
        gates = self.gate_mlp(concatenated)  # (B, 3)
        
        # Split gates for each modality
        g_text, g_image, g_metadata = gates.chunk(3, dim=-1)  # each (B, 1)
        
        # Weighted sum of modalities
        fused = (
            g_text * text_proj +
            g_image * image_proj +
            g_metadata * metadata_proj
        )  # (B, 256)
        
        return fused


class OldGatingFusion(nn.Module):
    """
    Alternative gating fusion (kept for backwards compatibility).
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
