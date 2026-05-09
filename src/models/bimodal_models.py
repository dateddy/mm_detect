"""
Bimodal model classes for ablation studies.

Each class combines exactly two modalities with SIMPLE CONCAT FUSION (not dual cross-attention).
This keeps the ablation clean and focuses on modality contribution, not fusion architecture.

Models:
- TextImageModel: Text + Image (no metadata)
- TextMetadataModel: Text + Metadata (no image)
- ImageMetadataModel: Image + Metadata (no text)
"""
import torch
import torch.nn as nn
import timm
from typing import Dict

from .components import MetadataMLP
from .classification_head import build_auxiliary_head, initialize_classifier_prior_bias
from .text_encoder import TextEncoder


class _BimodalBase(nn.Module):
    """Shared logic for bimodal models — concat-based fusion."""
    
    def _build_classifier(
        self,
        proj_dim: int,
        dropout: float = 0.3,
        config: Dict | None = None,
    ):
        """Build classification head."""
        classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        initialize_classifier_prior_bias(classifier, config)
        return classifier
    
    def _build_simple_fusion(self, proj_dim: int):
        """Concat 2 modalities (2*proj_dim) and project back to proj_dim."""
        return nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )


class TextImageModel(_BimodalBase):
    """Text + Image bimodal model (no metadata).
    
    Architecture:
    - PhoBERT → Linear(768→256) → Text projection
    - ViT-B/16 → Linear(768→256) → Image projection
    - Concat projections → Linear(512→256) → Classification head
    
    Params: ~220M
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.ablation_mode = "text_image"
        
        cfg_model = config.get("model", {})
        cfg_train = config.get("training", {})
        
        text_model_name = cfg_model.get("text_model_name", "vinai/phobert-base-v2")
        image_model_name = cfg_model.get("image_model_name", "vit_base_patch16_224")
        proj_dim = cfg_model.get("projection_dim", 256)
        freeze_epochs = cfg_train.get("freeze_encoder_epochs", 0)
        
        # === Text Encoder (PhoBERT) ===
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            freeze=freeze_epochs > 0,
        )
        text_dim = self.text_encoder.output_dim
        
        # === Image Encoder (ViT-B/16) ===
        self.image_encoder = timm.create_model(
            image_model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
        )
        image_dim = self.image_encoder.num_features
        
        if freeze_epochs > 0:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        
        # === Projections ===
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(cfg_model.get("projection_dropout", 0.1)),
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(cfg_model.get("projection_dropout", 0.1)),
        )
        
        # === Simple Concat Fusion ===
        self.gated_fusion = self._build_simple_fusion(proj_dim)
        
        # === Classifier ===
        self.classifier = self._build_classifier(
            proj_dim, cfg_model.get("classifier_dropout", 0.3), config
        )
        aux_dropout = cfg_model.get("aux_dropout", 0.2)
        self.aux_text_head = build_auxiliary_head(proj_dim, aux_dropout, config)
        self.aux_image_head = build_auxiliary_head(proj_dim, aux_dropout, config)
        
        # Explicitly None for factory verification
        self.metadata_encoder = None
        self.meta_proj = None
        self.dual_cross_attn = None
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass for text+image model."""
        # Text encoding
        if "text_emb" in batch:
            text_repr = batch["text_emb"]
        else:
            text_repr = self.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        t_proj = self.text_proj(text_repr)
        
        # Image encoding
        if "image_emb" in batch:
            image_repr = batch["image_emb"]
        else:
            image_repr = self.image_encoder(batch["pixel_values"])
        i_proj = self.image_proj(image_repr)
        
        # Concat fusion
        fused = self.gated_fusion(torch.cat([t_proj, i_proj], dim=-1))
        
        # Classification
        logits = self.classifier(fused)
        
        return {
            "logits": logits,
            "t_proj": t_proj,
            "i_proj": i_proj,
            "aux_text_logits": self.aux_text_head(t_proj),
            "aux_image_logits": self.aux_image_head(i_proj),
        }
    
    def transition_to_phase2(self, k: int = 4):
        """Unfreeze top-k blocks of both encoders."""
        params = []
        
        # Unfreeze text encoder blocks
        self.text_encoder.unfreeze_top_k(k)
        text_blocks = self.text_encoder.model.encoder.layer
        for block in text_blocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True
                params.append(p)
        
        # Unfreeze image encoder blocks
        image_blocks = self.image_encoder.blocks
        for block in image_blocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True
                params.append(p)
        
        return params

    def unfreeze_encoders(self, top_k: int) -> None:
        """Trainer-compatible encoder unfreeze hook."""
        self.text_encoder.unfreeze_top_k(top_k)
        for p in self.transition_to_phase2(top_k):
            p.requires_grad = True
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count total or trainable parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class TextMetadataModel(_BimodalBase):
    """Text + Metadata bimodal model (no image).
    
    Architecture:
    - PhoBERT → Linear(768→256) → Text projection
    - MLP(17→256) → Linear(256→256) → Metadata projection
    - Concat projections → Linear(512→256) → Classification head
    
    Params: ~135M
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.ablation_mode = "text_metadata"
        
        cfg_model = config.get("model", {})
        cfg_train = config.get("training", {})
        
        text_model_name = cfg_model.get("text_model_name", "vinai/phobert-base-v2")
        proj_dim = cfg_model.get("projection_dim", 256)
        freeze_epochs = cfg_train.get("freeze_encoder_epochs", 0)
        
        # Get metadata features
        metadata_features = config.get("metadata_features", [])
        n_metadata_features = len(metadata_features) if metadata_features else 17
        
        # === Text Encoder (PhoBERT) ===
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            freeze=freeze_epochs > 0,
        )
        text_dim = self.text_encoder.output_dim
        
        # === Metadata Encoder (MLP) ===
        self.metadata_encoder = MetadataMLP(
            input_dim=n_metadata_features,
            output_dim=proj_dim,
            hidden_dims=[64, 128, 256],
            dropout=0.1,
        )
        
        # === Projections ===
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(cfg_model.get("projection_dropout", 0.1)),
        )
        
        self.meta_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(cfg_model.get("projection_dropout", 0.1)),
        )
        
        # === Simple Concat Fusion ===
        self.gated_fusion = self._build_simple_fusion(proj_dim)
        
        # === Classifier ===
        self.classifier = self._build_classifier(
            proj_dim, cfg_model.get("classifier_dropout", 0.3), config
        )
        aux_dropout = cfg_model.get("aux_dropout", 0.2)
        self.aux_text_head = build_auxiliary_head(proj_dim, aux_dropout, config)
        self.aux_meta_head = build_auxiliary_head(proj_dim, aux_dropout, config)
        
        # Explicitly None for factory verification
        self.image_encoder = None
        self.image_proj = None
        self.dual_cross_attn = None
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass for text+metadata model."""
        # Text encoding
        if "text_emb" in batch:
            text_repr = batch["text_emb"]
        else:
            text_repr = self.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        t_proj = self.text_proj(text_repr)
        
        # Metadata encoding
        meta_repr = self.metadata_encoder(batch["metadata"])
        m_proj = self.meta_proj(meta_repr)
        
        # Concat fusion
        fused = self.gated_fusion(torch.cat([t_proj, m_proj], dim=-1))
        
        # Classification
        logits = self.classifier(fused)
        
        return {
            "logits": logits,
            "t_proj": t_proj,
            "m_proj": m_proj,
            "aux_text_logits": self.aux_text_head(t_proj),
            "aux_meta_logits": self.aux_meta_head(m_proj),
        }
    
    def transition_to_phase2(self, k: int = 4):
        """Unfreeze top-k blocks of text encoder."""
        params = []
        
        # Unfreeze text encoder blocks
        self.text_encoder.unfreeze_top_k(k)
        text_blocks = self.text_encoder.model.encoder.layer
        for block in text_blocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True
                params.append(p)
        
        return params

    def unfreeze_encoders(self, top_k: int) -> None:
        """Trainer-compatible encoder unfreeze hook."""
        self.text_encoder.unfreeze_top_k(top_k)
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count total or trainable parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class ImageMetadataModel(_BimodalBase):
    """Image + Metadata bimodal model (no text).
    
    Architecture:
    - ViT-B/16 → Linear(768→256) → Image projection
    - MLP(17→256) → Linear(256→256) → Metadata projection
    - Concat projections → Linear(512→256) → Classification head
    
    Params: ~85M
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.ablation_mode = "image_metadata"
        
        cfg_model = config.get("model", {})
        cfg_train = config.get("training", {})
        
        image_model_name = cfg_model.get("image_model_name", "vit_base_patch16_224")
        proj_dim = cfg_model.get("projection_dim", 256)
        freeze_epochs = cfg_train.get("freeze_encoder_epochs", 0)
        
        # Get metadata features
        metadata_features = config.get("metadata_features", [])
        n_metadata_features = len(metadata_features) if metadata_features else 17
        
        # === Image Encoder (ViT-B/16) ===
        self.image_encoder = timm.create_model(
            image_model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
        )
        image_dim = self.image_encoder.num_features
        
        if freeze_epochs > 0:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        
        # === Metadata Encoder (MLP) ===
        self.metadata_encoder = MetadataMLP(
            input_dim=n_metadata_features,
            output_dim=proj_dim,
            hidden_dims=[64, 128, 256],
            dropout=0.1,
        )
        
        # === Projections ===
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(cfg_model.get("projection_dropout", 0.1)),
        )
        
        self.meta_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(cfg_model.get("projection_dropout", 0.1)),
        )
        
        # === Simple Concat Fusion ===
        self.gated_fusion = self._build_simple_fusion(proj_dim)
        
        # === Classifier ===
        self.classifier = self._build_classifier(
            proj_dim, cfg_model.get("classifier_dropout", 0.3), config
        )
        aux_dropout = cfg_model.get("aux_dropout", 0.2)
        self.aux_image_head = build_auxiliary_head(proj_dim, aux_dropout, config)
        self.aux_meta_head = build_auxiliary_head(proj_dim, aux_dropout, config)
        
        # Explicitly None for factory verification
        self.text_encoder = None
        self.text_proj = None
        self.dual_cross_attn = None
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass for image+metadata model."""
        # Image encoding
        if "image_emb" in batch:
            image_repr = batch["image_emb"]
        else:
            image_repr = self.image_encoder(batch["pixel_values"])
        i_proj = self.image_proj(image_repr)
        
        # Metadata encoding
        meta_repr = self.metadata_encoder(batch["metadata"])
        m_proj = self.meta_proj(meta_repr)
        
        # Concat fusion
        fused = self.gated_fusion(torch.cat([i_proj, m_proj], dim=-1))
        
        # Classification
        logits = self.classifier(fused)
        
        return {
            "logits": logits,
            "i_proj": i_proj,
            "m_proj": m_proj,
            "aux_image_logits": self.aux_image_head(i_proj),
            "aux_meta_logits": self.aux_meta_head(m_proj),
        }
    
    def transition_to_phase2(self, k: int = 4):
        """Unfreeze top-k blocks of image encoder."""
        params = []
        
        # Unfreeze image encoder blocks
        image_blocks = self.image_encoder.blocks
        for block in image_blocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True
                params.append(p)
        
        return params

    def unfreeze_encoders(self, top_k: int) -> None:
        """Trainer-compatible encoder unfreeze hook."""
        for p in self.transition_to_phase2(top_k):
            p.requires_grad = True
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count total or trainable parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
