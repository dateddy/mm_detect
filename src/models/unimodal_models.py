"""
Single-modality model classes for ablation studies.

Each class contains ONLY the components necessary for its modality:
- TextOnlyModel: PhoBERT → Linear(768→256) → Classifier (~134M params)
- ImageOnlyModel: ViT-B/16 → Linear(768→256) → Classifier (~85M params)
- MetadataOnlyModel: MLP(17→256) → Classifier (~400-800K params)

No dead code branches, no disabled encoders, no wasted components.
"""
import torch
import torch.nn as nn
from transformers import AutoModel
import timm
from typing import Dict

from .components import MetadataMLP


class TextOnlyModel(nn.Module):
    """Text-only ablation model.
    
    Architecture: PhoBERT → Linear(768→256) → LayerNorm+GELU+Dropout → Classifier
    
    Params: ~134M
    No image encoder, no metadata encoder, no cross-attention, no contrastive loss.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.ablation_mode = "text_only"
        
        cfg_model = config.get("model", {})
        cfg_train = config.get("training", {})
        
        # === Text Encoder (PhoBERT) ===
        text_model_name = cfg_model.get("text_model_name", "vinai/phobert-base-v2")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 768
        
        # Freeze in Phase 1
        freeze_epochs = cfg_train.get("freeze_encoder_epochs", 0)
        if freeze_epochs > 0:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        
        # === Text Projection ===
        proj_dim = cfg_model.get("projection_dim", 256)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(cfg_model.get("projection_dropout", 0.1)),
        )
        
        # === Classifier Head ===
        # Same architecture as full model classifier for fair comparison
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg_model.get("classifier_dropout", 0.3)),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(cfg_model.get("classifier_dropout", 0.3)),
            nn.Linear(64, 1),
        )
        
        # Explicitly mark non-existent modules as None for factory verification
        self.image_encoder = None
        self.metadata_encoder = None
        self.image_proj = None
        self.meta_proj = None
        self.dual_cross_attn = None
        self.gated_fusion = None
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass for text-only model."""
        # === Text path only ===
        text_out = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        # [CLS] token from pooler or first hidden state
        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            text_repr = text_out.pooler_output  # [B, 768]
        else:
            text_repr = text_out.last_hidden_state[:, 0, :]  # [B, 768]
        
        t_proj = self.text_proj(text_repr)  # [B, 256]
        
        # === Classifier ===
        logits = self.classifier(t_proj)  # [B, 1]
        
        return {
            "logits": logits,
            "t_proj": t_proj,  # exposed for diagnostic logging
        }
    
    def transition_to_phase2(self, k: int = 4):
        """Unfreeze top-k blocks of PhoBERT."""
        text_blocks = self.text_encoder.encoder.layer
        params = []
        for block in text_blocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True
                params.append(p)
        return params
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count total or trainable parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class ImageOnlyModel(nn.Module):
    """Image-only ablation model.
    
    Architecture: ViT-B/16 → Linear(768→256) → LayerNorm+GELU+Dropout → Classifier
    
    Params: ~85M
    No text encoder, no metadata encoder, no cross-attention.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.ablation_mode = "image_only"
        
        cfg_model = config.get("model", {})
        cfg_train = config.get("training", {})
        
        # === Image Encoder (ViT-B/16) ===
        image_model_name = cfg_model.get("image_model_name", "vit_base_patch16_224")
        self.image_encoder = timm.create_model(
            image_model_name,
            pretrained=True,
            num_classes=0,  # remove classification head
        )
        image_dim = self.image_encoder.num_features  # 768
        
        freeze_epochs = cfg_train.get("freeze_encoder_epochs", 0)
        if freeze_epochs > 0:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        
        # === Image Projection ===
        proj_dim = cfg_model.get("projection_dim", 256)
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(cfg_model.get("projection_dropout", 0.1)),
        )
        
        # === Classifier Head ===
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg_model.get("classifier_dropout", 0.3)),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(cfg_model.get("classifier_dropout", 0.3)),
            nn.Linear(64, 1),
        )
        
        # Explicitly None
        self.text_encoder = None
        self.metadata_encoder = None
        self.text_proj = None
        self.meta_proj = None
        self.dual_cross_attn = None
        self.gated_fusion = None
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass for image-only model."""
        # timm ViT returns features directly when num_classes=0
        image_repr = self.image_encoder(batch["pixel_values"])  # [B, 768]
        
        i_proj = self.image_proj(image_repr)  # [B, 256]
        logits = self.classifier(i_proj)
        
        return {
            "logits": logits,
            "i_proj": i_proj,
        }
    
    def transition_to_phase2(self, k: int = 4):
        """Unfreeze top-k blocks of ViT."""
        image_blocks = self.image_encoder.blocks
        params = []
        for block in image_blocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True
                params.append(p)
        return params
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count total or trainable parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class MetadataOnlyModel(nn.Module):
    """Metadata-only ablation model.
    
    Architecture: 17-feature MLP → Linear(256→256) → Classifier
    
    Params: ~400-800K (tiny)
    No text encoder, no image encoder, no fusion.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.ablation_mode = "metadata_only"
        
        cfg_model = config.get("model", {})
        
        # Get metadata features
        metadata_features = config.get("metadata_features", [])
        n_features = len(metadata_features) if metadata_features else 17
        proj_dim = cfg_model.get("projection_dim", 256)
        
        # === Metadata Encoder (MLP) ===
        self.metadata_encoder = MetadataMLP(
            input_dim=n_features,
            output_dim=proj_dim,
            hidden_dims=[64, 128, 256],
            dropout=0.1,
        )
        
        # === Metadata Projection ===
        self.meta_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(cfg_model.get("projection_dropout", 0.1)),
        )
        
        # === Classifier Head ===
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg_model.get("classifier_dropout", 0.3)),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(cfg_model.get("classifier_dropout", 0.3)),
            nn.Linear(64, 1),
        )
        
        # Explicitly None
        self.text_encoder = None
        self.image_encoder = None
        self.text_proj = None
        self.image_proj = None
        self.dual_cross_attn = None
        self.gated_fusion = None
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass for metadata-only model."""
        meta_repr = self.metadata_encoder(batch["metadata"])  # [B, 256]
        m_proj = self.meta_proj(meta_repr)
        logits = self.classifier(m_proj)
        
        return {
            "logits": logits,
            "m_proj": m_proj,
        }
    
    def transition_to_phase2(self, k: int = 4):
        """No-op for metadata-only (no encoders to unfreeze)."""
        return []
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count total or trainable parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
