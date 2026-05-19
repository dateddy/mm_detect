# src/models/full_model.py
"""Full multimodal misinformation detector model."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.models.text_encoder import TextEncoder
from src.models.image_encoder import ImageEncoder
from src.models.metadata_encoder import MetadataEncoder
from src.models.projection import ModalityProjection, ModalityDropout
from src.models.cross_attention import DualCrossAttention
from src.models.gated_fusion import GatedFusion
from src.models.classification_head import ClassificationHead

logger = logging.getLogger(__name__)


class MultimodalMisinfoDetector(nn.Module):
    """
    Full multimodal misinformation detector model.

    Orchestrates all submodules for end-to-end binary classification of Vietnamese
    Facebook ads as misinformation (1) or not (0).

    Supports two input modes:
    1. **Raw inputs**: Text (tokenized), images (pixel values), metadata features
    2. **Pre-extracted embeddings**: Text and image embeddings cached offline

    The model transparently handles both modes, selecting based on input keys.

    Architecture pipeline:
    1. Encode modalities (text, image, metadata) or use pre-extracted embeddings
    2. Project to common dimension (256)
    3. Apply modality dropout for regularization
    4. Dual cross-attention fusion
    5. Gated fusion with residual connections
    6. Classification head for binary prediction

    Attributes:
        All submodules exposed as named attributes for easy access.
    """

    def __init__(self, config: Dict, ablation_mode: str = "full"):
        """
        Initialize MultimodalMisinfoDetector.

        Args:
            config: Configuration dictionary with keys:
                - text_encoder_name, image_encoder_name (model identifiers)
                - image_size, max_text_len
                - proj_dim, num_attn_heads, attn_dropout, modality_dropout_p, head_dropout
                - metadata_features: list of metadata feature names (for determining input_dim)
            ablation_mode: Ablation mode. One of:
                - 'full': Multimodal (default)
                - 'text_only': Text modality only
                - 'image_only': Image modality only
                - full_no_*: Full architecture component/loss ablations
        """
        super().__init__()

        # Validate ablation mode
        valid_modes = {
            "full", "full_no_contrastive", "full_no_modality_dropout",
            "full_no_dropout", "full_no_metadata_in_fusion",
            "full_no_attention", "full_no_gating",
        }
        if ablation_mode not in valid_modes:
            raise ValueError(f"ablation_mode must be one of {valid_modes}, got {ablation_mode}")
        self.ablation_mode = ablation_mode

        logger.info(f"Initializing MultimodalMisinfoDetector in '{ablation_mode}' mode...")

        # Extract config parameters
        cfg_model = config.get("model", {})
        cfg_training = config.get("training", {})
        text_encoder_name = cfg_model.get(
            "text_model_name", config.get("text_encoder_name", "vinai/phobert-base-v2")
        )
        image_encoder_name = cfg_model.get(
            "image_model_name", config.get("image_encoder_name", "vit_base_patch16_224")
        )
        proj_dim = cfg_model.get("projection_dim", config.get("proj_dim", 256))
        num_attn_heads = cfg_model.get("num_attn_heads", config.get("num_attn_heads", 8))
        proj_dropout = cfg_model.get("projection_dropout", config.get("projection_dropout", 0.1))
        attn_dropout = cfg_model.get("attention_dropout", config.get("attn_dropout", 0.1))
        modality_dropout_p = cfg_training.get(
            "modality_dropout", config.get("modality_dropout_p", 0.15)
        )
        text_modality_dropout_p = cfg_training.get(
            "text_modality_dropout_p", config.get("text_modality_dropout_p", 0.0)
        )
        image_modality_dropout_p = cfg_training.get(
            "image_modality_dropout_p", modality_dropout_p
        )
        head_dropout = cfg_model.get("classifier_dropout", config.get("head_dropout", 0.3))

        if self.ablation_mode == "full_no_dropout":
            logger.info(
                "[MultimodalMisinfoDetector] mode='full_no_dropout': "
                "forcing all nn.Dropout layers to 0.0"
            )
            proj_dropout = 0.0
            attn_dropout = 0.0
            head_dropout = 0.0
            text_modality_dropout_p = 0.0
            image_modality_dropout_p = 0.0
        
        # Store proj_dim for later use
        self.proj_dim = proj_dim
        
        # Get number of metadata features from config
        metadata_features = config.get("metadata_features", [])
        num_metadata_features = len(metadata_features) if metadata_features else 0

        # Text encoder (PhoBERT, frozen initially)
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name, freeze=True
        )

        # Image encoder (ViT, frozen initially)
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_name, freeze=True
        )

        # Metadata encoder (only if metadata features exist)
        self.num_metadata_features = num_metadata_features
        if num_metadata_features > 0:
            self.metadata_encoder = MetadataEncoder(
                input_dim=num_metadata_features,
                output_dim=proj_dim,
                dropout=proj_dropout,
            )
        else:
            self.metadata_encoder = None

        # Projection layers (to common dimension)
        self.text_proj = ModalityProjection(
            in_dim=self.text_encoder.output_dim,
            out_dim=proj_dim,
            dropout=proj_dropout,
        )
        self.image_proj = ModalityProjection(
            in_dim=self.image_encoder.output_dim,
            out_dim=proj_dim,
            dropout=proj_dropout,
        )
        self.meta_proj = ModalityProjection(
            in_dim=proj_dim,
            out_dim=proj_dim,
            dropout=proj_dropout,
        )

        # Modality dropout (ISSUE-021 Option D):
        # Keep text dropout disabled while preserving image/metadata dropout.
        self.text_modality_dropout_p = text_modality_dropout_p
        self.image_modality_dropout_p = image_modality_dropout_p
        self.modality_dropout = ModalityDropout(p=image_modality_dropout_p)

        # Dual cross-attention, removed entirely for the no-attention ablation.
        if self.ablation_mode == "full_no_attention":
            self.dual_cross_attn = None
            logger.info(
                "[MultimodalMisinfoDetector] mode='full_no_attention': "
                "DualCrossAttention NOT instantiated"
            )
        else:
            self.dual_cross_attn = DualCrossAttention(
                embed_dim=proj_dim,
                num_heads=num_attn_heads,
                dropout=attn_dropout,
            )

        # Gated fusion, or parameter-free sum fusion for the no-gating ablation.
        if self.ablation_mode == "full_no_gating":
            self.gated_fusion = None
            self.fusion_layer_norm = nn.LayerNorm(proj_dim)
            logger.info(
                "[MultimodalMisinfoDetector] mode='full_no_gating': "
                "GatedFusion replaced with simple sum + LayerNorm"
            )
        else:
            self.gated_fusion = GatedFusion(dim=proj_dim)
            self.fusion_layer_norm = None

        # Classification head
        self.classifier = ClassificationHead(
            in_dim=proj_dim, dropout=head_dropout
        )

        if self.ablation_mode == "full_no_dropout":
            n_dropout = 0
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                    n_dropout += 1
            logger.info(
                f"[MultimodalMisinfoDetector] Set {n_dropout} nn.Dropout layers to p=0.0"
            )

        logger.info(
            f"Initialized MultimodalMisinfoDetector with {self.count_parameters()} total parameters"
        )
        logger.info(
            f"Modality dropout config: text={self.text_modality_dropout_p:.3f}, "
            f"image/meta={self.image_modality_dropout_p:.3f}"
        )

    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass through the model.

        Handles both raw-input and offline-embedding modes transparently.
        Supports ablation studies by bypassing multimodal components in unimodal modes.

        Args:
            batch: Dictionary containing:
            Raw mode:
                - 'input_ids': (B, seq_len) token IDs
                - 'attention_mask': (B, seq_len)
                - 'pixel_values': (B, 3, H, W) images
                - 'metadata': (B, 9) metadata features
                - 'missing_image': List[bool] missing image flags
            Offline mode:
                - 'text_emb': (B, 768) pre-extracted text embeddings
                - 'image_emb': (B, 768) pre-extracted image embeddings
                - 'metadata': (B, 9) metadata features
                - 'missing_image': List[bool] missing image flags

        Returns:
            Dictionary with:
            - 'logits': (B, 1) classification logits
            - 't_proj': (B, 256) text embeddings (None if not in unimodal text mode)
            - 'i_proj': (B, 256) image embeddings (None if not in unimodal image mode)
            - 't_prime': (B, 256) text after cross-attention (None in unimodal modes)
            - 'i_prime': (B, 256) image after cross-attention (None in unimodal modes)
            - 'image_valid': (B,) bool — True where image was NOT dropped (None in unimodal modes)
            - 'is_multimodal': bool — True if multimodal components were computed, False otherwise
        """
        device = next(self.parameters()).device

        # Detect mode from batch keys
        is_offline_mode = "text_emb" in batch

        if is_offline_mode:
            # Offline embeddings mode
            text_emb = batch["text_emb"].to(device)  # (B, 768)
            image_emb = batch["image_emb"].to(device)  # (B, 768)
        else:
            # Raw input mode
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            # === DIAGNOSTIC: Check if pixel_values are suspiciously zero ===
            if self.training and not getattr(self, "_first_batch_checked", False):
                pv_sum = pixel_values.abs().sum().item()
                pv_std = pixel_values.std().item()
                
                if pv_sum == 0:
                    raise RuntimeError(
                        "ALL pixel_values in this batch are zero. "
                        "Either every image is missing or image loading is broken. "
                        "Run `python scripts/diagnose_image_loading.py --config configs/base.yaml` "
                        "to diagnose the root cause."
                    )
                
                if pv_std < 1e-4:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"⚠ Suspiciously low pixel_values std: {pv_std:.6f}. "
                        f"Check image normalization, transforms, or data loading pipeline."
                    )
                
                self._first_batch_checked = True

            # Encode text and image
            text_emb = self.text_encoder(input_ids, attention_mask)  # (B, 768)
            image_emb = self.image_encoder(pixel_values)  # (B, 768)

        # Metadata features
        metadata = batch["metadata"].to(device)  # (B, num_metadata_features) or (B, 0)

        # Encode metadata (or create zero embeddings if no metadata features)
        if self.metadata_encoder is not None:
            m_raw = self.metadata_encoder(metadata)  # (B, 256)
        else:
            # No metadata features: create zero embeddings
            batch_size = text_emb.shape[0]
            m_raw = torch.zeros(batch_size, self.proj_dim, device=device, dtype=torch.float32)

        # Project all modalities to common dimension
        # CRITICAL: Do NOT detach here — gradients must flow back through projections
        t_proj = self.text_proj(text_emb)  # (B, 256) with requires_grad=True
        i_proj = self.image_proj(image_emb)  # (B, 256) with requires_grad=True
        m_proj = self.meta_proj(m_raw)  # (B, 256)

        # Zero-mask missing images AFTER projection
        missing_image_mask = torch.tensor(
            batch["missing_image"], dtype=torch.bool, device=device
        )  # (B,)
        i_proj[missing_image_mask] = 0.0  # Zero out missing image embeddings

        # Apply modality dropout — returns BOTH masked embeddings AND validity masks
        # (skip for full_no_modality_dropout mode)
        if self.ablation_mode == "full_no_modality_dropout":
            t_valid = i_valid = m_valid = None
        else:
            # Text dropout intentionally disabled (ISSUE-021 Option D).
            t_valid = torch.ones(t_proj.shape[0], dtype=torch.bool, device=t_proj.device)
            (i_proj, m_proj), (i_valid, m_valid) = self.modality_dropout(i_proj, m_proj)

        # === ABLATION LOGIC: Handle different modes ===
        is_multimodal = self.ablation_mode in (
            "full", "full_no_contrastive", "full_no_modality_dropout",
            "full_no_dropout", "full_no_metadata_in_fusion",
            "full_no_attention", "full_no_gating"
        )
        
        if is_multimodal:
            # Full multimodal (with optional component ablations)
            
            # Cross-attention (skip if full_no_attention mode)
            if self.ablation_mode == "full_no_attention":
                # No cross-attention: use projections directly
                t_prime, i_prime = t_proj, i_proj
            elif self.ablation_mode == "full_no_metadata_in_fusion":
                # Encode metadata for diagnostics/param control, but exclude it from attention.
                t_prime, i_prime = self.dual_cross_attn(t_proj, i_proj, m_proj=None)
            else:
                # Standard cross-attention
                t_prime, i_prime = self.dual_cross_attn(t_proj, i_proj, m_proj)
            
            # Fusion (skip gating if full_no_gating mode, use simple sum instead)
            if self.ablation_mode == "full_no_gating":
                # Simple sum fusion with the same residual sources, then normalize.
                fused = t_prime + i_prime + m_proj
                fused = self.fusion_layer_norm(fused + t_proj + i_proj + m_proj)
            elif self.ablation_mode == "full_no_metadata_in_fusion":
                fused = self.gated_fusion(t_prime, i_prime, None, t_proj, i_proj)
            else:
                # Standard gated fusion
                fused = self.gated_fusion(t_prime, i_prime, m_proj, t_proj, i_proj)
        else:
            # Unimodal ablation: use only the selected modality
            # No cross-attention, no fusion — direct pathway
            if self.ablation_mode == "text_only":
                fused = t_proj  # Use text projection directly
            elif self.ablation_mode == "image_only":
                fused = i_proj  # Use image projection directly
            elif self.ablation_mode == "metadata_only":
                fused = m_proj  # Use metadata projection directly
            else:
                raise ValueError(f"Unknown ablation mode: {self.ablation_mode}")
            
            # Set cross-attention outputs to None (not computed in unimodal modes)
            t_prime = None
            i_prime = None

        # Classification head
        logits = self.classifier(fused)  # (B, 1)

        return {
            "logits": logits,
            "t_proj": t_proj if self.ablation_mode in (
                "full", "full_no_contrastive", "full_no_modality_dropout",
                "full_no_dropout", "full_no_metadata_in_fusion",
                "full_no_attention", "full_no_gating", "text_only"
            ) else None,
            "i_proj": i_proj if self.ablation_mode in (
                "full", "full_no_contrastive", "full_no_modality_dropout",
                "full_no_dropout", "full_no_metadata_in_fusion",
                "full_no_attention", "full_no_gating", "image_only"
            ) else None,
            "m_proj": m_proj if self.ablation_mode in (
                "full", "full_no_contrastive", "full_no_modality_dropout",
                "full_no_dropout", "full_no_metadata_in_fusion",
                "full_no_attention", "full_no_gating", "metadata_only"
            ) else None,
            "t_prime": t_prime,  # None in unimodal modes
            "i_prime": i_prime,  # None in unimodal modes
            "image_valid": i_valid if is_multimodal else None,  # None in unimodal modes
            "is_multimodal": is_multimodal,  # Flag for loss and trainer
        }

    def unfreeze_encoders(self, top_k: int) -> None:
        """
        Unfreeze top-k transformer blocks of text and image encoders.

        Used for Phase 2 of training to fine-tune pretrained encoders.

        Args:
            top_k: Number of top transformer blocks to unfreeze.
        """
        # Track trainable parameters before
        trainable_before = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        # Unfreeze top-k blocks in both encoders
        self.text_encoder.unfreeze_top_k(top_k)
        self.image_encoder.unfreeze_top_k(top_k)

        # Track trainable parameters after
        trainable_after = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        trainable_delta = trainable_after - trainable_before
        logger.info(
            f"Unfroze top-{top_k} blocks in encoders. "
            f"Trainable parameters: {trainable_before:,} → {trainable_after:,} "
            f"(+{trainable_delta:,})"
        )

    def get_optimizer_param_groups(
        self,
        lr_fusion: float = 3.0e-4,
        lr_encoders: float = 1.0e-5,
        weight_decay: float = 0.01,
    ) -> List[Dict]:
        """
        Create optimizer parameter groups with differential learning rates.

        Enables Phase 2 training with lower learning rates for encoders and
        higher rates for fusion/classification components.

        Excludes LayerNorm and bias parameters from weight decay for stable training.

        Args:
            lr_fusion: Learning rate for fusion and classification (default: 3e-4).
            lr_encoders: Learning rate for encoder fine-tuning (default: 1e-5).
            weight_decay: L2 regularization coefficient (default: 0.01).

        Returns:
            List of parameter group dictionaries for torch.optim.AdamW.
        """
        # Encoder parameters
        encoder_params = []
        for name, param in self.named_parameters():
            if "text_encoder" in name or "image_encoder" in name:
                encoder_params.append((name, param))

        # Non-encoder parameters
        fusion_params = []
        for name, param in self.named_parameters():
            if "text_encoder" not in name and "image_encoder" not in name:
                fusion_params.append((name, param))

        # Helper to separate params with/without weight decay
        def _get_wd_params(params: List[Tuple]) -> Tuple[List, List]:
            """Separate parameters by whether they should have weight decay."""
            no_wd = []
            with_wd = []
            for name, param in params:
                if "LayerNorm" in name or "bias" in name:
                    no_wd.append(param)
                else:
                    with_wd.append(param)
            return with_wd, no_wd

        # Create param groups
        param_groups = []

        # Encoder group
        enc_with_wd, enc_no_wd = _get_wd_params(encoder_params)
        if enc_with_wd:
            param_groups.append(
                {
                    "params": enc_with_wd,
                    "lr": lr_encoders,
                    "weight_decay": weight_decay,
                }
            )
        if enc_no_wd:
            param_groups.append(
                {
                    "params": enc_no_wd,
                    "lr": lr_encoders,
                    "weight_decay": 0.0,
                }
            )

        # Fusion group
        fus_with_wd, fus_no_wd = _get_wd_params(fusion_params)
        if fus_with_wd:
            param_groups.append(
                {
                    "params": fus_with_wd,
                    "lr": lr_fusion,
                    "weight_decay": weight_decay,
                }
            )
        if fus_no_wd:
            param_groups.append(
                {
                    "params": fus_no_wd,
                    "lr": lr_fusion,
                    "weight_decay": 0.0,
                }
            )

        logger.info(
            f"Created {len(param_groups)} optimizer param groups "
            f"(lr_encoders={lr_encoders}, lr_fusion={lr_fusion}, wd={weight_decay})"
        )

        return param_groups

    def count_parameters(
        self, trainable_only: bool = False
    ) -> int:
        """
        Count total or trainable parameters.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


def model_summary(model: MultimodalMisinfoDetector) -> str:
    """
    Generate a summary of model parameter counts per submodule.

    Args:
        model: MultimodalMisinfoDetector instance.

    Returns:
        Formatted summary string.
    """
    lines = ["=" * 80]
    lines.append("MODEL SUMMARY: MultimodalMisinfoDetector")
    lines.append("=" * 80)

    # Overall stats
    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)
    frozen_params = total_params - trainable_params

    lines.append(f"\nOverall:")
    lines.append(f"  Total parameters:     {total_params:,}")
    lines.append(f"  Trainable parameters: {trainable_params:,}")
    lines.append(f"  Frozen parameters:    {frozen_params:,}")

    # Per-module stats
    lines.append("\nPer-module breakdown:")

    for name, module in model.named_children():
        if hasattr(module, "parameters"):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            frozen = total - trainable
            lines.append(
                f"  {name:20s} | Total: {total:9,} | Trainable: {trainable:9,} | Frozen: {frozen:9,}"
            )

    lines.append("=" * 80 + "\n")

    return "\n".join(lines)
