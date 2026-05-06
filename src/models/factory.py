"""
Model factory for ablation-aware model construction.

The factory ensures that different ablation_mode values produce models with
truly different architectures, not just zeroed-out inputs or disabled branches.
"""
import logging
from typing import Dict, Any

import torch.nn as nn

logger = logging.getLogger(__name__)


VALID_ABLATION_MODES = {
    "full",
    "text_only",
    "image_only",
    "metadata_only",
    "text_image",
    "text_metadata",
    "image_metadata",
    "full_no_contrastive",          # full architecture, contrastive_lambda=0
    "full_no_modality_dropout",     # full architecture, modality_dropout=0
    "full_no_attention",            # full architecture without cross-attention
    "full_no_gating",               # full architecture with simple sum fusion
}


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a model based on config['ablation_mode'].
    
    Returns a different nn.Module subclass for each mode, with only the
    required modules instantiated. Logs the param breakdown for verification.
    
    Args:
        config: Configuration dictionary with 'ablation_mode' key
        
    Returns:
        nn.Module: Model instance specialized for the ablation mode
        
    Raises:
        ValueError: If ablation_mode is not recognized
    """
    mode = config.get("ablation_mode", "full")
    
    if mode not in VALID_ABLATION_MODES:
        raise ValueError(
            f"Unknown ablation_mode: '{mode}'. "
            f"Valid modes: {sorted(VALID_ABLATION_MODES)}"
        )
    
    # Import model classes here to avoid circular imports
    from .full_model import MultimodalMisinfoDetector
    from .unimodal_models import TextOnlyModel, ImageOnlyModel, MetadataOnlyModel
    from .bimodal_models import TextImageModel, TextMetadataModel, ImageMetadataModel
    
    # Map mode to model class
    if mode in ("full", "full_no_contrastive", "full_no_modality_dropout",
                "full_no_attention", "full_no_gating"):
        model = MultimodalMisinfoDetector(config)
    elif mode == "text_only":
        model = TextOnlyModel(config)
    elif mode == "image_only":
        model = ImageOnlyModel(config)
    elif mode == "metadata_only":
        model = MetadataOnlyModel(config)
    elif mode == "text_image":
        model = TextImageModel(config)
    elif mode == "text_metadata":
        model = TextMetadataModel(config)
    elif mode == "image_metadata":
        model = ImageMetadataModel(config)
    else:
        raise NotImplementedError(f"Mode '{mode}' not yet implemented")
    
    # Verify architecture matches the requested mode
    _verify_model_matches_mode(model, mode)
    
    # Log summary
    _log_model_summary(model, mode)
    
    return model


def _verify_model_matches_mode(model: nn.Module, mode: str):
    """
    Sanity check: verify that disabled modalities are NOT present in the model.
    
    This catches the most common bug: instantiating a full model but calling it 'text_only'.
    
    Args:
        model: The constructed model
        mode: The ablation_mode that was requested
        
    Raises:
        RuntimeError: If model architecture doesn't match the requested mode
    """
    has_text = hasattr(model, "text_encoder") and model.text_encoder is not None
    has_image = hasattr(model, "image_encoder") and model.image_encoder is not None
    has_meta = hasattr(model, "metadata_encoder") and model.metadata_encoder is not None
    has_fusion = (
        (hasattr(model, "dual_cross_attn") and model.dual_cross_attn is not None) or
        (hasattr(model, "gated_fusion") and model.gated_fusion is not None)
    )
    
    expected = {
        "full":                     {"text": True,  "image": True,  "meta": True,  "fusion": True},
        "full_no_contrastive":      {"text": True,  "image": True,  "meta": True,  "fusion": True},
        "full_no_modality_dropout": {"text": True,  "image": True,  "meta": True,  "fusion": True},
        "full_no_attention":        {"text": True,  "image": True,  "meta": True,  "fusion": True},
        "full_no_gating":           {"text": True,  "image": True,  "meta": True,  "fusion": True},
        "text_only":                {"text": True,  "image": False, "meta": False, "fusion": False},
        "image_only":               {"text": False, "image": True,  "meta": False, "fusion": False},
        "metadata_only":            {"text": False, "image": False, "meta": True,  "fusion": False},
        "text_image":               {"text": True,  "image": True,  "meta": False, "fusion": True},
        "text_metadata":            {"text": True,  "image": False, "meta": True,  "fusion": True},
        "image_metadata":           {"text": False, "image": True,  "meta": True,  "fusion": True},
    }
    
    exp = expected[mode]
    actual = {"text": has_text, "image": has_image, "meta": has_meta, "fusion": has_fusion}
    
    if actual != exp:
        raise RuntimeError(
            f"Model architecture does NOT match ablation_mode='{mode}'.\n"
            f"  Expected: {exp}\n"
            f"  Actual:   {actual}\n"
            f"This is a bug in the model factory or model class."
        )


def _log_model_summary(model: nn.Module, mode: str):
    """
    Print model summary with ablation-specific breakdown.
    
    Args:
        model: The constructed model
        mode: The ablation_mode
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Model built: ablation_mode = '{mode}'")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Total parameters:     {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Frozen parameters:    {total_params - trainable_params:,}")
    logger.info(f"{'-' * 70}")
    
    # Per-component breakdown
    components = [
        ("text_encoder",     "Text Encoder (PhoBERT)"),
        ("image_encoder",    "Image Encoder (ViT)"),
        ("metadata_encoder", "Metadata Encoder (MLP)"),
        ("text_proj",        "Text Projection"),
        ("image_proj",       "Image Projection"),
        ("meta_proj",        "Metadata Projection"),
        ("dual_cross_attn",  "Dual Cross-Attention"),
        ("gated_fusion",     "Gated Fusion"),
        ("classifier",       "Classifier Head"),
    ]
    
    for attr_name, display_name in components:
        if hasattr(model, attr_name):
            module = getattr(model, attr_name)
            if module is not None:
                n = sum(p.numel() for p in module.parameters())
                n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
                logger.info(f"  {display_name:30s} {n:>13,} ({n_train:,} trainable)")
    
    logger.info(f"{'=' * 70}\n")


def expected_param_range(mode: str) -> tuple:
    """
    Return (min, max) expected parameter count for a given ablation mode.
    
    Used by smoke tests to catch silent bugs in model instantiation.
    
    Args:
        mode: The ablation_mode
        
    Returns:
        Tuple of (min_params, max_params)
    """
    ranges = {
        "full":                     (220_000_000, 224_000_000),
        "full_no_contrastive":      (220_000_000, 224_000_000),
        "full_no_modality_dropout": (220_000_000, 224_000_000),
        "full_no_attention":        (219_000_000, 222_000_000),
        "full_no_gating":           (220_000_000, 222_000_000),
        "text_only":                (134_000_000, 137_000_000),
        "image_only":               ( 85_000_000,  88_000_000),
        "metadata_only":            (    400_000,     800_000),
        "text_image":               (220_000_000, 222_000_000),
        "text_metadata":            (135_000_000, 137_000_000),
        "image_metadata":           ( 85_000_000,  88_000_000),
    }
    return ranges.get(mode, (0, float('inf')))
