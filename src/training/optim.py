"""Optimizer construction for two-phase training protocol.

This module implements phase-aware optimizer initialization that avoids:
1. Stale momentum buffers for frozen parameters
2. Memory waste for frozen encoder parameters (~1.3 GB on GPU)
3. Silent failures when unfreezing parameters
"""

import torch
from torch.optim import AdamW
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


def build_optimizer_phase1(model, loss_fn, config: dict) -> Tuple[AdamW, Dict[str, List]]:
    """
    Build AdamW optimizer for Phase 1 (encoders frozen).
    
    Only fusion + projection + classifier + metadata_encoder parameters are optimized,
    plus the learnable temperature parameter from the contrastive loss.
    Encoder parameter groups are created as empty placeholders, to be populated
    at the Phase 1 → Phase 2 transition.
    
    This design prevents:
    - Allocating optimizer state for 220M frozen encoder parameters (~1.3 GB)
    - Stale momentum buffers for unfrozen parameters in Phase 2
    - Silent failures when unfreezing (params not registered in any group)
    
    Args:
        model: MultimodalMisinfoDetector instance with frozen encoders.
        loss_fn: CombinedLoss instance (contains learnable temperature parameter).
        config: Configuration dictionary with keys:
            - training.lr_fusion: Learning rate for fusion components
            - training.lr_encoders: Learning rate for encoders (Phase 2)
            - training.weight_decay: L2 regularization coefficient
    
    Returns:
        Tuple of:
        - optimizer: AdamW instance with 5 parameter groups
        - param_group_index: Dict mapping group_name → list of param tensors
                             (for future reference during unfreezing)
    
    Groups created:
        Group 0: text_encoder (empty in Phase 1, populated at Phase 2)
        Group 1: image_encoder (empty in Phase 1, populated at Phase 2)
        Group 2: fusion + projections + metadata (active in Phase 1)
        Group 3: classifier (active in Phase 1)
        Group 4: learnable temperature (active in Phase 1 and Phase 2)
    """
    # ===== VERIFICATION: Encoders are actually frozen =====
    text_frozen = all(not p.requires_grad for p in model.text_encoder.parameters())
    image_frozen = all(not p.requires_grad for p in model.image_encoder.parameters())
    
    if not text_frozen:
        logger.warning("[build_optimizer_phase1] text_encoder is not frozen, proceeding anyway")
    if not image_frozen:
        logger.warning("[build_optimizer_phase1] image_encoder is not frozen, proceeding anyway")
    
    # ===== COLLECT TRAINABLE PARAMETERS BY COMPONENT =====
    fusion_head_params = []
    classifier_params = []
    param_group_index = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # Skip frozen parameters — they will be added in Phase 2
            continue
        
        # Fusion + Projection + Metadata components
        if any(prefix in name for prefix in [
            "metadata_encoder",
            "text_proj",
            "image_proj", 
            "meta_proj",
            "dual_cross_attn",
            "gated_fusion"
        ]):
            fusion_head_params.append(param)
        
        # Classifier head
        elif "classifier" in name:
            classifier_params.append(param)
        
        else:
            # Catch-all: should not happen in Phase 1
            logger.warning(
                f"[build_optimizer_phase1] Unmatched trainable param: {name}, "
                f"adding to fusion group"
            )
            fusion_head_params.append(param)
    
    # ===== COLLECT LEARNABLE TEMPERATURE PARAMETER =====
    temperature_params = []
    if hasattr(loss_fn, "contrastive") and hasattr(loss_fn.contrastive, "logit_scale"):
        logit_scale = loss_fn.contrastive.logit_scale
        if logit_scale.requires_grad:
            temperature_params.append(logit_scale)
            init_temp = loss_fn.contrastive.temperature.item()
            logger.info(
                f"[Phase 1 Optimizer] Found learnable temperature: "
                f"logit_scale={logit_scale.item():.4f}, init_temp={init_temp:.4f}"
            )
    
    # ===== LOGGING: Parameter counts by component =====
    n_fusion = sum(p.numel() for p in fusion_head_params)
    n_cls = sum(p.numel() for p in classifier_params)
    
    logger.info(
        f"[Phase 1 Optimizer] Trainable parameters:"
        f" fusion+proj+meta={n_fusion:,} | classifier={n_cls:,} | temperature={len(temperature_params)}"
    )
    
    # ===== CREATE PARAMETER GROUPS =====
    param_groups = [
        # Group 0: text_encoder (placeholder, empty in Phase 1)
        {
            "params": [],
            "lr": 0.0,
            "weight_decay": config["training"].get("weight_decay", 0.01),
            "name": "text_encoder",
        },
        # Group 1: image_encoder (placeholder, empty in Phase 1)
        {
            "params": [],
            "lr": 0.0,
            "weight_decay": config["training"].get("weight_decay", 0.01),
            "name": "image_encoder",
        },
        # Group 2: fusion + projections + metadata (active in Phase 1)
        {
            "params": fusion_head_params,
            "lr": config["training"].get("lr_fusion", 3.0e-4),
            "weight_decay": config["training"].get("weight_decay", 0.01),
            "name": "fusion",
        },
        # Group 3: classifier (active in Phase 1)
        {
            "params": classifier_params,
            "lr": config["training"].get("lr_fusion", 3.0e-4),
            "weight_decay": 0.0,  # No weight decay on classifier biases
            "name": "classifier",
        },
        # Group 4: learnable temperature (active in Phase 1 and Phase 2)
        {
            "params": temperature_params,
            "lr": config["training"].get("lr_fusion", 3.0e-4),
            "weight_decay": 0.0,  # No weight decay on scalar parameter
            "name": "temperature",
        },
    ]
    
    # ===== CREATE OPTIMIZER =====
    optimizer = AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # ===== SANITY CHECKS =====
    assert len(optimizer.param_groups[0]["params"]) == 0, \
        "text_encoder group should be empty in Phase 1"
    assert len(optimizer.param_groups[1]["params"]) == 0, \
        "image_encoder group should be empty in Phase 1"
    assert len(optimizer.param_groups[2]["params"]) > 0, \
        "fusion group should have parameters in Phase 1"
    assert len(optimizer.param_groups[3]["params"]) > 0, \
        "classifier group should have parameters in Phase 1"
    
    logger.info(
        f"[Phase 1 Optimizer] Created with 5 groups: "
        f"text_encoder (empty), image_encoder (empty), "
        f"fusion ({len(optimizer.param_groups[2]['params'])}), "
        f"classifier ({len(optimizer.param_groups[3]['params'])}), "
        f"temperature ({len(optimizer.param_groups[4]['params'])})"
    )
    
    return optimizer, param_group_index
