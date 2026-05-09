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
    text_frozen = (model.text_encoder is None or 
                   all(not p.requires_grad for p in model.text_encoder.parameters()))
    image_frozen = (model.image_encoder is None or 
                    all(not p.requires_grad for p in model.image_encoder.parameters()))
    
    if model.text_encoder is not None and not text_frozen:
        logger.warning("[build_optimizer_phase1] text_encoder is not frozen, proceeding anyway")
    if model.image_encoder is not None and not image_frozen:
        logger.warning("[build_optimizer_phase1] image_encoder is not frozen, proceeding anyway")
    
    # ===== COLLECT TRAINABLE PARAMETERS BY COMPONENT =====
    fusion_head_params = []
    classifier_params = []
    aux_params = []
    param_group_index = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # Skip frozen parameters — they will be added in Phase 2
            continue
        
        # Fusion + Projection + Metadata components
        if "aux_" in name and "_head" in name:
            aux_params.append(param)

        elif any(prefix in name for prefix in [
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
    contrastive_module = getattr(loss_fn, "contrastive_loss", None)
    if (
        getattr(loss_fn, "_has_contrastive", False)
        and contrastive_module is not None
    ):
        for name, param in contrastive_module.named_parameters():
            if param.requires_grad:
                temperature_params.append(param)
                logger.info(
                    f"[Phase 1 Optimizer] Found learnable temperature: "
                    f"{name}={param.item():.4f}, "
                    f"init_temp={contrastive_module.temperature.item():.4f}"
                )
    else:
        logger.info("[Phase 1 Optimizer] Contrastive disabled - no temperature param")
    
    # ===== LOGGING: Parameter counts by component =====
    n_fusion = sum(p.numel() for p in fusion_head_params)
    n_cls = sum(p.numel() for p in classifier_params)
    n_aux = sum(p.numel() for p in aux_params)
    
    logger.info(
        f"[Phase 1 Optimizer] Trainable parameters:"
        f" fusion+proj+meta={n_fusion:,} | classifier={n_cls:,} | "
        f"aux_heads={n_aux:,} | temperature={len(temperature_params)}"
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
    ]

    if aux_params:
        param_groups.append(
            {
                "params": aux_params,
                "lr": config["training"].get("lr_fusion", 3.0e-4) * 1.5,
                "weight_decay": config["training"].get("weight_decay", 0.01) * 0.5,
                "name": "aux_heads",
            }
        )

    if temperature_params:
        param_groups.append(
            {
                "params": temperature_params,
                "lr": config["training"].get("lr_fusion", 3.0e-4),
                "weight_decay": 0.0,  # No weight decay on scalar parameter
                "name": "temperature",
            }
        )
    
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
        f"[Phase 1 Optimizer] Created with {len(optimizer.param_groups)} groups: "
        f"{[group['name'] for group in optimizer.param_groups]}"
    )
    
    return optimizer, param_group_index
