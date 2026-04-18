"""Ablation study configuration and variants."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AblationConfig:
    """Configuration for a single ablation study variant."""

    name: str
    config_overrides: Dict
    description: str


# MODALITY ABLATIONS: Test impact of each modality
MODALITY_ABLATIONS = [
    AblationConfig(
        name="text_only",
        config_overrides={
            "use_text": True,
            "use_image": False,
            "use_metadata": False,
        },
        description="Only text modality (PhoBERT)",
    ),
    AblationConfig(
        name="image_only",
        config_overrides={
            "use_text": False,
            "use_image": True,
            "use_metadata": False,
        },
        description="Only image modality (ViT)",
    ),
    AblationConfig(
        name="metadata_only",
        config_overrides={
            "use_text": False,
            "use_image": False,
            "use_metadata": True,
        },
        description="Only metadata modality (9 behavioral features)",
    ),
    AblationConfig(
        name="text_image",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": False,
        },
        description="Text + Image, no metadata",
    ),
    AblationConfig(
        name="text_metadata",
        config_overrides={
            "use_text": True,
            "use_image": False,
            "use_metadata": True,
        },
        description="Text + Metadata, no image",
    ),
    AblationConfig(
        name="image_metadata",
        config_overrides={
            "use_text": False,
            "use_image": True,
            "use_metadata": True,
        },
        description="Image + Metadata, no text",
    ),
    AblationConfig(
        name="full_model",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
        },
        description="All modalities (proposed multimodal approach)",
    ),
]

# FUSION ABLATIONS: Test different fusion mechanisms
FUSION_ABLATIONS = [
    AblationConfig(
        name="concat_mlp",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "fusion_type": "concat_mlp",
            "use_cross_attention": False,
            "use_gating": False,
        },
        description="Concatenation + MLP (simple baseline)",
    ),
    AblationConfig(
        name="simple_average",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "fusion_type": "simple_average",
            "use_cross_attention": False,
            "use_gating": False,
        },
        description="Simple average of modalities",
    ),
    AblationConfig(
        name="single_cross_attn",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "fusion_type": "cross_attention",
            "use_cross_attention": True,
            "dual_cross_attention": False,
            "use_gating": False,
        },
        description="Single cross-attention (text to image/metadata only)",
    ),
    AblationConfig(
        name="dual_cross_attn_no_gate",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "fusion_type": "cross_attention",
            "use_cross_attention": True,
            "dual_cross_attention": True,
            "use_gating": False,
        },
        description="Bidirectional cross-attention without gating",
    ),
    AblationConfig(
        name="dual_cross_attn_with_gate",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "fusion_type": "cross_attention",
            "use_cross_attention": True,
            "dual_cross_attention": True,
            "use_gating": True,
        },
        description="Bidirectional cross-attention with gated fusion (proposed)",
    ),
    AblationConfig(
        name="fusion_full_model",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "fusion_type": "cross_attention",
            "use_cross_attention": True,
            "dual_cross_attention": True,
            "use_gating": True,
        },
        description="Full model baseline for fusion comparison",
    ),
]

# METADATA FEATURE ABLATIONS: Test impact of metadata features
METADATA_ABLATIONS = [
    AblationConfig(
        name="top3_features",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "metadata_features": ["engagement_rate", "share_count", "comment_count"],
        },
        description="Top 3 engagement features only",
    ),
    AblationConfig(
        name="temporal_only",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "metadata_features": ["post_hour", "post_day_of_week", "days_since_posting"],
        },
        description="Temporal features only",
    ),
    AblationConfig(
        name="targeting_only",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "metadata_features": [
                "target_age_range_span",
                "target_gender_mixed",
                "target_countries_count",
            ],
        },
        description="Ad targeting features only",
    ),
    AblationConfig(
        name="selected_9",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "metadata_features": [
                "engagement_rate",
                "share_count",
                "comment_count",
                "post_hour",
                "post_day_of_week",
                "days_since_posting",
                "target_age_range_span",
                "target_gender_mixed",
                "target_countries_count",
            ],
        },
        description="9 engineered features (proposed)",
    ),
    AblationConfig(
        name="all_features",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "metadata_include_all_raw": True,
        },
        description="All available metadata features (no engineering)",
    ),
]

# CONTRASTIVE LOSS ABLATIONS: Test impact of InfoNCE loss weight
CONTRASTIVE_ABLATIONS = [
    AblationConfig(
        name="lambda_0",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "contrastive_weight": 0.0,
        },
        description="No contrastive loss (BCE only)",
    ),
    AblationConfig(
        name="lambda_0_05",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "contrastive_weight": 0.05,
        },
        description="Contrastive weight = 0.05 (weak)",
    ),
    AblationConfig(
        name="lambda_0_1",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "contrastive_weight": 0.1,
        },
        description="Contrastive weight = 0.1 (proposed)",
    ),
    AblationConfig(
        name="lambda_0_5",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "contrastive_weight": 0.5,
        },
        description="Contrastive weight = 0.5 (strong)",
    ),
]

# ENCODER FINE-TUNING ABLATIONS: Test different training protocols
FINETUNING_ABLATIONS = [
    AblationConfig(
        name="frozen",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "freeze_encoders_epoch1": True,
            "freeze_encoders_all": True,
        },
        description="Encoders frozen all epochs (no fine-tuning)",
    ),
    AblationConfig(
        name="partial_finetune",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "freeze_encoders_epoch1": True,
            "freeze_encoders_all": False,
            "unfreeze_blocks": 4,
            "encoder_lr": 1e-5,
            "fusion_head_lr": 3e-4,
        },
        description="Phase 1/2 protocol: freeze then unfreeze top-4 blocks (proposed)",
    ),
    AblationConfig(
        name="full_finetune_from_epoch1",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "freeze_encoders_epoch1": False,
            "freeze_encoders_all": False,
        },
        description="Unfreeze encoders from epoch 1",
    ),
    AblationConfig(
        name="full_finetune_after_warmup",
        config_overrides={
            "use_text": True,
            "use_image": True,
            "use_metadata": True,
            "freeze_encoders_epoch1": True,
            "freeze_encoders_all": False,
            "unfreeze_after_warmup": True,
        },
        description="Unfreeze encoders after warmup phase only",
    ),
]

# Group ablations by category
ABLATION_GROUPS = {
    "modality": MODALITY_ABLATIONS,
    "fusion": FUSION_ABLATIONS,
    "metadata": METADATA_ABLATIONS,
    "loss": CONTRASTIVE_ABLATIONS,
    "finetune": FINETUNING_ABLATIONS,
}

# All ablations flattened
ALL_ABLATIONS = (
    MODALITY_ABLATIONS
    + FUSION_ABLATIONS
    + METADATA_ABLATIONS
    + CONTRASTIVE_ABLATIONS
    + FINETUNING_ABLATIONS
)


def get_ablation_group(group_name: str) -> List[AblationConfig]:
    """
    Get ablation configs for a specific group.

    Args:
        group_name: One of "modality", "fusion", "metadata", "loss", "finetune", "all"

    Returns:
        List of AblationConfig for the group
    """
    if group_name == "all":
        return ALL_ABLATIONS
    elif group_name in ABLATION_GROUPS:
        return ABLATION_GROUPS[group_name]
    else:
        raise ValueError(
            f"Unknown ablation group: {group_name}. "
            f"Valid groups: {list(ABLATION_GROUPS.keys())}, all"
        )


def merge_config_with_ablation(base_config: dict, ablation: AblationConfig) -> dict:
    """
    Merge base config with ablation overrides.

    Args:
        base_config: Base configuration dict
        ablation: AblationConfig with overrides

    Returns:
        Merged configuration dict
    """
    merged = base_config.copy()
    merged.update(ablation.config_overrides)
    return merged
