"""Configuration dataclass schema for the multimodal model"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class TextEncoderConfig:
    """Configuration for text encoder (PhoBERT)"""
    model_name: str = "vinai/phobert-base-v2"
    max_length: int = 256
    output_dim: int = 768
    freeze_phase1: bool = True
    freeze_top_n_layers_phase2: int = 4


@dataclass
class ImageEncoderConfig:
    """Configuration for image encoder (ViT)"""
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    output_dim: int = 768
    freeze_phase1: bool = True
    freeze_top_n_layers_phase2: int = 4
    image_size: int = 224


@dataclass
class MetadataEncoderConfig:
    """Configuration for metadata encoder (MLP)"""
    num_features: int = 9
    hidden_dims: list = field(default_factory=lambda: [256, 256])
    output_dim: int = 256
    dropout: float = 0.0


@dataclass
class ProjectionConfig:
    """Shared projection layer configuration"""
    text_image_hidden_dim: int = 256
    metadata_hidden_dim: int = 256
    use_layer_norm: bool = True


@dataclass
class ModalityDropoutConfig:
    """Modality dropout configuration"""
    enabled: bool = True
    dropout_rate: float = 0.15


@dataclass
class CrossAttentionConfig:
    """Cross-attention configuration"""
    embed_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    batch_first: bool = True


@dataclass
class GatingFusionConfig:
    """Gating fusion configuration"""
    hidden_dim: int = 768
    use_gating: bool = True


@dataclass
class ClassificationHeadConfig:
    """Classification head configuration"""
    hidden_dim: int = 128
    dropout: float = 0.3
    use_gelu: bool = True


@dataclass
class LossConfig:
    """Loss function configuration"""
    use_class_weights: bool = True
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    type: str = "adamw"
    lr_phase1: float = 3e-4
    lr_phase2_encoder: float = 1e-5
    lr_phase2_head: float = 3e-4
    weight_decay: float = 0.01


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    type: str = "cosine_with_warmup"
    warmup_steps: int = 500
    num_epochs: int = 10
    total_steps: int = -1  # Set automatically during training


@dataclass
class TrainingConfig:
    """Training loop configuration"""
    num_epochs: int = 10
    phase1_epochs: int = 3
    batch_size: int = 32
    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    early_stopping_patience: int = 5
    early_stopping_metric: str = "macro_f1"
    log_freq: int = 10
    val_freq: int = 1


@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    embeddings_dir: Path = field(default_factory=lambda: Path("data/embeddings"))
    csv_file: str = "ads_vietnam_clean.csv"
    embeddings_file: str = "embedding_summary.json"
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint & logging configuration"""
    output_dir: Path = field(default_factory=lambda: Path("experiments"))
    save_best_only: bool = True
    save_freq: int = 1
    resume_from: Optional[Path] = None


@dataclass
class Config:
    """Complete configuration combining all sub-configs"""
    # Sub-configs
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    metadata_encoder: MetadataEncoderConfig = field(default_factory=MetadataEncoderConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    modality_dropout: ModalityDropoutConfig = field(default_factory=ModalityDropoutConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    gating_fusion: GatingFusionConfig = field(default_factory=GatingFusionConfig)
    classification_head: ClassificationHeadConfig = field(default_factory=ClassificationHeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Global settings
    device: str = "cuda"
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary recursively"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (Config, TextEncoderConfig, ImageEncoderConfig,
                                 MetadataEncoderConfig, ProjectionConfig,
                                 ModalityDropoutConfig, CrossAttentionConfig,
                                 GatingFusionConfig, ClassificationHeadConfig,
                                 LossConfig, OptimizerConfig, SchedulerConfig,
                                 TrainingConfig, DataConfig, CheckpointConfig)):
                result[key] = value.to_dict() if hasattr(value, 'to_dict') else vars(value)
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
