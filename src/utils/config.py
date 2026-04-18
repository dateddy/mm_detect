"""Configuration loading and management with YAML and dataclasses"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import is_dataclass, asdict
from omegaconf import OmegaConf, DictConfig
from src.config.schema import Config, TextEncoderConfig, ImageEncoderConfig, MetadataEncoderConfig
from src.config.schema import ProjectionConfig, ModalityDropoutConfig, CrossAttentionConfig
from src.config.schema import GatingFusionConfig, ClassificationHeadConfig, LossConfig
from src.config.schema import OptimizerConfig, SchedulerConfig, TrainingConfig
from src.config.schema import DataConfig, CheckpointConfig


def load_yaml(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_yaml(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary or Config dataclass
        output_path: Path where to save YAML
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict if needed
    if isinstance(config, Config):
        config = config.to_dict()
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """
    Convert dictionary to Config dataclass, creating nested dataclasses.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Config dataclass instance
    """
    # Create sub-config instances from dict values if they exist
    sub_configs = {
        'text_encoder': TextEncoderConfig,
        'image_encoder': ImageEncoderConfig,
        'metadata_encoder': MetadataEncoderConfig,
        'projection': ProjectionConfig,
        'modality_dropout': ModalityDropoutConfig,
        'cross_attention': CrossAttentionConfig,
        'gating_fusion': GatingFusionConfig,
        'classification_head': ClassificationHeadConfig,
        'loss': LossConfig,
        'optimizer': OptimizerConfig,
        'scheduler': SchedulerConfig,
        'training': TrainingConfig,
        'data': DataConfig,
        'checkpoint': CheckpointConfig,
    }
    
    config_copy = config_dict.copy()
    
    # Convert nested dicts to dataclass instances
    for key, dataclass_type in sub_configs.items():
        if key in config_copy and isinstance(config_copy[key], dict):
            config_copy[key] = dataclass_type(**config_copy[key])
    
    # Convert Path strings to Path objects
    if isinstance(config_copy.get('data'), DataConfig):
        if isinstance(config_copy['data'].data_dir, str):
            config_copy['data'].data_dir = Path(config_copy['data'].data_dir)
        if isinstance(config_copy['data'].raw_dir, str):
            config_copy['data'].raw_dir = Path(config_copy['data'].raw_dir)
        if isinstance(config_copy['data'].processed_dir, str):
            config_copy['data'].processed_dir = Path(config_copy['data'].processed_dir)
        if isinstance(config_copy['data'].embeddings_dir, str):
            config_copy['data'].embeddings_dir = Path(config_copy['data'].embeddings_dir)
    
    if isinstance(config_copy.get('checkpoint'), CheckpointConfig):
        if isinstance(config_copy['checkpoint'].output_dir, str):
            config_copy['checkpoint'].output_dir = Path(config_copy['checkpoint'].output_dir)
        if config_copy['checkpoint'].resume_from and isinstance(config_copy['checkpoint'].resume_from, str):
            config_copy['checkpoint'].resume_from = Path(config_copy['checkpoint'].resume_from)
    
    return Config(**config_copy)


def load_config(config_path: Path) -> Config:
    """
    Load YAML configuration file and return Config dataclass.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Config dataclass instance
    """
    config_dict = load_yaml(config_path)
    return _dict_to_config(config_dict)


def merge_configs(
    base_config: Dict[str, Any],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        base_config: Base configuration
        model_config: Model configuration (overrides base)
        training_config: Training configuration (overrides base)
        
    Returns:
        Merged configuration
    """
    merged = {**base_config}
    merged['model'] = {**base_config.get('model', {}), **model_config}
    merged['training'] = {**base_config.get('training', {}), **training_config}
    return merged


def config_to_omegaconf(config: Dict[str, Any]) -> DictConfig:
    """
    Convert dictionary config to OmegaConf DictConfig.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        OmegaConf DictConfig object
    """
    return OmegaConf.create(config)
