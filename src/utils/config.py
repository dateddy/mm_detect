"""Configuration loading and management with YAML and OmegaConf"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig


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
        config: Configuration dictionary
        output_path: Path where to save YAML
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(
    base_config: Dict[str, Any],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge multiple configuration files.
    
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
