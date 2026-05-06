"""
Model architectures and factory for ablation-aware model construction.

Main entry point:
    - build_model(config) - Factory function that returns the appropriate model class
      based on config['ablation_mode']

Exported classes:
    - MultimodalMisinfoDetector - Full multimodal model and all full_no_* variants
    - TextOnlyModel - Text-only baseline
    - ImageOnlyModel - Image-only baseline
    - MetadataOnlyModel - Metadata-only baseline
    - TextImageModel - Text + Image bimodal (no metadata)
    - TextMetadataModel - Text + Metadata bimodal (no image)
    - ImageMetadataModel - Image + Metadata bimodal (no text)
"""
from .factory import build_model, VALID_ABLATION_MODES, expected_param_range
from .full_model import MultimodalMisinfoDetector
from .unimodal_models import TextOnlyModel, ImageOnlyModel, MetadataOnlyModel
from .bimodal_models import TextImageModel, TextMetadataModel, ImageMetadataModel

__all__ = [
    "build_model",
    "VALID_ABLATION_MODES",
    "expected_param_range",
    "MultimodalMisinfoDetector",
    "TextOnlyModel",
    "ImageOnlyModel",
    "MetadataOnlyModel",
    "TextImageModel",
    "TextMetadataModel",
    "ImageMetadataModel",
]
