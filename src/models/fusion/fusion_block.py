"""Legacy fusion block module (deprecated - functionality moved to multimodal_model.py)

This module is kept for backwards compatibility only.
New code should use MultimodalFusionBlock from multimodal_model.py
"""

import torch
import torch.nn as nn


# Re-export for backwards compatibility
try:
    from src.models.multimodal_model import MultimodalFusionBlock
except ImportError:
    # If import fails, define a stub
    class MultimodalFusionBlock(nn.Module):
        """Stub for backwards compatibility"""
        
        def forward(self, *args, **kwargs):
            raise NotImplementedError(
                "MultimodalFusionBlock has been moved to multimodal_model.py"
            )


__all__ = ['MultimodalFusionBlock']

