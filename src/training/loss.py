"""Loss functions including BCEWithLogits and Focal Loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    From: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor in [0, 1] to balance positive vs negative examples
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            pos_weight: Manual rescaling weight given to positive class
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model logits (batch_size, num_classes)
            targets: Target labels (batch_size, num_classes)
            
        Returns:
            Scalar loss value
        """
        # Sigmoid for binary classification
        probs = torch.sigmoid(logits)
        
        # Compute BCE
        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Compute focal weight
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss
        loss = alpha_t * focal_weight * bce_loss
        
        return loss.mean()


class WeightedBCELoss(nn.Module):
    """
    Binary cross-entropy loss with positive class weighting.
    """
    
    def __init__(self, pos_weight: float = 1.0):
        """
        Initialize weighted BCE loss.
        
        Args:
            pos_weight: Weight for positive class
        """
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted binary cross-entropy.
        
        Args:
            logits: Model logits
            targets: Target labels
            
        Returns:
            Scalar loss value
        """
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device)
        )
