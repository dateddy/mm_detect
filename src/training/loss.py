"""Loss functions for multimodal misinformation detection

Combines classification loss (BCEWithLogitsLoss) with contrastive
auxiliary loss (InfoNCE) for text-image alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InfoNCELoss(nn.Module):
    """
    Contrastive InfoNCE loss for aligning text and image embeddings.
    
    Encourages similarity between paired text-image embeddings
    while pushing apart unpaired pairs. Uses normalized embeddings
    and scaled dot-product attention.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for softmax scaling.
                        Lower values → sharper softmax.
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between text and image embeddings.
        
        Args:
            text_embeddings: Text projection embeddings (batch_size, 256)
            image_embeddings: Image projection embeddings (batch_size, 256)
            
        Returns:
            Scalar loss value (mean of bidirectional losses)
        """
        # Normalize embeddings to unit vectors
        text_norm = F.normalize(text_embeddings, dim=-1)
        image_norm = F.normalize(image_embeddings, dim=-1)
        
        # Compute similarity matrix: (batch_size, batch_size)
        # logits[i, j] = similarity between text_i and image_j
        logits = torch.matmul(text_norm, image_norm.t()) / self.temperature
        
        batch_size = text_embeddings.shape[0]
        labels = torch.arange(batch_size, device=text_embeddings.device)
        
        # Text-to-image loss: predict which image matches this text
        loss_ti = F.cross_entropy(logits, labels)
        
        # Image-to-text loss: predict which text matches this image
        loss_it = F.cross_entropy(logits.t(), labels)
        
        # Average bidirectional losses
        return (loss_ti + loss_it) / 2


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
            logits: Model logits (batch_size, 1)
            targets: Target labels (batch_size, 1)
            
        Returns:
            Scalar loss value
        """
        # Sigmoid cross entropy loss
        p = torch.sigmoid(logits)
        targets = targets.float()
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal loss
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce
        
        # Apply alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_weight * focal_loss
        
        return focal_loss.mean()


class MultimodalLoss(nn.Module):
    """
    Combined loss for multimodal misinformation detection.
    
    Combines classification loss (BCEWithLogitsLoss) with
    contrastive auxiliary loss (InfoNCE).
    
    Total Loss = L_cls + contrastive_weight * L_con
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.07,
        use_focal: bool = False
    ):
        """
        Initialize combined loss.
        
        Args:
            pos_weight: Weight for positive class in BCEWithLogitsLoss
                       (computed from class imbalance if None)
            contrastive_weight: Weight for contrastive (InfoNCE) loss term (default: 0.1)
            contrastive_temperature: Temperature for InfoNCE softmax (default: 0.07)
            use_focal: Use Focal loss instead of BCEWithLogitsLoss
        """
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        
        # Classification loss
        if use_focal:
            self.classification_loss = FocalLoss(pos_weight=pos_weight)
        else:
            if pos_weight is not None:
                self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.classification_loss = nn.BCEWithLogitsLoss()
        
        # Contrastive loss
        self.contrastive_loss = InfoNCELoss(temperature=contrastive_temperature)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        text_proj: torch.Tensor,
        image_proj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            logits: Classification logits (batch_size, 1)
            targets: Binary labels (batch_size, 1) with values 0 or 1
            text_proj: Text projection embeddings (batch_size, 256)
            image_proj: Image projection embeddings (batch_size, 256)
            
        Returns:
            Tuple of (total_loss, classification_loss, contrastive_loss)
        """
        # Classification loss
        l_cls = self.classification_loss(logits, targets.float())
        
        # Contrastive loss
        l_con = self.contrastive_loss(text_proj, image_proj)
        
        # Combined loss
        total_loss = l_cls + self.contrastive_weight * l_con
        
        return total_loss, l_cls, l_con


def compute_class_weights(
    labels: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute class weights for imbalanced binary classification.
    
    Uses inverse frequency weighting:
    pos_weight = num_negative / num_positive
    
    Used with BCEWithLogitsLoss to rebalance classes.
    
    Args:
        labels: Binary labels tensor (N,) or (N, 1) with values 0 or 1
        device: Device to place weights on (cuda or cpu)
        
    Returns:
        Positive class weight tensor for BCEWithLogitsLoss
    """
    labels = labels.view(-1)
    
    # Count positive and negative samples
    num_pos = (labels == 1).sum().float()
    num_neg = (labels == 0).sum().float()
    
    # Avoid division by zero
    if num_pos < 1:
        return torch.tensor([1.0], device=device)
    
    # Weight for positive class
    # Higher weight when positive class is rare
    pos_weight = num_neg / num_pos
    
    return pos_weight.to(device).unsqueeze(0)
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
