"""Learning rate schedulers"""

import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'cosine',
    max_epochs: int = 20,
    warmup_epochs: int = 2,
    **kwargs
) -> Optional[LRScheduler]:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'linear', 'exponential', 'plateau')
        max_epochs: Maximum number of epochs
        warmup_epochs: Number of warmup epochs
        **kwargs: Additional arguments for scheduler
        
    Returns:
        LRScheduler instance or None
    """
    
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == 'linear':
        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get('start_factor', 0.1),
            total_iters=max_epochs
        )
    
    elif scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 3),
            verbose=True
        )
    
    elif scheduler_type == 'none':
        return None
    
    else:
        raise ValueError(f'Unknown scheduler type: {scheduler_type}')
