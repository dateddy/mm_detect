"""Random seed utilities for reproducibility"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode (can impact performance)
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic mode (may reduce performance)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
