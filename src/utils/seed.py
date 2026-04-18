# src/utils/seed.py
"""Reproducibility utilities for setting random seeds across all sources."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all sources.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch CPU and CUDA operations
    - cuDNN for deterministic behavior
    - PYTHONHASHSEED environment variable

    Args:
        seed: Seed value to use.
    """
    # Set Python built-in random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set cuDNN to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)
