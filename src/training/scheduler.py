# src/training/scheduler.py
"""Learning rate scheduler with warmup and cosine decay."""

import logging
import math
from typing import Callable, Optional

import torch
import torch.optim as optim

logger = logging.getLogger(__name__)


def get_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int = 500,
    total_steps: int = 10000,
    lr_min: float = 1.0e-7,
) -> optim.lr_scheduler.LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and cosine decay.

    Implements the schedule:
    1. **Warmup phase** (steps 0 → warmup_steps): Linear increase from 0 to 1
    2. **Decay phase** (steps warmup_steps → total_steps): Cosine decay from 1 to lr_min

    The cosine decay ensures smooth convergence to a minimum learning rate.

    Args:
        optimizer: PyTorch optimizer instance.
        warmup_steps: Number of steps for linear warmup (default: 500).
        total_steps: Total training steps (default: 10000).
        lr_min: Minimum learning rate multiplier (default: 1e-7).

    Returns:
        torch.optim.lr_scheduler.LambdaLR instance.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        >>> scheduler = get_scheduler(optimizer, warmup_steps=500, total_steps=50000)
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         loss = model(batch)
        ...         optimizer.zero_grad()
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()
    """

    def lr_lambda(current_step: int) -> float:
        """
        Compute learning rate multiplier for current step.

        Args:
            current_step: Current training step.

        Returns:
            Multiplier for the base learning rate.
        """
        # Handle edge case: total_steps <= warmup_steps
        if total_steps <= warmup_steps:
            # Only warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                return lr_min

        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)

        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(
            total_steps - warmup_steps
        )
        # Clamp progress to [0, 1]
        progress = min(progress, 1.0)

        # Cosine decay from 1 to lr_min
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = (1.0 - lr_min) * cosine_decay + lr_min

        return decayed

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(
        f"Initialized LambdaLR scheduler "
        f"(warmup_steps={warmup_steps}, total_steps={total_steps}, lr_min={lr_min})"
    )

    return scheduler
