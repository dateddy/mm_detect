# src/utils/checkpoint.py
"""Checkpoint saving and loading utilities."""

import re
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    metric: float,
    path: str,
) -> None:
    """
    Save model checkpoint with optimizer, scheduler, epoch, and metric.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: Learning rate scheduler state (optional).
        epoch: Current epoch number.
        metric: Metric value (e.g., validation loss, F1 score).
        path: Path to save checkpoint file.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": metric,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Create parent directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, float]:
    """
    Load checkpoint and restore model, optimizer, and scheduler states.

    Automatically handles device mapping (CPU/GPU).
    Uses strict=False to allow loading checkpoints with different module configurations
    (e.g., checkpoints with metadata_encoder into models without it).

    Args:
        path: Path to checkpoint file.
        model: Model to load state into.
        optimizer: Optimizer to load state into (optional).
        scheduler: Learning rate scheduler to load state into (optional).

    Returns:
        Dictionary with keys 'epoch' and 'metric'.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Automatically determine device of model
    device = next(model.parameters()).device

    checkpoint = torch.load(path, map_location=device)

    # Use strict=False to handle model configuration variations
    # (e.g., loading checkpoint with metadata_encoder into model without it)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )

    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

    if missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")

    # Try to load optimizer state dict, but handle mismatches gracefully
    # (e.g., when model architecture changed since checkpoint was saved)
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as e:
            if "parameter group" in str(e):
                logger.warning(
                    f"Optimizer state dict mismatch (likely due to model architecture change): {e}. "
                    f"Skipping optimizer state restoration."
                )
            else:
                raise

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except ValueError as e:
            logger.warning(f"Could not load scheduler state: {e}. Skipping scheduler restoration.")

    return {
        "epoch": checkpoint["epoch"],
        "metric": checkpoint["metric"],
    }


def get_best_checkpoint(
    checkpoint_dir: str, metric: str = "macro_f1"
) -> str:
    """
    Find and return path to checkpoint with best metric value.

    Scans checkpoint_dir for files matching pattern:
    "epoch_{N}_{metric}_{value:.4f}.pt"

    Returns the path with the highest metric value.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        metric: Name of the metric to search for in filenames.

    Returns:
        Path to the best checkpoint file (highest metric value).

    Raises:
        FileNotFoundError: If no matching checkpoints found.
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Pattern: epoch_N_metric_value.pt
    pattern = rf"epoch_\d+_{re.escape(metric)}_([\d.]+)\.pt"

    best_checkpoint = None
    best_value = float("-inf")

    for checkpoint_file in checkpoint_path.glob("*.pt"):
        match = re.search(pattern, checkpoint_file.name)
        if match:
            try:
                value = float(match.group(1))
                if value > best_value:
                    best_value = value
                    best_checkpoint = checkpoint_file
            except ValueError:
                continue

    if best_checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoints matching pattern with metric '{metric}' found in {checkpoint_dir}"
        )

    return str(best_checkpoint)
