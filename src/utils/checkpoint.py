"""Checkpoint saving and loading utilities"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """
    Manages model checkpointing and best model tracking.
    """
    
    def __init__(self, checkpoint_dir: Path, save_top_k: int = 3):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_top_k: Number of best checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.best_scores = []  # List of (score, path) tuples
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        primary_metric: str = 'val_loss'
    ) -> None:
        """
        Save checkpoint and manage best k models.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            primary_metric: Metric to track for best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Track best scores
        main_score = metrics.get(primary_metric, float('inf'))
        self.best_scores.append((main_score, checkpoint_path))
        
        # Remove worst checkpoints if exceeding save_top_k
        if len(self.best_scores) > self.save_top_k:
            self.best_scores.sort(key=lambda x: x[0])
            # Remove worst (highest index since sorted ascending)
            worst_score, worst_path = self.best_scores.pop()
            if worst_path.exists():
                worst_path.unlink()
    
    def load(self, checkpoint_path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optional optimizer to load state
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint or None
        """
        if not self.best_scores:
            return None
        # Return checkpoint with lowest score (best)
        return min(self.best_scores, key=lambda x: x[0])[1]
