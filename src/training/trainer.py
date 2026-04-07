"""Training loop with optimization, validation, and checkpointing"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import json
from pathlib import Path


class Trainer:
    """
    Trainer class for model training with validation and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: Optional[LRScheduler] = None,
        device: torch.device = torch.device('cpu'),
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler (optional)
            device: Device to use
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        grad_clip: float = 1.0
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            grad_clip: Gradient clipping value
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            text = batch['text'].to(self.device)
            image = batch['image'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1).float()
            
            # Forward pass
            logits = self.model(text, image, metadata)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (avg_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                text = batch['text'].to(self.device)
                image = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1).float()
                
                logits = self.model(text, image, metadata)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())
        
        avg_loss = total_loss / num_batches
        metrics = {
            'val_loss': avg_loss,
            'num_samples': sum(len(p) for p in all_preds)
        }
        
        return avg_loss, metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        early_stopping_patience: int = 5,
        grad_clip: float = 1.0
    ) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            grad_clip: Gradient clipping value
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, grad_clip)
            self.history['train_loss'].append(train_metrics['train_loss'])
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Learning rate scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save checkpoint
                if self.checkpoint_dir:
                    self.save_checkpoint(epoch, best_val_loss)
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        return self.history
    
    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
        """
        if self.checkpoint_dir is None:
            return
        
        checkpoint_path = self.checkpoint_dir / f'model_epoch_{epoch:03d}.pt'
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
