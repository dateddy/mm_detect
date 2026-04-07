"""Model evaluation on test set"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from pathlib import Path
import json

from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_classification_report,
    compute_per_class_metrics
)


class Evaluator:
    """
    Model evaluator for comprehensive test evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
    
    def evaluate(
        self,
        test_loader: DataLoader,
        threshold: float = 0.5,
        output_dir: Path = None
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            threshold: Classification threshold
            output_dir: Directory to save results
            
        Returns:
            Dictionary of evaluation results
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(self.device)
                image = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(text, image, metadata)
                probs = torch.sigmoid(logits).squeeze()
                
                all_preds.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        metrics = compute_metrics(predictions, targets, threshold)
        cm = compute_confusion_matrix(predictions, targets, threshold)
        per_class = compute_per_class_metrics(predictions, targets, threshold)
        report = compute_classification_report(predictions, targets, threshold)
        
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class,
            'classification_report': report,
            'num_samples': len(targets),
            'threshold': threshold
        }
        
        # Save results if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics and results
            with open(output_dir / 'results.json', 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'confusion_matrix': cm.tolist(),
                    'per_class_metrics': per_class,
                    'threshold': threshold
                }, f, indent=2)
            
            # Save report
            with open(output_dir / 'classification_report.txt', 'w') as f:
                f.write(report)
        
        return results
    
    def get_predictions(
        self,
        test_loader: DataLoader,
        return_targets: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions on test set.
        
        Args:
            test_loader: Test data loader
            return_targets: Whether to return target labels
            
        Returns:
            Tuple of (predictions, targets) if return_targets=True,
            else just predictions
        """
        self.model.eval()
        
        all_preds = []
        all_targets = [] if return_targets else None
        
        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(self.device)
                image = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                
                logits = self.model(text, image, metadata)
                probs = torch.sigmoid(logits).squeeze()
                
                all_preds.append(probs.cpu().numpy())
                
                if return_targets:
                    all_targets.append(batch['label'].cpu().numpy())
        
        predictions = np.concatenate(all_preds, axis=0)
        
        if return_targets:
            targets = np.concatenate(all_targets, axis=0)
            return predictions, targets
        else:
            return predictions
