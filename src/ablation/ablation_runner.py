"""Ablation study runner to execute multiple experiment configurations"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Callable
import pandas as pd
from pathlib import Path
import json

from ..training.trainer import Trainer
from ..evaluation.evaluator import Evaluator
from ..evaluation.metrics import compute_metrics
from .variants import get_model_variant


class AblationRunner:
    """
    Runner for systematic ablation studies.
    """
    
    def __init__(
        self,
        device: torch.device = torch.device('cpu'),
        output_dir: Path = None
    ):
        """
        Initialize ablation runner.
        
        Args:
            device: Device to use
            output_dir: Directory to save results
        """
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        self.results = []
    
    def run_experiment(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        variant_name: str,
        epochs: int = 20,
        patience: int = 5
    ) -> Dict:
        """
        Run single ablation experiment.
        
        Args:
            model: Model variant
            optimizer: Optimizer
            criterion: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            variant_name: Name of variant
            epochs: Number of epochs
            patience: Early stopping patience
            
        Returns:
            Dictionary of experiment results
        """
        # Create trainer
        trainer = Trainer(model, optimizer, criterion, device=self.device)
        
        # Train
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            early_stopping_patience=patience
        )
        
        # Evaluate
        evaluator = Evaluator(model, self.device)
        eval_results = evaluator.evaluate(test_loader, output_dir=self.output_dir)
        
        # Collect results
        result = {
            'variant': variant_name,
            'epochs_trained': len(history['train_loss']),
            'best_val_loss': min(history['val_loss']),
            'test_accuracy': eval_results['metrics']['accuracy'],
            'test_precision': eval_results['metrics']['precision'],
            'test_recall': eval_results['metrics']['recall'],
            'test_f1': eval_results['metrics']['f1'],
            'test_roc_auc': eval_results['metrics']['roc_auc'],
        }
        
        self.results.append(result)
        return result
    
    def run_ablation_study(
        self,
        variants: Dict[str, Dict],
        optimizer_fn: Callable,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 20,
        patience: int = 5
    ) -> pd.DataFrame:
        """
        Run full ablation study with multiple variants.
        
        Args:
            variants: Dict mapping variant names to configs
            optimizer_fn: Function to create optimizer given model
            criterion: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            epochs: Number of epochs per experiment
            patience: Early stopping patience
            
        Returns:
            DataFrame with results for all variants
        """
        results = []
        
        for variant_name, config in variants.items():
            print(f'Running ablation variant: {variant_name}')
            
            # Create model variant
            model = get_model_variant(variant_name, config)
            
            # Create optimizer
            optimizer = optimizer_fn(model.parameters())
            
            # Run experiment
            result = self.run_experiment(
                model,
                optimizer,
                criterion,
                train_loader,
                val_loader,
                test_loader,
                variant_name,
                epochs,
                patience
            )
            
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(self.output_dir / 'ablation_results.csv', index=False)
            
            with open(self.output_dir / 'ablation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        return results_df
