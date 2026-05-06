#!/usr/bin/env python3
"""Evaluate model on a test set and save metrics."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collate import collate_fn
from src.data.dataset import create_datasets
from src.evaluation.metrics import (
    compute_all_metrics,
    find_best_threshold,
    print_classification_report,
)
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from torch.utils.data import DataLoader

def main():
    """Load model, run inference on split, compute metrics."""
    parser = argparse.ArgumentParser(
        description='Evaluate multimodal misinformation detector'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML/JSON'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--threshold',
        type=str,
        default='0.5',
        help='Classification threshold (float or "auto" for val-based tuning)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(__name__)
    logger.info(f'Device: {device}')

    # Load config
    config_path = Path(args.config)
    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    logger.info(f'Config loaded from {config_path}')

    # Create output directory for results
    output_dir = Path('outputs/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine threshold strategy
    use_auto_threshold = args.threshold.lower() == 'auto'
    if use_auto_threshold:
        threshold = None  # Will be computed from val set
        logger.info('Threshold = auto (will be determined from validation set)')
    else:
        threshold = float(args.threshold)
        logger.info(f'Threshold = {threshold}')

    # Load model
    logger.info(f'Loading model from {args.checkpoint}')
    model = build_model(config)
    state_dict, _ = load_checkpoint(args.checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info('Model loaded and set to eval mode')

    # Create datasets
    logger.info('Creating datasets...')
    datasets = create_datasets(config)
    eval_dataset = datasets[args.split]
    logger.info(f'Eval dataset: {len(eval_dataset)} samples from split "{args.split}"')

    # If auto threshold, load validation set for threshold search
    if use_auto_threshold:
        logger.info('Loading validation set for threshold auto-tuning...')
        val_dataset = datasets['val']
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )

        # Inference on validation set
        logger.info('Running inference on validation set...')
        y_true_val = []
        y_pred_proba_val = []

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                logits = model(batch)
                proba = torch.sigmoid(logits).cpu().numpy().flatten()
                labels = batch['labels'].cpu().numpy()

                y_pred_proba_val.extend(proba)
                y_true_val.extend(labels)

        y_true_val = np.array(y_true_val)
        y_pred_proba_val = np.array(y_pred_proba_val)

        # Find best threshold on validation set
        logger.info('Finding best threshold on validation set...')
        threshold = find_best_threshold(y_true_val, y_pred_proba_val, metric='f1_macro')
        logger.info(f'Auto-tuned threshold: {threshold:.4f}')

    # Create eval dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    # Inference on eval split
    logger.info(f'Running inference on {args.split} set...')
    y_true = []
    y_pred_proba = []

    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            logits = model(batch)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = batch['labels'].cpu().numpy()

            y_pred_proba.extend(proba)
            y_true.extend(labels)

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    logger.info(f'Inference complete: {len(y_true)} samples')

    # Compute metrics
    logger.info(f'Computing metrics with threshold={threshold:.4f}')
    metrics = compute_all_metrics(y_true, y_pred_proba, threshold)

    # Print results
    logger.info('\n' + '=' * 60)
    logger.info(f'Evaluation Results ({args.split} split)')
    logger.info('=' * 60)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score (binary): {metrics['f1_binary']:.4f}")
    logger.info(f"F1 Score (macro):  {metrics['f1_macro']:.4f}")
    logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"AUC-PR:    {metrics['auc_pr']:.4f}")
    logger.info('=' * 60)

    # Print classification report
    y_pred = (y_pred_proba >= threshold).astype(int)
    print_classification_report(
        y_true,
        y_pred,
        label_names=['Not Misleading', 'Misleading']
    )

    # Save metrics to JSON
    results = {
        'split': args.split,
        'threshold': threshold,
        'metrics': metrics,
        'checkpoint': str(args.checkpoint),
    }

    output_path = output_dir / f'eval_{args.split}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    main()
