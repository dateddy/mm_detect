#!/usr/bin/env python3
"""Batch inference on new data: predict misinformation labels for a batch of ads."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collate import collate_fn
from src.data.dataset import AdDataset
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from torch.utils.data import DataLoader


def main():
    """Run inference on new data and save predictions to CSV."""
    parser = argparse.ArgumentParser(
        description='Run batch inference on new ads'
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
        '--csv',
        type=str,
        required=True,
        help='Path to input CSV with new data'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='data/raw/ad_images',
        help='Path to images directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV with predictions'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold for binary predictions'
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

    # Load input CSV (may lack 'misinformation' column if unlabeled)
    logger.info(f'Loading data from {args.csv}')
    df_input = pd.read_csv(args.csv)
    logger.info(f'Loaded {len(df_input)} records')

    # Verify required columns
    required_cols = ['text', 'ad_id', 'language']
    for col in required_cols:
        if col not in df_input.columns:
            raise ValueError(f'Missing required column: {col}')

    # Add dummy misinformation column if not present
    if 'misinformation' not in df_input.columns:
        logger.warning('no "misinformation" column found, using dummy labels for creation')
        df_input['misinformation'] = 0

    # Load model
    logger.info(f'Loading model from {args.checkpoint}')
    model = build_model(config)
    state_dict, _ = load_checkpoint(args.checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info('Model loaded and set to eval mode')

    # Create dataset
    logger.info('Creating dataset...')
    dataset = AdDataset(
        df_input,
        embeddings_dir=Path(config.get('embeddings_dir', 'data/embeddings')),
        images_dir=Path(args.images),
        cache_embeddings=False,  # Don't cache for new data
        mode='online',  # Always use online mode for new data
    )
    logger.info(f'Dataset created with {len(dataset)} samples')

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    # Run inference
    logger.info('Running inference...')
    predictions_proba = []
    predictions_binary = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            logits = model(batch)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            binary = (proba >= args.threshold).astype(int)

            predictions_proba.extend(proba)
            predictions_binary.extend(binary)

    logger.info(f'Inference complete: {len(predictions_proba)} predictions')

    # Prepare output dataframe
    df_output = df_input.copy()
    df_output['pred_proba'] = predictions_proba
    df_output['pred_label'] = predictions_binary
    df_output['pred_label_text'] = df_output['pred_label'].map({
        0: 'Not Misleading',
        1: 'Misleading'
    })

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)
    logger.info(f'Predictions saved to {output_path}')

    # Print summary statistics
    logger.info('\n' + '=' * 60)
    logger.info('Prediction Summary')
    logger.info('=' * 60)
    logger.info(f'Total records: {len(df_output)}')
    logger.info(f'Predicted Misleading: {(df_output["pred_label"] == 1).sum()} ({(df_output["pred_label"] == 1).mean() * 100:.1f}%)')
    logger.info(f'Predicted Not Misleading: {(df_output["pred_label"] == 0).sum()} ({(df_output["pred_label"] == 0).mean() * 100:.1f}%)')
    logger.info(f'Mean prob(Misleading): {np.mean(predictions_proba):.4f}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
