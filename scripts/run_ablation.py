"""Run ablation study with multiple model variants"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.dataset import MultimodalDataset
from src.ablation.ablation_runner import AblationRunner
from src.training.loss import FocalLoss
from torch.utils.data import DataLoader


def main(args):
    set_seed(42)
    logger = setup_logger('ablation', log_file=Path('logs/ablation.log'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load embeddings
    logger.info('Loading embeddings...')
    embeddings_dir = Path(args.embeddings_dir)
    text_emb = np.load(embeddings_dir / 'text' / 'all_embeddings.npy')
    image_emb = np.load(embeddings_dir / 'image' / 'all_embeddings.npy')
    metadata_emb = np.load(embeddings_dir / 'metadata' / 'all_features.npy')
    
    # Load splits
    id_mapping = np.load(embeddings_dir / 'id_mapping.npy', allow_pickle=True).item()
    train_indices = id_mapping['train_indices']
    val_indices = id_mapping['val_indices']
    test_indices = id_mapping['test_indices']
    labels = torch.from_numpy(id_mapping['labels']).float()
    
    # Create datasets
    train_dataset = MultimodalDataset(
        train_indices,
        torch.from_numpy(text_emb).float(),
        torch.from_numpy(image_emb).float(),
        torch.from_numpy(metadata_emb).float(),
        labels
    )
    
    val_dataset = MultimodalDataset(
        val_indices,
        torch.from_numpy(text_emb).float(),
        torch.from_numpy(image_emb).float(),
        torch.from_numpy(metadata_emb).float(),
        labels
    )
    
    test_dataset = MultimodalDataset(
        test_indices,
        torch.from_numpy(text_emb).float(),
        torch.from_numpy(image_emb).float(),
        torch.from_numpy(metadata_emb).float(),
        labels
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f'Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}')
    
    # Define ablation variants
    variants = {
        'text_only': {'embedding_dim': 768},
        'image_only': {'embedding_dim': 768},
        'metadata_only': {'embedding_dim': 768},
        'text_image': {'embedding_dim': 768},
        'full_multimodal': {'embedding_dim': 768},
    }
    
    # Run ablation
    logger.info('Running ablation study...')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    runner = AblationRunner(device=device, output_dir=output_dir)
    
    # Define optimizer factory
    def optimizer_fn(params):
        return torch.optim.Adam(params, lr=1e-3)
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Run all variants
    results_df = runner.run_ablation_study(
        variants,
        optimizer_fn,
        criterion,
        train_loader,
        val_loader,
        test_loader,
        epochs=args.epochs,
        patience=args.patience
    )
    
    logger.info('\nAblation Results:')
    logger.info(results_df.to_string())
    
    logger.info(f'Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='data/embeddings',
        help='Embeddings directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/ablation',
        help='Output directory'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs per variant'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience'
    )
    
    args = parser.parse_args()
    main(args)
