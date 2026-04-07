"""Main entry point for training"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.config import load_yaml, merge_configs
from src.models.encoders.text_encoder import TextEncoder
from src.models.encoders.image_encoder import ImageEncoder
from src.models.encoders.metadata_encoder import MetadataEncoder
from src.models.multimodal_model import MultimodalModel
from src.data.dataset import MultimodalDataset
from src.training.trainer import Trainer
from src.training.loss import FocalLoss
from src.training.scheduler import create_scheduler


def main(args):
    # Setup
    set_seed(42)
    logger = setup_logger('train', log_file=Path('logs/train.log'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load configuration
    configs = []
    configs.append(load_yaml(Path('configs/base.yaml')))
    if args.model_config:
        configs.append(load_yaml(Path(args.model_config)))
    if args.training_config:
        configs.append(load_yaml(Path(args.training_config)))
    
    config = merge_configs(*configs)
    logger.info(f'Configuration loaded')
    
    # Load embeddings
    embeddings_dir = Path(config.get('embeddings_dir', 'data/embeddings'))
    text_emb = np.load(embeddings_dir / 'text' / 'all_embeddings.npy')
    image_emb = np.load(embeddings_dir / 'image' / 'all_embeddings.npy')
    metadata_emb = np.load(embeddings_dir / 'metadata' / 'all_features.npy')
    
    # Load labels and splits
    id_mapping = np.load(embeddings_dir / 'id_mapping.npy', allow_pickle=True).item()
    train_indices = id_mapping['train_indices']
    val_indices = id_mapping['val_indices']
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
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False
    )
    
    # Model
    model = MultimodalModel(
        embedding_dim=768,
        num_heads=8,
        num_classes=1,
        hidden_dim=256,
        dropout=0.1,
        use_gating=True,
        use_fusion=True
    ).to(device)
    logger.info('Model created')
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3)
    )
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.get('scheduler', 'cosine'),
        max_epochs=config.get('num_epochs', 20)
    )
    
    # Trainer
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        scheduler,
        device,
        checkpoint_dir
    )
    
    # Train
    logger.info('Starting training...')
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=config.get('num_epochs', 20),
        early_stopping_patience=config.get('patience', 5)
    )
    
    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info('Training complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multimodal model')
    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/model/multimodal.yaml',
        help='Model config file'
    )
    parser.add_argument(
        '--training-config',
        type=str,
        default='configs/training/default.yaml',
        help='Training config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda/cpu)'
    )
    
    args = parser.parse_args()
    main(args)
