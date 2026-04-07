"""Model evaluation script"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.models.multimodal_model import MultimodalModel
from src.data.dataset import MultimodalDataset
from src.evaluation.evaluator import Evaluator
from torch.utils.data import DataLoader


def main(args):
    logger = setup_logger('evaluate', log_file=Path('logs/evaluate.log'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load embeddings
    logger.info('Loading embeddings...')
    embeddings_dir = Path(args.embeddings_dir)
    text_emb = np.load(embeddings_dir / 'text' / 'all_embeddings.npy')
    image_emb = np.load(embeddings_dir / 'image' / 'all_embeddings.npy')
    metadata_emb = np.load(embeddings_dir / 'metadata' / 'all_features.npy')
    
    # Load labels and splits
    id_mapping = np.load(embeddings_dir / 'id_mapping.npy', allow_pickle=True).item()
    test_indices = id_mapping['test_indices']
    labels = torch.from_numpy(id_mapping['labels']).float()
    
    # Create test dataset
    test_dataset = MultimodalDataset(
        test_indices,
        torch.from_numpy(text_emb).float(),
        torch.from_numpy(image_emb).float(),
        torch.from_numpy(metadata_emb).float(),
        labels
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Load model
    logger.info('Loading model...')
    model = MultimodalModel()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    
    # Evaluate
    logger.info('Evaluating...')
    evaluator = Evaluator(model, device)
    results = evaluator.evaluate(
        test_loader,
        threshold=0.5,
        output_dir=Path(args.output_dir)
    )
    
    # Print results
    logger.info('\nResults:')
    for metric, value in results['metrics'].items():
        logger.info(f'{metric}: {value:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Model checkpoint path'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='data/embeddings',
        help='Embeddings directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='Output directory'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    args = parser.parse_args()
    main(args)
