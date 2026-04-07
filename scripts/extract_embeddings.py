"""Extract and cache embeddings from pretrained models"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.models.encoders.text_encoder import TextEncoder
from src.models.encoders.image_encoder import ImageEncoder
from src.models.encoders.metadata_encoder import MetadataEncoder


def main(args):
    logger = setup_logger('extract_embeddings', log_file=Path('logs/embeddings.log'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load data
    logger.info('Loading data...')
    df = pd.read_csv(Path(args.input_dir) / 'all_data.csv')
    logger.info(f'Loaded {len(df)} samples')
    
    # Initialize encoders
    logger.info('Loading encoders...')
    text_encoder = TextEncoder().to(device)
    image_encoder = ImageEncoder().to(device)
    
    # Extract text embeddings
    logger.info('Extracting text embeddings...')
    text_embeddings = []
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['ad_text'].iloc[i:i+batch_size].tolist()
        with torch.no_grad():
            batch_emb = text_encoder(batch_texts, device)
        text_embeddings.append(batch_emb.cpu().numpy())
    
    text_embeddings = np.concatenate(text_embeddings)
    logger.info(f'Text embeddings shape: {text_embeddings.shape}')
    
    # Save embeddings
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'text').mkdir(exist_ok=True)
    np.save(output_dir / 'text' / 'all_embeddings.npy', text_embeddings)
    logger.info('Embeddings saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract embeddings')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processed',
        help='Input directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/embeddings',
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
