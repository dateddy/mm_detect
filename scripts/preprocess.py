"""Data preprocessing script"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.data.preprocessing import (
    load_raw_data,
    clean_text_features,
    normalize_numerical_features,
    encode_categorical_features,
    create_train_val_test_split,
    save_processed_splits
)


def main(args):
    logger = setup_logger('preprocess', log_file=Path('logs/preprocess.log'))
    
    # Load raw data
    logger.info('Loading raw data...')
    df = load_raw_data(Path(args.input_csv))
    logger.info(f'Loaded {len(df)} samples')
    
    # Clean text
    logger.info('Cleaning text features...')
    text_cols = ['ad_text_vn', 'ad_text_en']
    df = clean_text_features(df, text_cols)
    logger.info(f'After cleaning: {len(df)} samples')
    
    # Normalize numerical features
    logger.info('Normalizing numerical features...')
    numerical_cols = ['impressions', 'spend']
    df, scaler_info = normalize_numerical_features(df, numerical_cols)
    
    # Encode categorical
    logger.info('Encoding categorical features...')
    categorical_mappings = {
        'target_gender': {'M': 0, 'F': 1, 'U': 2},
        'publisher_platform': {
            'Facebook': 0,
            'Instagram': 1,
            'Audience Network': 2
        }
    }
    df = encode_categorical_features(df, categorical_mappings)
    
    # Create splits
    logger.info('Creating train/val/test splits...')
    train_df, val_df, test_df = create_train_val_test_split(
        df,
        label_col='misinformation',
        train_size=0.70,
        val_size=0.15,
        test_size=0.15
    )
    logger.info(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
    
    # Save processed splits
    logger.info(f'Saving to {args.output_dir}...')
    columns_to_save = ['ad_text', 'misinformation', 'target_gender_encoded', 'impressions_log', 'spend_log']
    save_processed_splits(train_df, val_df, test_df, Path(args.output_dir), columns_to_save)
    
    logger.info('Preprocessing complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument(
        '--input-csv',
        type=str,
        default='ads_vietnam_clean.csv',
        help='Input CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory'
    )
    
    args = parser.parse_args()
    main(args)
