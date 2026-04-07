"""Data preprocessing utilities"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """
    Load raw dataset from CSV.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(csv_path)


def clean_text_features(df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
    """
    Clean and combine text features.
    
    Args:
        df: Input DataFrame
        text_cols: List of text column names
        
    Returns:
        DataFrame with 'ad_text' column
    """
    df = df.copy()
    # Fill missing text with empty strings
    for col in text_cols:
        df[col] = df[col].fillna('')
    
    # Combine text columns
    df['ad_text'] = df[text_cols[0]]
    for col in text_cols[1:]:
        df['ad_text'] = df['ad_text'] + ' ' + df[col]
    
    # Remove rows with empty text
    df = df[df['ad_text'].str.strip().str.len() > 0]
    
    return df


def normalize_numerical_features(
    df: pd.DataFrame,
    numerical_cols: List[str]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize numerical features using log transformation and standardization.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        
    Returns:
        DataFrame with normalized features and scaler info
    """
    df = df.copy()
    scaler_info = {}
    
    for col in numerical_cols:
        # Handle missing values
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
        
        # Log transformation (for skewed distributions)
        log_col = f"{col}_log"
        df[log_col] = np.log1p(df[col])
        
        scaler_info[col] = {'min': df[col].min(), 'max': df[col].max()}
    
    return df, scaler_info


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_mappings: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        categorical_mappings: Dictionary mapping column names to value mappings
        
    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].fillna('U')  # Unknown
            df[f"{col}_encoded"] = df[col].map(mapping)
    
    return df


def create_train_val_test_split(
    df: pd.DataFrame,
    label_col: str = 'misinformation',
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation/test split.
    
    Args:
        df: Input DataFrame
        label_col: Name of label column
        train_size: Training set proportion
        val_size: Validation set proportion
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # First split: train+val vs test
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col]
    )
    
    # Second split: train vs val
    val_proportion = val_size / (train_size + val_size)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_proportion,
        random_state=random_state,
        stratify=df_train_val[label_col]
    )
    
    return df_train, df_val, df_test


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    columns_to_save: List[str]
) -> None:
    """
    Save processed datasets to CSV files.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output directory
        columns_to_save: Columns to include in saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df[columns_to_save].to_csv(output_dir / 'train.csv', index=False)
    val_df[columns_to_save].to_csv(output_dir / 'val.csv', index=False)
    test_df[columns_to_save].to_csv(output_dir / 'test.csv', index=False)

