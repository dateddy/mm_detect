# src/data/preprocessing.py
"""Data preprocessing pipeline for multimodal misinformation detection."""

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.feature_engineering import engineer_all_features, engineer_row_features, engineer_page_features

logger = logging.getLogger(__name__)

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def split_by_page(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ads by page_id with stratification on majority label per page.

    Groups rows by page_id, computes majority misinformation label per page,
    then stratifies the split of page_ids by this label. Asserts zero page overlap
    across all three splits before returning.

    Args:
        df: DataFrame with 'page_id' and 'misinformation' columns.
        train_ratio: Fraction of pages for training (e.g., 0.7).
        val_ratio: Fraction of pages for validation (e.g., 0.15).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If page_id overlap detected across splits.
    """
    if "page_id" not in df.columns:
        raise ValueError("DataFrame must have 'page_id' column")
    if "misinformation" not in df.columns:
        raise ValueError("DataFrame must have 'misinformation' column")

    # Compute majority label per page
    page_labels = df.groupby("page_id")["misinformation"].agg(
        lambda x: (x.sum() / len(x)) >= 0.5
    ).astype(int)

    # Get unique page_ids with their majority labels
    # Convert to numpy to handle PyArrow arrays from pandas
    page_ids = page_labels.index.to_numpy()
    page_majority_labels = page_labels.to_numpy()

    # First split: train + temp (val + test)
    test_ratio = 1.0 - train_ratio - val_ratio
    train_pages, temp_pages, train_labels, temp_labels = train_test_split(
        page_ids,
        page_majority_labels,
        train_size=train_ratio,
        stratify=page_majority_labels,
        random_state=seed,
    )

    # Second split: val and test from temp
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_pages, test_pages, val_labels, test_labels = train_test_split(
        temp_pages,
        temp_labels,
        train_size=val_ratio_adjusted,
        stratify=temp_labels,
        random_state=seed,
    )

    # Extract rows for each split
    train_df = df[df["page_id"].isin(train_pages)].reset_index(drop=True)
    val_df = df[df["page_id"].isin(val_pages)].reset_index(drop=True)
    test_df = df[df["page_id"].isin(test_pages)].reset_index(drop=True)

    # === ASSERT ZERO PAGE OVERLAP ===
    train_page_set = set(train_df["page_id"].unique())
    val_page_set = set(val_df["page_id"].unique())
    test_page_set = set(test_df["page_id"].unique())

    train_val_overlap = train_page_set & val_page_set
    train_test_overlap = train_page_set & test_page_set
    val_test_overlap = val_page_set & test_page_set

    if train_val_overlap or train_test_overlap or val_test_overlap:
        overlaps = list(train_val_overlap or train_test_overlap or val_test_overlap)[:5]
        raise ValueError(
            f"CRITICAL: Page ID overlap detected across splits! "
            f"Samples (first 5): {overlaps}. "
            f"Train: {len(train_page_set)}, Val: {len(val_page_set)}, Test: {len(test_page_set)}"
        )

    # Log statistics
    logger.info(f"Split by page_id (zero overlap verified):")
    logger.info(f"  Train: {len(train_pages)} pages, {len(train_df)} ads")
    logger.info(f"    Label distribution: {train_df['misinformation'].value_counts().to_dict()}")
    logger.info(f"  Val: {len(val_pages)} pages, {len(val_df)} ads")
    logger.info(f"    Label distribution: {val_df['misinformation'].value_counts().to_dict()}")
    logger.info(f"  Test: {len(test_pages)} pages, {len(test_df)} ads")
    logger.info(f"    Label distribution: {test_df['misinformation'].value_counts().to_dict()}")

    return train_df, val_df, test_df

def fit_metadata_scaler(
    train_df: pd.DataFrame, feature_cols: List[str], processed_dir: str
) -> RobustScaler:
    """
    Fit RobustScaler on training metadata columns and save to disk.

    Args:
        train_df: Training DataFrame.
        feature_cols: List of metadata feature column names.
        processed_dir: Directory to save the scaler pickle file.

    Returns:
        Fitted RobustScaler instance.
    """
    if not all(col in train_df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in train_df.columns]
        logger.warning(f"Missing columns: {missing} - skipping these in scaling")
    
    # Filter to only columns that exist
    available_cols = [col for col in feature_cols if col in train_df.columns]
    
    # Filter to only numeric columns (skip string/object columns like emojis_in_text)
    numeric_cols = [col for col in available_cols 
                   if pd.api.types.is_numeric_dtype(train_df[col])]
    
    if not numeric_cols:
        logger.warning("No numeric columns found for scaling - returning empty scaler")
        scaler = RobustScaler()
        return scaler
    
    logger.info(f"Scaling {len(numeric_cols)} numeric metadata features")
    
    scaler = RobustScaler()
    scaler.fit(train_df[numeric_cols])

    # Save scaler to disk
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    scaler_path = processed_path / "metadata_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    logger.info(f"Fitted and saved metadata scaler to {scaler_path}")

    return scaler

def apply_metadata_scaler(
    df: pd.DataFrame, feature_cols: List[str], scaler: RobustScaler
) -> pd.DataFrame:
    """
    Transform metadata columns using fitted RobustScaler.

    Args:
        df: DataFrame to transform.
        feature_cols: List of metadata feature column names.
        scaler: Fitted RobustScaler instance.

    Returns:
        DataFrame with scaled metadata features.
    """
    df_scaled = df.copy()
    
    # Filter to only columns that exist and are numeric (matching fit_metadata_scaler)
    available_cols = [col for col in feature_cols if col in df.columns]
    numeric_cols = [col for col in available_cols 
                   if pd.api.types.is_numeric_dtype(df_scaled[col])]
    
    if not numeric_cols:
        logger.debug("No numeric columns to scale - returning unchanged DataFrame")
        return df_scaled
    
    # Apply scaler to numeric columns only
    df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])

    logger.debug(f"Applied metadata scaler to {len(numeric_cols)} numeric features")

    return df_scaled

def get_image_transforms(split: str, image_size: int) -> transforms.Compose:
    """
    Get torchvision image transforms for the given split.

    Args:
        split: One of 'train', 'val', 'test'.
        image_size: Target image size (square, e.g., 224).

    Returns:
        torchvision.transforms.Compose object.

    Raises:
        ValueError: If split is not in ['train', 'val', 'test'].
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:  # val or test
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training labels.

    Args:
        train_df: Training DataFrame with 'misinformation' column.

    Returns:
        Tensor of shape (2,) with weights [weight_class_0, weight_class_1].
    """
    if "misinformation" not in train_df.columns:
        raise ValueError("DataFrame must have 'misinformation' column")

    counts = train_df["misinformation"].value_counts().sort_index()

    # Handle missing classes
    if len(counts) < 2:
        logger.warning("Less than 2 classes found in training data")
        return torch.tensor([1.0, 1.0])

    # Inverse frequency weighting
    class_0_weight = 1.0 / (counts[0] / len(train_df))
    class_1_weight = 1.0 / (counts[1] / len(train_df))

    # Normalize so they sum to 2 (maintaining relative scale)
    weights = torch.tensor([class_0_weight, class_1_weight], dtype=torch.float32)
    weights = weights / weights.sum() * 2.0

    logger.info(
        f"Computed class weights: {weights.tolist()} "
        f"(class 0 count: {counts[0]}, class 1 count: {counts[1]})"
    )

    return weights


def compute_pos_weight(train_df: pd.DataFrame) -> float:
    """
    Compute the positive-class weight for BCEWithLogitsLoss.

    PyTorch's pos_weight scales the positive term only. The class-balanced
    setting is n_negative / n_positive, not the inverse.
    """
    if "misinformation" not in train_df.columns:
        raise ValueError("DataFrame must have 'misinformation' column")

    labels = train_df["misinformation"].astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    if n_pos == 0:
        raise ValueError("No positive labels in training data")
    if n_neg == 0:
        raise ValueError("No negative labels in training data")

    pos_weight = n_neg / n_pos
    logger.info(
        f"[compute_pos_weight] n_pos={n_pos:,}, n_neg={n_neg:,}, "
        f"pos_rate={n_pos / (n_pos + n_neg):.4f}, pos_weight={pos_weight:.4f}"
    )
    return float(pos_weight)

# ============================================================================
# TEXT PROCESSING HELPER FUNCTIONS
# ============================================================================

def clean_text_basic(text: str) -> str:
    """
    Convert to lowercase and remove escape sequences like \\n, \\t, etc.
    
    Args:
        text: Input text string.
    
    Returns:
        Cleaned lowercase text with escape sequences removed.
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove escape sequences: \n, \r, \t, etc.
    text = text.replace('\\n', ' ')
    text = text.replace('\\r', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\xa0', ' ')
    
    # Remove other common escape sequences
    text = re.sub(r'\\[a-zA-Z]', ' ', text)
    
    return text

def remove_urls(text: str) -> str:
    """
    Remove URLs from text using regex pattern.
    Matches http, https, www, and other URL patterns.
    
    Args:
        text: Input text string.
    
    Returns:
        Text with URLs removed.
    """
    if pd.isna(text):
        return ""
    
    # Remove URLs starting with http, https, www
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '',
        text
    )
    
    # Remove www. URLs
    text = re.sub(
        r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '',
        text
    )
    
    return text

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: remove extra spaces, tabs, and newlines.
    Replace multiple spaces with single space.
    
    Args:
        text: Input text string.
    
    Returns:
        Text with normalized whitespace.
    """
    if pd.isna(text):
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text


def clean_text(df: pd.DataFrame, text_columns: List[str], preserve_raw: bool = False) -> pd.DataFrame:
    """
    Clean raw Vietnamese ad text in-place across specified columns.
    Must run before feature engineering so text-derived features
    operate on cleaned text.

    Operations applied in this exact order:
    1. Lowercase
       text.str.lower()

    2. Remove HTTP/HTTPS URLs
       regex: r"https?://\\S+"  â†’ replace with single space

    3. Remove HTML tags
       regex: r"<[^>]+>"  â†’ replace with single space

    4. Remove Facebook mention tags (@username)
       regex: r"@\\w+"  â†’ replace with single space

    5. Remove hashtags (#topic)
       regex: r"#\\w+"  â†’ replace with single space

    6. Normalize whitespace
       collapse multiple spaces/tabs/newlines into a single space,
       then strip leading/trailing whitespace
       regex: r"\\s+"  â†’ replace with single space, then .strip()

    7. Normalize Vietnamese misspelling patterns (digit/symbol substitution)
       Remove digits inside words: "g1áº£m" â†’ "gáº£m"
       Replace @ with 'a': "gi@m" â†’ "giam"

    8. Collapse repeated characters (emotional emphasis)
       "Sá»‘cccccc" â†’ "Sá»‘cc", "GhÃªeeee" â†’ "GhÃªe"
       Keep at most 2 repetitions: r'(.)\\1{2,}' â†’ r'\\1\\1'

    9. Normalize Vietnamese currency and number formats
       "1.000.000Ä‘", "1,000,000 VND", "1tr", "1 triá»‡u" â†’ __CURRENCY__

    10. Strip leading/trailing whitespace (always last)

    7. Handle null/NaN: fill NaN with empty string "" before cleaning,
       then after cleaning replace empty strings back with NaN
       so downstream null checks still work correctly.

    Parameters
    ----------
    df              : DataFrame containing the text columns
    text_columns    : list of column names to clean, e.g.
                      ["ad_creative_bodies", "ad_creative_link_titles"]
    preserve_raw    : If True, save raw text copy to "{col}_raw" before cleaning
                      (default: False). Allows raw-text features to be computed later.

    Returns
    -------
    The same DataFrame with the specified columns cleaned in-place.
    If preserve_raw=True, adds new columns "{col}_raw" with original text.
    Log a one-line summary per column:
    "Cleaned column '{col}': {n_null} nulls, {n_empty} empty after cleaning"
    """
    df_cleaned = df.copy()
    
    for col in text_columns:
        if col not in df_cleaned.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
        
        # Step 0: Preserve raw copy if requested (BEFORE any cleaning)
        if preserve_raw:
            raw_col_name = f"{col}_raw"
            df_cleaned[raw_col_name] = df_cleaned[col].copy()
            logger.info(f"Saved raw text copy to '{raw_col_name}'")
        
        # Count nulls before cleaning
        n_null = df_cleaned[col].isna().sum()
        
        # Step 1: Fill NaN with empty string for uniform processing
        df_cleaned[col] = df_cleaned[col].fillna("")
        
        # Step 2: Lowercase
        df_cleaned[col] = df_cleaned[col].str.lower()
        
        # Step 3: Remove HTTP/HTTPS URLs
        df_cleaned[col] = df_cleaned[col].str.replace(r'https?://\S+', ' ', regex=True)
        
        # Step 4: Remove HTML tags
        df_cleaned[col] = df_cleaned[col].str.replace(r'<[^>]+>', ' ', regex=True)
        
        # Step 5: Remove Facebook mention tags (@username)
        df_cleaned[col] = df_cleaned[col].str.replace(r'@\w+', ' ', regex=True)
        
        # Step 6: Remove hashtags (#topic)
        df_cleaned[col] = df_cleaned[col].str.replace(r'#\w+', ' ', regex=True)
        
        # Step 7: Normalize whitespace
        df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Step 8 (Vietnamese-specific): Normalize misspelling patterns (digit/symbol substitution)
        # Remove digits inside words: "g1ảm" → "gảm" using apply() to avoid PyArrow issues
        def _remove_digits_in_words(text):
            """Remove digits that appear between letters."""
            import re
            return re.sub(r'([a-z])\d+([a-z])', r'\1\2', text)
        
        df_cleaned[col] = df_cleaned[col].apply(_remove_digits_in_words)
        # Replace @ with 'a': "gi@m" → "giam"
        df_cleaned[col] = df_cleaned[col].str.replace('@', 'a', regex=False)
        
        # Step 9 (Vietnamese-specific): Collapse repeated characters
        # Keep at most 2 repetitions: use apply() to avoid PyArrow backreference issues
        def _collapse_repeated_chars(text):
            """Collapse runs of 3+ identical characters to 2."""
            import re
            return re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        df_cleaned[col] = df_cleaned[col].apply(_collapse_repeated_chars)
        
        # Step 10 (Vietnamese-specific): Normalize currency and number formats
        # Replace with placeholder __CURRENCY__ to prevent overfitting to price points
        # Pattern for 1.000.000 format or 1,000,000 format
        df_cleaned[col] = df_cleaned[col].str.replace(
            r'\d+[.,]\d+[.,]\d+\s*[đvnd]*', '__CURRENCY__', regex=True
        )
        # Pattern for 1,000,000+ formats (2+ comma-separated groups)
        df_cleaned[col] = df_cleaned[col].str.replace(
            r'\d+(?:,\d+)+\s*[đvnd]*', '__CURRENCY__', regex=True
        )
        # Pattern for "1tr" or "1 triệu" (Vietnamese million)
        df_cleaned[col] = df_cleaned[col].str.replace(r'\d+\s*tr(?:iệu)?', '__CURRENCY__', regex=True)
        
        # Step 11: Replace empty strings back with NaN
        df_cleaned[col] = df_cleaned[col].replace("", pd.NA)
        
        # Count empty/null values after cleaning
        n_empty = df_cleaned[col].isna().sum()
        
        logger.info(f"Cleaned column '{col}': {n_null} nulls, {n_empty} empty after cleaning")
    
    return df_cleaned


# ============================================================================
# DATA PREPROCESSING PIPELINE
# ============================================================================


def validate_config(config: Dict) -> None:
    """
    Validate that the configuration dictionary has all required keys and structure.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If required keys are missing or invalid.
    """
    required_keys = ["raw_csv", "processed_dir"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"config must contain '{key}' key")

    # Validate text configuration (required and must be non-empty)
    if "text" not in config:
        raise ValueError("config must contain 'text' key with text cleaning configuration")
    
    text_config = config["text"]
    if not isinstance(text_config, dict):
        raise ValueError("config['text'] must be a dictionary")

    if "columns" not in text_config:
        raise ValueError("config['text'] must contain 'columns' key")

    if not isinstance(text_config["columns"], list):
        raise ValueError("config['text']['columns'] must be a list")

    if len(text_config["columns"]) == 0:
        raise ValueError("config['text']['columns'] must be a non-empty list")

    # Validate metadata configuration
    if "model" in config and "metadata_input_dim" in config["model"]:
        metadata_input_dim = config["model"]["metadata_input_dim"]
        metadata_features = config.get("metadata_features", [])
        assert metadata_input_dim == len(metadata_features), (
            f"metadata_input_dim={metadata_input_dim} does not match "
            f"len(metadata_features)={len(metadata_features)}. "
            f"Update metadata_input_dim in configs/base.yaml to match "
            f"the number of metadata features in metadata_features list."
        )

    logger.debug("Config validation passed")


def run_preprocessing_pipeline(config: Dict,
                                raw_bodies: pd.Series | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate the full preprocessing pipeline.

    Steps:
    1. Validate config
    2. Load raw CSV
    3. Parse datetime columns
    4. Clean text
    5. Run feature engineering
    6. Split by page with stratification
    7. Fit and apply metadata scaler
    8. Save all split CSVs
    9. Log comprehensive statistics

    Args:
        config: Configuration dictionary with keys:
            - raw_csv: Path to raw ads CSV
            - processed_dir: Output directory for processed files
            - metadata_features: List of metadata feature names
            - text: Dict with "columns" key (list of text columns to clean)
            - train_ratio, val_ratio: Split ratios
            - random_seed: Random seed
        raw_bodies: Optional Series of raw ad bodies before text cleaning.
                   Used for advanced text feature computation. If provided,
                   passed to engineer_all_features() for full dataset only.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Starting preprocessing pipeline...")

    # Step 1: Validate config
    validate_config(config)

    # Extract config parameters
    raw_csv = config.get("raw_csv")
    processed_dir = config.get("processed_dir")
    metadata_features = config.get("metadata_features", [])
    text_columns = config.get("text", {}).get("columns", [])
    train_ratio = config.get("train_ratio", 0.7)
    val_ratio = config.get("val_ratio", 0.15)
    random_seed = config.get("random_seed", 42)

    if not raw_csv or not processed_dir:
        raise ValueError("config must contain 'raw_csv' and 'processed_dir' keys")

    # Create processed directory
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Step 2: Load raw CSV
    logger.info(f"Loading raw CSV from {raw_csv}")
    df_raw = pd.read_csv(raw_csv)
    logger.info(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
    
    # === LABEL VALIDATION: Drop rows with null misinformation label ===
    if "misinformation" in df_raw.columns:
        n_before = len(df_raw)
        df_raw = df_raw.dropna(subset=["misinformation"]).reset_index(drop=True)
        n_dropped = n_before - len(df_raw)
        if n_dropped > 0:
            logger.info(f"⚠ Dropped {n_dropped} rows with null misinformation label")
    else:
        logger.warning("⚠ No 'misinformation' column found in raw CSV")

    # Step 3: Parse datetime columns
    logger.info("Parsing datetime columns...")
    datetime_columns = [
        "ad_delivery_start_time",
        "ad_delivery_stop_time",
        "ad_creation_time",
    ]
    for col in datetime_columns:
        if col in df_raw.columns:
            df_raw[col] = pd.to_datetime(df_raw[col], errors='coerce')
    logger.info(f"Parsed {len(datetime_columns)} datetime columns")

    # Step 4: Clean text
    logger.info(f"Cleaning text in {len(text_columns)} columns...")
    df_raw = clean_text(df_raw, text_columns)
    logger.info(f"Text cleaning complete â€” {len(text_columns)} columns cleaned (lowercase, URLs/HTML/tags removed, whitespace normalized)")

    # Step 5: Engineer ROW-LEVEL features (safe on full dataset)
    logger.info("Step 5: Engineering row-level features (safe on full dataset)...")
    df_with_row_features = engineer_row_features(df_raw)
    logger.info(f"Row-level features engineered successfully")

    # Step 6: Split by page_id BEFORE computing page-level features
    logger.info("Step 6: Splitting by page_id with stratification...")
    train_df, val_df, test_df = split_by_page(
        df_with_row_features, train_ratio, val_ratio, random_seed
    )
    logger.info(f"Successfully split into 3 disjoint page sets")
    
    # === LABEL VALIDATION: Check label consistency across splits ===
    if "misinformation" in train_df.columns:
        # Force misinformation to int and validate values
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            split_df["misinformation"] = split_df["misinformation"].astype(int)
            unique_labels = set(split_df["misinformation"].unique())
            if unique_labels != {0, 1}:
                raise ValueError(
                    f"Labels in {split_name} split must be {{0, 1}}, got {unique_labels}. "
                    f"Check raw CSV for unexpected label values."
                )
        
        # Assert label distribution consistency
        train_pos = train_df["misinformation"].mean()
        val_pos = val_df["misinformation"].mean()
        test_pos = test_df["misinformation"].mean()
        max_diff = max(abs(train_pos - val_pos), abs(train_pos - test_pos))
        
        logger.info(f"Label distribution check:")
        logger.info(f"  Train: {train_pos:.4f} positive | Val: {val_pos:.4f} | Test: {test_pos:.4f}")
        
        if max_diff >= 0.10:
            raise AssertionError(
                f"Label distribution drifted > 10pp across splits: "
                f"train={train_pos:.4f}, val={val_pos:.4f}, test={test_pos:.4f}. "
                f"Stratification may have failed — re-check split_by_page()."
            )
        logger.info(f"  ✓ Distribution consistent (max_diff={max_diff:.4f} < 0.10)")

    # Step 7: Engineer PAGE-LEVEL features (using train statistics only)
    logger.info("Step 7: Engineering page-level features (using train reference)...")
    logger.info("  Train split...")
    train_df = engineer_page_features(train_df, reference_df=train_df)
    logger.info("  Val split (using train reference)...")
    val_df = engineer_page_features(val_df, reference_df=train_df)
    logger.info("  Test split (using train reference)...")
    test_df = engineer_page_features(test_df, reference_df=train_df)
    logger.info(f"Page-level features engineered successfully")

    # Step 8: Fit and apply metadata scaler on training split ONLY
    logger.info("Step 8: Fitting metadata scaler on training split only...")
    scaler = fit_metadata_scaler(train_df, metadata_features, processed_dir)

    # Apply scaler to val and test
    logger.info("Applying scaler to val split...")
    val_df = apply_metadata_scaler(val_df, metadata_features, scaler)
    logger.info("Applying scaler to test split...")
    test_df = apply_metadata_scaler(test_df, metadata_features, scaler)
    logger.info("Applied metadata scaler to all splits")

    # Step 9: Save split CSVs
    logger.info("Step 9: Saving split CSVs...")

    splits_dir = processed_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train split to {train_path}")
    logger.info(f"Saved val split to {val_path}")
    logger.info(f"Saved test split to {test_path}")

    # Also save copies to main processed_dir for backward compatibility with audit scripts
    train_path_main = processed_path / "train.csv"
    val_path_main = processed_path / "val.csv"
    test_path_main = processed_path / "test.csv"
    
    train_df.to_csv(train_path_main, index=False)
    val_df.to_csv(val_path_main, index=False)
    test_df.to_csv(test_path_main, index=False)
    
    logger.info(f"Also saved copies to main processed dir:")
    logger.info(f"  {train_path_main}")
    logger.info(f"  {val_path_main}")
    logger.info(f"  {test_path_main}")

    # Step 9: Log comprehensive statistics
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("="*80)

    logger.info("\nDataset Statistics:")
    total_ads = len(train_df) + len(val_df) + len(test_df)
    logger.info(f"  Total ads: {total_ads}")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/total_ads*100:.1f}%)")
    logger.info(f"  Val: {len(val_df)} ({len(val_df)/total_ads*100:.1f}%)")
    logger.info(f"  Test: {len(test_df)} ({len(test_df)/total_ads*100:.1f}%)")

    logger.info("\nMisinformation Label Distribution:")
    logger.info(f"  Train: {train_df['misinformation'].value_counts().to_dict()}")
    logger.info(f"  Val: {val_df['misinformation'].value_counts().to_dict()}")
    logger.info(f"  Test: {test_df['misinformation'].value_counts().to_dict()}")

    logger.info(f"\nMetadata Features Scaled:")
    for col in metadata_features:
        if col in train_df.columns:
            logger.info(
                f"  {col}: mean={train_df[col].mean():.4f}, "
                f"std={train_df[col].std():.4f}"
            )

    logger.info(f"\nOutput Directory: {processed_dir}")
    logger.info(f"  Splits saved to: {splits_dir}")
    logger.info(f"  Scaler saved to: {processed_path / 'metadata_scaler.pkl'}")

    logger.info("="*80 + "\n")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline on raw ads CSV."
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        required=True,
        help="Path to raw ads CSV file.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        required=True,
        help="Output directory for processed splits and scaler.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of pages for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of pages for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    # Build config dict
    config = {
        "raw_csv": args.raw_csv,
        "processed_dir": args.processed_dir,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "random_seed": args.seed,
        "metadata_features": [
            "ads_per_page",
            "platform_count",
            "FB_only_flag",
            "all_targeted",
            "burstiness",
            "avg_ad_duration",
            "launch_delay",
            "num_countries",
            "language_location_mismatch",
        ],
    }

    # Run pipeline
    run_preprocessing_pipeline(config)

