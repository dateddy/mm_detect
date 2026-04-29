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

from src.data.feature_engineering import engineer_row_features, engineer_page_features

logger = logging.getLogger(__name__)

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Metadata Feature Scaling Configuration
# ============================================================================
# Specify feature types based on domain knowledge for per-feature scaling
METADATA_FEATURE_TYPES = {
    # Binary features (no scaling needed - already in [0, 1])
    "FB_only_flag": "binary",
    "all_targeted": "binary",
    "language_location_mismatch": "binary",

    # Bounded ratios in [0, 1] (no scaling needed - already comparable)
    "repeated_text_ratio": "ratio",
    "exclamation_ratio": "ratio",
    "caps_word_ratio": "ratio",

    # Heavy-tailed counts (log1p transform, then RobustScaler)
    "ads_per_page": "log_robust",
    "text_length": "log_robust",
    "burstiness": "log_robust",
    "launch_delay": "log_robust",
    "ads_duration": "log_robust",
    "avg_ad_duration": "log_robust",

    # Discrete counts with moderate range (RobustScaler)
    "num_countries": "robust",
    "platform_count": "robust",
    "emoji_count": "robust",
    "repeated_punct_count": "robust",
    "url_count": "robust",
}


# ============================================================================
# MetadataScaler Class
# ============================================================================
class MetadataScaler:
    """
    Per-feature scaling for tabular metadata features.

    Strategy:
        - Binary features: no scaling (already in [0, 1])
        - Ratio features: no scaling (already in [0, 1])
        - Heavy-tailed counts: log1p transform, then RobustScaler
        - Discrete counts: RobustScaler (median/IQR)

    Fit on TRAIN ONLY. Apply to all splits to prevent leakage.
    """

    def __init__(
        self,
        feature_columns: List[str],
        feature_types: Dict[str, str] = None
    ):
        """
        Initialize the scaler.

        Args:
            feature_columns: List of metadata feature column names.
            feature_types: Dict mapping column name to scaling type.
                If None, uses METADATA_FEATURE_TYPES.
        """
        self.feature_columns = feature_columns
        self.feature_types = feature_types or METADATA_FEATURE_TYPES

        # Validate that all columns have a known type
        for col in feature_columns:
            if col not in self.feature_types:
                raise ValueError(
                    f"Feature '{col}' has no scaling type defined in METADATA_FEATURE_TYPES"
                )

        # One RobustScaler per scalable feature (allows per-feature inspection)
        self.scalers: Dict[str, RobustScaler] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "MetadataScaler":
        """
        Fit scalers on the TRAINING dataframe only.

        Args:
            df: Training DataFrame.

        Returns:
            self (for chaining).
        """
        for col in self.feature_columns:
            ftype = self.feature_types[col]

            if ftype in ("binary", "ratio"):
                # No scaler needed for binary/ratio features
                continue

            # Get column values
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame, skipping")
                continue

            values = df[col].values.astype(np.float32).reshape(-1, 1)

            # Handle missing values: fill with median before fitting
            mask = ~np.isnan(values).flatten()
            if mask.sum() < len(values):
                median_val = np.nanmedian(values)
                values = np.where(np.isnan(values), median_val, values)

            if ftype == "log_robust":
                # log1p handles zeros (log(1+0)=0) and compresses tails
                values = np.log1p(np.maximum(values, 0))

            scaler = RobustScaler(quantile_range=(25.0, 75.0))
            scaler.fit(values)
            self.scalers[col] = scaler

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a dataframe to a [N, len(feature_columns)] float32 array.

        Output is in roughly [-3, 3] range for scaled features,
        [0, 1] for binary/ratio features.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed array of shape (N, num_features) with dtype float32.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Scaler must be fit before transform. Call .fit(train_df) first."
            )

        out = np.zeros((len(df), len(self.feature_columns)), dtype=np.float32)

        for i, col in enumerate(self.feature_columns):
            ftype = self.feature_types[col]

            if col not in df.columns:
                logger.warning(
                    f"Column '{col}' not found in DataFrame, filling with zeros"
                )
                out[:, i] = 0.0
                continue

            values = df[col].values.astype(np.float32)

            # Scaling by feature type
            if ftype == "binary":
                # Fill NaN with 0 (assume "feature absent")
                values = np.nan_to_num(values, nan=0.0)
                values = np.clip(values, 0.0, 1.0)

            elif ftype == "ratio":
                # Fill NaN with 0, clip to [0, 1]
                values = np.nan_to_num(values, nan=0.0)
                values = np.clip(values, 0.0, 1.0)

            elif ftype == "log_robust":
                # Fill NaN with 0 in input space (log1p(0)=0)
                values = np.nan_to_num(values, nan=0.0)
                values = np.log1p(np.maximum(values, 0)).reshape(-1, 1)
                values = self.scalers[col].transform(values).flatten()
                # Clip to ±5 to prevent extreme outliers
                values = np.clip(values, -5.0, 5.0)

            elif ftype == "robust":
                values = np.nan_to_num(values, nan=0.0).reshape(-1, 1)
                values = self.scalers[col].transform(values).flatten()
                values = np.clip(values, -5.0, 5.0)

            else:
                raise ValueError(f"Unknown feature type: {ftype}")

            out[:, i] = values

        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(df).transform(df)

    def save(self, path: Path) -> None:
        """
        Save fitted scaler to disk for reuse on val/test/inference.

        Args:
            path: Path to save the scaler pickle file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "feature_types": self.feature_types,
                "scalers": self.scalers,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info(f"Saved MetadataScaler to {path}")

    @classmethod
    def load(cls, path: Path) -> "MetadataScaler":
        """
        Load a fitted scaler from disk.

        Args:
            path: Path to the saved scaler pickle file.

        Returns:
            Loaded MetadataScaler instance.
        """
        data = joblib.load(path)
        instance = cls(data["feature_columns"], data["feature_types"])
        instance.scalers = data["scalers"]
        instance.is_fitted = data["is_fitted"]
        return instance

    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Diagnostic: return summary of feature distributions before/after scaling.

        Args:
            df: DataFrame to analyze.

        Returns:
            DataFrame with raw and scaled statistics for each feature.
        """
        scaled = self.transform(df)
        rows = []

        for i, col in enumerate(self.feature_columns):
            if col not in df.columns:
                logger.warning(f"Column '{col}' not in DataFrame for stats")
                continue

            raw = df[col].values
            row = {
                "feature": col,
                "type": self.feature_types[col],
                "raw_mean": np.nanmean(raw),
                "raw_std": np.nanstd(raw),
                "raw_min": np.nanmin(raw),
                "raw_max": np.nanmax(raw),
                "scaled_mean": scaled[:, i].mean(),
                "scaled_std": scaled[:, i].std(),
                "scaled_min": scaled[:, i].min(),
                "scaled_max": scaled[:, i].max(),
            }
            rows.append(row)

        return pd.DataFrame(rows)

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


def clean_text(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Clean text columns: lowercase, remove URLs/HTML/tags, normalize whitespace.

    Args:
        df: DataFrame to clean.
        text_columns: List of column names to clean.

    Returns:
        DataFrame with cleaned text columns.
    """
    df_clean = df.copy()

    for col in text_columns:
        if col not in df_clean.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue

        # Remove URLs
        df_clean[col] = df_clean[col].str.replace(
            r'http\S+|www\S+|ftp\S+', '', regex=True
        )

        # Remove HTML tags
        df_clean[col] = df_clean[col].str.replace(r'<[^>]+>', '', regex=True)

        # Lowercase
        df_clean[col] = df_clean[col].str.lower()

        # Normalize whitespace
        df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True).str.strip()

    return df_clean


def validate_config(config: Dict) -> None:
    """
    Validate that config has required keys.

    Args:
        config: Configuration dictionary.

    Raises:
        ValueError: If required keys are missing.
    """
    required_keys = ['raw_csv', 'processed_dir', 'metadata_features']
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")


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


def run_preprocessing_pipeline(config: Dict,
                                raw_bodies: pd.Series | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate the full preprocessing pipeline with correct order to prevent data leakage.

    ANTI-LEAKAGE STRATEGY:
    1. Compute row-level features on full dataset (safe, no aggregation)
    2. SPLIT by page_id (page-level split, not row-level)
    3. Compute page-level features separately per split using train statistics only
    4. Fit scaler on train only, apply to val/test

    Steps:
    1. Load raw CSV and parse datetimes
    2. Preserve raw bodies before text cleaning
    3. Clean text
    4. Engineer row-level features (safe on full dataset)
    5. Split by page_id with stratification (assert zero page overlap)
    6. Engineer page-level features (per split, using train reference)
    7. Fit and apply metadata scaler
    8. Run leakage audit
    9. Save all split CSVs and scaler

    Args:
        config: Configuration dictionary with keys:
            - raw_csv: Path to raw ads CSV
            - processed_dir: Output directory for processed files
            - metadata_features: List of metadata feature names
            - text: Dict with "columns" key (list of text columns to clean)
            - train_ratio, val_ratio: Split ratios
            - random_seed: Random seed
        raw_bodies: Optional Series of raw ad bodies before text cleaning.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("\n" + "="*80)
    logger.info("STARTING PREPROCESSING PIPELINE (Anti-Leakage Strategy)")
    logger.info("="*80)

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

    # ========== STEP 1: Load raw CSV ==========
    logger.info("\n[STEP 1] Loading raw CSV...")
    df_raw = pd.read_csv(raw_csv)
    logger.info(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # ========== STEP 2: Preserve raw bodies ==========
    logger.info("\n[STEP 2] Preserving raw bodies before text cleaning...")
    raw_bodies_preserved = df_raw[text_columns[0]].copy() if text_columns else None
    logger.info(f"Preserved raw text from column: {text_columns[0] if text_columns else 'N/A'}")

    # ========== STEP 3: Parse datetimes and clean text ==========
    logger.info("\n[STEP 3] Parsing datetime columns...")
    datetime_columns = [
        "ad_delivery_start_time",
        "ad_delivery_stop_time",
        "ad_creation_time",
    ]
    for col in datetime_columns:
        if col in df_raw.columns:
            df_raw[col] = pd.to_datetime(df_raw[col], errors='coerce')
    logger.info(f"Parsed {len([c for c in datetime_columns if c in df_raw.columns])} datetime columns")

    logger.info(f"\nCleaning text in {len(text_columns)} columns...")
    df_raw = clean_text(df_raw, text_columns)
    logger.info(f"Text cleaning complete")

    # ========== STEP 4: Engineer row-level features (SAFE on full dataset) ==========
    logger.info("\n[STEP 4] Engineering row-level features on full dataset...")
    logger.info("(Row-level features are safe because they don't aggregate across rows)")
    df_with_row_features = engineer_row_features(df_raw)
    logger.info(f"Row-level features engineered successfully")

    # ========== STEP 5: Split by page_id (PAGE-LEVEL SPLIT, NOT ROW-LEVEL) ==========
    logger.info("\n[STEP 5] Splitting by page_id with stratification...")
    logger.info("(CRITICAL: Splitting at page_id level to prevent data leakage)")
    train_df, val_df, test_df = split_by_page(
        df_with_row_features, train_ratio, val_ratio, random_seed
    )
    logger.info(f"Successfully split into 3 disjoint page sets")

    # ========== STEP 6: Engineer page-level features (per split, using train reference) ==========
    logger.info("\n[STEP 6] Engineering page-level features (using train statistics only)...")
    logger.info("  Computing page aggregations for train split...")
    train_df = engineer_page_features(train_df, reference_df=train_df)
    
    logger.info("  Computing page aggregations for val split (using train reference)...")
    val_df = engineer_page_features(val_df, reference_df=train_df)
    
    logger.info("  Computing page aggregations for test split (using train reference)...")
    test_df = engineer_page_features(test_df, reference_df=train_df)
    logger.info(f"Page-level features engineered successfully")

    # ========== STEP 7: Fit and apply metadata scaler ==========
    logger.info("\n[STEP 7] Fitting metadata scaler on training split only...")
    
    scaler = MetadataScaler(
        feature_columns=metadata_features,
        feature_types=METADATA_FEATURE_TYPES,
    )
    scaler.fit(train_df)
    
    # Save scaler to disk
    scaler_path = processed_path / "metadata_scaler.joblib"
    scaler.save(scaler_path)
    logger.info(f"Metadata scaler saved to {scaler_path}")
    
    # Print diagnostic stats
    logger.info("\n  Scaled metadata feature statistics (TRAIN):")
    stats_df = scaler.get_feature_stats(train_df)
    logger.info("\n" + stats_df.to_string(index=False))
    
    # Apply scaler to all splits
    logger.info("\nApplying scaler to all splits...")
    
    # For train, val, test - keep the original features but also store scaled arrays
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
    # Save scaled metadata arrays for direct loading in dataset
    np.save(processed_path / "train_metadata_scaled.npy", train_scaled)
    np.save(processed_path / "val_metadata_scaled.npy", val_scaled)
    np.save(processed_path / "test_metadata_scaled.npy", test_scaled)
    logger.info(f"Saved scaled metadata arrays:")
    logger.info(f"  train_metadata_scaled.npy: shape {train_scaled.shape}")
    logger.info(f"  val_metadata_scaled.npy: shape {val_scaled.shape}")
    logger.info(f"  test_metadata_scaled.npy: shape {test_scaled.shape}")
    
    logger.info("Metadata scaler applied to all splits")

    # ========== STEP 8: Run leakage audit ==========
    logger.info("\n[STEP 8] Running leakage audit...")
    try:
        from src.data.feature_engineering import leakage_audit
        leakage_audit(train_df, val_df, test_df, metadata_features)
        logger.info("Leakage audit PASSED ✓")
    except Exception as e:
        logger.warning(f"Leakage audit warning: {e}")

    # ========== STEP 9: Save split CSVs and scaler ==========
    logger.info("\n[STEP 9] Saving split CSVs...")

    splits_dir = processed_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train split ({len(train_df)} rows) to {train_path}")
    logger.info(f"Saved val split ({len(val_df)} rows) to {val_path}")
    logger.info(f"Saved test split ({len(test_df)} rows) to {test_path}")

    # ========== FINAL LOGGING ==========
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("="*80)

    logger.info("\nDataset Statistics:")
    total_ads = len(train_df) + len(val_df) + len(test_df)
    logger.info(f"  Total ads: {total_ads}")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/total_ads*100:.1f}%)")
    logger.info(f"  Val: {len(val_df)} ({len(val_df)/total_ads*100:.1f}%)")
    logger.info(f"  Test: {len(test_df)} ({len(test_df)/total_ads*100:.1f}%)")

    logger.info("\nPage Isolation:")
    train_pages = set(train_df["page_id"].unique())
    val_pages = set(val_df["page_id"].unique())
    test_pages = set(test_df["page_id"].unique())
    logger.info(f"  Train pages: {len(train_pages)}")
    logger.info(f"  Val pages: {len(val_pages)}")
    logger.info(f"  Test pages: {len(test_pages)}")
    logger.info(f"  Overlap: {len(train_pages & val_pages)} (train-val), "
                f"{len(train_pages & test_pages)} (train-test), "
                f"{len(val_pages & test_pages)} (val-test)")

    logger.info("\nMisinformation Label Distribution:")
    logger.info(f"  Train: {train_df['misinformation'].value_counts().to_dict()}")
    logger.info(f"  Val: {val_df['misinformation'].value_counts().to_dict()}")
    logger.info(f"  Test: {test_df['misinformation'].value_counts().to_dict()}")

    logger.info(f"\nMetadata Features ({len(metadata_features)} total):")
    for col in metadata_features:
        if col in train_df.columns:
            logger.info(
                f"  {col}: mean={train_df[col].mean():.4f}, std={train_df[col].std():.4f}"
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
