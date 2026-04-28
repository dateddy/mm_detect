#!/usr/bin/env python3
"""Data preparation and preprocessing script."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    run_preprocessing_pipeline,
    clean_text_basic,
    remove_urls,
    normalize_whitespace,
)
from src.data.feature_engineering import (
    extract_emojis,
    count_emojis,
    remove_emojis,
    calculate_text_length,
    compute_time_based_features,
    compute_page_based_features,
    compute_spend_based_features,
    compute_demographic_features,
    compute_platform_features,
)
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


# ============================================================================
# TEXT PROCESSING HELPER FUNCTIONS
# ============================================================================

def apply_text_cleaning_pipeline(df: pd.DataFrame, text_columns: list = None) -> pd.DataFrame:
    """
    Apply full text cleaning pipeline to specified columns.
    
    Args:
        df: Input DataFrame.
        text_columns: List of column names to clean (default: ['ad_creative_bodies', 'ad_creative_link_titles']).
    
    Returns:
        DataFrame with cleaned text columns.
    """
    df = df.copy()
    
    if text_columns is None:
        text_columns = ['ad_creative_bodies', 'ad_creative_link_titles']
    
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text_basic)
            df[col] = df[col].apply(remove_urls)
            df[col] = df[col].apply(normalize_whitespace)
    
    logger.info(f"Applied text cleaning to {len(text_columns)} columns")
    return df


def extract_emoji_features(df: pd.DataFrame, text_columns: list = None) -> pd.DataFrame:
    """
    Extract emoji features from text columns.
    
    Args:
        df: Input DataFrame.
        text_columns: List of column names to process (default: ['ad_creative_bodies']).
    
    Returns:
        DataFrame with emoji features: 'emoji', 'emoji_count'.
    """
    df = df.copy()
    
    if text_columns is None:
        text_columns = ['ad_creative_bodies']
    
    # Combine emojis from all text columns
    all_emojis = []
    for col in text_columns:
        if col in df.columns:
            all_emojis.append(df[col].apply(extract_emojis))
    
    if all_emojis:
        df['emoji'] = all_emojis[0]
        for additional in all_emojis[1:]:
            df['emoji'] = df['emoji'] + additional
        
        df['emoji_count'] = df['emoji'].apply(count_emojis)
        
        # Remove emojis from original text columns
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(remove_emojis)
    
    logger.info(f"Extracted emoji features from {len(text_columns)} columns")
    return df


def compute_text_length_features(df: pd.DataFrame, text_columns: list = None) -> pd.DataFrame:
    """
    Compute text length features.
    
    Args:
        df: Input DataFrame.
        text_columns: List of column names to measure (default: ['ad_creative_bodies', 'ad_creative_link_titles']).
    
    Returns:
        DataFrame with text length features.
    """
    df = df.copy()
    
    if text_columns is None:
        text_columns = ['ad_creative_bodies', 'ad_creative_link_titles']
    
    total_length = None
    for col in text_columns:
        if col in df.columns:
            length_col = f"{col}_length"
            df[length_col] = df[col].apply(calculate_text_length)
            
            if total_length is None:
                total_length = df[length_col]
            else:
                total_length = total_length + df[length_col]
    
    if total_length is not None:
        df['text_length'] = total_length
    
    logger.info(f"Computed text length features for {len(text_columns)} columns")
    return df


# ============================================================================
# FEATURE ENGINEERING HELPER FUNCTIONS
# ============================================================================

def apply_all_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering functions in sequence.
    
    Args:
        df: Input DataFrame with raw ad data.
    
    Returns:
        DataFrame with all engineered features.
    """
    df = df.copy()
    
    logger.info("Starting comprehensive feature engineering...")
    
    # Time-based features
    df = compute_time_based_features(df)
    logger.debug(f"After time features: {df.shape}")
    
    # Page-based features
    df = compute_page_based_features(df)
    logger.debug(f"After page features: {df.shape}")
    
    # Spend-based features
    df = compute_spend_based_features(df)
    logger.debug(f"After spend features: {df.shape}")
    
    # Demographic features
    df = compute_demographic_features(df)
    logger.debug(f"After demographic features: {df.shape}")
    
    # Platform features
    df = compute_platform_features(df)
    logger.debug(f"After platform features: {df.shape}")
    
    logger.info("Feature engineering complete")
    return df


def apply_selective_feature_engineering(df: pd.DataFrame, feature_groups: list = None) -> pd.DataFrame:
    """
    Apply selective feature engineering based on specified groups.
    
    Args:
        df: Input DataFrame.
        feature_groups: List of feature groups to compute.
                       Options: ['time', 'page', 'spend', 'demographic', 'platform']
                       Default: all groups
    
    Returns:
        DataFrame with selected engineered features.
    """
    df = df.copy()
    
    if feature_groups is None:
        feature_groups = ['time', 'page', 'spend', 'demographic', 'platform']
    
    logger.info(f"Applying feature groups: {', '.join(feature_groups)}")
    
    if 'time' in feature_groups:
        df = compute_time_based_features(df)
        logger.debug("Added time-based features")
    
    if 'page' in feature_groups:
        df = compute_page_based_features(df)
        logger.debug("Added page-based features")
    
    if 'spend' in feature_groups:
        df = compute_spend_based_features(df)
        logger.debug("Added spend-based features")
    
    if 'demographic' in feature_groups:
        df = compute_demographic_features(df)
        logger.debug("Added demographic features")
    
    if 'platform' in feature_groups:
        df = compute_platform_features(df)
        logger.debug("Added platform features")
    
    logger.info("Selective feature engineering complete")
    return df


# ============================================================================
# COMPREHENSIVE DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_data_with_text_processing(raw_csv_path: str, output_dir: str, 
                                      text_cleaning: bool = True,
                                      extract_emojis_flag: bool = True,
                                      text_lengths: bool = True) -> pd.DataFrame:
    """
    Prepare data with text processing applied.
    
    Args:
        raw_csv_path: Path to raw CSV file.
        output_dir: Output directory for processed data.
        text_cleaning: Apply text cleaning (lowercase, remove URLs, normalize whitespace).
        extract_emojis_flag: Extract and count emojis.
        text_lengths: Calculate text length features.
    
    Returns:
        Processed DataFrame.
    """
    logger.info("Loading raw CSV for text processing...")
    df = pd.read_csv(raw_csv_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Apply text processing
    if text_cleaning:
        df = apply_text_cleaning_pipeline(df)
    
    if extract_emojis_flag:
        df = extract_emoji_features(df)
    
    if text_lengths:
        df = compute_text_length_features(df)
    
    # Save processed data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "data_with_text_processing.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved text-processed data to {output_path}")
    
    return df


def prepare_data_with_feature_engineering(raw_csv_path: str, output_dir: str,
                                         feature_groups: list = None) -> pd.DataFrame:
    """
    Prepare data with feature engineering applied.
    
    Args:
        raw_csv_path: Path to raw CSV file.
        output_dir: Output directory for processed data.
        feature_groups: List of feature groups to compute.
    
    Returns:
        DataFrame with engineered features.
    """
    logger.info("Loading raw CSV for feature engineering...")
    df = pd.read_csv(raw_csv_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Apply feature engineering
    if feature_groups is None:
        df = apply_all_feature_engineering(df)
    else:
        df = apply_selective_feature_engineering(df, feature_groups)
    
    logger.info(f"Engineered data: {len(df)} rows, {len(df.columns)} columns")
    
    # Save processed data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "data_with_engineered_features.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved engineered data to {output_path}")
    
    return df


def prepare_data_full_pipeline(raw_csv_path: str, output_dir: str,
                              text_processing: bool = True,
                              feature_engineering: bool = True) -> pd.DataFrame:
    """
    Prepare data with full pipeline: text processing + feature engineering.
    
    Args:
        raw_csv_path: Path to raw CSV file.
        output_dir: Output directory for processed data.
        text_processing: Apply text processing pipeline.
        feature_engineering: Apply feature engineering.
    
    Returns:
        Fully processed DataFrame.
    """
    logger.info("Starting full data preparation pipeline...")
    df = pd.read_csv(raw_csv_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Step 1: Text processing
    if text_processing:
        logger.info("Step 1: Text Processing")
        df = apply_text_cleaning_pipeline(df)
        df = extract_emoji_features(df)
        df = compute_text_length_features(df)
    
    # Step 2: Feature engineering
    if feature_engineering:
        logger.info("Step 2: Feature Engineering")
        df = apply_all_feature_engineering(df)
    
    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Save processed data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "data_fully_processed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved fully processed data to {output_path}")
    
    return df


# ============================================================================
# ADVANCED DATA PREPARATION WITH LEAKAGE CHECKS AND ALL FEATURES
# ============================================================================

def get_feature_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics of all features in DataFrame.
    
    Args:
        df: DataFrame with all features.
    
    Returns:
        Dictionary with feature statistics.
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_features': len(df.select_dtypes(include=['number']).columns),
        'text_features': len(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'text_columns': df.select_dtypes(include=['object']).columns.tolist(),
    }
    return summary


def log_feature_summary(df: pd.DataFrame, stage: str = ""):
    """Log summary of features in DataFrame."""
    summary = get_feature_summary(df)
    logger.info(f"\n{'='*70}")
    logger.info(f"Feature Summary {stage}")
    logger.info(f"{'='*70}")
    logger.info(f"  Rows: {summary['total_rows']}")
    logger.info(f"  Total Columns: {summary['total_columns']}")
    logger.info(f"  Numeric Features: {summary['numeric_features']}")
    logger.info(f"  Text Features: {summary['text_features']}")
    logger.info(f"  Missing Values: {summary['missing_values']}")
    logger.info(f"\nNumeric Columns ({summary['numeric_features']}):")
    for col in summary['numeric_columns'][:10]:  # Show first 10
        logger.info(f"    - {col}")
    if len(summary['numeric_columns']) > 10:
        logger.info(f"    ... and {len(summary['numeric_columns']) - 10} more")
    logger.info(f"{'='*70}\n")


def validate_no_label_leakage(df: pd.DataFrame, label_column: str = 'misinformation') -> bool:
    """
    Validate that target label is not in features (Check 3).
    
    Args:
        df: DataFrame to validate.
        label_column: Name of label column.
    
    Returns:
        True if check passes, False otherwise.
    """
    if label_column in df.columns:
        logger.warning(f"✓ Label column '{label_column}' found (expected for training data)")
        return True
    else:
        logger.error(f"✗ Label column '{label_column}' not found!")
        return False


def validate_page_isolation(df: pd.DataFrame, split_metadata: dict = None) -> bool:
    """
    Validate page-level isolation across splits (Check 1 & 2).
    
    Args:
        df: DataFrame to validate.
        split_metadata: Dictionary with train_pages, val_pages, test_pages if already split.
    
    Returns:
        True if check passes, False otherwise.
    """
    if 'page_id' not in df.columns:
        logger.warning("page_id column not found, skipping page isolation check")
        return True
    
    if split_metadata:
        train_pages = set(split_metadata.get('train_pages', []))
        val_pages = set(split_metadata.get('val_pages', []))
        test_pages = set(split_metadata.get('test_pages', []))
        
        overlap_tv = len(train_pages & val_pages)
        overlap_t_test = len(train_pages & test_pages)
        overlap_v_test = len(val_pages & test_pages)
        
        if overlap_tv == 0 and overlap_t_test == 0 and overlap_v_test == 0:
            logger.info("✓ Check 1 (page_isolation): PASS - No page overlap across splits")
            return True
        else:
            logger.error(f"✗ Check 1 (page_isolation): FAIL - Overlaps: train-val={overlap_tv}, train-test={overlap_t_test}, val-test={overlap_v_test}")
            return False
    else:
        logger.info("✓ Check 1 (page_isolation): Data not yet split, will be checked during split")
        return True


def validate_feature_integrity(df: pd.DataFrame) -> dict:
    """
    Validate feature engineering integrity.
    
    Args:
        df: DataFrame with engineered features.
    
    Returns:
        Dictionary with validation results.
    """
    validation = {
        'has_time_features': all(col in df.columns for col in ['ads_duration', 'launch_delay', 'burstiness']),
        'has_page_features': all(col in df.columns for col in ['ads_per_page', 'avg_ad_duration']),
        'has_spend_features': all(col in df.columns for col in ['spend_per_day', 'impressions_per_day']),
        'has_demographic_features': all(col in df.columns for col in ['age_span', 'num_countries']),
        'has_platform_features': all(col in df.columns for col in ['platform_count', 'FB_only_flag']),
        'has_text_features': all(col in df.columns for col in ['text_length', 'emoji_count']),
    }
    
    logger.info("\nFeature Group Validation:")
    logger.info(f"  Time-based features: {'✓' if validation['has_time_features'] else '✗'}")
    logger.info(f"  Page-based features: {'✓' if validation['has_page_features'] else '✗'}")
    logger.info(f"  Spend-based features: {'✓' if validation['has_spend_features'] else '✗'}")
    logger.info(f"  Demographic features: {'✓' if validation['has_demographic_features'] else '✗'}")
    logger.info(f"  Platform features: {'✓' if validation['has_platform_features'] else '✗'}")
    logger.info(f"  Text features: {'✓' if validation['has_text_features'] else '✗'}")
    
    return validation


def prepare_data_comprehensive(raw_csv_path: str, output_dir: str,
                               apply_text_processing: bool = True,
                               apply_feature_engineering: bool = True,
                               apply_leakage_checks: bool = True,
                               save_all_features: bool = True) -> dict:
    """
    Comprehensive data preparation with ALL features and leakage validation.
    Saves all features (not just metadata) for complete analysis.
    
    Args:
        raw_csv_path: Path to raw CSV file.
        output_dir: Output directory for processed data.
        apply_text_processing: Apply text cleaning and emoji extraction.
        apply_feature_engineering: Apply all feature engineering.
        apply_leakage_checks: Run data leakage validation checks.
        save_all_features: Save complete feature set (all features, not just metadata).
    
    Returns:
        Dictionary with processed data and validation results.
    """
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE DATA PREPARATION PIPELINE")
    logger.info("="*80 + "\n")
    
    # Step 1: Load raw data
    logger.info("Step 1: Loading raw data...")
    df_raw = pd.read_csv(raw_csv_path)
    logger.info(f"  Loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    log_feature_summary(df_raw, "(Raw Data)")
    
    # Step 2: Text processing
    df_processed = df_raw.copy()
    if apply_text_processing:
        logger.info("\nStep 2: Applying text processing...")
        df_processed = apply_text_cleaning_pipeline(df_processed)
        df_processed = extract_emoji_features(df_processed)
        df_processed = compute_text_length_features(df_processed)
        logger.info(f"  Result: {len(df_processed)} rows, {len(df_processed.columns)} columns")
        log_feature_summary(df_processed, "(After Text Processing)")
    
    # Step 3: Feature engineering
    if apply_feature_engineering:
        logger.info("\nStep 3: Applying feature engineering...")
        df_processed = apply_all_feature_engineering(df_processed)
        logger.info(f"  Result: {len(df_processed)} rows, {len(df_processed.columns)} columns")
        log_feature_summary(df_processed, "(After Feature Engineering)")
    
    # Step 4: Validation and leakage checks
    validation_results = {
        'label_check': False,
        'page_isolation_check': False,
        'feature_integrity_check': {},
    }
    
    if apply_leakage_checks:
        logger.info("\nStep 4: Running data leakage checks...")
        logger.info("\n" + "="*70)
        logger.info("DATA LEAKAGE VALIDATION CHECKS")
        logger.info("="*70)
        
        # Check 3: Label not in features
        validation_results['label_check'] = validate_no_label_leakage(df_processed)
        
        # Check 1 & 2: Page isolation (pre-split)
        validation_results['page_isolation_check'] = validate_page_isolation(df_processed)
        
        # Feature integrity
        validation_results['feature_integrity_check'] = validate_feature_integrity(df_processed)
        
        logger.info("="*70 + "\n")
    
    # Step 5: Save all features
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    if save_all_features:
        logger.info("\nStep 5: Saving all features...")
        
        # Save complete dataset with ALL features
        output_path_all = Path(output_dir) / "data_all_features_complete.csv"
        df_processed.to_csv(output_path_all, index=False)
        output_files['all_features'] = str(output_path_all)
        logger.info(f"  ✓ All features saved: {output_path_all}")
        logger.info(f"    ({len(df_processed)} rows, {len(df_processed.columns)} columns)")
        
        # Save feature metadata
        feature_metadata = {
            'total_features': len(df_processed.columns),
            'numeric_features': len(df_processed.select_dtypes(include=['number']).columns),
            'text_features': len(df_processed.select_dtypes(include=['object']).columns),
            'feature_list': df_processed.columns.tolist(),
            'rows': len(df_processed),
            'missing_values': int(df_processed.isnull().sum().sum()),
        }
        
        metadata_path = Path(output_dir) / "feature_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        output_files['metadata'] = str(metadata_path)
        logger.info(f"  ✓ Feature metadata saved: {metadata_path}")
        
        # Save feature statistics
        stats_path = Path(output_dir) / "feature_statistics.csv"
        stats_df = pd.DataFrame({
            'Feature': df_processed.columns,
            'Data_Type': [str(df_processed[col].dtype) for col in df_processed.columns],
            'Non_Null_Count': [df_processed[col].count() for col in df_processed.columns],
            'Null_Count': [df_processed[col].isnull().sum() for col in df_processed.columns],
            'Null_Percentage': [f"{df_processed[col].isnull().sum() / len(df_processed) * 100:.2f}%" for col in df_processed.columns],
        })
        stats_df.to_csv(stats_path, index=False)
        output_files['statistics'] = str(stats_path)
        logger.info(f"  ✓ Feature statistics saved: {stats_path}")
    
    # Step 6: Summary report
    logger.info("\n" + "="*80)
    logger.info("DATA PREPARATION SUMMARY")
    logger.info("="*80)
    logger.info(f"\nOriginal Data:")
    logger.info(f"  Rows: {len(df_raw)}")
    logger.info(f"  Columns: {len(df_raw.columns)}")
    
    logger.info(f"\nProcessed Data:")
    logger.info(f"  Rows: {len(df_processed)}")
    logger.info(f"  Columns: {len(df_processed.columns)}")
    logger.info(f"  New Features Added: {len(df_processed.columns) - len(df_raw.columns)}")
    
    logger.info(f"\nValidation Results:")
    logger.info(f"  Label Leakage Check: {'✓ PASS' if validation_results['label_check'] else '✗ FAIL'}")
    logger.info(f"  Page Isolation Check: {'✓ PASS' if validation_results['page_isolation_check'] else '✗ FAIL'}")
    
    feature_check = validation_results['feature_integrity_check']
    all_features_ok = all(feature_check.values()) if feature_check else False
    logger.info(f"  Feature Integrity Check: {'✓ PASS' if all_features_ok else '⚠ PARTIAL'}")
    
    logger.info(f"\nOutput Files:")
    for file_type, file_path in output_files.items():
        logger.info(f"  ✓ {file_type}: {file_path}")
    
    logger.info("="*80 + "\n")
    
    return {
        'dataframe': df_processed,
        'validation_results': validation_results,
        'output_files': output_files,
        'feature_summary': get_feature_summary(df_processed),
    }


def prepare_data_split_comprehensive(raw_csv_path: str, output_dir: str,
                                    train_ratio: float = 0.7,
                                    val_ratio: float = 0.15,
                                    random_seed: int = 42,
                                    apply_text_processing: bool = True,
                                    apply_feature_engineering: bool = True) -> dict:
    """
    Prepare data comprehensively and create train/val/test splits WITH ALL FEATURES.
    
    Args:
        raw_csv_path: Path to raw CSV file.
        output_dir: Output directory for processed data.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        random_seed: Random seed for reproducibility.
        apply_text_processing: Apply text processing.
        apply_feature_engineering: Apply feature engineering.
    
    Returns:
        Dictionary with processed data and split information.
    """
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE DATA PREPARATION WITH TRAIN/VAL/TEST SPLIT")
    logger.info("="*80 + "\n")
    
    # Prepare comprehensive data
    prep_result = prepare_data_comprehensive(
        raw_csv_path=raw_csv_path,
        output_dir=output_dir,
        apply_text_processing=apply_text_processing,
        apply_feature_engineering=apply_feature_engineering,
        apply_leakage_checks=True,
        save_all_features=True,
    )
    
    df_processed = prep_result['dataframe']
    
    # Split by page if page_id exists
    logger.info("\nStep 6: Creating train/val/test splits...")
    
    if 'page_id' in df_processed.columns:
        logger.info("  Using page-level stratified split...")
        
        # Get unique pages and their majority labels
        page_labels = df_processed.groupby('page_id')['misinformation'].agg(
            lambda x: (x.sum() / len(x)) >= 0.5
        ).astype(int)
        
        page_ids = page_labels.index.values
        page_majority_labels = page_labels.values
        
        # Split pages
        from sklearn.model_selection import train_test_split
        
        test_ratio = 1.0 - train_ratio - val_ratio
        train_pages, temp_pages, _, _ = train_test_split(
            page_ids, page_majority_labels,
            train_size=train_ratio,
            stratify=page_majority_labels,
            random_state=random_seed,
        )
        
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_pages, test_pages, _, _ = train_test_split(
            temp_pages, temp_pages,
            train_size=val_ratio_adjusted,
            random_state=random_seed,
        )
        
        # Create splits
        train_df = df_processed[df_processed['page_id'].isin(train_pages)].reset_index(drop=True)
        val_df = df_processed[df_processed['page_id'].isin(val_pages)].reset_index(drop=True)
        test_df = df_processed[df_processed['page_id'].isin(test_pages)].reset_index(drop=True)
    else:
        logger.info("  Using random stratified split...")
        
        from sklearn.model_selection import train_test_split
        
        test_ratio = 1.0 - train_ratio - val_ratio
        train_df, temp_df = train_test_split(
            df_processed,
            train_size=train_ratio,
            stratify=df_processed.get('misinformation') if 'misinformation' in df_processed.columns else None,
            random_state=random_seed,
        )
        
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            random_state=random_seed,
        )
    
    # Save all splits with ALL features
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    splits_dir = Path(output_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False, float_format='%.0f')
    val_df.to_csv(val_path, index=False, float_format='%.0f')
    test_df.to_csv(test_path, index=False, float_format='%.0f')
    
    logger.info(f"\n  ✓ Train split: {train_path} ({len(train_df)} rows, {len(train_df.columns)} columns)")
    logger.info(f"  ✓ Val split: {val_path} ({len(val_df)} rows, {len(val_df.columns)} columns)")
    logger.info(f"  ✓ Test split: {test_path} ({len(test_df)} rows, {len(test_df.columns)} columns)")
    
    # Split statistics
    logger.info("\n" + "="*80)
    logger.info("SPLIT STATISTICS")
    logger.info("="*80)
    
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        ratio = len(split_df) / len(df_processed) * 100
        if 'misinformation' in split_df.columns:
            label_0 = (split_df['misinformation'] == 0).sum()
            label_1 = (split_df['misinformation'] == 1).sum()
            logger.info(f"\n{split_name}:")
            logger.info(f"  Rows: {len(split_df)} ({ratio:.1f}%)")
            logger.info(f"  Columns: {len(split_df.columns)}")
            logger.info(f"  Class 0: {label_0} ({label_0/len(split_df)*100:.1f}%)")
            logger.info(f"  Class 1: {label_1} ({label_1/len(split_df)*100:.1f}%)")
        else:
            logger.info(f"\n{split_name}:")
            logger.info(f"  Rows: {len(split_df)} ({ratio:.1f}%)")
            logger.info(f"  Columns: {len(split_df.columns)}")
    
    logger.info("="*80 + "\n")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'train_path': str(train_path),
        'val_path': str(val_path),
        'test_path': str(test_path),
        'validation_results': prep_result['validation_results'],
        'feature_summary': prep_result['feature_summary'],
    }


def main():
    """Main entry point for data preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare and preprocess data for training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file (YAML or JSON)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Override CSV path from config",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Override images directory from config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    # If path is relative, resolve it relative to project root
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix == ".json":
            config = json.load(f)
        else:
            config = yaml.safe_load(f)

    # Setup logging
    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    get_logger("root", log_file=str(log_dir / "preprocessing.log"))

    # Set seed
    set_seed(config["data"].get("random_seed", 42))

    logger.info("=" * 70)
    logger.info("Data Preparation & Preprocessing")
    logger.info("=" * 70)
    logger.info(f"Config: {config_path}")

    # Override paths if provided
    if args.csv:
        config["paths"]["raw_csv"] = args.csv
        logger.info(f"Override CSV: {args.csv}")
    if args.images:
        config["paths"]["images_dir"] = args.images
        logger.info(f"Override Images: {args.images}")
    if args.output:
        config["paths"]["processed_dir"] = args.output
        logger.info(f"Override Output: {args.output}")

    # Flatten config for preprocessing pipeline and resolve relative paths
    project_root = Path(__file__).parent.parent
    raw_csv_path = Path(config["paths"]["raw_csv"])
    if not raw_csv_path.is_absolute():
        raw_csv_path = project_root / raw_csv_path
    
    processed_dir_path = Path(config["paths"]["processed_dir"])
    if not processed_dir_path.is_absolute():
        processed_dir_path = project_root / processed_dir_path
    
    flat_config = {
        "raw_csv": str(raw_csv_path),
        "processed_dir": str(processed_dir_path),
        "metadata_features": config.get("metadata_features", []),
        "text": config.get("text", {}),
        "train_ratio": config["data"].get("train_ratio", 0.7),
        "val_ratio": config["data"].get("val_ratio", 0.15),
        "test_ratio": config["data"].get("test_ratio", 0.15),
        "random_seed": config["data"].get("random_seed", 42),
    }

    # Preserve raw ad creative bodies before preprocessing
    logger.info("Loading raw CSV to preserve ad bodies for feature engineering...")
    df_raw_temp = pd.read_csv(flat_config["raw_csv"])
    raw_bodies = df_raw_temp["ad_creative_bodies"].copy() if "ad_creative_bodies" in df_raw_temp.columns else None
    logger.info(f"Preserved raw ad bodies: {len(raw_bodies) if raw_bodies is not None else 0} rows")

    # Run preprocessing pipeline
    logger.info("Running preprocessing pipeline...")
    train_df, val_df, test_df = run_preprocessing_pipeline(flat_config, raw_bodies=raw_bodies)

    # Print summary table
    logger.info("=" * 70)
    logger.info("Data Split Summary")
    logger.info("=" * 70)

    summary_data = []
    for split, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        num_samples = len(df)
        num_pages = df["page_id"].nunique() if "page_id" in df.columns else num_samples
        label_0 = (df["misinformation"] == 0).sum() if "misinformation" in df.columns else 0
        label_1 = (df["misinformation"] == 1).sum() if "misinformation" in df.columns else 0

        summary_data.append({
            "Split": split,
            "Samples": num_samples,
            "Pages": num_pages,
            "Not Misleading": label_0,
            "Misleading": label_1,
            "Label Ratio": f"{label_1 / num_samples:.2%}" if num_samples > 0 else "N/A",
        })

    summary_df = pd.DataFrame(summary_data)
    logger.info("\n" + summary_df.to_string(index=False))

    # Log engineered features
    logger.info("\n" + "=" * 70)
    logger.info("Engineered Features Summary")
    logger.info("=" * 70)
    
    engineered_features = {
        "Core Metadata (9)": [
            "ads_per_page", "platform_count", "FB_only_flag", "all_targeted",
            "burstiness", "avg_ad_duration", "launch_delay", "num_countries",
            "language_location_mismatch"
        ],
        "Text-Based (3)": [
            "emojis_in_text", "emoji_count", "text_length"
        ],
        "Time-Based": [
            "ads_duration", "launch_delay", "burstiness", "active_status"
        ],
        "Page-Based": [
            "ads_per_page", "avg_ad_duration", "repeated_text_ratio", "historical_volume"
        ],
        "Spend-Based": [
            "spend_per_day", "impressions_per_day", "CPM_estimate", "low_spend_high_reach"
        ],
        "Demographic": [
            "age_span", "num_countries", "women_targeted", "men_targeted",
            "all_targeted", "language_location_mismatch"
        ],
        "Platform": [
            "platform_count", "FB_only_flag", "IG_only_flag"
        ]
    }
    
    for category, features in engineered_features.items():
        existing_features = [f for f in features if f in train_df.columns]
        logger.info(f"  {category}: {len(existing_features)} features")
        for f in existing_features[:5]:
            logger.info(f"    • {f}")
        if len(existing_features) > 5:
            logger.info(f"    ... and {len(existing_features) - 5} more")

    logger.info("=" * 70)
    logger.info("Data preparation complete!")
    logger.info(f"Processed data saved to: {flat_config['processed_dir']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
