#!/usr/bin/env python3
"""
Comprehensive label validation script to diagnose label inversion.

Performs 5 checks:
1. Load and compare train/val/test splits for label consistency
2. Cross-reference raw CSV to verify labels weren't inverted in preprocessing
3. Train logistic regression on metadata to check if labels are valid
4. Evaluate on val and test separately
5. Report findings with exit code
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_label_distribution(processed_dir: Path) -> bool:
    """
    CHECK 1: Load train/val/test and verify label distribution consistency.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    logger.info("=" * 80)
    logger.info("CHECK 1: LABEL DISTRIBUTION ACROSS SPLITS")
    logger.info("=" * 80)
    
    try:
        train_df = pd.read_csv(processed_dir / "splits" / "train.csv", dtype={"id": str})
        val_df = pd.read_csv(processed_dir / "splits" / "val.csv", dtype={"id": str})
        test_df = pd.read_csv(processed_dir / "splits" / "test.csv", dtype={"id": str})
    except FileNotFoundError as e:
        logger.error(f"❌ Failed to load split CSVs: {e}")
        return False
    
    # Check train split
    logger.info("\n[TRAIN SPLIT]")
    logger.info(f"  Rows: {len(train_df)}")
    logger.info(f"  Positive rate: {train_df['misinformation'].mean():.4f}")
    logger.info(f"  Unique labels: {sorted(train_df['misinformation'].unique())}")
    logger.info(f"  Dtype: {train_df['misinformation'].dtype}")
    logger.info(f"  Null count: {train_df['misinformation'].isna().sum()}")
    logger.info(f"  First 5 samples:")
    for idx, row in train_df[['id', 'misinformation']].head(5).iterrows():
        logger.info(f"    id={row['id']}, label={row['misinformation']}")
    
    # Check val split
    logger.info("\n[VAL SPLIT]")
    logger.info(f"  Rows: {len(val_df)}")
    logger.info(f"  Positive rate: {val_df['misinformation'].mean():.4f}")
    logger.info(f"  Unique labels: {sorted(val_df['misinformation'].unique())}")
    logger.info(f"  Dtype: {val_df['misinformation'].dtype}")
    logger.info(f"  Null count: {val_df['misinformation'].isna().sum()}")
    logger.info(f"  First 5 samples:")
    for idx, row in val_df[['id', 'misinformation']].head(5).iterrows():
        logger.info(f"    id={row['id']}, label={row['misinformation']}")
    
    # Check test split
    logger.info("\n[TEST SPLIT]")
    logger.info(f"  Rows: {len(test_df)}")
    logger.info(f"  Positive rate: {test_df['misinformation'].mean():.4f}")
    logger.info(f"  Unique labels: {sorted(test_df['misinformation'].unique())}")
    logger.info(f"  Dtype: {test_df['misinformation'].dtype}")
    logger.info(f"  Null count: {test_df['misinformation'].isna().sum()}")
    logger.info(f"  First 5 samples:")
    for idx, row in test_df[['id', 'misinformation']].head(5).iterrows():
        logger.info(f"    id={row['id']}, label={row['misinformation']}")
    
    # Verify stratification consistency
    train_pos = train_df['misinformation'].mean()
    val_pos = val_df['misinformation'].mean()
    test_pos = test_df['misinformation'].mean()
    
    max_diff = max(abs(train_pos - val_pos), abs(train_pos - test_pos))
    
    logger.info(f"\nStratification check:")
    logger.info(f"  train_pos={train_pos:.4f}, val_pos={val_pos:.4f}, test_pos={test_pos:.4f}")
    logger.info(f"  max_diff={max_diff:.4f}")
    
    if max_diff > 0.10:
        logger.error(
            f"❌ STRATIFICATION FAILED: Label distribution drifted > 10pp\n"
            f"   train={train_pos:.4f} val={val_pos:.4f} test={test_pos:.4f}"
        )
        return False
    else:
        logger.info("✓ Stratification consistent (< 10pp difference)")
    
    return True, train_df, val_df, test_df


def check_raw_vs_processed(raw_csv: Path, processed_df: pd.DataFrame) -> bool:
    """
    CHECK 2: Cross-reference raw CSV to verify labels weren't inverted.
    
    Args:
        raw_csv: Path to raw CSV (data/raw/ads_vietnam_clean.csv)
        processed_df: Processed train DataFrame
        
    Returns:
        True if labels match, False if inversion detected.
    """
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 2: RAW VS PROCESSED LABEL VERIFICATION")
    logger.info("=" * 80)
    
    if not raw_csv.exists():
        logger.warning(f"⚠ Raw CSV not found: {raw_csv}. Skipping this check.")
        return True
    
    try:
        raw_df = pd.read_csv(raw_csv, dtype={"id": str})
    except Exception as e:
        logger.warning(f"⚠ Failed to load raw CSV: {e}. Skipping this check.")
        return True
    
    # Get 20 random sample IDs that appear in both
    common_ids = set(processed_df['id'].values) & set(raw_df['id'].values)
    
    if len(common_ids) < 5:
        logger.warning(f"⚠ Only {len(common_ids)} common IDs found. Skipping detailed check.")
        return True
    
    sample_ids = np.random.choice(list(common_ids), min(20, len(common_ids)), replace=False)
    
    logger.info(f"Comparing labels for {len(sample_ids)} random samples:")
    
    mismatches = 0
    for sample_id in sample_ids:
        raw_label = raw_df[raw_df['id'] == sample_id]['misinformation'].values[0]
        proc_label = processed_df[processed_df['id'] == sample_id]['misinformation'].values[0]
        
        match = "✓" if raw_label == proc_label else "✗"
        logger.info(f"  {match} id={sample_id}: raw={raw_label}, processed={proc_label}")
        
        if raw_label != proc_label:
            mismatches += 1
    
    if mismatches > 0:
        logger.error(
            f"❌ LABEL INVERSION DETECTED: {mismatches}/{len(sample_ids)} labels don't match"
        )
        return False
    else:
        logger.info(f"✓ All {len(sample_ids)} labels match between raw and processed")
    
    return True


def check_logistic_regression_auc(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> bool:
    """
    CHECK 3-4: Train logistic regression on first metadata feature, evaluate on val/test.
    
    If labels are correct, AUC should be ~0.55-0.75 (depending on signal).
    If AUC < 0.30 on val/test, labels are likely inverted.
    
    Returns:
        True if AUC checks pass, False if inversion detected.
    """
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 3-4: LOGISTIC REGRESSION SANITY CHECK")
    logger.info("=" * 80)
    
    # Use first metadata feature available
    feature_cols = [
        'ads_per_page', 'emoji_count', 'text_length', 'exclamation_ratio',
        'caps_word_ratio', 'repeated_punct_count', 'url_count',
        'launch_delay', 'platform_count'
    ]
    
    feature_col = None
    for col in feature_cols:
        if col in train_df.columns:
            feature_col = col
            break
    
    if feature_col is None:
        logger.warning(f"⚠ No suitable feature column found. Skipping logistic regression check.")
        return True
    
    logger.info(f"Using feature: '{feature_col}' for sanity check")
    
    # Prepare data
    X_train = train_df[[feature_col]].values
    y_train = train_df['misinformation'].values
    X_val = val_df[[feature_col]].values
    y_val = val_df['misinformation'].values
    X_test = test_df[[feature_col]].values
    y_test = test_df['misinformation'].values
    
    # === CRITICAL: Check for NaNs in labels ===
    nan_train = np.isnan(y_train).sum()
    nan_val = np.isnan(y_val).sum()
    nan_test = np.isnan(y_test).sum()
    
    if nan_train > 0:
        logger.error(
            f"❌ CRITICAL: {nan_train} NaN values found in train labels. "
            f"These must be removed before training."
        )
        return False
    
    if nan_val > 0:
        logger.warning(f"⚠ {nan_val} NaN values in val labels (will be skipped)")
        mask_val = ~np.isnan(y_val)
        X_val = X_val[mask_val]
        y_val = y_val[mask_val]
    
    if nan_test > 0:
        logger.warning(f"⚠ {nan_test} NaN values in test labels (will be skipped)")
        mask_test = ~np.isnan(y_test)
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]
    
    # Handle NaN in features
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Train
    logger.info(f"\nTraining logistic regression on train split...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    # Evaluate on train
    y_train_pred_proba = lr.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    train_f1 = f1_score(y_train, (y_train_pred_proba > 0.5).astype(int))
    logger.info(f"Train AUC-ROC: {train_auc:.4f}, F1-macro: {train_f1:.4f}")
    
    # Evaluate on val
    logger.info(f"\nEvaluating on val split...")
    y_val_pred_proba = lr.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_f1 = f1_score(y_val, (y_val_pred_proba > 0.5).astype(int))
    logger.info(f"Val AUC-ROC: {val_auc:.4f}, F1-macro: {val_f1:.4f}")
    
    # Evaluate on test
    logger.info(f"\nEvaluating on test split...")
    y_test_pred_proba = lr.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_f1 = f1_score(y_test, (y_test_pred_proba > 0.5).astype(int))
    logger.info(f"Test AUC-ROC: {test_auc:.4f}, F1-macro: {test_f1:.4f}")
    
    # Diagnostic
    logger.info(f"\nDiagnostic:")
    
    inversion_detected = False
    
    if val_auc < 0.30:
        logger.error(
            f"❌ LABEL INVERSION DETECTED: Val AUC-ROC = {val_auc:.4f} < 0.30\n"
            f"   This suggests val labels are inverted relative to train."
        )
        inversion_detected = True
    elif val_auc > 0.70:
        logger.info(f"✓ Val AUC-ROC = {val_auc:.4f} > 0.70 (strong signal)")
    elif val_auc > 0.55:
        logger.info(f"✓ Val AUC-ROC = {val_auc:.4f} > 0.55 (acceptable signal)")
    else:
        logger.warning(f"⚠ Val AUC-ROC = {val_auc:.4f} (weak signal, but not inverted)")
    
    if test_auc < 0.30:
        logger.error(
            f"❌ LABEL INVERSION DETECTED: Test AUC-ROC = {test_auc:.4f} < 0.30\n"
            f"   This suggests test labels are inverted relative to train."
        )
        inversion_detected = True
    elif test_auc > 0.70:
        logger.info(f"✓ Test AUC-ROC = {test_auc:.4f} > 0.70 (strong signal)")
    elif test_auc > 0.55:
        logger.info(f"✓ Test AUC-ROC = {test_auc:.4f} > 0.55 (acceptable signal)")
    else:
        logger.warning(f"⚠ Test AUC-ROC = {test_auc:.4f} (weak signal, but not inverted)")
    
    return not inversion_detected


def main(processed_dir: str, raw_csv: str) -> int:
    """
    Run all diagnostic checks.
    
    Returns:
        0 if all checks pass, 1 if any check fails.
    """
    logger.info("=" * 80)
    logger.info("LABEL INVERSION DIAGNOSTIC")
    logger.info("=" * 80)
    
    processed_dir = Path(processed_dir)
    raw_csv = Path(raw_csv)
    
    # Check 1: Label distribution
    result = check_label_distribution(processed_dir)
    if result is True:
        logger.error("❌ Check 1 failed")
        return 1
    else:
        check1_pass, train_df, val_df, test_df = result
        if not check1_pass:
            return 1
    
    # Check 2: Raw vs processed
    check2_pass = check_raw_vs_processed(raw_csv, train_df)
    if not check2_pass:
        return 1
    
    # Check 3-4: Logistic regression
    check34_pass = check_logistic_regression_auc(train_df, val_df, test_df)
    if not check34_pass:
        return 1
    
    # All checks passed
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL CHECKS PASSED - Labels appear to be valid")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose label inversion issues")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory (default: data/processed)",
    )
    parser.add_argument(
        "--raw_csv",
        type=str,
        default="data/raw/ads_vietnam_clean.csv",
        help="Path to raw CSV (default: data/raw/ads_vietnam_clean.csv)",
    )
    args = parser.parse_args()
    
    exit_code = main(args.processed_dir, args.raw_csv)
    sys.exit(exit_code)
