#!/usr/bin/env python3
"""
Identify metadata features that leak label information.

A feature leaks if it achieves high validation AUC in a single-feature
logistic regression model, indicating it encodes the label directly or
through page identity (since 95.7% of pages have pure labels).
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=UserWarning)


def load_config(config_path: str) -> dict:
    """Load base configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(processed_dir: str):
    """Load train and val CSVs with scaled features."""
    # Try to load from the main processed directory (has scaled features)
    # Fall back to splits subdirectory if not found
    main_dir = Path(processed_dir)
    splits_dir = main_dir / "splits"
    
    if (main_dir / "train.csv").exists():
        train_df = pd.read_csv(main_dir / "train.csv")
        val_df = pd.read_csv(main_dir / "val.csv")
        # Remove rows with NaN labels
        train_df = train_df.dropna(subset=["misinformation"]).reset_index(drop=True)
        val_df = val_df.dropna(subset=["misinformation"]).reset_index(drop=True)
        print("  (loaded from data/processed/ with scaled features)")
    elif (splits_dir / "train.csv").exists():
        train_df = pd.read_csv(splits_dir / "train.csv", dtype={"id": str})
        val_df = pd.read_csv(splits_dir / "val.csv", dtype={"id": str})
        train_df = train_df.dropna(subset=["misinformation"]).reset_index(drop=True)
        val_df = val_df.dropna(subset=["misinformation"]).reset_index(drop=True)
        print("  (loaded from data/processed/splits/ without scaling)")
    else:
        raise FileNotFoundError(f"No CSVs found in {processed_dir}")
    
    return train_df, val_df


def evaluate_single_feature(train_df: pd.DataFrame, val_df: pd.DataFrame, feature: str):
    """
    Train logistic regression on a single feature and return validation AUC.
    
    Returns:
        (auc, train_auc, coef)
    """
    # Handle missing values in features
    Xt = train_df[[feature]].fillna(0).values.astype(float)
    Xv = val_df[[feature]].fillna(0).values.astype(float)
    
    # Ensure labels are int and have no NaN
    yt = train_df["misinformation"].fillna(0).values.astype(int)
    yv = val_df["misinformation"].fillna(0).values.astype(int)
    
    # Skip if all values are NaN
    if np.isnan(Xt).all() or np.isnan(Xv).all():
        return None, None, None
    
    try:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(Xt, yt)
        
        train_proba = model.predict_proba(Xt)[:, 1]
        val_proba = model.predict_proba(Xv)[:, 1]
        
        train_auc = roc_auc_score(yt, train_proba)
        val_auc = roc_auc_score(yv, val_proba)
        coef = model.coef_[0, 0]
        
        return val_auc, train_auc, coef
    except Exception as e:
        print(f"Warning: Failed to train on feature '{feature}': {e}", file=sys.stderr)
        return None, None, None


def evaluate_feature_pair(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feat1: str,
    feat2: str,
):
    """
    Train logistic regression on two features and return validation AUC.
    
    Returns:
        (auc, train_auc)
    """
    Xt = train_df[[feat1, feat2]].fillna(0).values.astype(float)
    Xv = val_df[[feat1, feat2]].fillna(0).values.astype(float)
    yt = train_df["misinformation"].fillna(0).values.astype(int)
    yv = val_df["misinformation"].fillna(0).values.astype(int)
    
    try:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(Xt, yt)
        
        train_proba = model.predict_proba(Xt)[:, 1]
        val_proba = model.predict_proba(Xv)[:, 1]
        
        train_auc = roc_auc_score(yt, train_proba)
        val_auc = roc_auc_score(yv, val_proba)
        
        return val_auc, train_auc
    except Exception as e:
        return None, None


def compute_feature_stats(train_df: pd.DataFrame, feature: str):
    """
    Compute feature statistics for leak analysis.
    
    Returns:
        (correlation, mean_pos, mean_neg, feature_type)
    """
    # Handle missing values and ensure correct types
    X = train_df[feature].fillna(0).values.astype(float)
    y = train_df["misinformation"].fillna(0).values.astype(int)
    
    # Correlation
    try:
        corr, _ = pearsonr(X, y)
    except Exception:
        corr = 0.0
    
    # Mean by label
    mean_pos = train_df[train_df["misinformation"] == 1][feature].mean()
    mean_neg = train_df[train_df["misinformation"] == 0][feature].mean()
    
    # Feature type (page-aggregated vs row-level)
    page_aggregated_features = [
        "ads_per_page",
        "burstiness",
        "avg_ad_duration",
        "launch_delay",
    ]
    feature_type = "page-aggregated" if feature in page_aggregated_features else "row-level"
    
    return corr, mean_pos, mean_neg, feature_type


def get_verdict(auc: float):
    """Get verdict based on AUC threshold."""
    if auc > 0.95:
        return "LEAK — REMOVE"
    elif auc > 0.85:
        return "SUSPICIOUS"
    else:
        return "OK"


def evaluate_all_features(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list):
    """
    Train logistic regression on ALL features and return validation metrics.
    
    Returns:
        (accuracy, auc, f1, train_auc)
    """
    Xt = train_df[features].fillna(0).values.astype(float)
    Xv = val_df[features].fillna(0).values.astype(float)
    
    # Ensure labels are int and have no NaN
    yt = train_df["misinformation"].fillna(0).values.astype(int)
    yv = val_df["misinformation"].fillna(0).values.astype(int)
    
    try:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(Xt, yt)
        
        train_proba = model.predict_proba(Xt)[:, 1]
        val_proba = model.predict_proba(Xv)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)
        
        train_auc = roc_auc_score(yt, train_proba)
        val_auc = roc_auc_score(yv, val_proba)
        val_accuracy = accuracy_score(yv, val_pred)
        val_f1 = f1_score(yv, val_pred, average="macro")
        
        return val_accuracy, val_auc, val_f1, train_auc
    except Exception as e:
        print(f"Error training on all features: {e}", file=sys.stderr)
        return None, None, None, None


def main():
    """Main diagnostic pipeline."""
    print("=" * 80)
    print("METADATA FEATURE LEAKAGE DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    config = load_config(str(config_path))
    metadata_features = config.get("metadata_features", [])
    
    # Load data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    train_df, val_df = load_data(str(processed_dir))
    
    print(f"Loaded train: {len(train_df)} rows, val: {len(val_df)} rows")
    print(f"Metadata features: {len(metadata_features)}")
    print()
    
    # ========== TEST ALL FEATURES COMBINED ==========
    print("=" * 80)
    print("ALL-FEATURES COMBINED MODEL (CHECK 5 BASELINE)")
    print("=" * 80)
    print()
    
    available_features = [f for f in metadata_features if f in train_df.columns]
    all_acc, all_auc, all_f1, all_train_auc = evaluate_all_features(train_df, val_df, available_features)
    
    if all_acc is not None:
        print(f"Using all {len(available_features)} metadata features:")
        print(f"  Train AUC:       {all_train_auc:.4f}")
        print(f"  Val AUC:         {all_auc:.4f}")
        print(f"  Val Accuracy:    {all_acc:.4f}")
        print(f"  Val F1 (macro):  {all_f1:.4f}")
        print()
        if all_acc > 0.98:
            print("  ⚠ ALERT: Combined metadata achieves >98% accuracy (LEAKAGE DETECTED)")
        elif all_acc > 0.90:
            print("  ⚠ WARNING: Combined metadata achieves >90% accuracy (high overfit)")
        print()
    print()
    
    # ========== SINGLE FEATURE ANALYSIS ==========
    print("=" * 80)
    print("SINGLE-FEATURE LEAKAGE ANALYSIS")
    print("=" * 80)
    print()
    
    results = []
    for feature in metadata_features:
        val_auc, train_auc, coef = evaluate_single_feature(train_df, val_df, feature)
        if val_auc is not None:
            results.append({
                "feature": feature,
                "val_auc": val_auc,
                "train_auc": train_auc,
                "coef": coef,
                "distance_from_0_5": abs(val_auc - 0.5),
            })
    
    # Sort by distance from 0.5 (most predictive first)
    results.sort(key=lambda x: x["distance_from_0_5"], reverse=True)
    
    # Print table
    print(f"{'Feature':<30} {'Val AUC':>10} {'Verdict':<25}")
    print("-" * 80)
    
    leaking_features = []
    suspicious_features = []
    
    for res in results:
        feature = res["feature"]
        auc = res["val_auc"]
        verdict = get_verdict(auc)
        
        print(f"{feature:<30} {auc:>10.4f}  {verdict:<25}")
        
        if verdict == "LEAK — REMOVE":
            leaking_features.append(feature)
        elif verdict == "SUSPICIOUS":
            suspicious_features.append(feature)
    
    print()
    
    # ========== DETAILED LEAK ANALYSIS ==========
    if leaking_features:
        print("=" * 80)
        print("DETAILED ANALYSIS OF LEAKING FEATURES")
        print("=" * 80)
        print()
        
        for feature in leaking_features:
            res = next(r for r in results if r["feature"] == feature)
            corr, mean_pos, mean_neg, feat_type = compute_feature_stats(train_df, feature)
            
            print(f"FEATURE: {feature}")
            print(f"  Type: {feat_type}")
            print(f"  Validation AUC: {res['val_auc']:.4f}")
            print(f"  Training AUC: {res['train_auc']:.4f}")
            print(f"  Pearson correlation with label: {corr:.4f}")
            print(f"  Mean when label=0: {mean_neg:.4f}")
            print(f"  Mean when label=1: {mean_pos:.4f}")
            print(f"  Difference: {abs(mean_pos - mean_neg):.4f}")
            print()
    
    # ========== PAIRWISE INTERACTION ANALYSIS ==========
    print("=" * 80)
    print("PAIRWISE FEATURE INTERACTIONS (Top-3 Most Predictive)")
    print("=" * 80)
    print()
    
    top_3_features = [r["feature"] for r in results[:3]]
    
    for i, feat1 in enumerate(top_3_features):
        for feat2 in top_3_features[i+1:]:
            auc, train_auc = evaluate_feature_pair(train_df, val_df, feat1, feat2)
            if auc is not None:
                if auc > 0.99:
                    print(f"{feat1} + {feat2}: Val AUC = {auc:.4f}  [INTERACTION LEAK]")
                elif auc > 0.95:
                    print(f"{feat1} + {feat2}: Val AUC = {auc:.4f}  [INTERACTION SUSPICIOUS]")
                else:
                    print(f"{feat1} + {feat2}: Val AUC = {auc:.4f}")
    
    print()
    
    # ========== RECOMMENDATIONS ==========
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    if leaking_features:
        # Classify leaking features
        page_agg_leaks = []
        row_level_leaks = []
        
        for feature in leaking_features:
            _, _, _, feat_type = compute_feature_stats(train_df, feature)
            if feat_type == "page-aggregated":
                page_agg_leaks.append(feature)
            else:
                row_level_leaks.append(feature)
        
        if page_agg_leaks:
            print("ACTION 1: REMOVE PAGE-AGGREGATED LEAKING FEATURES")
            print(f"  These features encode page identity, which is essentially the label")
            print(f"  since 95.7% of pages have pure labels (all ads misinformation or all OK).")
            print()
            print(f"  REMOVE from metadata_features in configs/base.yaml:")
            print(f"    {page_agg_leaks}")
            print()
        
        if row_level_leaks:
            print("ACTION 2: INVESTIGATE ROW-LEVEL LEAKING FEATURES")
            print(f"  These features may use information derived from the label or be")
            print(f"  computed across the full dataset including val/test rows.")
            print()
            print(f"  INSPECT feature engineering for:")
            for feature in row_level_leaks:
                print(f"    - {feature}")
            print()
            print(f"  Check src/data/feature_engineering.py for data leakage")
            print()
        
        print("NEXT STEPS:")
        print("  1. Update metadata_features in configs/base.yaml")
        print("  2. Update model.metadata_input_dim (currently:", len(metadata_features), ")")
        print("  3. Run: python scripts/prepare_data.py")
        print("  4. Run: python scripts/diagnose_leakage.py")
        print()
    
    else:
        print("✓ NO LEAKING FEATURES DETECTED")
        print()
        if suspicious_features:
            print(f"Note: {len(suspicious_features)} features are SUSPICIOUS (AUC > 0.85)")
            print(f"Consider monitoring or investigating further:")
            for feature in suspicious_features:
                res = next(r for r in results if r["feature"] == feature)
                print(f"  - {feature}: {res['val_auc']:.4f}")
            print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()
