#!/usr/bin/env python3
"""
Verification tests for robust metadata feature scaling (FIX 3).

Tests the MetadataScaler implementation to ensure:
1. Per-feature scaling with proper type handling
2. Binary/ratio features remain unchanged
3. Heavy-tailed features are log-transformed then scaled
4. No NaN or Inf in output
5. Scaler fit on train only, not on val/test
6. All features have comparable scale after transformation
"""

import sys
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing_fixed import MetadataScaler, METADATA_FEATURE_TYPES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def create_synthetic_metadata(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create synthetic metadata DataFrame for testing."""
    np.random.seed(seed)
    
    df = pd.DataFrame({
        # Binary features (0 or 1)
        "FB_only_flag": np.random.binomial(1, 0.3, n_samples),
        "all_targeted": np.random.binomial(1, 0.5, n_samples),
        
        # Ratio features (0 to 1)
        "repeated_text_ratio": np.random.uniform(0, 1, n_samples),
        "exclamation_ratio": np.random.uniform(0, 0.5, n_samples),
        "caps_word_ratio": np.random.uniform(0, 1, n_samples),
        
        # Heavy-tailed counts (log-normal)
        "ads_per_page": np.random.exponential(10, n_samples),
        "text_length": np.random.exponential(500, n_samples),
        "burstiness": np.random.exponential(5, n_samples),
        "launch_delay": np.random.exponential(50, n_samples),
        "ads_duration": np.random.exponential(100, n_samples),
        "avg_ad_duration": np.random.exponential(30, n_samples),
        
        # Discrete counts
        "num_countries": np.random.poisson(10, n_samples) + 1,
        "platform_count": np.random.randint(1, 6, n_samples),
        "emoji_count": np.random.poisson(2, n_samples),
        "repeated_punct_count": np.random.poisson(3, n_samples),
        "url_count": np.random.poisson(2, n_samples),
    })
    
    return df


def test_binary_features_unchanged():
    """Test 1: Binary features remain in {0, 1}."""
    logger.info("=" * 70)
    logger.info("TEST 1: Binary Features Unchanged")
    logger.info("=" * 70)
    
    train_df = create_synthetic_metadata(n_samples=500)
    test_df = create_synthetic_metadata(n_samples=100, seed=123)
    
    scaler = MetadataScaler(
        feature_columns=list(train_df.columns),
        feature_types=METADATA_FEATURE_TYPES,
    )
    scaler.fit(train_df)
    
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    binary_cols = ["FB_only_flag", "all_targeted"]
    
    for col in binary_cols:
        idx = list(train_df.columns).index(col)
        unique_vals = np.unique(train_scaled[:, idx])
        
        logger.info(f"  {col}: unique values = {sorted(unique_vals)}")
        
        assert set(unique_vals).issubset({0.0, 1.0}), \
            f"Binary feature {col} has non-binary values: {unique_vals}"
    
    logger.info("✓ PASSED: All binary features remain in {0, 1}\n")
    return True


def test_ratio_features_unchanged():
    """Test 2: Ratio features remain in [0, 1]."""
    logger.info("=" * 70)
    logger.info("TEST 2: Ratio Features Unchanged")
    logger.info("=" * 70)
    
    train_df = create_synthetic_metadata(n_samples=500)
    
    scaler = MetadataScaler(
        feature_columns=list(train_df.columns),
        feature_types=METADATA_FEATURE_TYPES,
    )
    scaler.fit(train_df)
    
    scaled = scaler.transform(train_df)
    
    ratio_cols = ["repeated_text_ratio", "exclamation_ratio", "caps_word_ratio"]
    
    for col in ratio_cols:
        idx = list(train_df.columns).index(col)
        min_val = scaled[:, idx].min()
        max_val = scaled[:, idx].max()
        
        logger.info(f"  {col}: min={min_val:.4f}, max={max_val:.4f}")
        
        assert 0.0 <= min_val, f"{col}: min {min_val} < 0.0"
        assert max_val <= 1.0, f"{col}: max {max_val} > 1.0"
    
    logger.info("✓ PASSED: All ratio features remain in [0, 1]\n")
    return True


def test_scaled_features_comparable_magnitude():
    """Test 3: Scaled features have comparable magnitude (std ~ 0.5-1.5)."""
    logger.info("=" * 70)
    logger.info("TEST 3: Scaled Features Have Comparable Magnitude")
    logger.info("=" * 70)
    
    train_df = create_synthetic_metadata(n_samples=500)
    
    scaler = MetadataScaler(
        feature_columns=list(train_df.columns),
        feature_types=METADATA_FEATURE_TYPES,
    )
    scaler.fit(train_df)
    
    scaled = scaler.transform(train_df)
    
    logger.info("  Scaled feature statistics:")
    for i, col in enumerate(train_df.columns):
        ftype = METADATA_FEATURE_TYPES[col]
        mean = scaled[:, i].mean()
        std = scaled[:, i].std()
        
        logger.info(f"    {col:30s} ({ftype:12s}): mean={mean:7.4f}, std={std:.4f}")
        
        # Check that std is reasonable
        if ftype in ("binary", "ratio"):
            # Binary/ratio features may have lower std
            assert std <= 0.6, f"{col}: std {std} too high for {ftype}"
        else:
            # Scaled features should have std in reasonable range
            assert 0.3 <= std <= 2.0, f"{col}: std {std} out of reasonable range [0.3, 2.0]"
    
    logger.info("✓ PASSED: All scaled features have comparable magnitude\n")
    return True


def test_no_nan_or_inf():
    """Test 4: No NaN or Inf in scaled output."""
    logger.info("=" * 70)
    logger.info("TEST 4: No NaN or Inf in Output")
    logger.info("=" * 70)
    
    train_df = create_synthetic_metadata(n_samples=500)
    test_df = create_synthetic_metadata(n_samples=100, seed=123)
    
    # Add some NaN values to test handling
    train_df.iloc[0, 0] = np.nan
    test_df.iloc[5, 2] = np.nan
    
    scaler = MetadataScaler(
        feature_columns=list(train_df.columns),
        feature_types=METADATA_FEATURE_TYPES,
    )
    scaler.fit(train_df)
    
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    logger.info(f"  Train scaled shape: {train_scaled.shape}")
    logger.info(f"  Test scaled shape: {test_scaled.shape}")
    
    assert not np.isnan(train_scaled).any(), "Train scaled contains NaN"
    assert not np.isinf(train_scaled).any(), "Train scaled contains Inf"
    assert not np.isnan(test_scaled).any(), "Test scaled contains NaN"
    assert not np.isinf(test_scaled).any(), "Test scaled contains Inf"
    
    logger.info("✓ PASSED: No NaN or Inf in any scaled data\n")
    return True


def test_deterministic_scaling():
    """Test 5: Scaler is deterministic (same input -> same output)."""
    logger.info("=" * 70)
    logger.info("TEST 5: Deterministic Scaling")
    logger.info("=" * 70)
    
    train_df = create_synthetic_metadata(n_samples=500)
    test_df = create_synthetic_metadata(n_samples=100, seed=123)
    
    scaler = MetadataScaler(
        feature_columns=list(train_df.columns),
        feature_types=METADATA_FEATURE_TYPES,
    )
    scaler.fit(train_df)
    
    out1 = scaler.transform(test_df.head(50))
    out2 = scaler.transform(test_df.head(50))
    
    logger.info(f"  Output 1 shape: {out1.shape}")
    logger.info(f"  Output 2 shape: {out2.shape}")
    
    try:
        np.testing.assert_array_equal(out1, out2)
        logger.info("✓ PASSED: Scaler is deterministic\n")
        return True
    except AssertionError:
        logger.error("✗ FAILED: Outputs differ")
        return False


def test_scaler_persistence():
    """Test 6: Scaler can be saved and loaded with identical behavior."""
    logger.info("=" * 70)
    logger.info("TEST 6: Scaler Persistence (Save/Load)")
    logger.info("=" * 70)
    
    train_df = create_synthetic_metadata(n_samples=500)
    test_df = create_synthetic_metadata(n_samples=100, seed=123)
    
    scaler = MetadataScaler(
        feature_columns=list(train_df.columns),
        feature_types=METADATA_FEATURE_TYPES,
    )
    scaler.fit(train_df)
    
    # Transform with original scaler
    out_original = scaler.transform(test_df.head(50))
    
    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        scaler_path = Path(tmpdir) / "scaler_test.joblib"
        scaler.save(scaler_path)
        logger.info(f"  Saved scaler to {scaler_path}")
        
        scaler_reloaded = MetadataScaler.load(scaler_path)
        logger.info(f"  Reloaded scaler")
        
        # Transform with reloaded scaler
        out_reloaded = scaler_reloaded.transform(test_df.head(50))
    
    logger.info(f"  Original output shape: {out_original.shape}")
    logger.info(f"  Reloaded output shape: {out_reloaded.shape}")
    
    try:
        np.testing.assert_array_almost_equal(out_original, out_reloaded, decimal=5)
        logger.info("✓ PASSED: Save/Load produces identical results\n")
        return True
    except AssertionError:
        logger.error("✗ FAILED: Outputs differ after reload")
        return False


def test_train_only_fitting():
    """Test 7: Scaler fit on train only, not modified when applied to val/test."""
    logger.info("=" * 70)
    logger.info("TEST 7: Train-Only Fitting (No Data Leakage)")
    logger.info("=" * 70)
    
    train_df = create_synthetic_metadata(n_samples=500, seed=42)
    val_df = create_synthetic_metadata(n_samples=100, seed=123)
    test_df = create_synthetic_metadata(n_samples=100, seed=456)
    
    scaler = MetadataScaler(
        feature_columns=list(train_df.columns),
        feature_types=METADATA_FEATURE_TYPES,
    )
    
    logger.info(f"  Fitting scaler on train_df ({len(train_df)} samples)...")
    scaler.fit(train_df)
    
    # Capture scaler state
    scaler_state_1 = {col: (s.center_.copy(), s.scale_.copy()) 
                      for col, s in scaler.scalers.items()}
    
    logger.info(f"  Transforming val_df ({len(val_df)} samples)...")
    val_scaled = scaler.transform(val_df)
    
    # Check scaler state hasn't changed
    scaler_state_2 = {col: (s.center_.copy(), s.scale_.copy()) 
                      for col, s in scaler.scalers.items()}
    
    for col in scaler_state_1:
        assert np.allclose(scaler_state_1[col][0], scaler_state_2[col][0]), \
            f"Scaler state changed for {col} after transform"
    
    logger.info(f"  Transforming test_df ({len(test_df)} samples)...")
    test_scaled = scaler.transform(test_df)
    
    logger.info("✓ PASSED: Scaler not modified by transform calls\n")
    return True


def test_heavy_tailed_compression():
    """Test 8: Heavy-tailed features are compressed (log-transform + scaling)."""
    logger.info("=" * 70)
    logger.info("TEST 8: Heavy-Tailed Feature Compression")
    logger.info("=" * 70)
    
    train_df = create_synthetic_metadata(n_samples=500)
    
    scaler = MetadataScaler(
        feature_columns=list(train_df.columns),
        feature_types=METADATA_FEATURE_TYPES,
    )
    scaler.fit(train_df)
    
    scaled = scaler.transform(train_df)
    
    heavy_tailed_cols = [
        "ads_per_page", "text_length", "burstiness", 
        "launch_delay", "ads_duration", "avg_ad_duration"
    ]
    
    logger.info("  Heavy-tailed feature compression:")
    for col in heavy_tailed_cols:
        idx = list(train_df.columns).index(col)
        raw_values = train_df[col].values
        scaled_values = scaled[:, idx]
        
        raw_max = raw_values.max()
        scaled_max = np.abs(scaled_values).max()
        
        # Compression ratio: large values compressed
        logger.info(f"    {col:20s}: raw_max={raw_max:10.2f}, scaled_max={scaled_max:6.2f}")
        
        # After log1p + scaling, max should be much smaller
        assert scaled_max <= 5.0, f"{col}: max after scaling {scaled_max} > 5.0"
    
    logger.info("✓ PASSED: Heavy-tailed features compressed effectively\n")
    return True


def test_feature_stats_diagnostic():
    """Test 9: get_feature_stats returns comprehensive diagnostics."""
    logger.info("=" * 70)
    logger.info("TEST 9: Feature Stats Diagnostic")
    logger.info("=" * 70)
    
    train_df = create_synthetic_metadata(n_samples=500)
    
    scaler = MetadataScaler(
        feature_columns=list(train_df.columns),
        feature_types=METADATA_FEATURE_TYPES,
    )
    scaler.fit(train_df)
    
    stats = scaler.get_feature_stats(train_df)
    
    logger.info(f"  Stats shape: {stats.shape}")
    logger.info(f"  Stats columns: {list(stats.columns)}")
    logger.info(f"\n  Stats table (first 5 rows):\n{stats.head().to_string(index=False)}")
    
    assert len(stats) == len(train_df.columns), "Stats should have one row per feature"
    assert "feature" in stats.columns, "Stats should have 'feature' column"
    assert "type" in stats.columns, "Stats should have 'type' column"
    assert "raw_mean" in stats.columns, "Stats should have 'raw_mean' column"
    assert "scaled_mean" in stats.columns, "Stats should have 'scaled_mean' column"
    
    logger.info("✓ PASSED: Feature stats diagnostic works correctly\n")
    return True


def test_output_dtype_and_shape():
    """Test 10: Scaler output is correct dtype and shape."""
    logger.info("=" * 70)
    logger.info("TEST 10: Output Dtype and Shape")
    logger.info("=" * 70)
    
    train_df = create_synthetic_metadata(n_samples=500)
    test_df = create_synthetic_metadata(n_samples=100, seed=123)
    
    scaler = MetadataScaler(
        feature_columns=list(train_df.columns),
        feature_types=METADATA_FEATURE_TYPES,
    )
    scaler.fit(train_df)
    
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    logger.info(f"  Train output: dtype={train_scaled.dtype}, shape={train_scaled.shape}")
    logger.info(f"  Test output: dtype={test_scaled.dtype}, shape={test_scaled.shape}")
    
    assert train_scaled.dtype == np.float32, f"Expected float32, got {train_scaled.dtype}"
    assert train_scaled.shape == (len(train_df), len(train_df.columns)), \
        f"Expected shape ({len(train_df)}, {len(train_df.columns)}), got {train_scaled.shape}"
    
    assert test_scaled.dtype == np.float32, f"Expected float32, got {test_scaled.dtype}"
    assert test_scaled.shape == (len(test_df), len(test_df.columns)), \
        f"Expected shape ({len(test_df)}, {len(test_df.columns)}), got {test_scaled.shape}"
    
    logger.info("✓ PASSED: Output dtype and shape are correct\n")
    return True


def main():
    """Run all verification tests."""
    logger.info("\n" + "=" * 70)
    logger.info("ROBUST METADATA FEATURE SCALING - VERIFICATION TESTS")
    logger.info("=" * 70 + "\n")
    
    tests = [
        ("Binary Features Unchanged", test_binary_features_unchanged),
        ("Ratio Features Unchanged", test_ratio_features_unchanged),
        ("Scaled Features Comparable Magnitude", test_scaled_features_comparable_magnitude),
        ("No NaN or Inf in Output", test_no_nan_or_inf),
        ("Deterministic Scaling", test_deterministic_scaling),
        ("Scaler Persistence (Save/Load)", test_scaler_persistence),
        ("Train-Only Fitting", test_train_only_fitting),
        ("Heavy-Tailed Feature Compression", test_heavy_tailed_compression),
        ("Feature Stats Diagnostic", test_feature_stats_diagnostic),
        ("Output Dtype and Shape", test_output_dtype_and_shape),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ FAILED: {test_name}")
            logger.error(f"  Error: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Metadata scaling is working correctly.\n")
        return 0
    else:
        logger.error(f"\n⚠ {total - passed} test(s) failed. Please review the errors above.\n")
        return 1


if __name__ == "__main__":
    exit(main())
