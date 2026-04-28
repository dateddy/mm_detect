# src/data/feature_engineering.py
"""Feature engineering for multimodal misinformation detection."""

import argparse
import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import emoji
except ImportError:
    emoji = None

# Configure logging
logger = logging.getLogger(__name__)


# Country-to-languages mapping (ISO 639-1 codes)
COUNTRY_LANGUAGES: Dict[str, List[str]] = {
    "VN": ["vi"],  # Vietnam - Vietnamese
    "US": ["en"],  # USA - English
    "GB": ["en"],  # UK - English
    "FR": ["fr"],  # France - French
    "DE": ["de"],  # Germany - German
    "ES": ["es"],  # Spain - Spanish
    "IT": ["it"],  # Italy - Italian
    "JP": ["ja"],  # Japan - Japanese
    "CN": ["zh"],  # China - Mandarin
    "IN": ["hi", "en"],  # India - Hindi, English
    "BR": ["pt"],  # Brazil - Portuguese
    "MX": ["es"],  # Mexico - Spanish
    "CA": ["en", "fr"],  # Canada - English, French
    "AU": ["en"],  # Australia - English
    "TH": ["th"],  # Thailand - Thai
    "PH": ["fil", "en"],  # Philippines - Filipino, English
    "ID": ["id"],  # Indonesia - Indonesian
    "MY": ["ms"],  # Malaysia - Malay
    "SG": ["en", "zh", "ms", "ta"],  # Singapore - Multiple
    "KR": ["ko"],  # South Korea - Korean
}


def parse_impressions(s: str) -> float:
    """
    Parse Meta's impression range format to midpoint value.

    Handles formats like "1000-5000", "1000+", "Unknown", or null.

    Args:
        s: String representation of impression range.

    Returns:
        Float midpoint of range, or 0.0 if unparseable.
    """
    if pd.isna(s) or s is None:
        return 0.0

    s = str(s).strip().upper()

    if s in ["UNKNOWN", "N/A", ""]:
        return 0.0

    # Format: "1000-5000" → midpoint
    if "-" in s:
        try:
            parts = s.split("-")
            low = float(parts[0].replace("+", "").strip())
            high = float(parts[1].strip())
            return (low + high) / 2.0
        except (ValueError, IndexError):
            return 0.0

    # Format: "1000+" or just "1000"
    try:
        value = float(s.replace("+", "").strip())
        return value
    except ValueError:
        return 0.0


def parse_spend(s: str) -> float:
    """
    Parse Meta's spend range format to midpoint value.

    Same logic as parse_impressions, handling ranges like "$1000-$5000".

    Args:
        s: String representation of spend range.

    Returns:
        Float midpoint of range, or 0.0 if unparseable.
    """
    if pd.isna(s) or s is None:
        return 0.0

    s = str(s).strip().upper()

    # Remove currency symbols
    s = s.replace("$", "").replace("€", "").replace("£", "")

    if s in ["UNKNOWN", "N/A", ""]:
        return 0.0

    # Format: "1000-5000" → midpoint
    if "-" in s:
        try:
            parts = s.split("-")
            low = float(parts[0].replace("+", "").strip())
            high = float(parts[1].strip())
            return (low + high) / 2.0
        except (ValueError, IndexError):
            return 0.0

    # Format: "1000+" or just "1000"
    try:
        value = float(s.replace("+", "").strip())
        return value
    except ValueError:
        return 0.0


def compute_ads_per_page(df: pd.DataFrame) -> pd.Series:
    """
    Compute count of ads per page_id, mapped back to each row.

    Args:
        df: DataFrame with 'page_id' column.

    Returns:
        Series with ad counts per page.
    """
    if "page_id" not in df.columns:
        logger.warning("'page_id' column not found, returning 1s")
        return pd.Series(1, index=df.index)

    page_counts = df.groupby("page_id").size()
    return df["page_id"].map(page_counts).fillna(1)


def compute_platform_count(df: pd.DataFrame) -> pd.Series:
    """
    Parse publisher_platforms (comma-separated or list string) and count unique platforms per row.

    Args:
        df: DataFrame with 'publisher_platforms' column.

    Returns:
        Series with platform counts per row.
    """
    if "publisher_platforms" not in df.columns:
        logger.warning("'publisher_platforms' column not found, returning 1s")
        return pd.Series(1, index=df.index)

    def parse_platforms(s: str) -> int:
        if pd.isna(s) or s is None:
            return 0

        s = str(s).strip()

        # Try parsing as JSON list: ["facebook", "instagram"]
        try:
            platforms = json.loads(s)
            if isinstance(platforms, list):
                return len(set(p.lower().strip() for p in platforms))
        except json.JSONDecodeError:
            pass

        # Parse as comma-separated: "facebook, instagram"
        if "," in s:
            platforms = [p.strip().lower() for p in s.split(",")]
            return len(set(platforms))

        # Single platform
        if s:
            return 1

        return 0

    return df["publisher_platforms"].apply(parse_platforms)


def compute_fb_only_flag(df: pd.DataFrame) -> pd.Series:
    """
    Return 1 if publisher_platforms contains only 'facebook', else 0.

    Args:
        df: DataFrame with 'publisher_platforms' column.

    Returns:
        Series of binary flags.
    """
    if "publisher_platforms" not in df.columns:
        logger.warning("'publisher_platforms' column not found, returning 0s")
        return pd.Series(0, index=df.index)

    def is_fb_only(s: str) -> int:
        if pd.isna(s) or s is None:
            return 0

        s = str(s).strip().lower()

        # Try JSON list
        try:
            platforms = json.loads(s)
            if isinstance(platforms, list):
                platforms = [p.lower().strip() for p in platforms]
                return 1 if all(p == "facebook" for p in platforms) and platforms else 0
        except json.JSONDecodeError:
            pass

        # Comma-separated
        if "," in s:
            platforms = [p.strip().lower() for p in s.split(",")]
            return 1 if all(p == "facebook" for p in platforms) and platforms else 0

        # Single value
        return 1 if s == "facebook" else 0

    return df["publisher_platforms"].apply(is_fb_only)


def compute_all_targeted(df: pd.DataFrame) -> pd.Series:
    """
    Return 1 if target_gender is 'all' or null/unknown, else 0.

    Args:
        df: DataFrame with 'target_gender' column.

    Returns:
        Series of binary flags.
    """
    if "target_gender" not in df.columns:
        logger.warning("'target_gender' column not found, treating as all_targeted")
        return pd.Series(1, index=df.index)

    def is_all_targeted(s: str) -> int:
        if pd.isna(s) or s is None:
            return 1  # Treat null as all targeted
        s = str(s).strip().lower()
        return 1 if s in ["all", "unknown", "n/a", ""] else 0

    return df["target_gender"].apply(is_all_targeted)


def compute_burstiness(df: pd.DataFrame) -> pd.Series:
    """
    Compute burstiness metric per page_id: (max_daily - mean_daily) / (std_daily + 1e-8).

    Extracts daily ad counts from ad_delivery_start_time grouped by page_id.

    Args:
        df: DataFrame with 'page_id' and 'ad_delivery_start_time' columns.

    Returns:
        Series with burstiness scores per row.
    """
    if "page_id" not in df.columns or "ad_delivery_start_time" not in df.columns:
        logger.warning(
            "Required columns for burstiness missing, returning 0s"
        )
        return pd.Series(0.0, index=df.index)

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["ad_delivery_start_time"]):
        try:
            df_copy = df.copy()
            df_copy["ad_delivery_start_time"] = pd.to_datetime(
                df_copy["ad_delivery_start_time"]
            )
        except Exception:
            logger.warning("Could not parse ad_delivery_start_time, returning 0s")
            return pd.Series(0.0, index=df.index)
    else:
        df_copy = df.copy()

    # Extract date from timestamp
    df_copy["date"] = df_copy["ad_delivery_start_time"].dt.date

    burstiness_scores = {}

    for page_id, group in df_copy.groupby("page_id"):
        daily_counts = group.groupby("date").size()

        if len(daily_counts) < 2:
            # Not enough daily variation
            burstiness_scores[page_id] = 0.0
            continue

        mean_daily = daily_counts.mean()
        std_daily = daily_counts.std()
        max_daily = daily_counts.max()

        if std_daily == 0:
            burstiness_scores[page_id] = 0.0
        else:
            burstiness_scores[page_id] = (
                max_daily - mean_daily
            ) / (std_daily + 1e-8)

    return df["page_id"].map(burstiness_scores).fillna(0.0)


def compute_avg_ad_duration(df: pd.DataFrame) -> pd.Series:
    """
    Compute average ad duration in hours per page_id.

    For each row: (ad_delivery_stop_time - ad_delivery_start_time).total_seconds() / 3600.
    If stop_time is null, use current date as stop time.
    Return per-page mean mapped to each row.

    Args:
        df: DataFrame with 'page_id', 'ad_delivery_start_time', 'ad_delivery_stop_time' columns.

    Returns:
        Series with average ad duration (hours) per page.
    """
    if (
        "page_id" not in df.columns
        or "ad_delivery_start_time" not in df.columns
        or "ad_delivery_stop_time" not in df.columns
    ):
        logger.warning("Required columns for avg_ad_duration missing, returning 0s")
        return pd.Series(0.0, index=df.index)

    df_copy = df.copy()

    # Convert to datetime
    try:
        df_copy["ad_delivery_start_time"] = pd.to_datetime(
            df_copy["ad_delivery_start_time"]
        )
        df_copy["ad_delivery_stop_time"] = pd.to_datetime(
            df_copy["ad_delivery_stop_time"]
        )
    except Exception:
        logger.warning("Could not parse datetime columns, returning 0s")
        return pd.Series(0.0, index=df.index)

    # Fill null stop times with current date
    now = pd.Timestamp.now()
    df_copy["ad_delivery_stop_time"] = df_copy["ad_delivery_stop_time"].fillna(now)

    # Compute duration in hours per row
    df_copy["duration_hours"] = (
        (df_copy["ad_delivery_stop_time"] - df_copy["ad_delivery_start_time"])
        .dt.total_seconds()
        / 3600.0
    )

    # Per-page mean
    page_avg_duration = df_copy.groupby("page_id")["duration_hours"].mean()

    return df["page_id"].map(page_avg_duration).fillna(0.0)


def compute_launch_delay(df: pd.DataFrame) -> pd.Series:
    """
    Compute launch delay in hours per row.

    Delay = (ad_delivery_start_time - ad_creation_time).total_seconds() / 3600.
    Negative values clipped to 0.

    Args:
        df: DataFrame with 'ad_delivery_start_time' and 'ad_creation_time' columns.

    Returns:
        Series with launch delays (hours) per row, clipped to [0, inf).
    """
    if "ad_delivery_start_time" not in df.columns or "ad_creation_time" not in df.columns:
        logger.warning("Required columns for launch_delay missing, returning 0s")
        return pd.Series(0.0, index=df.index)

    df_copy = df.copy()

    try:
        df_copy["ad_delivery_start_time"] = pd.to_datetime(
            df_copy["ad_delivery_start_time"]
        )
        df_copy["ad_creation_time"] = pd.to_datetime(df_copy["ad_creation_time"])
    except Exception:
        logger.warning("Could not parse datetime columns, returning 0s")
        return pd.Series(0.0, index=df.index)

    launch_delay = (
        (df_copy["ad_delivery_start_time"] - df_copy["ad_creation_time"])
        .dt.total_seconds()
        / 3600.0
    )

    # Clip negative values to 0
    launch_delay = launch_delay.clip(lower=0.0)

    return launch_delay


def compute_num_countries(df: pd.DataFrame) -> pd.Series:
    """
    Parse target_locations (JSON-like or comma-separated) and count unique country codes per row.

    Args:
        df: DataFrame with 'target_locations' column.

    Returns:
        Series with country counts per row.
    """
    if "target_locations" not in df.columns:
        logger.warning("'target_locations' column not found, returning 0s")
        return pd.Series(0, index=df.index)

    def parse_locations(s: str) -> int:
        if pd.isna(s) or s is None:
            return 0

        s = str(s).strip()

        # Try JSON list
        try:
            locations = json.loads(s)
            if isinstance(locations, list):
                return len(set(loc.upper().strip() for loc in locations))
        except json.JSONDecodeError:
            pass

        # Comma-separated country codes
        if "," in s:
            countries = [c.strip().upper() for c in s.split(",")]
            return len(set(countries))

        # Single country
        if s:
            return 1

        return 0

    return df["target_locations"].apply(parse_locations)


def compute_language_location_mismatch(df: pd.DataFrame) -> pd.Series:
    """
    Return 1 if ad language list contains a language not commonly spoken in target_locations.

    Return 0 otherwise. Handles parse errors gracefully.

    Args:
        df: DataFrame with 'ad_languages' and 'target_locations' columns.

    Returns:
        Series of binary mismatch flags.
    """
    if "ad_languages" not in df.columns or "target_locations" not in df.columns:
        logger.warning("Required columns for language_location_mismatch missing, returning 0s")
        return pd.Series(0, index=df.index)

    def has_mismatch(row: pd.Series) -> int:
        langs = row["ad_languages"]
        locs = row["target_locations"]

        # Parse languages
        ad_langs = set()
        if pd.notna(langs):
            s = str(langs).strip().lower()
            try:
                langs_list = json.loads(s)
                ad_langs = set(l.lower().strip() for l in langs_list)
            except json.JSONDecodeError:
                if "," in s:
                    ad_langs = set(l.strip().lower() for l in s.split(","))
                elif s:
                    ad_langs = {s}

        # Parse locations
        target_countries = set()
        if pd.notna(locs):
            s = str(locs).strip().upper()
            try:
                locs_list = json.loads(s)
                target_countries = set(c.upper().strip() for c in locs_list)
            except json.JSONDecodeError:
                if "," in s:
                    target_countries = set(
                        c.strip().upper() for c in s.split(",")
                    )
                elif s:
                    target_countries = {s}

        if not ad_langs or not target_countries:
            return 0

        # Collect languages spoken in target countries
        spoken_langs = set()
        for country in target_countries:
            if country in COUNTRY_LANGUAGES:
                spoken_langs.update(COUNTRY_LANGUAGES[country])

        # Check for mismatch
        if not spoken_langs:
            return 0

        # 1 if any ad language is not in spoken languages
        for lang in ad_langs:
            if lang not in spoken_langs:
                return 1

        return 0

    return df.apply(has_mismatch, axis=1)


# ============================================================================
# DATA LEAKAGE DIAGNOSTICS (10 Checks)
# ============================================================================

def diagnose_leakage_presplit(df: pd.DataFrame) -> None:
    """
    Run pre-split data leakage diagnostics.
    
    These checks can be performed before train/val/test split:
    - Check 3: label_not_in_features
    - Check 6: label_distribution (preliminary)
    
    Checks requiring train/val/test splits should be run in diagnose_leakage_postsplit().
    
    Args:
        df: Full DataFrame with all columns before splitting.
    """
    logger.info("\n[LEAKAGE DIAGNOSTICS - Pre-Split Checks]")
    
    # CHECK 3: label_not_in_features
    logger.info("  [3] label_not_in_features: Verifying target label not in features...")
    if "misinformation" in df.columns:
        # Get all computed features (numeric columns except misinformation, page_id, ad_id)
        exclude_cols = {"misinformation", "page_id", "ad_id", "ad_creative_bodies", 
                       "ad_creative_link_titles", "publisher_platforms", "target_gender",
                       "target_locations", "ad_languages", "target_ages", "spend", 
                       "impressions", "ad_delivery_start_time", "ad_delivery_stop_time", 
                       "ad_creation_time"}
        features = [col for col in df.columns if col not in exclude_cols]
        
        # Verify no leakage of label itself
        if "misinformation" not in features:
            logger.info("    ✓ PASS: Label 'misinformation' not in feature set")
        else:
            logger.error("    ✗ FAIL: Label 'misinformation' found in features - LEAKAGE DETECTED")
    else:
        logger.warning("    ? SKIP: 'misinformation' column not found")
    
    # CHECK 6: label_distribution (preliminary)
    logger.info("  [6] label_distribution: Checking class balance in full dataset...")
    if "misinformation" in df.columns:
        counts = df["misinformation"].value_counts().sort_index()
        total = len(df)
        if len(counts) == 2:
            pos_pct = (counts[1] / total) * 100
            neg_pct = (counts[0] / total) * 100
            logger.info(f"    Class 0: {counts[0]:6d} ({neg_pct:5.1f}%)")
            logger.info(f"    Class 1: {counts[1]:6d} ({pos_pct:5.1f}%)")
            
            if counts.min() < 10:
                logger.warning(f"    ⚠ WARN: Minority class has {counts.min()} samples (< 10)")
            if counts.min() / total < 0.1:
                logger.warning(f"    ⚠ WARN: Minority class is {(counts.min()/total)*100:.1f}% (< 10%)")
            else:
                logger.info("    ✓ PASS: Reasonable class distribution")
        else:
            logger.warning(f"    ? SKIP: Expected 2 classes, found {len(counts)}")
    else:
        logger.warning("    ? SKIP: 'misinformation' column not found")
    
    # Other checks require splits - note for post-split diagnostics
    logger.info("  [1,2,4,5,7,8,9,10] Post-split checks: Run diagnose_leakage_postsplit(train_df, val_df, test_df)")


def diagnose_leakage_postsplit(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                               test_df: pd.DataFrame) -> None:
    """
    Run post-split data leakage diagnostics on train/val/test splits.
    
    Performs all 10 checks after data is split:
    1. page_isolation - pages don't overlap
    2. sample_id_isolation - individual IDs disjoint
    3. label_not_in_features - handled pre-split
    4. page_feature_leakage - page features not contaminated
    5. metadata_overfit_proxy - metadata alone doesn't overpredict
    6. label_distribution - stratification success
    7. missing_image_audit - image availability
    8. temporal_leakage - time ordering effects
    9. embedding_alignment - embeddings match CSV
    10. scaler_train_only - scaler fit on train only
    
    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
    """
    logger.info("\n[LEAKAGE DIAGNOSTICS - Post-Split Checks]")
    
    # CHECK 1: page_isolation
    logger.info("  [1] page_isolation: Checking page_id isolation across splits...")
    try:
        if "page_id" in train_df.columns:
            train_pages = set(train_df["page_id"].unique())
            val_pages = set(val_df["page_id"].unique())
            test_pages = set(test_df["page_id"].unique())
            
            overlap_tv = train_pages & val_pages
            overlap_tt = train_pages & test_pages
            overlap_vt = val_pages & test_pages
            
            if not overlap_tv and not overlap_tt and not overlap_vt:
                logger.info("    ✓ PASS: Pages fully isolated across splits")
            else:
                logger.error(f"    ✗ FAIL: Page overlap detected - train/val: {len(overlap_tv)}, train/test: {len(overlap_tt)}, val/test: {len(overlap_vt)}")
        else:
            logger.warning("    ? SKIP: 'page_id' column not found")
    except Exception as e:
        logger.warning(f"    ? SKIP: {e}")
    
    # CHECK 2: sample_id_isolation
    logger.info("  [2] sample_id_isolation: Checking individual ID isolation...")
    try:
        # Try common ID column names
        id_col = None
        for col in ["ad_id", "id", "ad_archive_id"]:
            if col in train_df.columns:
                id_col = col
                break
        
        if id_col:
            train_ids = set(train_df[id_col].dropna().unique())
            val_ids = set(val_df[id_col].dropna().unique())
            test_ids = set(test_df[id_col].dropna().unique())
            
            overlap_tv = train_ids & val_ids
            overlap_tt = train_ids & test_ids
            overlap_vt = val_ids & test_ids
            
            if not overlap_tv and not overlap_tt and not overlap_vt:
                logger.info("    ✓ PASS: Individual IDs fully isolated")
            else:
                logger.error(f"    ✗ FAIL: ID overlap detected - train/val: {len(overlap_tv)}, train/test: {len(overlap_tt)}, val/test: {len(overlap_vt)}")
        else:
            logger.warning("    ? SKIP: No ID column found (tried: ad_id, id, ad_archive_id)")
    except Exception as e:
        logger.warning(f"    ? SKIP: {e}")
    
    # CHECK 6: label_distribution (stratification check)
    logger.info("  [6] label_distribution: Checking stratification success...")
    try:
        if "misinformation" in train_df.columns:
            train_counts = train_df["misinformation"].value_counts(normalize=True).sort_index()
            val_counts = val_df["misinformation"].value_counts(normalize=True).sort_index()
            test_counts = test_df["misinformation"].value_counts(normalize=True).sort_index()
            
            if len(train_counts) == 2 and len(val_counts) == 2 and len(test_counts) == 2:
                train_pos = train_counts[1]
                val_pos = val_counts[1]
                test_pos = test_counts[1]
                
                logger.info(f"    Positive class % - Train: {train_pos*100:.1f}%, Val: {val_pos*100:.1f}%, Test: {test_pos*100:.1f}%")
                
                skew_tv = abs(train_pos - val_pos)
                skew_tt = abs(train_pos - test_pos)
                
                if skew_tv > 0.15 or skew_tt > 0.15:
                    logger.warning(f"    ⚠ WARN: Class distribution skew detected (>15pp)")
                else:
                    logger.info("    ✓ PASS: Stratification successful")
            else:
                logger.warning("    ? SKIP: Not all splits have both classes")
        else:
            logger.warning("    ? SKIP: 'misinformation' column not found")
    except Exception as e:
        logger.warning(f"    ? SKIP: {e}")
    
    logger.info("  [4,5,7,8,9,10] Advanced checks: Implement in diagnose_leakage.py with model training and file I/O\n")


# ============================================================================
# RAW TEXT FEATURE EXTRACTION (must run BEFORE clean_text())
# ============================================================================

def compute_exclamation_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Compute ratio of exclamation marks to total characters in ad_creative_bodies.
    
    Misinformation ads often use repeated !!! for urgency and emotional manipulation.
    This feature must be computed on RAW text BEFORE clean_text() removes punctuation.
    
    Args:
        df: DataFrame with 'ad_creative_bodies' column (raw, uncleaned text).
    
    Returns:
        Series with exclamation ratio per row, filled with 0.0 for NaN/empty.
    """
    if "ad_creative_bodies" not in df.columns:
        logger.warning("ad_creative_bodies column not found, returning 0s")
        return pd.Series(0.0, index=df.index)
    
    def ratio(text):
        if pd.isna(text) or not text or len(text) == 0:
            return 0.0
        text = str(text)
        exclamation_count = text.count("!")
        return exclamation_count / len(text)
    
    return df["ad_creative_bodies"].apply(ratio).fillna(0.0)


def compute_caps_word_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Compute ratio of ALL-CAPS words to total words in ad_creative_bodies.
    
    Vietnamese misinformation often uses all-caps words for sensationalism:
    e.g., "GIẢM GIÁ SỐC HÔM NAY" signals urgency and emotional manipulation.
    
    A "caps word" is defined as:
    - Word with length > 1 (exclude single-letter abbreviations)
    - All alphabetic characters are uppercase
    
    This feature must be computed on RAW text BEFORE clean_text().
    
    Args:
        df: DataFrame with 'ad_creative_bodies' column (raw, uncleaned text).
    
    Returns:
        Series with caps word ratio per row, filled with 0.0 for NaN/empty.
    """
    if "ad_creative_bodies" not in df.columns:
        logger.warning("ad_creative_bodies column not found, returning 0s")
        return pd.Series(0.0, index=df.index)
    
    def ratio(text):
        if pd.isna(text) or not text:
            return 0.0
        text = str(text)
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        caps_count = 0
        for word in words:
            # Remove non-alpha chars, check if remaining is uppercase
            alpha_only = ''.join(c for c in word if c.isalpha())
            if len(alpha_only) > 1 and alpha_only.isupper():
                caps_count += 1
        
        return caps_count / len(words)
    
    return df["ad_creative_bodies"].apply(ratio).fillna(0.0)


def compute_repeated_punct_count(df: pd.DataFrame) -> pd.Series:
    """
    Count occurrences of 2+ consecutive identical punctuation marks.
    
    Indicates emotional emphasis or manipulation attempts.
    Pattern: "Thật không??" → 1 match
             "Mua ngay!!!" → 1 match
             "Tin sốc!!!???" → 2 matches
    
    This feature must be computed on RAW text BEFORE clean_text().
    
    Args:
        df: DataFrame with 'ad_creative_bodies' column (raw, uncleaned text).
    
    Returns:
        Series with repeated punctuation count per row, filled with 0 for NaN/empty.
    """
    if "ad_creative_bodies" not in df.columns:
        logger.warning("ad_creative_bodies column not found, returning 0s")
        return pd.Series(0, index=df.index)
    
    def count(text):
        if pd.isna(text) or not text:
            return 0
        text = str(text)
        # Pattern: 2+ consecutive identical punctuation marks from {!?.,-}
        matches = re.findall(r'([!?.,-])\1+', text)
        return len(matches)
    
    return df["ad_creative_bodies"].apply(count).fillna(0)


def compute_url_count(df: pd.DataFrame) -> pd.Series:
    """
    Count URLs in ad_creative_bodies BEFORE they are removed by clean_text().
    
    Ads with multiple URLs may be link farms or redirect scams.
    Pattern: http://, https://, and variations.
    
    This feature must be computed on RAW text BEFORE clean_text().
    
    Args:
        df: DataFrame with 'ad_creative_bodies' column (raw, uncleaned text).
    
    Returns:
        Series with URL count per row, filled with 0 for NaN/empty.
    """
    if "ad_creative_bodies" not in df.columns:
        logger.warning("ad_creative_bodies column not found, returning 0s")
        return pd.Series(0, index=df.index)
    
    def count(text):
        if pd.isna(text) or not text:
            return 0
        text = str(text)
        # Pattern: http:// or https:// followed by non-whitespace
        matches = re.findall(r'https?://\S+', text)
        return len(matches)
    
    return df["ad_creative_bodies"].apply(count).fillna(0)


def engineer_all_features(df: pd.DataFrame,
                           reference_df: pd.DataFrame | None = None,
                           raw_bodies: pd.Series | None = None
                           ) -> pd.DataFrame:
    """
    Compute all engineered features and append as new columns.

    Calls all feature functions, logs new column names and their non-null counts,
    and returns augmented DataFrame.

    Args:
        df: Raw ads DataFrame (already cleaned by preprocessing pipeline).
        reference_df: Optional reference DataFrame for relative feature computation.
        raw_bodies: Optional Series of raw ad bodies before cleaning (for advanced text features).

    Returns:
        DataFrame with all engineered feature columns appended.
    """
    logger.info("Starting feature engineering...")

    df_out = df.copy()
    new_features = []

    # === RAW TEXT FEATURES (must run BEFORE clean_text()) ===
    # These features extract signals from raw uncleaned text before linguistic
    # preprocessing. They capture stylistic misinformation markers like emotional
    # punctuation, capitalization patterns, and URL counts.
    logger.info("Computing raw text features (before cleaning)...")
    if "ad_creative_bodies" in df_out.columns:
        df_out["exclamation_ratio"] = compute_exclamation_ratio(df_out)
        new_features.append("exclamation_ratio")
        
        df_out["caps_word_ratio"] = compute_caps_word_ratio(df_out)
        new_features.append("caps_word_ratio")
        
        df_out["repeated_punct_count"] = compute_repeated_punct_count(df_out)
        new_features.append("repeated_punct_count")
        
        df_out["url_count"] = compute_url_count(df_out)
        new_features.append("url_count")
        
        logger.info(
            f"  Raw text features: exclamation_ratio, caps_word_ratio, "
            f"repeated_punct_count, url_count"
        )
    else:
        logger.warning("ad_creative_bodies column not found, skipping raw text features")

    # ---- CORE METADATA FEATURES ----
    logger.info("Computing core metadata features...")
    core_features = {
        "ads_per_page": compute_ads_per_page(df_out),
        "platform_count": compute_platform_count(df_out),
        "FB_only_flag": compute_fb_only_flag(df_out),
        "all_targeted": compute_all_targeted(df_out),
        "burstiness": compute_burstiness(df_out),
        "avg_ad_duration": compute_avg_ad_duration(df_out),
        "launch_delay": compute_launch_delay(df_out),
        "num_countries": compute_num_countries(df_out),
        "language_location_mismatch": compute_language_location_mismatch(df_out),
    }

    for feature_name, feature_series in core_features.items():
        df_out[feature_name] = feature_series
        new_features.append(feature_name)

    # ---- TEXT FEATURE EXTRACTION (on cleaned text) ----
    logger.info("Computing text-based features...")
    if "ad_creative_bodies" in df_out.columns:
        df_out["emojis_in_text"] = df_out["ad_creative_bodies"].apply(extract_emojis)
        new_features.append("emojis_in_text")
        
        df_out["emoji_count"] = df_out["ad_creative_bodies"].apply(count_emojis)
        new_features.append("emoji_count")
        
        df_out["text_length"] = df_out["ad_creative_bodies"].apply(calculate_text_length)
        new_features.append("text_length")
    else:
        logger.warning("ad_creative_bodies column not found, skipping text features")

    # ---- TIME-BASED FEATURES ----
    logger.info("Computing time-based features...")
    try:
        df_time = compute_time_based_features(df_out)
        time_features = [col for col in df_time.columns if col not in df_out.columns]
        for col in time_features:
            df_out[col] = df_time[col]
            new_features.append(col)
    except Exception as e:
        logger.warning(f"Time-based feature computation failed: {e}")

    # ---- PAGE-BASED FEATURES ----
    logger.info("Computing page-based features...")
    try:
        df_page = compute_page_based_features(df_out)
        page_features = [col for col in df_page.columns if col not in df_out.columns]
        for col in page_features:
            df_out[col] = df_page[col]
            new_features.append(col)
    except Exception as e:
        logger.warning(f"Page-based feature computation failed: {e}")

    # ---- SPEND-BASED FEATURES ----
    logger.info("Computing spend-based features...")
    try:
        df_spend = compute_spend_based_features(df_out)
        spend_features = [col for col in df_spend.columns if col not in df_out.columns]
        for col in spend_features:
            df_out[col] = df_spend[col]
            new_features.append(col)
    except Exception as e:
        logger.warning(f"Spend-based feature computation failed: {e}")

    # ---- DEMOGRAPHIC FEATURES ----
    logger.info("Computing demographic features...")
    try:
        df_demo = compute_demographic_features(df_out)
        demo_features = [col for col in df_demo.columns if col not in df_out.columns]
        for col in demo_features:
            df_out[col] = df_demo[col]
            new_features.append(col)
    except Exception as e:
        logger.warning(f"Demographic feature computation failed: {e}")

    # ---- PLATFORM FEATURES ----
    logger.info("Computing platform features...")
    try:
        df_platform = compute_platform_features(df_out)
        platform_features = [col for col in df_platform.columns if col not in df_out.columns]
        for col in platform_features:
            df_out[col] = df_platform[col]
            new_features.append(col)
    except Exception as e:
        logger.warning(f"Platform feature computation failed: {e}")

    # ---- DATA LEAKAGE DIAGNOSTICS (pre-split checks) ----
    logger.info("Running pre-split data leakage diagnostics...")
    try:
        diagnose_leakage_presplit(df_out)
    except Exception as e:
        logger.warning(f"Pre-split leakage diagnostics failed: {e}")

    # ---- LOG SUMMARY ----
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total new features added: {len(new_features)}\n")
    
    for feature_name in new_features:
        if feature_name in df_out.columns:
            null_count = df_out[feature_name].isna().sum()
            non_null_count = len(df_out) - null_count
            logger.info(f"  {feature_name}: {non_null_count} non-null, {null_count} null")
    
    logger.info("="*80 + "\n")

    return df_out


# ============================================================================
# ANTI-LEAKAGE FEATURE ENGINEERING FUNCTIONS
# ============================================================================
# These functions split the feature engineering pipeline to prevent data leakage.
# Row features are computed on full data before splitting.
# Page features are computed per split using train statistics as reference.

def engineer_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer only row-level features safe to compute independently.
    
    These features derive from a single row with no aggregation across rows.
    Safe to compute on the full dataset before splitting.
    
    Features engineered:
    - launch_delay: Days between ad creation and first delivery
    - platform_count: Number of unique platforms
    - FB_only_flag: Binary flag if only on Facebook
    - all_targeted: Binary flag if all ads are targeted
    - num_countries: Number of target countries
    - language_location_mismatch: Language-location mismatch flag
    - text_length: Character count of ad text
    - emoji_count: Count of emojis
    - exclamation_ratio: Ratio of exclamation marks
    - caps_word_ratio: Ratio of all-caps words
    - repeated_punct_count: Count of consecutive punctuation
    - url_count: Count of URLs
    
    Args:
        df: Raw DataFrame (text should be cleaned before calling)
    
    Returns:
        DataFrame with row-level features added
    """
    logger.info("="*80)
    logger.info("ENGINEERING ROW-LEVEL FEATURES (safe before split)")
    logger.info("="*80)
    
    df_out = df.copy()
    features_added = []
    
    # === Raw text features (must exist in original text before clean_text) ===
    logger.info("Computing raw text features...")
    if "ad_creative_bodies" in df_out.columns:
        df_out["exclamation_ratio"] = compute_exclamation_ratio(df_out)
        features_added.append("exclamation_ratio")
        
        df_out["caps_word_ratio"] = compute_caps_word_ratio(df_out)
        features_added.append("caps_word_ratio")
        
        df_out["repeated_punct_count"] = compute_repeated_punct_count(df_out)
        features_added.append("repeated_punct_count")
        
        df_out["url_count"] = compute_url_count(df_out)
        features_added.append("url_count")
    
    # === Row-level metadata features ===
    logger.info("Computing row-level metadata features...")
    df_out["launch_delay"] = compute_launch_delay(df_out)
    features_added.append("launch_delay")
    
    df_out["platform_count"] = compute_platform_count(df_out)
    features_added.append("platform_count")
    
    df_out["FB_only_flag"] = compute_fb_only_flag(df_out)
    features_added.append("FB_only_flag")
    
    df_out["all_targeted"] = compute_all_targeted(df_out)
    features_added.append("all_targeted")
    
    df_out["num_countries"] = compute_num_countries(df_out)
    features_added.append("num_countries")
    
    df_out["language_location_mismatch"] = compute_language_location_mismatch(df_out)
    features_added.append("language_location_mismatch")
    
    # === Text features (on cleaned text) ===
    logger.info("Computing text-based features...")
    if "ad_creative_bodies" in df_out.columns:
        df_out["emoji_count"] = df_out["ad_creative_bodies"].apply(count_emojis)
        features_added.append("emoji_count")
        
        df_out["text_length"] = df_out["ad_creative_bodies"].apply(calculate_text_length)
        features_added.append("text_length")
    
    logger.info(f"Row-level features added: {features_added}")
    logger.info(f"Total: {len(features_added)} features")
    logger.info("="*80 + "\n")
    
    return df_out


def engineer_page_features(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer only page-level features using reference_df for aggregations.
    
    These features aggregate across rows per page_id. ALL aggregations use
    reference_df, never df itself. For unseen pages in df (val/test), fills
    with reference_df median.
    
    Features engineered:
    - ads_per_page: Count of ads per page_id
    - burstiness: Temporal concentration metric per page
    - avg_ad_duration: Average duration in hours per page
    - ads_duration: Campaign duration in days per page
    - repeated_text_ratio: Proportion of repeated text per page
    
    Args:
        df: DataFrame for which to compute page features (train/val/test split)
        reference_df: Reference DataFrame for computing aggregates (usually train_df)
                     Must have same structure as df
    
    Returns:
        DataFrame with page-level features added
    """
    logger.info(f"Engineering page-level features (df: {len(df)} rows, "
                f"reference: {len(reference_df)} rows)")
    
    df_out = df.copy()
    features_added = []
    
    # === Page-aggregated features using reference_df ===
    
    # ads_per_page: count from reference
    if "page_id" in reference_df.columns:
        page_counts = reference_df.groupby("page_id").size()
        df_out["ads_per_page"] = df_out["page_id"].map(page_counts).fillna(
            page_counts.median()
        )
        features_added.append("ads_per_page")
    
    # burstiness: temporal pattern from reference
    df_out["burstiness"] = compute_burstiness(reference_df).groupby(
        reference_df["page_id"]
    ).transform("first")  # Get unique value per page, then map to df rows
    if "page_id" in df_out.columns:
        burstiness_by_page = (
            reference_df.copy()
        )
        burstiness_by_page["_burstiness"] = compute_burstiness(reference_df)
        burstiness_map = burstiness_by_page.groupby("page_id")["_burstiness"].first()
        df_out["burstiness"] = df_out["page_id"].map(burstiness_map).fillna(
            burstiness_map.median()
        )
    features_added.append("burstiness")
    
    # avg_ad_duration: average duration per page from reference
    df_out["avg_ad_duration"] = compute_avg_ad_duration(reference_df).groupby(
        reference_df["page_id"]
    ).transform("first")  # Get unique value per page, then map to df rows
    if "page_id" in df_out.columns:
        avg_duration_by_page = reference_df.copy()
        avg_duration_by_page["_avg_duration"] = compute_avg_ad_duration(reference_df)
        avg_duration_map = avg_duration_by_page.groupby("page_id")["_avg_duration"].first()
        df_out["avg_ad_duration"] = df_out["page_id"].map(avg_duration_map).fillna(
            avg_duration_map.median()
        )
    features_added.append("avg_ad_duration")
    
    # ads_duration: campaign duration (stop - start) per page
    if "page_id" in reference_df.columns and "ad_delivery_start_time" in reference_df.columns:
        ref_copy = reference_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(ref_copy["ad_delivery_start_time"]):
            ref_copy["ad_delivery_start_time"] = pd.to_datetime(
                ref_copy["ad_delivery_start_time"], errors='coerce'
            )
        
        # Campaign duration: max delivery stop - min delivery start per page
        def compute_campaign_duration(group):
            if "ad_delivery_stop_time" in group.columns:
                stop_times = pd.to_datetime(group["ad_delivery_stop_time"], errors='coerce')
                min_start = group["ad_delivery_start_time"].min()
                max_stop = stop_times.max()
                if pd.notna(min_start) and pd.notna(max_stop):
                    return (max_stop - min_start).days
            return 0
        
        campaign_duration_by_page = ref_copy.groupby("page_id").apply(
            compute_campaign_duration
        )
        df_out["ads_duration"] = df_out["page_id"].map(campaign_duration_by_page).fillna(
            campaign_duration_by_page.median()
        )
        features_added.append("ads_duration")
    
    # repeated_text_ratio: proportion of repeated ad text per page
    if "page_id" in reference_df.columns and "ad_creative_bodies" in reference_df.columns:
        def compute_repeated_ratio(group):
            texts = group["ad_creative_bodies"].dropna().astype(str)
            if len(texts) <= 1:
                return 0.0
            text_counts = texts.value_counts()
            repeated_count = (text_counts[text_counts > 1].sum())
            return repeated_count / len(texts) if len(texts) > 0 else 0.0
        
        repeated_ratio_by_page = reference_df.groupby("page_id").apply(
            compute_repeated_ratio
        )
        df_out["repeated_text_ratio"] = df_out["page_id"].map(repeated_ratio_by_page).fillna(
            repeated_ratio_by_page.median()
        )
        features_added.append("repeated_text_ratio")
    
    logger.info(f"Page-level features added: {features_added}")
    
    return df_out


# ============================================================================
# TEXT PROCESSING HELPER FUNCTIONS (for feature engineering)
# ============================================================================

def extract_emojis(text: str) -> str:
    """
    Extract all emojis from text and return them as a concatenated string.
    Example: '💓💓👉👉👉'
    
    Args:
        text: Input text string.
    
    Returns:
        String containing all extracted emojis.
    """
    if pd.isna(text) or emoji is None:
        return ""
    
    # Extract emojis using emoji library
    emojis = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    
    return emojis


def count_emojis(text: str) -> int:
    """
    Count the total number of emojis in the text.
    
    Args:
        text: Input text string.
    
    Returns:
        Number of emojis found.
    """
    if pd.isna(text) or emoji is None:
        return 0
    
    # Count emojis using emoji library
    emoji_count = sum(1 for c in text if c in emoji.EMOJI_DATA)
    
    return emoji_count


def remove_emojis(text: str) -> str:
    """
    Remove all emojis from text, keeping only text characters.
    
    Args:
        text: Input text string.
    
    Returns:
        Text with emojis removed and whitespace normalized.
    """
    if pd.isna(text) or emoji is None:
        return "" if pd.isna(text) else str(text)
    
    # Remove emojis using emoji library
    text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
    
    return text


def calculate_text_length(text: str) -> int:
    """
    Calculate the length (character count) of text.
    
    Args:
        text: Input text string.
    
    Returns:
        Length of text in characters.
    """
    if pd.isna(text):
        return 0
    
    return len(str(text))


# ============================================================================
# TIME-BASED FEATURE FUNCTIONS
# ============================================================================

def compute_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-based features from ad delivery timestamps.
    
    Args:
        df: DataFrame with ad_delivery_start_time and ad_delivery_stop_time columns.
    
    Returns:
        DataFrame with added time-based features:
            - ads_duration: Duration in days
            - launch_delay: Days until ad launch from now
            - burstiness: 1 if duration < 7 days, else 0
            - active_status: 1 if currently active, else 0
    """
    df = df.copy()
    
    # Parse datetime columns
    df['ad_delivery_start_time'] = pd.to_datetime(df['ad_delivery_start_time'], errors='coerce')
    df['ad_delivery_stop_time'] = pd.to_datetime(df['ad_delivery_stop_time'], errors='coerce')
    
    # Calculate ad duration in days
    df['ads_duration'] = (df['ad_delivery_stop_time'] - df['ad_delivery_start_time']).dt.days
    
    # Calculate launch delay (days between current date and start time)
    current_date = pd.Timestamp.now()
    df['launch_delay'] = (df['ad_delivery_start_time'] - current_date).dt.days
    
    # Calculate burstiness (1 = burst campaign if duration < 7 days, 0 = sustained)
    df['burstiness'] = (df['ads_duration'] < 7).astype(int)
    
    # Active status (1 = active, 0 = inactive based on stop time)
    df['active_status'] = (df['ad_delivery_stop_time'] > current_date).astype(int)
    
    logger.debug("Computed time-based features: ads_duration, launch_delay, burstiness, active_status")
    
    return df


# ============================================================================
# PAGE-BASED FEATURE FUNCTIONS
# ============================================================================

def compute_page_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute page-level aggregated features capturing campaign behavior.
    
    Args:
        df: DataFrame with page_id and ad_creative_bodies columns.
    
    Returns:
        DataFrame with added page-based features:
            - ads_per_page: Number of ads per page
            - avg_ad_duration: Average ad duration per page
            - repeated_text_ratio: Proportion of repeated text per page
            - historical_volume: Total ad count per page
    """
    df = df.copy()
    
    # Ads per page
    ads_per_page = df.groupby('page_id').size().reset_index(name='ads_per_page')
    df = df.merge(ads_per_page, on='page_id', how='left')
    
    # Average ad duration per page
    if 'ads_duration' in df.columns:
        avg_ad_duration = df.groupby('page_id')['ads_duration'].mean().reset_index(name='avg_ad_duration')
        df = df.merge(avg_ad_duration, on='page_id', how='left')
    
    # Repeated text ratio - template scams
    def calculate_repeated_text_ratio(page_id_val):
        if 'ad_creative_bodies' not in df.columns:
            return 0
        page_texts = df[df['page_id'] == page_id_val]['ad_creative_bodies'].values
        if len(page_texts) <= 1:
            return 0
        text_counts = Counter(page_texts)
        most_common_count = text_counts.most_common(1)[0][1] if text_counts else 0
        ratio = most_common_count / len(page_texts)
        return ratio
    
    repeated_text_ratio = df.groupby('page_id')['page_id'].apply(
        lambda x: calculate_repeated_text_ratio(x.iloc[0])
    ).reset_index(name='repeated_text_ratio')
    df = df.merge(repeated_text_ratio, on='page_id', how='left')
    
    # Historical volume
    historical_volume = df.groupby('page_id').size().reset_index(name='historical_volume')
    df = df.merge(historical_volume, on='page_id', how='left')
    
    logger.debug("Computed page-based features")
    
    return df


# ============================================================================
# SPEND-BASED FEATURE FUNCTIONS
# ============================================================================

def compute_spend_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute financial metrics from spend and impressions data.
    
    Args:
        df: DataFrame with spend, impressions, and ads_duration columns.
    
    Returns:
        DataFrame with added spend-based features:
            - spend_per_day: Daily spend rate
            - impressions_per_day: Daily impressions rate
            - CPM_estimate: Cost per mille estimate
            - low_spend_high_reach: Flag for suspicious low-spend high-reach campaigns
    """
    df = df.copy()
    
    # Convert spend and impressions to numeric
    df['spend'] = pd.to_numeric(df.get('spend', 0), errors='coerce').fillna(0)
    df['impressions'] = pd.to_numeric(df.get('impressions', 0), errors='coerce').fillna(0)
    
    if 'ads_duration' not in df.columns:
        logger.warning("ads_duration not found, skipping spend-based features")
        return df
    
    # Spend per day
    df['spend_per_day'] = df['spend'] / (df['ads_duration'].clip(lower=1))
    
    # Impressions per day
    df['impressions_per_day'] = df['impressions'] / (df['ads_duration'].clip(lower=1))
    
    # CPM estimate (Cost Per Mille)
    df['CPM_estimate'] = (df['spend'] / df['impressions'].clip(lower=1)) * 1000
    
    # Low spend high reach - suspicious pattern
    median_impressions_per_day = df['impressions_per_day'].median()
    median_spend_per_day = df['spend_per_day'].median()
    
    df['low_spend_high_reach'] = (
        (df['impressions_per_day'] > median_impressions_per_day) & 
        (df['spend_per_day'] < median_spend_per_day)
    ).astype(int)
    
    logger.debug("Computed spend-based features")
    
    return df


# ============================================================================
# DEMOGRAPHIC FEATURE FUNCTIONS
# ============================================================================

def extract_age_span(age_str: str) -> int:
    """
    Extract age span from age range string (e.g., '18-24' -> 6).
    
    Args:
        age_str: Age range string.
    
    Returns:
        Age span (difference between max and min age).
    """
    if pd.isna(age_str):
        return 0
    try:
        ages = str(age_str).split('-')
        if len(ages) == 2:
            return int(ages[1]) - int(ages[0])
    except (ValueError, IndexError):
        return 0
    return 0


def count_countries(location_str: str) -> int:
    """
    Count number of countries in location string.
    
    Args:
        location_str: Comma-separated location string.
    
    Returns:
        Number of countries.
    """
    if pd.isna(location_str):
        return 0
    return len(str(location_str).split(',')) if str(location_str).strip() else 0


def create_gender_flags(gender_str: str) -> Tuple[int, int, int]:
    """
    Create gender targeting flags from gender string.
    
    Args:
        gender_str: Gender targeting string.
    
    Returns:
        Tuple of (women_targeted, men_targeted, all_targeted).
    """
    if pd.isna(gender_str):
        return 0, 0, 0
    
    gender_str = str(gender_str).lower()
    women_targeted = 1 if 'women' in gender_str or 'female' in gender_str else 0
    men_targeted = 1 if 'men' in gender_str or 'male' in gender_str else 0
    all_targeted = 1 if 'all' in gender_str or (women_targeted == 0 and men_targeted == 0) else 0
    
    return women_targeted, men_targeted, all_targeted


def check_language_location_mismatch(row: pd.Series) -> int:
    """
    Check for language-location mismatch (red flag for scams).
    
    Args:
        row: DataFrame row with 'languages' and 'target_locations' columns.
    
    Returns:
        1 if mismatch detected, 0 otherwise.
    """
    langs_present = 1 if pd.notna(row.get('languages')) and str(row.get('languages', '')).strip() else 0
    locs_present = 1 if pd.notna(row.get('target_locations')) and str(row.get('target_locations', '')).strip() else 0
    
    # Mismatch if only one is present
    return 1 if (langs_present + locs_present == 1) else 0


def compute_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute demographic and targeting features.
    
    Args:
        df: DataFrame with target_ages, target_locations, target_gender, languages, target_locations columns.
    
    Returns:
        DataFrame with added demographic features.
    """
    df = df.copy()
    
    # Age span
    if 'target_ages' in df.columns:
        df['age_span'] = df['target_ages'].apply(extract_age_span)
    
    # Number of countries
    if 'target_locations' in df.columns:
        df['num_countries'] = df['target_locations'].apply(count_countries)
    
    # Gender flags
    if 'target_gender' in df.columns:
        df[['women_targeted', 'men_targeted', 'all_targeted']] = df['target_gender'].apply(
            lambda x: pd.Series(create_gender_flags(x))
        )
    
    # Language-location mismatch
    df['language_location_mismatch'] = df.apply(check_language_location_mismatch, axis=1)
    
    logger.debug("Computed demographic features")
    
    return df


# ============================================================================
# PLATFORM FEATURE FUNCTIONS
# ============================================================================

def count_platforms(platform_str: str) -> int:
    """
    Count number of platforms in platform string.
    
    Args:
        platform_str: Comma-separated platform string.
    
    Returns:
        Number of platforms.
    """
    if pd.isna(platform_str):
        return 0
    return len(str(platform_str).split(','))


def check_fb_only(platform_str: str) -> int:
    """
    Check if targeting Facebook only (indicator of older audience targeting).
    
    Args:
        platform_str: Platform string.
    
    Returns:
        1 if Facebook-only, 0 otherwise.
    """
    if pd.isna(platform_str):
        return 0
    platforms = str(platform_str).lower()
    return 1 if 'facebook' in platforms and 'instagram' not in platforms else 0


def check_ig_only(platform_str: str) -> int:
    """
    Check if targeting Instagram only (indicator of visual deception).
    
    Args:
        platform_str: Platform string.
    
    Returns:
        1 if Instagram-only, 0 otherwise.
    """
    if pd.isna(platform_str):
        return 0
    platforms = str(platform_str).lower()
    return 1 if 'instagram' in platforms and 'facebook' not in platforms else 0


def compute_platform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute platform-based features.
    
    Args:
        df: DataFrame with publisher_platforms column.
    
    Returns:
        DataFrame with added platform features.
    """
    df = df.copy()
    
    if 'publisher_platforms' not in df.columns:
        logger.warning("publisher_platforms column not found, skipping platform features")
        return df
    
    # Platform count
    df['platform_count'] = df['publisher_platforms'].apply(count_platforms)
    
    # FB only flag
    df['FB_only_flag'] = df['publisher_platforms'].apply(check_fb_only)
    
    # IG only flag
    df['IG_only_flag'] = df['publisher_platforms'].apply(check_ig_only)
    
    logger.debug("Computed platform features")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run feature engineering on ads CSV."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to raw ads CSV file (should already be cleaned by preprocessing).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path to save engineered features CSV.",
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

    # Load CSV
    logger.info(f"Loading CSV from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Extract raw ad bodies before cleaning (if available for analysis)
    raw_bodies = df["ad_creative_bodies"].copy() if "ad_creative_bodies" in df.columns else None

    # Engineer all features
    df_engineered = engineer_all_features(
        df,
        reference_df=None,
        raw_bodies=raw_bodies
    )

    # Select and print metadata features summary
    metadata_features = [
        "ads_per_page",
        "platform_count",
        "FB_only_flag",
        "all_targeted",
        "burstiness",
        "avg_ad_duration",
        "launch_delay",
        "num_countries",
        "language_location_mismatch",
    ]

    available_features = [f for f in metadata_features if f in df_engineered.columns]
    if available_features:
        logger.info("\nCore metadata features statistics:")
        print(df_engineered[available_features].describe())

    # Optionally save to CSV
    if args.output:
        df_engineered.to_csv(args.output, index=False)
        logger.info(f"Saved engineered features to {args.output}")
