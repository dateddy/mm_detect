#!/usr/bin/env python3
"""
Production-grade data leakage and quality audit for machine learning pipelines.

This script performs comprehensive checks for data leakage, label contamination,
and data quality issues BEFORE any model training begins.

When to run:
    After prepare_data.py, before scripts/train.py
    $ python scripts/diagnose_leakage.py --config configs/base.yaml

Exit codes:
    0 — All checks passed, safe to train
    1 — One or more failed checks, do NOT train
    2 — Missing required data file, run prepare_data.py first

Leakage detection methodology follows:
    Kaufman et al. (2012) "Leakage in Data Mining"
    Kapoor & Narayanan (2023) "Leakage and the Reproducibility Crisis in ML-based Science"
"""

import argparse
import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from sklearn.preprocessing import RobustScaler

# Suppress FutureWarning and DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Setup logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single leakage audit check."""

    name: str
    status: str  # "PASS", "WARN", "FAIL", "SKIP"
    details: Dict[str, Any]
    warning_only: bool  # True = WARN doesn't block training, False = FAIL blocks training
    fix_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization, handling numpy types."""
        def convert_value(obj):
            """Recursively convert numpy types to Python natives."""
            if isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_value(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        result = asdict(self)
        return convert_value(result)


class LeakageAuditor:
    """Comprehensive leakage and quality audit of prepared datasets."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize auditor and load all required data.

        Args:
            config: Configuration dict from base.yaml

        Raises:
            FileNotFoundError: If required files are missing
        """
        self.config = config
        self.processed_dir = Path(config.get("processed_dir", "data/processed"))
        self.embeddings_dir = Path(config.get("embeddings_dir", "data/embeddings"))
        self.image_dir = Path(config.get("image_dir", "data/raw/ad_images"))

        # Load all data
        self._load_data()

    def _load_data(self) -> None:
        """Load train/val/test CSVs and embeddings metadata."""
        # Load CSVs
        for split in ["train", "val", "test"]:
            csv_path = self.processed_dir / f"{split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"ERROR: {csv_path} not found. Run prepare_data.py first.")

        self.train_df = pd.read_csv(self.processed_dir / "train.csv")
        self.val_df = pd.read_csv(self.processed_dir / "val.csv")
        self.test_df = pd.read_csv(self.processed_dir / "test.csv")

        logger.info(f"Loaded datasets: train={len(self.train_df)}, val={len(self.val_df)}, test={len(self.test_df)}")

    def run_all(self) -> List[CheckResult]:
        """Run all 10 checks and return results."""
        results = [
            self._check_page_isolation(),
            self._check_sample_id_isolation(),
            self._check_label_not_in_features(),
            self._check_page_feature_leakage(),
            self._check_metadata_overfit_proxy(),
            self._check_label_distribution(),
            self._check_missing_image_audit(),
            self._check_temporal_leakage(),
            self._check_embedding_alignment(),
            self._check_scaler_train_only(),
        ]
        return results

    # ========== CHECK 1: Page Isolation ==========
    def _check_page_isolation(self) -> CheckResult:
        """Verify that page_ids do not overlap across splits."""
        try:
            train_pages = set(self.train_df["page_id"].unique())
            val_pages = set(self.val_df["page_id"].unique())
            test_pages = set(self.test_df["page_id"].unique())

            train_val_overlap = train_pages & val_pages
            train_test_overlap = train_pages & test_pages
            val_test_overlap = val_pages & test_pages

            details = {
                "train_pages": len(train_pages),
                "val_pages": len(val_pages),
                "test_pages": len(test_pages),
                "train_val_overlap": len(train_val_overlap),
                "train_test_overlap": len(train_test_overlap),
                "val_test_overlap": len(val_test_overlap),
            }

            if train_val_overlap or train_test_overlap or val_test_overlap:
                overlaps = []
                if train_val_overlap:
                    overlaps.extend(list(train_val_overlap)[:5])
                if train_test_overlap:
                    overlaps.extend(list(train_test_overlap)[:5])
                details["overlapping_page_ids_sample"] = overlaps[:10]
                return CheckResult(
                    name="page_isolation",
                    status="FAIL",
                    details=details,
                    warning_only=False,
                    fix_hint="Re-run prepare_data.py — split_by_page() must group by page_id before splitting.",
                )

            return CheckResult(
                name="page_isolation",
                status="PASS",
                details=details,
                warning_only=False,
            )

        except Exception as e:
            return CheckResult(
                name="page_isolation",
                status="FAIL",
                details={"error": str(e)},
                warning_only=False,
            )

    # ========== CHECK 2: Sample ID Isolation ==========
    def _check_sample_id_isolation(self) -> CheckResult:
        """Verify that individual ad IDs (id column) are disjoint across splits."""
        try:
            train_ids = set(self.train_df["id"].unique())
            val_ids = set(self.val_df["id"].unique())
            test_ids = set(self.test_df["id"].unique())

            train_val_overlap = train_ids & val_ids
            train_test_overlap = train_ids & test_ids
            val_test_overlap = val_ids & test_ids

            details = {
                "train_samples": len(self.train_df),
                "val_samples": len(self.val_df),
                "test_samples": len(self.test_df),
                "train_val_overlap": len(train_val_overlap),
                "train_test_overlap": len(train_test_overlap),
                "val_test_overlap": len(val_test_overlap),
            }

            if train_val_overlap or train_test_overlap or val_test_overlap:
                overlaps = []
                if train_val_overlap:
                    overlaps.extend(list(train_val_overlap)[:5])
                if train_test_overlap:
                    overlaps.extend(list(train_test_overlap)[:5])
                details["overlapping_ids_sample"] = overlaps[:10]
                return CheckResult(
                    name="sample_id_isolation",
                    status="FAIL",
                    details=details,
                    warning_only=False,
                    fix_hint="Individual ad IDs overlap across splits. Check split_by_page() logic.",
                )

            return CheckResult(
                name="sample_id_isolation",
                status="PASS",
                details=details,
                warning_only=False,
            )

        except Exception as e:
            return CheckResult(
                name="sample_id_isolation",
                status="FAIL",
                details={"error": str(e)},
                warning_only=False,
            )

    # ========== CHECK 3: Label Not in Features ==========
    def _check_label_not_in_features(self) -> CheckResult:
        """Verify that target label is not in metadata features list."""
        try:
            forbidden_names = [
                "misinformation",
                "label",
                "target",
                "y",
                "is_fake",
                "fake",
                "mislead",
                "ground_truth",
                "gt",
                "truth",
            ]

            # Check config
            config_features = self.config.get("metadata_features", [])
            config_violations = [f for f in config_features if f.lower() in forbidden_names]

            # Check actual CSV columns
            csv_columns = self.train_df.columns.tolist()
            csv_violations = [c for c in csv_columns if c.lower() in forbidden_names]

            details = {
                "config_features_ok": len(config_features) - len(config_violations),
                "csv_columns_ok": len(csv_columns) - len(csv_violations),
            }

            if config_violations or csv_violations:
                violations = config_violations + csv_violations
                details["violations"] = violations
                return CheckResult(
                    name="label_not_in_features",
                    status="FAIL",
                    details=details,
                    warning_only=False,
                    fix_hint="Remove target column from metadata_features in configs/base.yaml.",
                )

            return CheckResult(
                name="label_not_in_features",
                status="PASS",
                details=details,
                warning_only=False,
            )

        except Exception as e:
            return CheckResult(
                name="label_not_in_features",
                status="FAIL",
                details={"error": str(e)},
                warning_only=False,
            )

    # ========== CHECK 4: Page Feature Leakage ==========
    def _check_page_feature_leakage(self) -> CheckResult:
        """Verify page-aggregated features not contaminated with val/test statistics."""
        try:
            # Step A: Compute train-only page statistics
            train_page_counts = self.train_df.groupby("page_id").size().to_dict()

            # Step B/C: Check that val/test pages are disjoint from train
            val_page_ids = set(self.val_df["page_id"].unique())
            test_page_ids = set(self.test_df["page_id"].unique())
            train_page_ids = set(train_page_counts.keys())

            val_in_train = val_page_ids & train_page_ids
            test_in_train = test_page_ids & train_page_ids

            details = {
                "train_pages_with_stats": len(train_page_counts),
                "val_pages_in_train": len(val_in_train),
                "test_pages_in_train": len(test_in_train),
            }

            if val_in_train or test_in_train:
                details["overlapping_pages"] = list(val_in_train | test_in_train)[:10]
                return CheckResult(
                    name="page_feature_leakage",
                    status="FAIL",
                    details=details,
                    warning_only=False,
                    fix_hint="Page isolation check failed — see CHECK 1.",
                )

            # Step D: Verify ads_per_page values in val/test match train statistics
            if "ads_per_page" in self.val_df.columns:
                val_actual_counts = self.val_df.groupby("page_id").size()
                val_stored_counts = self.val_df.groupby("page_id")["ads_per_page"].first()

                mismatches = (val_actual_counts != val_stored_counts).sum()
                if mismatches > 0:
                    details["val_ads_per_page_mismatches"] = int(mismatches)
                    return CheckResult(
                        name="page_feature_leakage",
                        status="FAIL",
                        details=details,
                        warning_only=False,
                        fix_hint="Pass reference_df=train_df to compute_ads_per_page() in feature_engineering.py.",
                    )

            return CheckResult(
                name="page_feature_leakage",
                status="PASS",
                details=details,
                warning_only=False,
            )

        except Exception as e:
            return CheckResult(
                name="page_feature_leakage",
                status="FAIL",
                details={"error": str(e)},
                warning_only=False,
            )

    # ========== CHECK 5: Metadata Overfit Proxy ==========
    def _check_metadata_overfit_proxy(self) -> CheckResult:
        """Quick sanity check: LogisticRegression on metadata features alone."""
        try:
            config_features = self.config.get("metadata_features", [])
            if not config_features:
                return CheckResult(
                    name="metadata_overfit_proxy",
                    status="SKIP",
                    details={"reason": "No metadata features configured"},
                    warning_only=False,
                )

            # Get features that exist in both train and val
            available_features = [f for f in config_features if f in self.train_df.columns]
            if not available_features:
                return CheckResult(
                    name="metadata_overfit_proxy",
                    status="SKIP",
                    details={"reason": f"None of configured features found in CSV"},
                    warning_only=False,
                )

            X_train = self.train_df[available_features].fillna(0).values
            y_train = self.train_df["misinformation"].values
            X_val = self.val_df[available_features].fillna(0).values
            y_val = self.val_df["misinformation"].values

            # Train LogisticRegression
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)

            # Evaluate
            train_acc = accuracy_score(y_train, lr.predict(X_train))
            val_acc = accuracy_score(y_val, lr.predict(X_val))
            val_f1 = f1_score(y_val, lr.predict(X_val), average="macro")
            y_pred_proba = lr.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            val_auc = auc(fpr, tpr)

            # Top features by coefficient
            feature_importance = list(zip(available_features, lr.coef_[0]))
            top_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:3]

            details = {
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "val_f1_macro": float(val_f1),
                "val_auc_roc": float(val_auc),
                "top_features": [[name, float(coef)] for name, coef in top_features],
            }

            # Determine status based on accuracy threshold
            if val_acc > 0.98:
                return CheckResult(
                    name="metadata_overfit_proxy",
                    status="FAIL",
                    details=details,
                    warning_only=False,
                    fix_hint="Metadata alone predicts val perfectly (>98%). Check for label contamination.",
                )
            elif val_acc > 0.90:
                return CheckResult(
                    name="metadata_overfit_proxy",
                    status="WARN",
                    details=details,
                    warning_only=True,
                    fix_hint="Metadata alone predicts val too accurately. Inspect page_feature_leakage.",
                )

            return CheckResult(
                name="metadata_overfit_proxy",
                status="PASS",
                details=details,
                warning_only=False,
            )

        except Exception as e:
            return CheckResult(
                name="metadata_overfit_proxy",
                status="FAIL",
                details={"error": str(e)},
                warning_only=False,
            )

    # ========== CHECK 6: Label Distribution ==========
    def _check_label_distribution(self) -> CheckResult:
        """Verify class balance across splits."""
        try:
            details = {}
            distribution_table = []

            for split_name, split_df in [("train", self.train_df), ("val", self.val_df), ("test", self.test_df)]:
                n_total = len(split_df)
                n_positive = (split_df["misinformation"] == 1).sum()
                n_negative = n_total - n_positive
                pos_pct = 100.0 * n_positive / n_total if n_total > 0 else 0
                neg_pct = 100.0 - pos_pct

                distribution_table.append({
                    "split": split_name,
                    "total": n_total,
                    "misleading": n_positive,
                    "not_misleading": n_negative,
                    "misleading_pct": pos_pct,
                })

                details[f"{split_name}_total"] = n_total
                details[f"{split_name}_positive"] = n_positive
                details[f"{split_name}_negative"] = n_negative
                details[f"{split_name}_positive_pct"] = pos_pct

            details["distribution_table"] = distribution_table

            # Check warnings
            warnings_list = []
            for split_name, split_df in [("train", self.train_df), ("val", self.val_df), ("test", self.test_df)]:
                n_total = len(split_df)
                n_positive = (split_df["misinformation"] == 1).sum()
                n_negative = n_total - n_positive
                pos_pct = 100.0 * n_positive / n_total if n_total > 0 else 0

                if n_positive < 10 or n_negative < 10:
                    warnings_list.append(f"{split_name}: minority class has < 10 samples")
                if pos_pct < 10 or pos_pct > 90:
                    warnings_list.append(f"{split_name}: severe class imbalance ({pos_pct:.1f}%)")

            # Check stratification
            train_pos_pct = details.get("train_positive_pct", 0)
            for split_name in ["val", "test"]:
                split_pos_pct = details.get(f"{split_name}_positive_pct", 0)
                if abs(split_pos_pct - train_pos_pct) > 15:
                    warnings_list.append(
                        f"{split_name} pos_pct ({split_pos_pct:.1f}%) differs from train ({train_pos_pct:.1f}%) by >15pp"
                    )

            if warnings_list:
                details["warnings"] = warnings_list
                return CheckResult(
                    name="label_distribution",
                    status="WARN",
                    details=details,
                    warning_only=True,
                    fix_hint="Re-run prepare_data.py with stratified=True.",
                )

            return CheckResult(
                name="label_distribution",
                status="PASS",
                details=details,
                warning_only=False,
            )

        except Exception as e:
            return CheckResult(
                name="label_distribution",
                status="FAIL",
                details={"error": str(e)},
                warning_only=False,
            )

    # ========== CHECK 7: Missing Image Audit ==========
    def _check_missing_image_audit(self) -> CheckResult:
        """Audit image availability across splits."""
        try:
            if not self.image_dir.exists():
                return CheckResult(
                    name="missing_image_audit",
                    status="SKIP",
                    details={"reason": f"Image directory not found: {self.image_dir}"},
                    warning_only=False,
                )

            # Get all images in directory
            image_files = set(f.stem for f in self.image_dir.glob("*.png"))

            details = {}
            image_stats = []
            warnings_list = []

            for split_name, split_df in [("train", self.train_df), ("val", self.val_df), ("test", self.test_df)]:
                n_total = len(split_df)
                n_has_image = split_df["id"].isin(image_files).sum()
                n_missing = n_total - n_has_image
                missing_pct = 100.0 * n_missing / n_total if n_total > 0 else 0

                image_stats.append({
                    "split": split_name,
                    "total": n_total,
                    "has_image": n_has_image,
                    "missing": n_missing,
                    "missing_pct": missing_pct,
                })

                details[f"{split_name}_total"] = n_total
                details[f"{split_name}_has_image"] = n_has_image
                details[f"{split_name}_missing"] = n_missing
                details[f"{split_name}_missing_pct"] = missing_pct

                if missing_pct > 20:
                    warnings_list.append(f"{split_name}: missing > 20% ({missing_pct:.1f}%)")

            details["image_stats"] = image_stats

            # Check consistency
            train_missing_pct = details.get("train_missing_pct", 0)
            for split_name in ["val", "test"]:
                split_missing_pct = details.get(f"{split_name}_missing_pct", 0)
                if split_missing_pct - train_missing_pct > 10:
                    warnings_list.append(
                        f"{split_name} missing ({split_missing_pct:.1f}%) > train ({train_missing_pct:.1f}%) + 10pp"
                    )

            if warnings_list:
                details["warnings"] = warnings_list
                return CheckResult(
                    name="missing_image_audit",
                    status="WARN",
                    details=details,
                    warning_only=True,
                    fix_hint="High missing rate — verify images_dir in config and re-run data preparation.",
                )

            return CheckResult(
                name="missing_image_audit",
                status="PASS",
                details=details,
                warning_only=False,
            )

        except Exception as e:
            return CheckResult(
                name="missing_image_audit",
                status="FAIL",
                details={"error": str(e)},
                warning_only=False,
            )

    # ========== CHECK 8: Temporal Leakage ==========
    def _check_temporal_leakage(self) -> CheckResult:
        """Detect time-ordering of splits."""
        try:
            # Check if ad_creation_time exists
            if "ad_creation_time" not in self.train_df.columns:
                return CheckResult(
                    name="temporal_leakage",
                    status="SKIP",
                    details={"reason": "ad_creation_time column not found"},
                    warning_only=False,
                )

            # Convert to timestamps
            try:
                train_times = pd.to_datetime(self.train_df["ad_creation_time"]).astype(np.int64).values
                val_times = pd.to_datetime(self.val_df["ad_creation_time"]).astype(np.int64).values
            except Exception:
                train_times = self.train_df["ad_creation_time"].astype(np.int64).values
                val_times = self.val_df["ad_creation_time"].astype(np.int64).values

            # Fit LR to detect time-ordering
            X_combined = np.concatenate([train_times, val_times]).reshape(-1, 1)
            y_combined = np.concatenate([np.zeros(len(train_times)), np.ones(len(val_times))])

            lr_time = LogisticRegression(max_iter=1000, random_state=42)
            lr_time.fit(X_combined, y_combined)

            # AUC for detecting split membership by time
            y_pred_proba = lr_time.predict_proba(X_combined)[:, 1]
            fpr, tpr, _ = roc_curve(y_combined, y_pred_proba)
            time_auc = auc(fpr, tpr)

            details = {
                "time_ordering_auc": float(time_auc),
                "train_time_range": [float(train_times.min()), float(train_times.max())],
                "val_time_range": [float(val_times.min()), float(val_times.max())],
            }

            if time_auc > 0.85:
                return CheckResult(
                    name="temporal_leakage",
                    status="WARN",
                    details=details,
                    warning_only=True,
                    fix_hint="Splits appear time-ordered. Valid if intentional, but document your design choice.",
                )

            return CheckResult(
                name="temporal_leakage",
                status="PASS",
                details=details,
                warning_only=False,
            )

        except Exception as e:
            return CheckResult(
                name="temporal_leakage",
                status="FAIL",
                details={"error": str(e)},
                warning_only=False,
            )

    # ========== CHECK 9: Embedding Alignment ==========
    def _check_embedding_alignment(self) -> CheckResult:
        """Verify embeddings match CSV row order."""
        try:
            if not self.embeddings_dir.exists():
                return CheckResult(
                    name="embedding_alignment",
                    status="SKIP",
                    details={"reason": "Embeddings not yet extracted"},
                    warning_only=False,
                )

            details = {}
            mismatches = []

            for split in ["train", "val", "test"]:
                ids_file = self.embeddings_dir / f"ids_{split}.npy"
                if not ids_file.exists():
                    continue

                try:
                    ids_npy = np.load(ids_file, allow_pickle=True)
                    split_df = getattr(self, f"{split}_df")

                    if len(ids_npy) != len(split_df):
                        details[f"{split}_length_mismatch"] = {
                            "npy": len(ids_npy),
                            "csv": len(split_df),
                        }
                        mismatches.append(f"{split}: length mismatch")
                        continue

                    # Check order
                    csv_ids = split_df["id"].values
                    order_matches = (ids_npy == csv_ids).sum()
                    if order_matches != len(ids_npy):
                        first_mismatches = []
                        for i in range(min(5, len(ids_npy))):
                            if ids_npy[i] != csv_ids[i]:
                                first_mismatches.append({
                                    "position": i,
                                    "id_in_npy": str(ids_npy[i]),
                                    "id_in_csv": str(csv_ids[i]),
                                })
                        details[f"{split}_order_mismatches"] = first_mismatches
                        mismatches.append(f"{split}: {len(ids_npy) - order_matches} order mismatches")

                except Exception as e:
                    mismatches.append(f"{split}: {str(e)}")

            if mismatches:
                details["mismatches"] = mismatches
                return CheckResult(
                    name="embedding_alignment",
                    status="FAIL",
                    details=details,
                    warning_only=False,
                    fix_hint="Re-run extract_embeddings.py — CSV was modified after embeddings were saved.",
                )

            return CheckResult(
                name="embedding_alignment",
                status="PASS" if not mismatches else "SKIP",
                details=details,
                warning_only=False,
            )

        except Exception as e:
            return CheckResult(
                name="embedding_alignment",
                status="FAIL",
                details={"error": str(e)},
                warning_only=False,
            )

    # ========== CHECK 10: Scaler Train Only ==========
    def _check_scaler_train_only(self) -> CheckResult:
        """Verify RobustScaler was fit on train data only."""
        try:
            scaler_path = self.processed_dir / "metadata_scaler.pkl"
            if not scaler_path.exists():
                return CheckResult(
                    name="scaler_train_only",
                    status="SKIP",
                    details={"reason": "Scaler file not found"},
                    warning_only=False,
                )

            import pickle

            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            # Get config features
            config_features = self.config.get("metadata_features", [])
            available_features = [f for f in config_features if f in self.train_df.columns]

            if not available_features:
                return CheckResult(
                    name="scaler_train_only",
                    status="SKIP",
                    details={"reason": "No metadata features to check"},
                    warning_only=False,
                )

            # Compute actual train statistics
            train_X = self.train_df[available_features].fillna(0).values
            train_center = np.median(train_X, axis=0)
            train_scale = np.percentile(train_X, 75, axis=0) - np.percentile(train_X, 25, axis=0)

            # Compare with scaler
            if hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
                center_match = np.allclose(scaler.center_, train_center, rtol=1e-3)
                scale_match = np.allclose(scaler.scale_, train_scale, rtol=1e-3)

                details = {
                    "center_matches_train": bool(center_match),
                    "scale_matches_train": bool(scale_match),
                }

                if not center_match or not scale_match:
                    details["max_center_error"] = float(np.abs(scaler.center_ - train_center).max())
                    details["max_scale_error"] = float(np.abs(scaler.scale_ - train_scale).max())
                    return CheckResult(
                        name="scaler_train_only",
                        status="FAIL",
                        details=details,
                        warning_only=False,
                        fix_hint="Scaler fit on full dataset, not train only. Re-run prepare_data.py.",
                    )

                return CheckResult(
                    name="scaler_train_only",
                    status="PASS",
                    details=details,
                    warning_only=False,
                )

            return CheckResult(
                name="scaler_train_only",
                status="SKIP",
                details={"reason": "Scaler type not recognized"},
                warning_only=False,
            )

        except Exception as e:
            return CheckResult(
                name="scaler_train_only",
                status="FAIL",
                details={"error": str(e)},
                warning_only=False,
            )


def run(config: Dict[str, Any], verbose: bool = False, export_path: Optional[str] = None) -> int:
    """
    Run complete leakage audit and return exit code.

    Args:
        config: Configuration dict from YAML
        verbose: Print detailed results for each check
        export_path: Optional path to save JSON report

    Returns:
        0 if all checks passed
        1 if any check failed
        2 if required files missing
    """
    try:
        auditor = LeakageAuditor(config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Run all checks
    results = auditor.run_all()

    # Print formatted report
    print("\n" + "=" * 70)
    print("  LEAKAGE AUDIT REPORT")
    print(f"  Config : {config.get('_config_path', 'unknown')}")
    print(f"  Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    for i, result in enumerate(results, 1):
        status_str = f"{result.status:4s}"
        check_str = f"CHECK {i:2d}  {result.name:25s}"
        print(f"  {check_str} .... {status_str}", end="")

        if result.status == "SKIP":
            print()
        else:
            # Add detail suffix if available
            if "val_accuracy" in result.details:
                print(f"  val_acc={result.details['val_accuracy']:.2f}")
            elif "mismatches" in result.details:
                print(f"  {len(result.details['mismatches'])} mismatches")
            else:
                print()

        # Print verbose details if requested
        if verbose and result.status in ["FAIL", "WARN"]:
            for key, val in result.details.items():
                if key not in ["distribution_table", "image_stats", "top_features"]:
                    print(f"    {key}: {val}")

    print("\n" + "-" * 70)

    # Summary
    pass_count = sum(1 for r in results if r.status == "PASS")
    warn_count = sum(1 for r in results if r.status == "WARN")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    skip_count = sum(1 for r in results if r.status == "SKIP")

    print(f"  RESULT: {pass_count} PASS   {warn_count} WARN   {fail_count} FAIL   {skip_count} SKIP")

    # Determine overall status and exit code
    if fail_count > 0:
        print("  STATUS: DO NOT TRAIN — fix the above failures first")
        exit_code = 1
    elif warn_count > 0:
        print("  STATUS: SAFE TO TRAIN (warnings present — review before proceeding)")
        exit_code = 0
    else:
        print("  STATUS: SAFE TO TRAIN [OK]")
        exit_code = 0

    print("-" * 70 + "\n")

    # Export JSON if requested
    if export_path:
        export_dict = {
            "config": str(config.get("_config_path", "unknown")),
            "timestamp": datetime.now().isoformat(),
            "overall_status": "FAIL" if fail_count > 0 else ("WARN" if warn_count > 0 else "PASS"),
            "safe_to_train": exit_code == 0,
            "summary": {
                "pass": pass_count,
                "warn": warn_count,
                "fail": fail_count,
                "skip": skip_count,
            },
            "checks": [r.to_dict() for r in results],
        }

        export_path_obj = Path(export_path)
        export_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path_obj, "w") as f:
            json.dump(export_dict, f, indent=2)
        print(f"✓ Report exported to {export_path}\n")

    return exit_code


def main():
    """Parse arguments and run audit."""
    parser = argparse.ArgumentParser(
        description="Comprehensive data leakage and quality audit for ML pipelines",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--export", type=str, default=None, help="Path to save JSON report")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["_config_path"] = str(config_path)

    # Run audit
    exit_code = run(config, verbose=args.verbose, export_path=args.export)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
