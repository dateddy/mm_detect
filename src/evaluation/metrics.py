# src/evaluation/metrics.py
"""Evaluation metrics computation and logging."""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1), shape (N,).
        y_pred_proba: Predicted probabilities in [0, 1], shape (N,).
        threshold: Decision threshold for converting probabilities to binary predictions
                   (default: 0.5).

    Returns:
        Dictionary with keys:
        - accuracy: Classification accuracy
        - precision: Precision (TP / (TP + FP))
        - recall: Recall / Sensitivity (TP / (TP + FN))
        - f1_binary: F1 score (binary average)
        - f1_macro: F1 score (macro average)
        - auc_roc: Area under ROC curve
        - auc_pr: Area under Precision-Recall curve
        - threshold: Decision threshold used
    """
    # === DEFENSIVE CHECK: Verify y_pred_proba is in [0, 1] ===
    assert y_pred_proba.min() >= 0.0 and y_pred_proba.max() <= 1.0, (
        f"y_pred_proba must be in [0,1]. Got min={y_pred_proba.min():.6f}, max={y_pred_proba.max():.6f}. "
        f"Did you forget to apply sigmoid to logits?"
    )
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1_binary = f1_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Compute AUC metrics
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = auc(recall_curve, precision_curve)
    
    # === DIAGNOSTIC: Detect label inversion automatically ===
    if auc_roc < 0.3:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"⚠ AUC-ROC = {auc_roc:.4f} is significantly below 0.5. "
            f"This suggests label inversion or systematically wrong predictions. "
            f"Check: (a) y_true is correct, (b) y_pred_proba is sigmoid(logit) not 1-sigmoid(logit), "
            f"(c) labels weren't inverted in preprocessing."
        )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_macro": float(f1_macro),
        "f1_binary": float(f1_binary),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "threshold": float(threshold),
    }


def find_best_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, metric: str = "f1_binary"
) -> float:
    """
    Find the best decision threshold that maximizes the specified metric.

    Intended for validation set only—never use on test set to avoid data leakage.

    Sweeps thresholds from 0.1 to 0.9 in steps of 0.01 and returns the threshold
    that maximizes the specified metric.

    Args:
        y_true: Ground truth binary labels (0 or 1), shape (N,).
        y_pred_proba: Predicted probabilities in [0, 1], shape (N,).
        metric: Metric to optimize. Options: "f1_binary", "f1_macro", "accuracy",
                "precision", "recall" (default: "f1_binary").

    Returns:
        Best threshold value in [0.1, 0.9].

    Raises:
        ValueError: If metric is not recognized.
    """
    best_threshold = 0.5
    best_value = -1.0

    for threshold in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == "f1_binary":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "f1_macro":
            value = f1_score(y_true, y_pred, average="macro", zero_division=0)
        elif metric == "accuracy":
            value = accuracy_score(y_true, y_pred)
        elif metric == "precision":
            value = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            value = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(
                f"Unknown metric: {metric}. "
                'Options: "f1_binary", "f1_macro", "accuracy", "precision", "recall"'
            )

        if value > best_value:
            best_value = value
            best_threshold = threshold

    logger.info(
        f"Best threshold for {metric}: {best_threshold:.2f} (value: {best_value:.4f})"
    )
    return float(best_threshold)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] = ["not misleading", "misleading"],
) -> None:
    """
    Print classification report and confusion matrix in a readable format.

    Args:
        y_true: Ground truth binary labels (0 or 1), shape (N,).
        y_pred: Predicted binary labels (0 or 1), shape (N,).
        label_names: Names for the label classes
                     (default: ["not misleading", "misleading"]).
    """
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    print("\nConfusion Matrix")
    print("-" * 70)
    cm = confusion_matrix(y_true, y_pred)
    print(f"{'':20s} Predicted Negative  Predicted Positive")
    print(f"{'Actual Negative':20s} {cm[0, 0]:>19d}  {cm[0, 1]:>19d}")
    print(f"{'Actual Positive':20s} {cm[1, 0]:>19d}  {cm[1, 1]:>19d}")
    print("=" * 70 + "\n")


def log_metrics_to_file(metrics: dict, epoch: int, split: str, log_dir: str) -> None:
    """
    Append metrics as a JSON line to metrics.jsonl.

    Each call appends one line with epoch, split, timestamp, and all metric values.
    File is created at log_dir/metrics.jsonl.

    Args:
        metrics: Dictionary of metric values (e.g., from compute_all_metrics).
        epoch: Epoch number (0-indexed or 1-indexed, as used in training).
        split: Dataset split name ("train", "val", "test").
        log_dir: Directory path to save metrics.jsonl.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "metrics.jsonl"

    # Prepare log entry with epoch, split, timestamp, and all metrics
    log_entry = {
        "epoch": int(epoch),
        "split": split,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }

    # Append as JSON line (one metric dict per line)
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    logger.debug(f"Logged metrics for epoch {epoch}, split {split} to {log_file}")
