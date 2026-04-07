"""Evaluation metrics: F1, precision, recall, ROC-AUC, etc."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted probabilities (values in [0, 1])
        targets: Ground truth binary labels
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    # Convert to binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    binary_targets = targets.astype(int)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(binary_targets, binary_preds),
        'precision': precision_score(binary_targets, binary_preds, zero_division=0),
        'recall': recall_score(binary_targets, binary_preds, zero_division=0),
        'f1': f1_score(binary_targets, binary_preds, zero_division=0),
        'roc_auc': roc_auc_score(binary_targets, predictions),
    }
    
    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted probabilities
        targets: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Confusion matrix (2x2 for binary classification)
    """
    binary_preds = (predictions >= threshold).astype(int)
    binary_targets = targets.astype(int)
    
    return confusion_matrix(binary_targets, binary_preds)


def compute_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> str:
    """
    Generate classification report.
    
    Args:
        predictions: Predicted probabilities
        targets: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Classification report string
    """
    binary_preds = (predictions >= threshold).astype(int)
    binary_targets = targets.astype(int)
    
    return classification_report(
        binary_targets,
        binary_preds,
        target_names=['Authentic', 'Misinformation']
    )


def compute_per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.
    
    Args:
        predictions: Predicted probabilities
        targets: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Dictionary with metrics for each class
    """
    binary_preds = (predictions >= threshold).astype(int)
    binary_targets = targets.astype(int)
    
    cm = confusion_matrix(binary_targets, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'authentic': {
            'tn': int(tn),
            'fp': int(fp),
            'precision': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        },
        'misinformation': {
            'tp': int(tp),
            'fn': int(fn),
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        }
    }
    
    return metrics
