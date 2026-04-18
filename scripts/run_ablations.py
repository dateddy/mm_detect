#!/usr/bin/env python3
"""Run ablation studies: train and evaluate multiple model variants."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ablation import (
    AblationConfig,
    get_ablation_group,
    merge_config_with_ablation,
)
from src.evaluation.metrics import compute_all_metrics, find_best_threshold
from src.models.full_model import MultimodalMisinfoDetector
from src.training.trainer import Trainer
from src.data.dataset import create_datasets
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from torch.utils.data import DataLoader
from src.data.collate import collate_fn


def train_ablation_variant(
    config: dict,
    ablation: AblationConfig,
    device: torch.device,
    logger,
    output_dir: Path,
) -> Dict:
    """
    Train a single ablation variant and evaluate on test set.

    Args:
        config: Base configuration dict with ablation overrides applied
        ablation: AblationConfig metadata
        device: Torch device
        logger: Logger instance
        output_dir: Output directory for this variant

    Returns:
        Dict with metrics and metadata
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Training ablation: {ablation.name}")
    logger.info(f"Description: {ablation.description}")
    logger.info(f"{'=' * 70}")

    # Set seed for reproducibility
    set_seed(config.get("seed", 42))

    # Create output dir for this variant
    variant_dir = output_dir / ablation.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Save ablation config
    ablation_config_path = variant_dir / "ablation_config.json"
    with open(ablation_config_path, "w") as f:
        json.dump(
            {
                "name": ablation.name,
                "description": ablation.description,
                "overrides": ablation.config_overrides,
            },
            f,
            indent=2,
        )

    # Create datasets
    logger.info("Creating datasets...")
    datasets = create_datasets(config)
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]
    logger.info(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Build model
    logger.info("Building model...")
    model = MultimodalMisinfoDetector(config)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {num_params} total params, {num_trainable} trainable")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=variant_dir,
    )

    # Train
    logger.info("Starting training...")
    best_checkpoint_path = trainer.train()
    logger.info(f"Training complete. Best checkpoint: {best_checkpoint_path}")

    # Load best checkpoint and evaluate on test set
    logger.info("Evaluating on test set...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Inference on test set
    y_true = []
    y_pred_proba = []

    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            logits = model(batch)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = batch["labels"].cpu().numpy()

            y_pred_proba.extend(proba)
            y_true.extend(labels)

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    # Find best threshold on validation set
    # (inference on val set to determine threshold)
    y_true_val = []
    y_pred_proba_val = []

    with torch.no_grad():
        for batch in val_loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            logits = model(batch)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = batch["labels"].cpu().numpy()

            y_pred_proba_val.extend(proba)
            y_true_val.extend(labels)

    y_true_val = np.array(y_true_val)
    y_pred_proba_val = np.array(y_pred_proba_val)

    threshold = find_best_threshold(y_true_val, y_pred_proba_val, metric="f1_macro")
    logger.info(f"Best threshold (from val set): {threshold:.4f}")

    # Compute metrics on test set
    metrics = compute_all_metrics(y_true, y_pred_proba, threshold)
    logger.info(f"Test Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    # Save metrics
    metrics_path = variant_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "ablation": ablation.name,
                "description": ablation.description,
                "threshold": threshold,
                "test_metrics": metrics,
            },
            f,
            indent=2,
        )

    # Copy best checkpoint to variant directory
    final_checkpoint_path = variant_dir / "best_model.pt"
    import shutil

    shutil.copy(best_checkpoint_path, final_checkpoint_path)
    logger.info(f"Checkpoint saved to {final_checkpoint_path}")

    # Return summary row
    return {
        "ablation": ablation.name,
        "description": ablation.description,
        **metrics,
    }


def main():
    """Run ablation studies."""
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, required=True, help="Base config YAML/JSON")
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["modality", "fusion", "metadata", "loss", "finetune", "all"],
        help="Ablation group to run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ablations",
        help="Output directory for ablation results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(__name__)
    logger.info(f"Device: {device}")

    # Load base config
    config_path = Path(args.config)
    # If path is relative, resolve it relative to project root
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    if config_path.suffix == ".json":
        with open(config_path, "r") as f:
            base_config = json.load(f)
    else:
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)
    logger.info(f"Base config loaded from {config_path}")

    # Get ablation group
    ablations = get_ablation_group(args.group)
    logger.info(f"Running {len(ablations)} ablations from group: {args.group}")
    logger.info(f"Ablations: {[a.name for a in ablations]}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.group
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Run each ablation
    results = []
    for i, ablation in enumerate(ablations, 1):
        logger.info(f"\n[{i}/{len(ablations)}] Running ablation: {ablation.name}")

        # Merge config with ablation overrides
        merged_config = merge_config_with_ablation(base_config, ablation)

        try:
            # Train and evaluate
            result = train_ablation_variant(
                merged_config, ablation, device, logger, output_dir
            )
            results.append(result)
            logger.info(f"✓ Ablation {ablation.name} complete")

        except Exception as e:
            logger.error(f"✗ Ablation {ablation.name} failed: {str(e)}")
            # Log partial result with NaN metrics
            failed_result = {
                "ablation": ablation.name,
                "description": ablation.description,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1_macro": np.nan,
                "auc_roc": np.nan,
            }
            results.append(failed_result)

    # Create summary table
    logger.info(f"\n{'=' * 70}")
    logger.info("ABLATION SUMMARY")
    logger.info(f"{'=' * 70}")

    df_summary = pd.DataFrame(results)

    # Sort by F1 macro in descending order
    df_summary_sorted = df_summary.sort_values("f1_macro", ascending=False, na_position="last")

    # Display summary
    logger.info("\nResults sorted by F1 (macro):")
    logger.info(
        df_summary_sorted[
            ["ablation", "description", "f1_macro", "accuracy", "auc_roc"]
        ].to_string(index=False)
    )

    # Save summary to CSV
    summary_path = output_dir / "summary.csv"
    df_summary_sorted.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to {summary_path}")

    # Print best result
    if not df_summary_sorted.empty:
        best_row = df_summary_sorted.iloc[0]
        logger.info(f"\n{'=' * 70}")
        logger.info("BEST ABLATION")
        logger.info(f"{'=' * 70}")
        logger.info(f"Name: {best_row['ablation']}")
        logger.info(f"Description: {best_row['description']}")
        logger.info(f"F1 (macro): {best_row['f1_macro']:.4f}")
        logger.info(f"Accuracy: {best_row['accuracy']:.4f}")
        logger.info(f"Precision: {best_row['precision']:.4f}")
        logger.info(f"Recall: {best_row['recall']:.4f}")
        logger.info(f"AUC-ROC: {best_row['auc_roc']:.4f}")

    logger.info(f"\nAll ablations complete. Results in {output_dir}")


if __name__ == "__main__":
    main()
