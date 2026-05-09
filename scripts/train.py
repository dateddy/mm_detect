#!/usr/bin/env python3
"""End-to-end training script for multimodal misinformation detector."""

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_datasets
from src.data.preprocessing import compute_class_weights, compute_pos_weight
from src.losses.combined_loss import CombinedLoss
from src.models import build_model, MultimodalMisinfoDetector
from src.models.full_model import model_summary
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.seed import set_seed


class LimitedDataLoader:
    """Wrapper around DataLoader that limits number of batches per epoch (for dry_run)."""

    def __init__(self, dataloader, limit: int = 2):
        self.dataloader = dataloader
        self.limit = limit

    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            if i >= self.limit:
                break
            yield batch

    def __len__(self):
        return min(self.limit, len(self.dataloader))


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multimodal misinformation detector end-to-end"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name appended to output dir (default: timestamp)",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default=None,
        choices=[
            "full", "text_only", "image_only", "metadata_only",
            "text_image", "text_metadata", "image_metadata",
            "full_no_contrastive", "full_no_modality_dropout",
            "full_no_dropout", "full_no_metadata_in_fusion",
            "full_no_attention", "full_no_gating",
        ],
        help="Optional ablation mode override. Defaults to the value in the config file.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run only 2 batches per epoch for 2 epochs to verify pipeline",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to use for training (default: cuda if available, else cpu)",
    )
    return parser.parse_args()


def main():
    """End-to-end training pipeline."""
    args = parse_args()

    config_path = Path(args.config)
    # If path is relative, resolve it relative to project root
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load base config first
    base_config_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Load the specified config (could be an ablation or override config)
    with open(config_path, "r") as f:
        override_config = yaml.safe_load(f)

    # Merge: override_config takes precedence over base_config
    config = deep_merge_dicts(base_config, override_config)
    if args.ablation_mode is not None:
        config["ablation_mode"] = args.ablation_mode

    logger_temp = logging.getLogger(__name__)
    logger_temp.info(f"Loaded base config from {base_config_path}")
    logger_temp.info(f"Loaded override config from {config_path}")

    # ========== 2. Setup output directories ==========
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"
    else:
        experiment_name = args.experiment_name

    output_base = Path(config["paths"].get("checkpoint_dir", "experiments"))
    output_base.mkdir(parents=True, exist_ok=True)

    experiment_dir = output_base / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"
    results_dir = experiment_dir / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========== 3. Setup seed ==========
    random_seed = config["data"].get("random_seed", 42)
    set_seed(random_seed)

    # ========== 4. Setup logger ==========
    logger = get_logger(
        __name__,
        log_file=log_dir / "training.log",
    )

    logger.info("=" * 80)
    logger.info("TRAINING MULTIMODAL MISINFORMATION DETECTOR")
    logger.info("=" * 80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Log dir: {log_dir}")
    logger.info(f"Results dir: {results_dir}")

    # ========== 5. Log config ==========
    config_json = json.dumps(config, indent=2)
    logger.info("Config:")
    logger.info("\n" + config_json)

    # Save config to JSON for reproducibility
    config_save_path = experiment_dir / "config.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_save_path}")

    # ========== 6. Setup device ==========
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ========== 7. Load data ==========
    logger.info("Creating datasets...")
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    processed_dir = Path(config["paths"]["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir
    
    embeddings_dir = Path(config["paths"]["embeddings_dir"])
    if not embeddings_dir.is_absolute():
        embeddings_dir = project_root / embeddings_dir
    
    images_dir = Path(config["paths"]["images_dir"])
    if not images_dir.is_absolute():
        images_dir = project_root / images_dir
    
    # Initialize tokenizer and transforms
    from transformers import AutoTokenizer
    import torchvision.transforms as transforms
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_model_name"])
    image_size = config["model"].get("image_size", 224)
    
    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_csv=str(processed_dir / "splits" / "train.csv"),
        val_csv=str(processed_dir / "splits" / "val.csv"),
        test_csv=str(processed_dir / "splits" / "test.csv"),
        images_dir=str(images_dir),
        tokenizer=tokenizer,
        image_transforms=image_transforms,
        metadata_cols=config.get("metadata_features", []),
        offline_embeddings_dir=(
            str(embeddings_dir)
            if config.get("data", {}).get("use_offline_embeddings", False)
            and embeddings_dir.exists()
            else None
        ),
        ablation_mode=config.get("ablation_mode", "full"),
        text_cols=config.get("text", {}).get("columns"),
        max_text_len=config.get("model", {}).get("max_text_len", 256),
    )

    logger.info(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # ========== 8. Compute class weights ==========
    logger.info("Computing class weights...")
    train_csv_path = processed_dir / "splits" / "train.csv"
    train_df = pd.read_csv(train_csv_path)
    class_weights_list = compute_class_weights(train_df)
    pos_weight = compute_pos_weight(train_df)
    actual_pos_rate = float(train_df["misinformation"].mean())
    config.setdefault("data", {})["estimated_pos_rate"] = actual_pos_rate
    # Keep class_weights on CPU initially for loss function initialization
    class_weights = class_weights_list
    logger.info(f"Class weights: {class_weights.numpy()}")
    logger.info(f"BCE pos_weight (n_neg/n_pos): {pos_weight:.4f}")
    logger.info(f"Classifier prior initialized from train pos_rate: {actual_pos_rate:.4f}")

    # ========== 9. Create dataloaders ==========
    from src.data.collate import collate_fn

    batch_size = config["training"].get("batch_size", 32)
    num_workers = config.get("data", {}).get("num_workers", 2)
    pin_memory = config.get("data", {}).get("pin_memory", False) and device.type == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ========== 10. DRY RUN: limit batches ==========
    if args.dry_run:
        logger.info("DRY RUN MODE: limiting to 2 batches per epoch, 2 epochs total")
        train_loader = LimitedDataLoader(train_loader, limit=2)
        val_loader = LimitedDataLoader(val_loader, limit=2)
        config["max_epochs"] = 2
        config["early_stopping_patience"] = 1

    logger.info(f"DataLoader sizes: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # ========== 11. Build model ==========
    logger.info("Building model...")
    logger.info(f"Ablation mode: {config.get('ablation_mode', 'full')}")
    model = build_model(config)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {num_params:,} total params, {num_trainable:,} trainable")

    # Log model summary
    try:
        summary = model_summary(model)
        logger.info("Model summary:")
        logger.info("\n" + summary)
    except Exception as e:
        logger.warning(f"Could not generate model summary: {e}")

    # ========== 12. Build loss ==========
    logger.info("Building loss function...")
    loss_fn = CombinedLoss(
        class_weights=None,
        pos_weight=pos_weight,
        contrastive_lambda=config["loss"].get("contrastive_lambda", 0.1),
        contrastive_temperature_init=config["loss"].get("contrastive_temperature_init", 0.07),
        label_smoothing=config["loss"].get("label_smoothing", 0.0),
        cls_loss_type=config["loss"].get("cls_loss_type", "bce"),
        focal_alpha=config["loss"].get("focal_alpha", 0.5),
        focal_gamma=config["loss"].get("focal_gamma", 2.0),
        focal_gamma_pos=config["loss"].get("focal_gamma_pos", 1.0),
        focal_gamma_neg=config["loss"].get("focal_gamma_neg", 4.0),
        focal_clip=config["loss"].get("focal_clip", 0.05),
        ablation_mode=config.get("ablation_mode", "full"),
        aux_lambda=config["loss"].get("aux_lambda", 0.1),
    )
    logger.info(f"Loss: CombinedLoss (type={config['loss'].get('cls_loss_type', 'bce')}), label_smoothing={config['loss'].get('label_smoothing', 0.0)}")

    # ========== 13. Resume from checkpoint if specified ==========
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            from src.utils.checkpoint import load_checkpoint

            state_dict, metadata = load_checkpoint(args.resume)
            model.load_state_dict(state_dict)
            start_epoch = metadata.get("epoch", 0)
            logger.info(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")

    # ========== 14. Build trainer ==========
    logger.info("Building trainer...")
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        experiment_name=experiment_name,
        logger_obj=logger,
    )
    logger.info("Trainer initialized")

    # ========== 15. Train ==========
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    trainer.train()

    # ========== 16. Evaluate on test set ==========
    # Verify trainer recorded a best checkpoint
    if trainer.best_checkpoint_path is None:
        raise RuntimeError(
            "Trainer ended without recording a best checkpoint. "
            "Check that early_stopping_metric matches a key in val_metrics dict."
        )

    logger.info("=" * 80)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 80)
    logger.info(f"Loading best checkpoint: {trainer.best_checkpoint_path}")
    logger.info(f"Best val metric:         {trainer.best_metric:.4f} "
                f"(epoch {trainer.best_epoch + 1})")

    # Load best checkpoint into model (trainer already did this, but ensure consistency)
    from src.utils.checkpoint import load_checkpoint
    load_checkpoint(trainer.best_checkpoint_path, trainer.model)
    model.eval()

    # Find best threshold on validation set
    logger.info("Finding best threshold on validation set...")
    import numpy as np

    val_labels = []
    val_preds = []

    with torch.no_grad():
        for batch in val_loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            outputs = model(batch)
            
            # Handle both dict and direct tensor outputs
            if isinstance(outputs, dict):
                logits = outputs["logits"].squeeze(-1)  # (B, 1) -> (B,)
            else:
                logits = outputs.squeeze(-1)  # (B, 1) -> (B,)
            
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = batch["label"].cpu().numpy()

            val_preds.extend(proba)
            val_labels.extend(labels)

    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)

    # Compute metrics
    from src.evaluation.metrics import (
        compute_all_metrics,
        find_best_threshold,
        print_classification_report,
        log_metrics_to_file,
    )

    best_threshold = find_best_threshold(val_labels, val_preds, metric="f1_macro")
    logger.info(f"Best threshold (from val): {best_threshold:.4f}")

    # Run inference on test set
    logger.info("Running inference on test set...")

    test_labels = []
    test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            outputs = model(batch)
            
            # Handle both dict and direct tensor outputs
            if isinstance(outputs, dict):
                logits = outputs["logits"].squeeze(-1)  # (B, 1) -> (B,)
            else:
                logits = outputs.squeeze(-1)  # (B, 1) -> (B,)
            
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = batch["label"].cpu().numpy()

            test_preds.extend(proba)
            test_labels.extend(labels)

    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    logger.info(
        "Test probability stats: "
        f"min={test_preds.min():.4f}, mean={test_preds.mean():.4f}, "
        f"max={test_preds.max():.4f}, std={test_preds.std():.4f}, "
        f"pos_rate@0.5={(test_preds >= 0.5).mean():.3f}"
    )

    # Compute test metrics using best threshold from validation
    test_metrics_at_05 = compute_all_metrics(test_labels, test_preds, threshold=0.5)
    test_metrics = compute_all_metrics(test_labels, test_preds, best_threshold)
    test_metrics["f1_macro_at_05"] = test_metrics_at_05["f1_macro"]
    test_metrics["accuracy_at_05"] = test_metrics_at_05["accuracy"]
    test_metrics["best_threshold"] = best_threshold

    logger.info("=" * 80)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 80)
    logger.info(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"F1 (binary): {test_metrics['f1_binary']:.4f}")
    logger.info(f"F1 (macro):  {test_metrics['f1_macro']:.4f}")
    logger.info(f"F1@0.5:      {test_metrics['f1_macro_at_05']:.4f}")
    logger.info(f"AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    logger.info(f"AUC-PR:    {test_metrics['auc_pr']:.4f}")

    # Print classification report
    test_binary = (test_preds >= best_threshold).astype(int)
    print_classification_report(
        test_labels,
        test_binary,
        label_names=["Not Misleading", "Misleading"],
    )

    # Save test metrics
    test_metrics["threshold"] = best_threshold
    test_metrics_path = results_dir / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved to {test_metrics_path}")

    # Log to metrics file
    try:
        log_metrics_to_file(test_metrics, epoch=-1, split="test", log_dir=results_dir)
    except Exception as e:
        logger.warning(f"Could not log metrics to file: {e}")

    try:
        gate_analysis_path = results_dir / "sample_gate_analysis.npz"
        gate_analysis = trainer.analyze_sample_gates(
            val_loader,
            save_path=gate_analysis_path,
        )
        if gate_analysis is not None:
            logger.info(f"Sample gate analysis saved to {gate_analysis_path}")
    except Exception as e:
        logger.warning(f"Could not run sample gate analysis: {e}")

    # ========== 17. Final summary ==========
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Best checkpoint: {trainer.best_checkpoint_path}")
    logger.info(f"Best epoch: {trainer.best_epoch + 1}")
    logger.info(f"Best val metric: {trainer.best_metric:.4f}")
    logger.info(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Results directory: {results_dir}")
    print(f"Best F1 (macro): {test_metrics['f1_macro']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
