# src/training/trainer.py
"""Training orchestration for the multimodal misinformation detector."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics, log_metrics_to_file
from src.losses.combined_loss import CombinedLoss
from src.models.full_model import MultimodalMisinfoDetector
from src.training.early_stopping import EarlyStopping
from src.training.scheduler import get_scheduler
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for the multimodal misinformation detector.

    Supports two-phase training protocol:
    - Phase 1: Frozen encoders, train fusion + classification head
    - Phase 2: Unfrozen top-k encoder blocks, train all with differential learning rates
    """

    def __init__(
        self,
        model: MultimodalMisinfoDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: CombinedLoss,
        config: dict,
        device: torch.device,
        experiment_name: str = "default",
        logger_obj=None,
    ):
        """
        Initialize trainer.

        Args:
            model: MultimodalMisinfoDetector instance.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            loss_fn: CombinedLoss instance.
            config: Configuration dictionary.
            device: Device to run on (cpu or cuda).
            experiment_name: Experiment name for checkpoint directories (default: "default").
            logger_obj: Logger instance (optional).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.criterion = loss_fn  # Alias for backward compatibility
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        self.logger = logger_obj or logger

        # Initialize optimizer with Phase 1 setup (encoders frozen)
        # Note: Encoders are already frozen by model.__init__
        param_groups = self.model.get_optimizer_param_groups(
            lr_fusion=config["training"]["lr_fusion"],
            lr_encoders=config["training"]["lr_encoders"],
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )
        self.optimizer = torch.optim.AdamW(param_groups)

        # Initialize scheduler (step-based warmup + cosine decay)
        total_steps = (
            len(train_loader)
            * config["training"]["max_epochs"]
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            warmup_steps=config["training"].get("warmup_steps", 500),
            total_steps=total_steps,
            lr_min=config["training"].get("lr_min", 1e-7),
        )

        # Initialize mixed precision scaler if enabled
        self.mixed_precision = config["training"].get("mixed_precision", False)
        self.scaler = GradScaler() if self.mixed_precision else None

        # Initialize early stopping with config metric
        early_stop_patience = config["training"].get("early_stopping_patience", 8)
        early_stop_metric = config["training"].get("early_stopping_metric", "f1_macro")
        ema_alpha = config["training"].get("early_stopping_ema_alpha", 0.7)
        
        self.early_stopping = EarlyStopping(
            patience=early_stop_patience,
            metric=early_stop_metric,
            mode="max",
            ema_alpha=ema_alpha,
        )

        # Initialize best checkpoint tracking
        self.best_metric = -float("inf")
        self.best_checkpoint_path = None
        self.best_epoch = None

        # Step counter for scheduler
        self.global_step = 0
        self.phase_2_started = False
        self.phase2_start_step = None
        self.phase2_warmup_steps = None

        self.logger.info("Trainer initialized successfully")

    def _transition_to_phase2(self) -> None:
        """
        Transition from Phase 1 (frozen encoders) to Phase 2 (fine-tuning encoders).
        
        Called exactly once at the start of Phase 2 (epoch freeze_encoder_epochs + 1).
        
        Actions:
        - Unfreeze top-k encoder blocks in model
        - Update encoder parameter group LR to target value with initial_lr
        - Record current step counter for Phase 2 warmup tracking
        - Set Phase 2 warmup duration
        - Log the transition and encoder LR
        """
        self.model.unfreeze_encoders(self.config["training"]["unfreeze_top_k_blocks"])

        # Update encoder parameter group LR (already in optimizer from init,
        # but was at 0 / frozen — now set to target LR)
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") == "encoders":
                param_group["lr"] = self.config["training"]["lr_encoders"]
                param_group["initial_lr"] = self.config["training"]["lr_encoders"]

        # Record step counter for Phase 2 scheduler
        self.phase2_start_step = self.global_step

        # Phase 2 warmup: linearly ramp encoder LR from 0 to lr_encoders
        # over warmup_steps steps before cosine decay takes over
        self.phase2_warmup_steps = self.config["training"].get("warmup_steps", 800)

        self.logger.info("=" * 70)
        self.logger.info(f"Phase 2 Started: Encoder Fine-tuning Initialized")
        self.logger.info(f"  Encoder LR: {self.config['training']['lr_encoders']:.2e}")
        self.logger.info(f"  Phase 2 warmup steps: {self.phase2_warmup_steps}")
        self.logger.info(f"  Start step: {self.phase2_start_step}")
        self.logger.info("=" * 70)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """
        Run one training epoch.

        Args:
            epoch: Epoch number (0-indexed).

        Returns:
            Dictionary with mean loss values for the epoch.
        """
        self.model.train()
        
        # Reset diagnostic flag so we check the first batch of this epoch
        if hasattr(self.model, "module"):
            # DataParallel case
            self.model.module._first_batch_checked = False
        else:
            # Single GPU case
            self.model._first_batch_checked = False
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_con_loss = 0.0
        total_cosine_sim = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    output = self.model(batch)
                    loss_dict = self.criterion(
                        logits=output["logits"],
                        labels=batch["label"].to(self.device),
                        text_emb=output["t_proj"],        # NOT detached, gradient attached
                        image_emb=output["i_proj"],        # NOT detached, gradient attached
                        valid_mask=output["image_valid"],  # Exclude dropout-zeroed images
                    )
                    loss = loss_dict["loss"]

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                output = self.model(batch)
                loss_dict = self.criterion(
                    logits=output["logits"],
                    labels=batch["label"].to(self.device),
                    text_emb=output["t_proj"],        # NOT detached, gradient attached
                    image_emb=output["i_proj"],        # NOT detached, gradient attached
                    valid_mask=output["image_valid"],  # Exclude dropout-zeroed images
                )
                loss = loss_dict["loss"]
                loss.backward()

            # Diagnostic logging: embedding norm and similarity
            if epoch <= 5 or epoch % 5 == 0:
                with torch.no_grad():
                    t = output["t_proj"].float()
                    i = output["i_proj"].float()
                    t_n = torch.nn.functional.normalize(t, dim=-1, eps=1e-8)
                    i_n = torch.nn.functional.normalize(i, dim=-1, eps=1e-8)
                    cos_sim = (t_n * i_n).sum(dim=-1)
                    self.logger.info(
                        f"[Diag] t_proj norm: {t.norm(dim=-1).mean():.4f} | "
                        f"i_proj norm: {i.norm(dim=-1).mean():.4f} | "
                        f"cos_sim: mean={cos_sim.mean():.4f} std={cos_sim.std():.4f} | "
                        f"t_proj std: {t.std():.6f} | i_proj std: {i.std():.6f}"
                    )

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config["training"].get("gradient_clip", 1.0),
            )

            # Optimizer step
            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

            # Accumulate losses
            total_loss += loss.item()
            total_cls_loss += loss_dict["cls_loss"].item()
            total_con_loss += loss_dict["con_loss"].item()
            num_batches += 1

            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                self.logger.info(
                    f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {total_loss/num_batches:.6f}"
                )

        mean_loss = total_loss / num_batches
        mean_cls_loss = total_cls_loss / num_batches
        mean_con_loss = total_con_loss / num_batches
        mean_cosine_sim = total_cosine_sim / num_batches

        self.logger.info(
            f"[Epoch {epoch+1}] Train Loss: {mean_loss:.6f} "
            f"(cls={mean_cls_loss:.6f}, con={mean_con_loss:.6f}) | "
            f"Mean intra-batch cosine sim: {mean_cosine_sim:.4f}"
        )

        return {
            "loss": mean_loss,
            "cls_loss": mean_cls_loss,
            "con_loss": mean_con_loss,
        }

    def evaluate(self, loader: DataLoader, split: str = "val") -> dict[str, float]:
        """
        Run inference on a dataset and compute metrics.

        Args:
            loader: DataLoader for evaluation.
            split: Split name ("val" or "test") for logging.

        Returns:
            Dictionary with all metrics from compute_all_metrics.
        """
        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                outputs = self.model(batch)
                logits = outputs["logits"].squeeze(-1)  # (B, 1) -> (B,)
                labels = batch["label"]

                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Concatenate all batches
        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        # Convert logits to probabilities via sigmoid
        proba_np = 1.0 / (1.0 + np.exp(-logits_np))

        # Compute metrics
        metrics = compute_all_metrics(labels_np, proba_np, threshold=0.5)

        self.logger.info(
            f"[{split.upper()}] Accuracy: {metrics['accuracy']:.4f} | "
            f"F1-Macro: {metrics['f1_macro']:.4f} | "
            f"ROC-AUC: {metrics['auc_roc']:.4f}"
        )

        return metrics

    def train(self) -> None:
        """
        Main training loop with two-phase protocol.

        Phase 1: Encoders frozen, train fusion + classification head
        Phase 2: Unfrozen top-k encoder blocks, train all with differential learning rates
        """
        num_epochs = self.config["training"]["max_epochs"]
        freeze_epochs = self.config["training"]["freeze_encoder_epochs"]
        
        # Create experiment-specific checkpoint directory
        base_checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
        checkpoint_dir = base_checkpoint_dir / self.experiment_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            # === Phase 2 Transition ===
            if epoch == freeze_epochs and not self.phase_2_started:
                self._transition_to_phase2()
                self.phase_2_started = True

            # --- Log Parameter Group Learning Rates ---
            self.logger.info(f"[Epoch {epoch+1}/{num_epochs}] Learning Rates:")
            for i, pg in enumerate(self.optimizer.param_groups):
                pg_name = pg.get("name", f"group_{i}")
                pg_lr = pg["lr"]
                pg_params = sum(p.numel() for p in pg["params"])
                self.logger.info(
                    f"  {pg_name}: lr={pg_lr:.2e} | params={pg_params:,}"
                )

            # --- Train Epoch ---
            self.logger.info(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"Phase {'2' if epoch >= freeze_epochs else '1'}"
            )
            train_metrics = self.train_epoch(epoch)

            # --- Validate ---
            val_metrics = self.evaluate(self.val_loader, split="val")

            # --- Early Stopping Check ---
            early_stop_metric_key = self.config["training"].get("early_stopping_metric", "f1_macro")
            current_metric = val_metrics[early_stop_metric_key]
            should_stop = self.early_stopping(current_metric, epoch)

            # --- Save Checkpoint (Every Epoch) ---
            ckpt_path = checkpoint_dir / f"epoch_{epoch+1:03d}_{early_stop_metric_key}_{current_metric:.4f}.pt"
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                current_metric,
                ckpt_path,
            )
            self.logger.info(f"Checkpoint saved: {ckpt_path}")

            # --- Update Best Checkpoint Pointer ---
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_checkpoint_path = ckpt_path
                self.best_epoch = epoch
                self.logger.info(
                    f"New best {early_stop_metric_key}: "
                    f"{self.best_metric:.4f} at {self.best_checkpoint_path}"
                )

            # --- Log Metrics ---
            log_metrics_to_file(
                {"train_loss": train_metrics["loss"], **val_metrics},
                epoch=epoch,
                split="val",
                log_dir=str(self.config["paths"]["log_dir"]),
            )

            # --- Early Stopping ---
            if should_stop:
                self.logger.warning(f"Early stopping triggered at epoch {epoch+1}")
                break

        # === Load Best Checkpoint ===
        self.logger.info("=" * 70)
        self.logger.info("Loading best checkpoint for final evaluation")
        self.logger.info("=" * 70)

        if self.best_checkpoint_path is None:
            raise RuntimeError(
                "No best checkpoint was recorded during training. "
                "Check that early_stopping_metric key matches a key in val_metrics."
            )

        load_checkpoint(self.best_checkpoint_path, self.model)
        self.logger.info(f"Loaded best checkpoint: {self.best_checkpoint_path}")
        self.logger.info(
            f"Best val {self.config['training'].get('early_stopping_metric', 'f1_macro')}: "
            f"{self.best_metric:.4f} at epoch {self.early_stopping.best_epoch}"
        )

        self.logger.info("=" * 70)
        self.logger.info(f"Best checkpoint: {self.best_checkpoint_path}")
        self.logger.info(f"Best val F1-macro: {self.best_metric:.4f}")
        self.logger.info("Training complete!")
        self.logger.info("=" * 70)

    def _move_batch_to_device(self, batch: dict) -> dict:
        """
        Move batch tensors to device, handling mixed types (tensors, lists, etc.).

        Args:
            batch: Batch dictionary.

        Returns:
            Batch dictionary with all tensors moved to device.
        """
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, list):
                # Handle lists (e.g., missing_image flags, sample_ids)
                result[k] = v
            else:
                result[k] = v
        return result


def train():
    """
    Main entry point for training.

    Parses command-line arguments, loads configuration, initializes all components,
    and runs the training loop.
    """
    parser = argparse.ArgumentParser(
        description="Train multimodal misinformation detector"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file (YAML or JSON)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
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
    get_logger("root", log_file=str(log_dir / "training.log"))

    # Set seed
    set_seed(config["data"]["seed"])

    logger.info("=" * 70)
    logger.info("Starting Training")
    logger.info("=" * 70)
    logger.info(f"Config: {config_path}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # NOTE: User must provide train_loader and val_loader
    # Example:
    # from src.data.dataset import create_datasets
    # from src.data.collate import collate_fn
    # train_dataset, val_dataset, _ = create_datasets(...)
    # train_loader = DataLoader(train_dataset, batch_size=..., collate_fn=collate_fn, ...)
    # val_loader = DataLoader(val_dataset, batch_size=..., collate_fn=collate_fn, ...)
    train_loader = None  # TODO: Populate from config
    val_loader = None  # TODO: Populate from config

    if train_loader is None or val_loader is None:
        raise ValueError(
            "train_loader and val_loader must be provided. "
            "See comments in train() for instructions."
        )

    # Build model
    model = MultimodalMisinfoDetector(
        text_model_name=config["encoders"]["text_encoder"],
        image_model_name=config["encoders"]["image_encoder"],
        feat_dim=config["model"]["feat_dim"],
        num_heads=config["model"]["num_heads"],
        attn_dropout=config["model"].get("attn_dropout", 0.1),
        modality_dropout=config["model"].get("modality_dropout", 0.15),
        head_dropout=config["model"].get("head_dropout", 0.3),
        device=device,
    )
    logger.info(f"Model built: {model.__class__.__name__}")

    # Log model summary
    from src.models.full_model import model_summary
    logger.info("Model Summary:")
    model_summary(model)

    # Build loss function
    # NOTE: class_weights should be computed from training data
    # Example: from src.data.preprocessing import compute_class_weights
    # class_weights = compute_class_weights(train_df)
    class_weights = torch.tensor([1.0, 1.0], device=device)  # Default equal weights
    loss_fn = CombinedLoss(
        class_weights=class_weights,
        contrastive_lambda=config["loss"]["contrastive_lambda"],
        temperature=config["loss"]["temperature"],
    )
    logger.info("Loss function initialized")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        device=device,
        logger_obj=logger,
    )

    # Start training
    trainer.train()

    logger.info("=" * 70)
    logger.info("All training complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    train()
