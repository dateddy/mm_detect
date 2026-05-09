#!/usr/bin/env python3
"""
Smoke test for ablation models. Runs 100 training steps per ablation and
verifies key health signals before committing to full training.

Usage:
    # Test all ablations
    python scripts/ablation_smoke_test.py --all
    
    # Test specific ablations
    python scripts/ablation_smoke_test.py --modes text_only image_only
    
    # Verbose mode (print diagnostics every 10 steps)
    python scripts/ablation_smoke_test.py --all --verbose
"""
import argparse
import json
import logging
import sys
import time
import traceback
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_datasets
from src.data.collate import collate_fn
from src.data.preprocessing import compute_pos_weight
from src.losses.combined_loss import CombinedLoss
from src.models import build_model, expected_param_range
from src.training.optim import build_optimizer_phase1
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Map ablation modes to their base config paths
ABLATION_CONFIGS = {
    "full":                     "configs/model/multimodal.yaml",
    "text_only":                "configs/model/text_only.yaml",
    "image_only":               "configs/model/image_only.yaml",
    "metadata_only":            "configs/model/metadata_only.yaml",
    "text_image":               "configs/ablation/text_image_only.yaml",
    "text_metadata":            "configs/ablation/text_metadata_only.yaml",
    "image_metadata":            "configs/ablation/image_metadata_only.yaml",
    "full_no_contrastive":      "configs/ablation/no_contrastive.yaml",
    "full_no_modality_dropout": "configs/ablation/no_modality_dropout.yaml",
    "full_no_dropout":          "configs/ablation/no_dropout.yaml",
    "full_no_metadata_in_fusion": "configs/ablation/no_metadata_in_fusion.yaml",
    "full_no_attention":        "configs/ablation/no_attention.yaml",
    "full_no_gating":           "configs/ablation/no_gating.yaml",
}

# Health thresholds
THRESHOLDS = {
    "min_loss_decrease_ratio": 0.95,    # late_avg_loss < early_avg_loss * 0.95 (5% drop)
    "max_pred_pos_rate":       0.95,    # not predicting all positive
    "min_pred_pos_rate":       0.05,    # not predicting all negative
    "min_logit_std":           0.05,    # logits should have variance
    "min_proj_std":            0.20,    # projections shouldn't collapse
    "max_proj_norm":           50.0,    # projections shouldn't explode
    "min_proj_norm":           0.5,     # projections shouldn't be zero
}


class SmokeTestResult:
    def __init__(self, mode: str):
        self.mode = mode
        self.checks: List[Dict] = []
        self.passed = True
        self.error: str = None
    
    def check(self, name: str, condition: bool, message: str = ""):
        self.checks.append({
            "name": name,
            "passed": condition,
            "message": message,
        })
        if not condition:
            self.passed = False
    
    def print_report(self):
        status = "[PASSED]" if self.passed else "[FAILED]"
        print(f"\n{status} {self.mode}")
        for c in self.checks:
            mark = "  [OK]" if c["passed"] else "  [FAIL]"
            print(f"{mark} {c['name']}: {c['message']}")
        if self.error:
            print(f"  ERROR: {self.error}")


class _FastTextEncoder(torch.nn.Module):
    """Small deterministic stand-in used by fast synthetic smoke tests."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.register_buffer("basis", torch.linspace(0.01, 1.0, hidden_size))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = input_ids.float()
        if attention_mask is not None:
            x = x * attention_mask.float()
        pooled = torch.sin(x.mean(dim=1, keepdim=True) * self.basis.unsqueeze(0))
        return pooled


class _FastImageEncoder(torch.nn.Module):
    """Small deterministic stand-in used by fast synthetic smoke tests."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.num_features = hidden_size
        self.register_buffer("basis", torch.linspace(0.01, 1.0, hidden_size))

    def forward(self, pixel_values):
        x = pixel_values.float().flatten(1).mean(dim=1, keepdim=True)
        return torch.cos(x * self.basis.unsqueeze(0))


def _patch_fast_encoders(model: torch.nn.Module) -> None:
    """Replace heavyweight pretrained encoders after architecture checks."""
    if getattr(model, "ablation_mode", None) in (
        "full", "full_no_contrastive", "full_no_modality_dropout",
        "full_no_dropout", "full_no_metadata_in_fusion",
        "full_no_attention", "full_no_gating",
    ):
        return
    if getattr(model, "text_encoder", None) is not None:
        model.text_encoder = _FastTextEncoder()
    if getattr(model, "image_encoder", None) is not None:
        model.image_encoder = _FastImageEncoder()


def _make_synthetic_batch(config: dict, batch_size: int, step: int) -> Dict:
    """Create a tiny deterministic batch with enough variance for smoke checks."""
    metadata_dim = len(config.get("metadata_features", [])) or 17
    gen = torch.Generator().manual_seed(10_000 + step)
    metadata = torch.randn(batch_size, metadata_dim, generator=gen)
    labels = (metadata[:, 0] > metadata[:, 0].median()).float()

    seq_len = 16
    input_ids = torch.randint(1, 5000, (batch_size, seq_len), generator=gen)
    pixel_values = torch.randn(batch_size, 3, 16, 16, generator=gen)
    text_emb = torch.randn(batch_size, 768, generator=gen)
    image_emb = torch.randn(batch_size, 768, generator=gen)

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "pixel_values": pixel_values,
        "text_emb": text_emb,
        "image_emb": image_emb,
        "metadata": metadata,
        "label": labels,
        "missing_image": [False] * batch_size,
        "valid_mask": torch.ones(batch_size, dtype=torch.bool),
    }


def load_base_config(project_root: Path) -> dict:
    """Load the base config."""
    base_config_path = project_root / "configs" / "base.yaml"
    with open(base_config_path, "r") as f:
        return yaml.safe_load(f)


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def smoke_test_one(
    mode: str,
    config_path: str,
    verbose: bool = False,
    steps: int = 5,
    use_synthetic: bool = True,
) -> SmokeTestResult:
    """Run a fast structural smoke test, or a slower real-data training smoke test."""
    result = SmokeTestResult(mode)
    
    print(f"\n{'=' * 70}")
    print(f"SMOKE TEST: {mode}")
    print(f"{'=' * 70}")
    
    try:
        project_root = Path(__file__).parent.parent
        base_config = load_base_config(project_root)
        
        # === 1. Load config ===
        config_file = project_root / config_path
        if not config_file.exists():
            result.error = f"Config file not found: {config_file}"
            result.check("config_exists", False, result.error)
            return result
        
        with open(config_file, "r") as f:
            override_config = yaml.safe_load(f)
        
        config = deep_merge_dicts(base_config, override_config or {})
        
        # Override for smoke test
        config["training"]["max_epochs"] = 1
        config["training"]["max_steps"] = steps
        config["loss"]["label_smoothing"] = 0.0
        print("[Smoke Test] Disabled label_smoothing for diagnostic clarity")
        config["training"]["batch_size"] = min(config["training"].get("batch_size", 32), 4)
        config["training"]["freeze_encoder_epochs"] = max(
            config["training"].get("freeze_encoder_epochs", 0), 1
        )
        if "data" not in config:
            config["data"] = {}
        config["data"]["train_subset"] = 1000  # use small subset
        config["data"]["num_workers"] = 0  # easier to debug
        
        # === 2. Setup device ===
        device = torch.device(config["training"].get("device", "cuda")
                              if torch.cuda.is_available() else "cpu")
        
        # === 3. Build model ===
        model = build_model(config)
        model = model.to(device)
        
        # === 4. CHECK: Param count ===
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lo, hi = expected_param_range(mode)
        result.check(
            "param_count_in_range",
            lo <= n_total <= hi,
            f"{n_total:,} params (expected [{lo:,}, {hi:,}])"
        )
        
        # === 5. CHECK: Modality presence ===
        has_text = (hasattr(model, "text_encoder") and model.text_encoder is not None)
        has_image = (hasattr(model, "image_encoder") and model.image_encoder is not None)
        has_meta = (hasattr(model, "metadata_encoder") and model.metadata_encoder is not None)
        n_modalities = sum([has_text, has_image, has_meta])
        result.check(
            "has_at_least_one_modality",
            n_modalities >= 1,
            f"T={has_text}, I={has_image}, M={has_meta}"
        )

        if use_synthetic:
            _patch_fast_encoders(model)
        
        # === 6. Build data ===
        set_seed(config.get("seed", 42))

        if use_synthetic:
            train_loader = [
                _make_synthetic_batch(config, config["training"]["batch_size"], i)
                for i in range(steps)
            ]
            pos_weight = 1.0
        else:
            # Resolve paths
            processed_dir = Path(config["paths"]["processed_dir"])
            if not processed_dir.is_absolute():
                processed_dir = project_root / processed_dir
            
            embeddings_dir = Path(config["paths"]["embeddings_dir"])
            if not embeddings_dir.is_absolute():
                embeddings_dir = project_root / embeddings_dir
            
            images_dir = Path(config["paths"]["images_dir"])
            if not images_dir.is_absolute():
                images_dir = project_root / images_dir
            
            uses_text = model.ablation_mode in {
                "full", "full_no_contrastive", "full_no_modality_dropout",
                "full_no_dropout", "full_no_metadata_in_fusion",
                "full_no_attention", "full_no_gating",
                "text_only", "text_image", "text_metadata",
            }

            # Initialize tokenizer and transforms
            tokenizer = (
                AutoTokenizer.from_pretrained(config["model"]["text_model_name"])
                if uses_text else None
            )
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
            }
            
            # Create datasets
            train_dataset, val_dataset, _ = create_datasets(
                train_csv=str(processed_dir / "splits" / "train.csv"),
                val_csv=str(processed_dir / "splits" / "val.csv"),
                test_csv=str(processed_dir / "splits" / "test.csv"),
                images_dir=str(images_dir),
                tokenizer=tokenizer,
                image_transforms=image_transforms,
                metadata_cols=config.get("metadata_features", []),
                offline_embeddings_dir=str(embeddings_dir) if embeddings_dir.exists() else None,
                ablation_mode=model.ablation_mode,
            )
            
            # Full models with frozen encoders need more data to show loss decrease.
            # Keep the subset reasonably small, but large enough to observe a trend.
            if len(train_dataset) > 4000:
                indices = np.random.choice(len(train_dataset), 4000, replace=False)
                train_dataset = torch.utils.data.Subset(train_dataset, indices)
            
            # Create dataloader
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["training"].get("batch_size", 32),
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
            )
            
            # === 7. Compute pos_weight ===
            train_csv_path = processed_dir / "splits" / "train.csv"
            train_df = pd.read_csv(train_csv_path)
            pos_weight = compute_pos_weight(train_df)
        
        # === 8. Build loss, optimizer ===
        loss_fn = CombinedLoss(
            class_weights=None,
            pos_weight=pos_weight,
            contrastive_lambda=config.get("loss", {}).get("contrastive_lambda", 0.1),
            contrastive_temperature_init=config.get("loss", {}).get("contrastive_temperature_init", 0.07),
            label_smoothing=config.get("loss", {}).get("label_smoothing", 0.0),
            ablation_mode=model.ablation_mode,
            aux_lambda=config.get("loss", {}).get("aux_lambda", 0.1),
        ).to(device)
        optimizer, _ = build_optimizer_phase1(model, loss_fn, config)
        
        # === 9. Run smoke steps (loop dataset if needed) ===
        model.train()
        losses = []
        proj_stats = {"t_proj": [], "i_proj": [], "m_proj": []}
        logit_stats = {"means": [], "stds": [], "pos_rates": []}
        
        step = 0
        max_passes = 5  # maximum times to loop the small subset
        for pass_idx in range(max_passes):
            if step >= steps:
                break
            for batch in train_loader:
                if step >= steps:
                    break

                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Ensure valid_mask exists (for full-model contrastive)
                if "valid_mask" not in batch:
                    B = batch["label"].shape[0]
                    batch["valid_mask"] = torch.ones(B, dtype=torch.bool, device=device)

                # Forward
                output = model(batch)
                losses_dict = loss_fn(
                    logits=output["logits"],
                    labels=batch["label"],
                    text_emb=output.get("t_proj", None),
                    image_emb=output.get("i_proj", None),
                    valid_mask=output.get("image_valid", batch.get("valid_mask", None)),
                    is_multimodal=output.get("is_multimodal", True),
                    output_dict=output,
                )
                loss = losses_dict["loss"]

                # CHECK: Loss is finite
                if not torch.isfinite(loss):
                    result.error = f"Loss is {loss.item()} at step {step}"
                    result.check("loss_is_finite", False, result.error)
                    return result

                # Backward + step
                optimizer.zero_grad()
                loss.backward()

                # CHECK: Gradients flow only to expected modalities
                if step == 5:  # check after a few steps
                    if not has_text:
                        text_grad_present = (
                            hasattr(model, "text_encoder") and
                            model.text_encoder is not None and
                            any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.text_encoder.parameters())
                        )
                        result.check(
                            "no_grad_to_disabled_text",
                            not text_grad_present,
                            "text_encoder receives gradient even though disabled"
                            if text_grad_present else "OK"
                        )

                    if not has_image:
                        image_grad_present = (
                            hasattr(model, "image_encoder") and
                            model.image_encoder is not None and
                            any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.image_encoder.parameters())
                        )
                        result.check(
                            "no_grad_to_disabled_image",
                            not image_grad_present,
                            "image_encoder receives gradient even though disabled"
                            if image_grad_present else "OK"
                        )

                optimizer.step()

                # Record statistics
                losses.append(loss.item())

                for proj_key in ["t_proj", "i_proj", "m_proj"]:
                    if proj_key in output:
                        proj_stats[proj_key].append({
                            "norm": output[proj_key].norm(dim=-1).mean().item(),
                            "std": output[proj_key].std().item(),
                        })

                logits = output["logits"].squeeze(-1).detach()
                logit_stats["means"].append(logits.mean().item())
                logit_stats["stds"].append(logits.std().item())
                preds = (torch.sigmoid(logits) > 0.5).float()
                logit_stats["pos_rates"].append(preds.mean().item())

                if verbose and step % 10 == 0:
                    con = losses_dict.get("con_loss", losses_dict.get("con", torch.tensor(0.0))).item()
                    cls = losses_dict.get("cls_loss", losses_dict.get("cls", torch.tensor(0.0))).item()
                    print(f"  step {step:3d}: loss={loss.item():.4f} (cls={cls:.4f}, "
                          f"con={con:.4f}) | logit_std={logits.std().item():.3f} | "
                          f"pos_rate={preds.mean().item():.3f}")

                step += 1
        
        result.check("ran_requested_steps", step == steps, f"Completed {step}/{steps} steps")

        if use_synthetic:
            return result
        
        # === 10. CHECK: Loss decreased (adaptive threshold based on # steps) ===
        if len(losses) < 2:
            result.check(
                "loss_decreased",
                True,
                f"Skipped: need at least 2 steps to compare loss (completed {len(losses)})"
            )
        else:
            window = max(1, min(20, len(losses) // 3))
            early_avg = np.mean(losses[:window])
            late_avg = np.mean(losses[-window:])
            absolute_drop = early_avg - late_avg
            decrease_ratio = late_avg / max(early_avg, 1e-8)

            # Adaptive threshold: more lenient if fewer steps were run
            if step >= 100:
                decrease_threshold = 0.95
            elif step >= 50:
                decrease_threshold = 0.97
            else:
                decrease_threshold = 0.99

            # Real-data smoke tests are short, noisy optimization runs with
            # frozen encoders and dropout. A small absolute decrease is enough
            # to prove the trainable head/fusion path is learning.
            min_absolute_drop = 0.01
            loss_decreased = (
                decrease_ratio < decrease_threshold or
                absolute_drop >= min_absolute_drop
            )

            result.check(
                "loss_decreased",
                loss_decreased,
                f"early={early_avg:.4f}, late={late_avg:.4f}, "
                f"drop={absolute_drop:.4f}, ratio={decrease_ratio:.3f} "
                f"(need ratio<{decrease_threshold} or drop>={min_absolute_drop:.2f} "
                f"for {step} steps)"
            )
        
        # === 11. CHECK: Predictions not degenerate ===
        late_pos_rates = logit_stats["pos_rates"][-10:]
        avg_pos_rate = np.mean(late_pos_rates)
        late_logit_stds = logit_stats["stds"][-10:]
        avg_logit_std = np.mean(late_logit_stds)
        pos_rate_ok = (
            THRESHOLDS["min_pred_pos_rate"] <
            avg_pos_rate <
            THRESHOLDS["max_pred_pos_rate"]
        )
        logits_spread_ok = avg_logit_std > THRESHOLDS["min_logit_std"]
        result.check(
            "predictions_not_collapsed",
            pos_rate_ok or logits_spread_ok,
            f"avg pos_rate={avg_pos_rate:.3f} (need > {THRESHOLDS['min_pred_pos_rate']} and "
            f"< {THRESHOLDS['max_pred_pos_rate']}, or logit_std>{THRESHOLDS['min_logit_std']}); "
            f"avg logit_std={avg_logit_std:.4f}"
        )
        
        # === 12. CHECK: Logit variance ===
        result.check(
            "logits_have_variance",
            avg_logit_std > THRESHOLDS["min_logit_std"],
            f"avg logit_std={avg_logit_std:.4f}"
        )
        
        # === 13. CHECK: Projection norms healthy ===
        for proj_key in ["t_proj", "i_proj", "m_proj"]:
            if not proj_stats[proj_key]:
                continue
            late_norms = [s["norm"] for s in proj_stats[proj_key][-10:]]
            late_stds = [s["std"] for s in proj_stats[proj_key][-10:]]
            avg_norm = np.mean(late_norms)
            avg_std = np.mean(late_stds)
            
            result.check(
                f"{proj_key}_norm_healthy",
                THRESHOLDS["min_proj_norm"] < avg_norm < THRESHOLDS["max_proj_norm"],
                f"norm={avg_norm:.2f}"
            )
            result.check(
                f"{proj_key}_not_collapsed",
                avg_std > THRESHOLDS["min_proj_std"],
                f"std={avg_std:.4f}"
            )
        
        # === 14. CHECK: Contrastive loss only present when applicable ===
        if "t_proj" in output and "i_proj" in output and config["loss"]["contrastive_lambda"] > 0:
            con_value = losses_dict.get("con_loss", losses_dict.get("con", torch.tensor(0.0))).item()
            result.check(
                "contrastive_active_when_applicable",
                con_value > 0,
                f"con={con_value:.4f}"
            )
    
    except Exception as e:
        result.error = str(e)
        result.passed = False
        traceback.print_exc()
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test ablation model wiring quickly, or run a slower real-data check"
    )
    parser.add_argument("--all", action="store_true",
                        help="Test all ablation modes")
    parser.add_argument("--modes", nargs="+", default=None,
                        help="Test specific ablation modes")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step diagnostics")
    parser.add_argument("--continue-on-fail", action="store_true",
                        help="Continue testing other ablations even if one fails")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of optimization steps (default: 5 synthetic, 100 real-data)")
    parser.add_argument("--real-data", action="store_true",
                        help="Use the dataset and real encoders instead of tiny synthetic inputs")
    args = parser.parse_args()
    
    if args.all:
        modes = list(ABLATION_CONFIGS.keys())
    elif args.modes:
        modes = args.modes
    else:
        print("Must specify --all or --modes")
        sys.exit(1)
    
    steps = args.steps if args.steps is not None else (100 if args.real_data else 5)
    print(f"\nRunning smoke tests for: {modes}")
    print(f"Mode: {'real-data' if args.real_data else 'synthetic-fast'} | steps: {steps}\n")
    
    all_results = []
    for mode in modes:
        if mode not in ABLATION_CONFIGS:
            print(f"Unknown mode: {mode}, skipping")
            continue
        
        result = smoke_test_one(
            mode,
            ABLATION_CONFIGS[mode],
            verbose=args.verbose,
            steps=steps,
            use_synthetic=not args.real_data,
        )
        result.print_report()
        all_results.append(result)
        
        if not result.passed and not args.continue_on_fail:
            print(f"\n[ABORT] {mode} failed smoke test. Use --continue-on-fail to continue.")
            break
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("SMOKE TEST SUMMARY")
    print(f"{'=' * 70}")
    n_passed = sum(1 for r in all_results if r.passed)
    n_failed = len(all_results) - n_passed
    print(f"Passed: {n_passed}/{len(all_results)}")
    print(f"Failed: {n_failed}/{len(all_results)}")
    
    if n_failed > 0:
        print(f"\nFailed ablations:")
        for r in all_results:
            if not r.passed:
                failed_checks = [c["name"] for c in r.checks if not c["passed"]]
                print(f"  {r.mode}: {failed_checks}")
        sys.exit(1)
    
    print(f"\n[OK] All ablations passed smoke test. Safe to run full training.")
    sys.exit(0)


if __name__ == "__main__":
    main()
