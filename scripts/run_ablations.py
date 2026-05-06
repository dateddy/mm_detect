#!/usr/bin/env python3
"""
Run ablation studies sequentially with sanity checks.

Usage:
    # Run all ablations
    python scripts/run_ablations.py --all
    
    # Run specific ablations
    python scripts/run_ablations.py --modes text_only image_only metadata_only
    
    # Run with smoke test only (100 steps each)
    python scripts/run_ablations.py --all --smoke-test
    
    # Skip already-completed runs
    python scripts/run_ablations.py --all --skip-existing
    
    # Run on specific GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ablations.py --all
"""
import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_datasets
from src.data.collate import collate_fn
from src.data.preprocessing import compute_class_weights
from src.losses.combined_loss import CombinedLoss
from src.models import build_model, expected_param_range
from src.training.optim import build_optimizer_phase1
from src.training.trainer import Trainer
from src.utils.config import load_config_with_inheritance, deep_merge_dicts
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Mapping: ablation name -> config path
ABLATION_CONFIGS = {
    "full":                     "configs/model/multimodal.yaml",
    "text_only":                "configs/model/text_only.yaml",
    "image_only":               "configs/model/image_only.yaml",
    "metadata_only":            "configs/model/metadata_only.yaml",
    "text_image":               "configs/ablation/text_image_only.yaml",
    "text_metadata":            "configs/ablation/text_metadata_only.yaml",
    "image_metadata":           "configs/ablation/image_metadata_only.yaml",
    "no_contrastive":           "configs/ablation/no_contrastive.yaml",
    "no_modality_dropout":      "configs/ablation/no_modality_dropout.yaml",
    "no_attention":             "configs/ablation/no_attention.yaml",
    "no_gating":                "configs/ablation/no_gating.yaml",
}

# Order matters: cheapest ablations first to fail fast on bugs
RECOMMENDED_ORDER = [
    "metadata_only",            # smallest, fastest, catches data leakage early
    "text_only",                # ~135M params
    "image_only",               # ~86M params
    "text_metadata",
    "image_metadata",
    "text_image",
    "no_contrastive",
    "no_modality_dropout",
    "no_attention",
    "no_gating",
    "full",                     # full model last (longest)
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--all", action="store_true",
                        help="Run all ablations in RECOMMENDED_ORDER")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=list(ABLATION_CONFIGS.keys()),
                        help="Specific ablation modes to run")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Only run 100 steps per ablation (for sanity check)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip ablations whose output_dir already has a results.json")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max_epochs in all configs (e.g., for quick experiments)")
    parser.add_argument("--results-dir", default="outputs/ablation_results",
                        help="Directory to save aggregated results")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue running other ablations if one crashes")
    return parser.parse_args()


def verify_model_construction(config: Dict, model: torch.nn.Module) -> Dict:
    """Verify the constructed model matches expected ablation properties."""
    mode = config["ablation_mode"]
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Check param count is in expected range
    lo, hi = expected_param_range(mode)
    if not (lo <= n_total <= hi):
        raise RuntimeError(
            f"\n[VERIFICATION FAILED] Mode '{mode}' has {n_total:,} params, "
            f"expected in [{lo:,}, {hi:,}].\n"
            f"This indicates the model architecture does NOT match the ablation mode.\n"
            f"Check src/models/factory.py and the model class for '{mode}'."
        )
    
    # Check expected branches are present/absent
    has_text = (hasattr(model, "text_encoder") and model.text_encoder is not None)
    has_image = (hasattr(model, "image_encoder") and model.image_encoder is not None)
    has_meta = (hasattr(model, "metadata_encoder") and model.metadata_encoder is not None)
    
    expected_modalities = {
        "full":                     (True, True, True),
        "full_no_contrastive":      (True, True, True),
        "full_no_modality_dropout": (True, True, True),
        "full_no_attention":        (True, True, True),
        "full_no_gating":           (True, True, True),
        "text_only":                (True, False, False),
        "image_only":               (False, True, False),
        "metadata_only":            (False, False, True),
        "text_image":               (True, True, False),
        "text_metadata":            (True, False, True),
        "image_metadata":           (False, True, True),
    }
    
    expected = expected_modalities.get(mode)
    actual = (has_text, has_image, has_meta)
    
    if expected != actual:
        raise RuntimeError(
            f"\n[VERIFICATION FAILED] Mode '{mode}' has modalities (T={has_text}, "
            f"I={has_image}, M={has_meta}) but expected (T={expected[0]}, "
            f"I={expected[1]}, M={expected[2]})."
        )
    
    print(f"[VERIFY] mode='{mode}' OK: {n_total:,} total, {n_train:,} trainable, "
          f"modalities=(T={has_text}, I={has_image}, M={has_meta})")
    
    return {
        "mode": mode,
        "total_params": n_total,
        "trainable_params": n_train,
        "has_text": has_text,
        "has_image": has_image,
        "has_metadata": has_meta,
    }


def run_single_ablation(
    mode: str,
    config_path: str,
    project_root: Path,
    smoke_test: bool = False,
    max_epochs_override: Optional[int] = None,
) -> Dict:
    """Run training for a single ablation and return final metrics."""
    print(f"\n{'#' * 70}")
    print(f"# RUNNING ABLATION: {mode}")
    print(f"# Config: {config_path}")
    print(f"{'#' * 70}\n")
    
    start_time = time.time()
    
    try:
        # Load config
        config = load_config_with_inheritance(str(project_root / config_path))
        
        # Apply overrides
        if smoke_test:
            config["training"]["max_epochs"] = 1
            config["training"]["max_steps"] = 100
            if "data" not in config:
                config["data"] = {}
            config["data"]["train_subset"] = 1000
        elif max_epochs_override is not None:
            config["training"]["max_epochs"] = max_epochs_override
        
        # Ensure ablation_mode is set
        if "ablation_mode" not in config:
            config["ablation_mode"] = mode
        
        # Build data loaders
        print(f"[DATA] Building dataloaders for {mode}...")
        
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
        
        # Initialize tokenizer and transforms
        tokenizer = AutoTokenizer.from_pretrained(config["encoders"]["text_encoder_name"])
        image_size = config["encoders"].get("image_size", 224)
        
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
            offline_embeddings_dir=str(embeddings_dir) if embeddings_dir.exists() else None,
        )
        
        # Create dataloaders
        batch_size = config["training"].get("batch_size", 32)
        num_workers = config.get("num_workers", 2)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        
        # Build model via factory
        print(f"[MODEL] Building model for {mode}...")
        model = build_model(config)
        verification = verify_model_construction(config, model)
        
        # Build trainer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DEVICE] Using device: {device}")
        
        trainer = Trainer(
            model,
            config,
            train_loader,
            val_loader,
            device=device,
        )
        
        # Run training
        print(f"[TRAIN] Starting training for {mode}...")
        train_metrics = trainer.train()
        
        # Final test evaluation
        print(f"[FINAL TEST] Loading best checkpoint and evaluating...")
        test_metrics = trainer.evaluate(test_loader, split="test", load_best=True)
        
        duration = time.time() - start_time
        
        # Bundle results
        result = {
            "mode": mode,
            "config_path": str(config_path),
            "status": "success",
            "duration_seconds": duration,
            "duration_human": f"{duration / 60:.1f} min",
            "verification": verification,
            "best_val_metrics": train_metrics.get("best_val", {}),
            "test_metrics": test_metrics,
            "best_epoch": train_metrics.get("best_epoch"),
            "early_stopped": train_metrics.get("early_stopped", False),
        }
        
        # Save results to ablation's output_dir
        output_dir = Path(config.get("output_dir", f"outputs/{mode}"))
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n[DONE] {mode} finished in {duration / 60:.1f} min. "
              f"Test F1-macro = {test_metrics.get('f1_macro', 'N/A'):.4f}")
        
        return result
        
    except Exception as e:
        print(f"\n[TRAINING FAILED] {mode}: {e}")
        traceback.print_exc()
        duration = time.time() - start_time
        return {
            "mode": mode,
            "status": "failed",
            "error": str(e),
            "duration_seconds": duration,
        }


def main():
    args = parse_args()
    
    # Determine which modes to run
    if args.all:
        modes = RECOMMENDED_ORDER
    elif args.modes:
        modes = args.modes
    else:
        print("ERROR: Must specify --all or --modes")
        sys.exit(1)
    
    project_root = Path(__file__).parent.parent
    
    print(f"\n{'=' * 70}")
    print(f"ABLATION STUDY RUN")
    print(f"{'=' * 70}")
    print(f"Modes to run:    {modes}")
    print(f"Smoke test:      {args.smoke_test}")
    print(f"Max epochs:      {args.max_epochs or '(from config)'}")
    print(f"Skip existing:   {args.skip_existing}")
    print(f"Results dir:     {args.results_dir}")
    print(f"Continue on err: {args.continue_on_error}")
    print(f"{'=' * 70}\n")
    
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for mode in modes:
        if mode not in ABLATION_CONFIGS:
            print(f"[SKIP] Unknown mode: {mode}")
            continue
        
        config_path = ABLATION_CONFIGS[mode]
        
        # Skip if already done
        if args.skip_existing:
            config = load_config_with_inheritance(str(project_root / config_path))
            output_dir = Path(config.get("output_dir", f"outputs/{mode}"))
            results_file = output_dir / "results.json"
            if results_file.exists():
                print(f"[SKIP-EXISTING] {mode}: results.json already exists at {results_file}")
                try:
                    with open(results_file) as f:
                        all_results.append(json.load(f))
                except Exception as e:
                    print(f"  Error reading {results_file}: {e}")
                continue
        
        # Run the ablation
        try:
            result = run_single_ablation(
                mode=mode,
                config_path=config_path,
                project_root=project_root,
                smoke_test=args.smoke_test,
                max_epochs_override=args.max_epochs,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[FATAL ERROR] {mode}: {e}")
            traceback.print_exc()
            if not args.continue_on_error:
                print("Aborting. Use --continue-on-error to skip failures.")
                sys.exit(1)
            all_results.append({
                "mode": mode,
                "status": "fatal_error",
                "error": str(e),
            })
    
    # Save aggregated results
    aggregate_path = Path(args.results_dir) / "all_ablations.json"
    with open(aggregate_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print(f"ALL ABLATIONS COMPLETE")
    print(f"Aggregated results: {aggregate_path}")
    print(f"{'=' * 70}\n")
    
    # Print summary table
    print_summary_table(all_results)


def print_summary_table(results: List[Dict]):
    """Print a markdown-formatted summary table to stdout."""
    print("| Ablation Mode | Status | Total Params | Trainable | Test F1-macro | Test AUC |")
    print("|---|---|---|---|---|---|")
    for r in results:
        if r.get("status") != "success":
            print(f"| {r['mode']} | {r.get('status', '?')} | - | - | - | - |")
            continue
        
        v = r.get("verification", {})
        t = r.get("test_metrics", {})
        print(f"| {r['mode']} | OK | {v.get('total_params', 0):,} | "
              f"{v.get('trainable_params', 0):,} | "
              f"{t.get('f1_macro', 0):.4f} | {t.get('auc_roc', 0):.4f} |")


if __name__ == "__main__":
    main()
