"""
Diagnose why contrastive loss appears to be stuck at 0.0000.

Run:
    python scripts/debug_contrastive.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from src.losses.combined_loss import CombinedLoss
from src.models import build_model
from src.utils.config import load_config_with_inheritance


def main() -> None:
    config = load_config_with_inheritance("configs/model/multimodal.yaml")
    config.setdefault("training", {})["device"] = "cpu"

    model = build_model(config).eval()
    loss_fn = CombinedLoss(
        contrastive_lambda=config.get("loss", {}).get("contrastive_lambda", 0.1),
        contrastive_temperature_init=config.get("loss", {}).get("contrastive_temperature_init", 0.07),
        ablation_mode=config.get("ablation_mode", "full"),
        cls_loss_type=config.get("loss", {}).get("cls_loss_type", "bce"),
        pos_weight=0.6288,
    )

    batch_size = 8
    metadata_dim = len(config.get("metadata_features", []))
    batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, 256)),
        "attention_mask": torch.ones(batch_size, 256, dtype=torch.long),
        "pixel_values": torch.randn(batch_size, 3, 224, 224),
        "metadata": torch.randn(batch_size, metadata_dim),
        "label": torch.randint(0, 2, (batch_size,)),
        "valid_mask": torch.ones(batch_size, dtype=torch.bool),
    }

    print("=" * 70)
    print("DEBUG: Contrastive Loss Forward")
    print("=" * 70)

    output = model(batch)
    print(f"\n[1] Model output keys: {list(output.keys())}")

    t_proj = output.get("t_proj")
    i_proj = output.get("i_proj")
    if t_proj is None or i_proj is None:
        print("    ✗ Missing t_proj or i_proj from model output")
        return

    print("\n[2] Projection statistics:")
    print(f"    t_proj: shape={tuple(t_proj.shape)}, requires_grad={t_proj.requires_grad}")
    print(f"            norm.mean={t_proj.norm(dim=-1).mean():.4f}")
    print(f"            std={t_proj.std():.4f}")
    print(f"            any_nan={torch.isnan(t_proj).any().item()}")
    print(f"    i_proj: shape={tuple(i_proj.shape)}, requires_grad={i_proj.requires_grad}")
    print(f"            norm.mean={i_proj.norm(dim=-1).mean():.4f}")
    print(f"            std={i_proj.std():.4f}")
    print(f"            any_nan={torch.isnan(i_proj).any().item()}")

    print("\n[3] Manual InfoNCE (bypass class):")
    t_norm = F.normalize(t_proj, p=2, dim=-1)
    i_norm = F.normalize(i_proj, p=2, dim=-1)
    sim = t_norm @ i_norm.T
    logit_scale = loss_fn.contrastive.logit_scale.exp().item() if loss_fn.contrastive is not None else 1.0 / 0.07
    logits = logit_scale * sim
    manual_loss = 0.5 * (
        F.cross_entropy(logits, torch.arange(batch_size))
        + F.cross_entropy(logits.T, torch.arange(batch_size))
    )
    print(f"    Manual InfoNCE loss: {manual_loss.item():.4f}")
    print(f"    Expected (≈ log(B) for random): {torch.log(torch.tensor(float(batch_size))).item():.4f}")

    print("\n[4] Through CombinedLoss.forward():")
    loss_dict = loss_fn(
        logits=output["logits"],
        labels=batch["label"],
        text_emb=t_proj,
        image_emb=i_proj,
        valid_mask=output.get("image_valid", None),
        is_multimodal=output.get("is_multimodal", True),
    )
    print(f"    loss_dict['loss']: {loss_dict['loss'].item():.4f}")
    print(f"    loss_dict['cls_loss']: {loss_dict['cls_loss'].item():.4f}")
    print(f"    loss_dict['con_loss']: {0.0 if loss_dict['con_loss'] is None else loss_dict['con_loss'].item():.6f}")

    print("\n[5] valid_mask check:")
    vm = batch.get("valid_mask")
    if vm is None:
        print("    valid_mask is None")
    else:
        print(f"    valid_mask: dtype={vm.dtype}, sum={vm.sum().item()}/{len(vm)}")

    print("\n[6] Forward without valid_mask:")
    batch_no_mask = {k: v for k, v in batch.items() if k != "valid_mask"}
    output_no_mask = model(batch_no_mask)
    loss_dict_no_mask = loss_fn(
        logits=output_no_mask["logits"],
        labels=batch_no_mask["label"],
        text_emb=output_no_mask.get("t_proj", None),
        image_emb=output_no_mask.get("i_proj", None),
        valid_mask=output_no_mask.get("image_valid", None),
        is_multimodal=output_no_mask.get("is_multimodal", True),
    )
    print(f"    con (no valid_mask): {0.0 if loss_dict_no_mask['con_loss'] is None else loss_dict_no_mask['con_loss'].item():.4f}")


if __name__ == "__main__":
    main()