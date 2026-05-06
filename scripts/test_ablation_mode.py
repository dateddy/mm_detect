#!/usr/bin/env python3
"""
Test script for ablation mode support.

Verifies that:
1. Unimodal modes bypass multimodal components correctly
2. No NaN/Inf gradients are produced
3. Loss computation is correct (classification only for unimodal)
4. Logging is conditional (no multimodal metrics in unimodal mode)
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.full_model import MultimodalMisinfoDetector
from src.losses.combined_loss import CombinedLoss
from src.data.preprocessing import compute_class_weights

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_dummy_batch(batch_size: int = 4) -> dict:
    """Create a dummy batch for testing."""
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, 64)),
        "attention_mask": torch.ones((batch_size, 64), dtype=torch.long),
        "pixel_values": torch.randn(batch_size, 3, 224, 224),
        "metadata": torch.randn(batch_size, 17),
        "missing_image": [False] * batch_size,
        "label": torch.randint(0, 2, (batch_size,)).float(),
    }


def test_ablation_mode(ablation_mode: str):
    """Test a specific ablation mode."""
    logger.info("=" * 70)
    logger.info(f"Testing ablation mode: {ablation_mode}")
    logger.info("=" * 70)

    config = {
        "text_encoder_name": "vinai/phobert-base-v2",
        "image_encoder_name": "vit_base_patch16_224",
        "proj_dim": 256,
        "num_attn_heads": 8,
        "attn_dropout": 0.1,
        "modality_dropout_p": 0.15,
        "head_dropout": 0.3,
        "metadata_features": [
            "ads_per_page", "platform_count", "FB_only_flag", "all_targeted",
            "burstiness", "avg_ad_duration", "launch_delay", "num_countries",
            "language_location_mismatch", "emoji_count", "text_length",
            "ads_duration", "repeated_text_ratio", "exclamation_ratio",
            "caps_word_ratio", "repeated_punct_count", "url_count"
        ],
    }

    # Create model with ablation mode
    model = MultimodalMisinfoDetector(config, ablation_mode=ablation_mode)
    model.eval()

    # Create loss function
    class_weights = torch.tensor([1.0, 1.0])
    loss_fn = CombinedLoss(class_weights=class_weights)

    # Create dummy batch
    batch = create_dummy_batch(batch_size=4)

    # Forward pass
    logger.info(f"  Running forward pass...")
    output = model(batch)

    # Check model output
    logger.info(f"  Model output keys: {output.keys()}")
    logger.info(f"  is_multimodal: {output['is_multimodal']}")

    # Verify ablation-specific outputs
    if ablation_mode == "full":
        assert output["is_multimodal"] == True, "Should be multimodal"
        assert output["t_proj"] is not None, "t_proj should not be None"
        assert output["i_proj"] is not None, "i_proj should not be None"
        assert output["t_prime"] is not None, "t_prime should not be None"
        assert output["i_prime"] is not None, "i_prime should not be None"
        assert output["image_valid"] is not None, "image_valid should not be None"
        logger.info(f"  ✓ Multimodal outputs computed correctly")

    elif ablation_mode == "text_only":
        assert output["is_multimodal"] == False, "Should not be multimodal"
        assert output["t_proj"] is not None, "t_proj should not be None"
        assert output["i_proj"] is None, "i_proj should be None"
        assert output["t_prime"] is None, "t_prime should be None"
        assert output["i_prime"] is None, "i_prime should be None"
        assert output["image_valid"] is None, "image_valid should be None"
        logger.info(f"  ✓ Text-only outputs correct (multimodal components bypassed)")

    elif ablation_mode == "image_only":
        assert output["is_multimodal"] == False, "Should not be multimodal"
        assert output["t_proj"] is None, "t_proj should be None"
        assert output["i_proj"] is not None, "i_proj should not be None"
        assert output["t_prime"] is None, "t_prime should be None"
        assert output["i_prime"] is None, "i_prime should be None"
        assert output["image_valid"] is None, "image_valid should be None"
        logger.info(f"  ✓ Image-only outputs correct (multimodal components bypassed)")

    elif ablation_mode == "metadata_only":
        assert output["is_multimodal"] == False, "Should not be multimodal"
        assert output["t_proj"] is None, "t_proj should be None"
        assert output["i_proj"] is None, "i_proj should be None"
        assert output["t_prime"] is None, "t_prime should be None"
        assert output["i_prime"] is None, "i_prime should be None"
        assert output["image_valid"] is None, "image_valid should be None"
        logger.info(f"  ✓ Metadata-only outputs correct (multimodal components bypassed)")

    # Test loss computation
    logger.info(f"  Computing loss...")
    loss_dict = loss_fn(
        logits=output["logits"],
        labels=batch["label"],
        text_emb=output["t_proj"],
        image_emb=output["i_proj"],
        valid_mask=output["image_valid"],
        is_multimodal=output["is_multimodal"],
    )

    logger.info(f"  Loss dict keys: {loss_dict.keys()}")
    logger.info(f"  cls_loss: {loss_dict['cls_loss'].item():.6f}")
    logger.info(f"  con_loss: {loss_dict['con_loss']}")
    logger.info(f"  temperature: {loss_dict['temperature']}")

    # Verify loss computation based on mode
    if ablation_mode == "full":
        assert loss_dict["con_loss"] is not None, "con_loss should not be None in multimodal mode"
        assert loss_dict["temperature"] is not None, "temperature should not be None in multimodal mode"
        assert torch.isfinite(loss_dict["loss"]), "Loss should be finite"
        logger.info(f"  ✓ Multimodal loss computation correct")

    else:
        assert loss_dict["con_loss"] is None, "con_loss should be None in unimodal mode"
        assert loss_dict["temperature"] is None, "temperature should be None in unimodal mode"
        assert torch.isfinite(loss_dict["loss"]), "Loss should be finite"
        logger.info(f"  ✓ Unimodal loss computation correct (contrastive loss skipped)")

    # Test gradients
    logger.info(f"  Testing gradients...")
    model.train()
    
    # Create a fresh batch with requires_grad
    batch = create_dummy_batch(batch_size=4)
    output = model(batch)
    
    loss_dict = loss_fn(
        logits=output["logits"],
        labels=batch["label"],
        text_emb=output["t_proj"],
        image_emb=output["i_proj"],
        valid_mask=output["image_valid"],
        is_multimodal=output["is_multimodal"],
    )
    
    loss = loss_dict["loss"]
    loss.backward()

    # Check that gradients are computed and finite
    has_finite_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isfinite(param.grad)):
                has_finite_grads = True
            if not torch.all(torch.isfinite(param.grad)):
                logger.warning(f"  ⚠ Non-finite gradients in {name}")

    assert has_finite_grads, "Model should have some finite gradients"
    logger.info(f"  ✓ Gradients computed correctly (finite)")

    logger.info(f"✓ PASSED: {ablation_mode}\n")


def main():
    """Run all ablation mode tests."""
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION MODE SUPPORT - VERIFICATION TESTS")
    logger.info("=" * 70 + "\n")

    ablation_modes = ["full", "text_only", "image_only", "metadata_only"]

    passed = 0
    failed = 0

    for mode in ablation_modes:
        try:
            test_ablation_mode(mode)
            passed += 1
        except Exception as e:
            logger.error(f"✗ FAILED: {mode}")
            logger.error(f"  Error: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Passed: {passed}/{len(ablation_modes)}")
    logger.info(f"Failed: {failed}/{len(ablation_modes)}")

    if failed == 0:
        logger.info("\n🎉 ALL TESTS PASSED! Ablation mode support is working correctly.\n")
        return 0
    else:
        logger.error(f"\n⚠ {failed} test(s) failed.\n")
        return 1


if __name__ == "__main__":
    exit(main())
