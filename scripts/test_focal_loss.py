#!/usr/bin/env python3
"""
Verification tests for Focal Loss implementations (FIX 4).

Tests the Focal Loss implementations to ensure:
1. FocalLoss with γ=0 reduces to BCE
2. FocalLoss down-weights easy examples
3. AsymmetricFocalLoss emphasizes FN over FP
4. All loss types produce finite, differentiable outputs
5. Config loading works for all loss types
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.losses.combined_loss import (
    FocalLossWithLogits,
    AsymmetricFocalLossWithLogits,
    CombinedLoss,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def test_focal_gamma_0_equals_bce():
    """Test 1: Focal Loss with γ=0 reduces to standard BCE."""
    logger.info("=" * 70)
    logger.info("TEST 1: Focal Loss with γ=0 Reduces to BCE")
    logger.info("=" * 70)
    
    # Focal loss with γ=0, α=0.5
    focal_loss = FocalLossWithLogits(alpha=0.5, gamma=0.0, reduction="none")
    
    # Standard BCE (weighted by alpha)
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    
    logits = torch.randn(100, requires_grad=True)
    labels = torch.randint(0, 2, (100,)).float()
    
    focal_out = focal_loss(logits, labels)
    bce_out = bce_loss(logits, labels)
    
    # Focal with α=0.5 and γ=0 should be 0.5 * BCE
    ratio = focal_out / (bce_out + 1e-8)
    
    # Check that ratio is close to 0.5 (accounting for numerical precision)
    expected_ratio = 0.5 * torch.ones_like(ratio)
    close_enough = torch.allclose(ratio, expected_ratio, atol=0.01)
    
    logger.info(f"  Focal (γ=0) / BCE ratio: mean={ratio.mean():.4f}, std={ratio.std():.4f}")
    logger.info(f"  Expected ratio: 0.5 (within tolerance)")
    
    assert close_enough, f"Focal γ=0 should equal 0.5*BCE, got ratio {ratio.mean():.4f}"
    
    logger.info("✓ PASSED: Focal Loss with γ=0 equals 0.5*BCE\n")
    return True


def test_focal_downweights_easy_examples():
    """Test 2: Focal Loss produces smaller loss for high-confidence correct predictions."""
    logger.info("=" * 70)
    logger.info("TEST 2: Focal Loss Down-Weights Easy Examples")
    logger.info("=" * 70)
    
    focal_loss = FocalLossWithLogits(alpha=0.5, gamma=2.0, reduction="none")
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    
    # Very confident correct predictions (easy examples)
    logits_easy = torch.tensor([10.0, -10.0])
    labels_easy = torch.tensor([1.0, 0.0])
    
    focal_easy = focal_loss(logits_easy, labels_easy)
    bce_easy = bce_loss(logits_easy, labels_easy)
    
    logger.info(f"  Easy examples (high confidence correct):")
    logger.info(f"    logits: {logits_easy.tolist()}")
    logger.info(f"    labels: {labels_easy.tolist()}")
    logger.info(f"    BCE loss: {bce_easy.tolist()}")
    logger.info(f"    Focal loss: {focal_easy.tolist()}")
    logger.info(f"    Focal/BCE ratio: {(focal_easy / (bce_easy + 1e-8)).tolist()}")
    
    # Focal should be significantly smaller (down-weighted)
    ratio = focal_easy / (bce_easy + 1e-8)
    assert (ratio < 0.1).all(), f"Focal should be << BCE for easy examples, got ratio {ratio.tolist()}"
    
    logger.info("✓ PASSED: Focal Loss down-weights easy examples\n")
    return True


def test_asymmetric_emphasizes_fn_over_fp():
    """Test 3: Asymmetric Focal Loss gives stronger gradient for missed positives (FN) than false positives (FP)."""
    logger.info("=" * 70)
    logger.info("TEST 3: Asymmetric Focal Loss Emphasizes FN over FP")
    logger.info("=" * 70)
    
    asym_loss = AsymmetricFocalLossWithLogits(
        alpha=0.5,
        gamma_pos=1.0,
        gamma_neg=4.0,
        clip=0.05,
        reduction="none",
    )
    
    # Misclassified positive (predicts 0.1, true=1) — False Negative
    # Convert prob to logit: logit = log(p / (1-p))
    p_fn = 0.1
    logit_fn = np.log(p_fn / (1 - p_fn))
    logits_fn = torch.tensor([logit_fn], requires_grad=True)
    labels_fn = torch.tensor([1.0])
    
    # Misclassified negative (predicts 0.9, true=0) — False Positive
    p_fp = 0.9
    logit_fp = np.log(p_fp / (1 - p_fp))
    logits_fp = torch.tensor([logit_fp], requires_grad=True)
    labels_fp = torch.tensor([0.0])
    
    # Compute losses
    loss_fn = asym_loss(logits_fn, labels_fn)
    loss_fp = asym_loss(logits_fp, labels_fp)
    
    # Backward to get gradients
    loss_fn.backward()
    loss_fp.backward()
    
    grad_fn = logits_fn.grad.item()
    grad_fp = logits_fp.grad.item()
    
    logger.info(f"  Missed positive (FN): p={p_fn}, logit={logit_fn:.4f}, loss={loss_fn.item():.4f}, grad={grad_fn:.4f}")
    logger.info(f"  False positive (FP): p={p_fp}, logit={logit_fp:.4f}, loss={loss_fp.item():.4f}, grad={grad_fp:.4f}")
    logger.info(f"  |grad_FN| / |grad_FP| = {abs(grad_fn) / (abs(grad_fp) + 1e-8):.2f}x")
    
    # FN should have stronger gradient magnitude (higher γ_pos than γ_neg effect)
    assert abs(grad_fn) > abs(grad_fp), \
        f"Asymmetric focal should penalize missed positives (FN) more than false positives (FP)"
    
    logger.info("✓ PASSED: Asymmetric Focal Loss emphasizes FN over FP\n")
    return True


def test_no_nan_or_inf():
    """Test 4: All loss types produce finite, non-NaN outputs."""
    logger.info("=" * 70)
    logger.info("TEST 4: No NaN or Inf in Loss Output")
    logger.info("=" * 70)
    
    logits = torch.randn(100, requires_grad=True)
    labels = torch.randint(0, 2, (100,)).float()
    
    losses = [
        ("BCE", nn.BCEWithLogitsLoss()),
        ("Focal (γ=2)", FocalLossWithLogits(alpha=0.5, gamma=2.0)),
        ("Asymmetric Focal", AsymmetricFocalLossWithLogits(
            alpha=0.5, gamma_pos=1.0, gamma_neg=4.0, clip=0.05
        )),
    ]
    
    for name, loss_fn in losses:
        output = loss_fn(logits.detach(), labels)
        
        assert torch.isfinite(output), f"{name}: loss is not finite: {output.item()}"
        assert not torch.isnan(output), f"{name}: loss is NaN"
        assert not torch.isinf(output), f"{name}: loss is Inf"
        
        logger.info(f"  {name:20s}: loss={output.item():.6f} ✓")
    
    logger.info("✓ PASSED: All loss types produce finite outputs\n")
    return True


def test_gradients_flow():
    """Test 5: Gradients flow correctly through all loss types."""
    logger.info("=" * 70)
    logger.info("TEST 5: Gradients Flow Correctly")
    logger.info("=" * 70)
    
    logits = torch.randn(32, requires_grad=True)
    labels = torch.randint(0, 2, (32,)).float()
    
    losses = [
        ("BCE", nn.BCEWithLogitsLoss()),
        ("Focal (γ=2)", FocalLossWithLogits(alpha=0.5, gamma=2.0)),
        ("Asymmetric Focal", AsymmetricFocalLossWithLogits(
            alpha=0.5, gamma_pos=1.0, gamma_neg=4.0, clip=0.05
        )),
    ]
    
    for name, loss_fn in losses:
        logits_copy = logits.detach().requires_grad_(True)
        output = loss_fn(logits_copy, labels)
        output.backward()
        
        assert logits_copy.grad is not None, f"{name}: gradient is None"
        assert torch.any(logits_copy.grad != 0), f"{name}: gradient is all zeros"
        assert torch.all(torch.isfinite(logits_copy.grad)), f"{name}: gradient contains NaN/Inf"
        
        grad_norm = logits_copy.grad.norm().item()
        logger.info(f"  {name:20s}: grad_norm={grad_norm:.6f} ✓")
    
    logger.info("✓ PASSED: Gradients flow correctly\n")
    return True


def test_combined_loss_all_types():
    """Test 6: CombinedLoss works with all classification loss types."""
    logger.info("=" * 70)
    logger.info("TEST 6: CombinedLoss with All Classification Loss Types")
    logger.info("=" * 70)
    
    class_weights = torch.tensor([0.88, 1.12])
    
    loss_types = [
        ("bce", {"cls_loss_type": "bce"}),
        ("focal", {"cls_loss_type": "focal", "focal_gamma": 2.0, "focal_alpha": 0.5}),
        ("asymmetric_focal", {
            "cls_loss_type": "asymmetric_focal",
            "focal_alpha": 0.5,
            "focal_gamma_pos": 1.0,
            "focal_gamma_neg": 4.0,
            "focal_clip": 0.05,
        }),
    ]
    
    batch_size = 32
    logits = torch.randn(batch_size, 1, requires_grad=True)
    labels = torch.randint(0, 2, (batch_size,)).float()
    text_emb = torch.randn(batch_size, 256, requires_grad=True)
    image_emb = torch.randn(batch_size, 256, requires_grad=True)
    
    for name, kwargs in loss_types:
        loss_fn = CombinedLoss(
            class_weights=class_weights,
            contrastive_lambda=0.1,
            **kwargs
        )
        
        logits_copy = logits.detach().requires_grad_(True)
        text_copy = text_emb.detach().requires_grad_(True)
        image_copy = image_emb.detach().requires_grad_(True)
        
        output = loss_fn(logits_copy, labels, text_copy, image_copy)
        
        assert "loss" in output, f"{name}: missing 'loss' key"
        assert torch.isfinite(output["loss"]), f"{name}: loss is not finite"
        
        output["loss"].backward()
        
        assert logits_copy.grad is not None, f"{name}: logits grad is None"
        assert torch.any(logits_copy.grad != 0), f"{name}: logits grad is all zeros"
        
        logger.info(f"  {name:20s}: total_loss={output['loss'].item():.6f}, "
                   f"cls_loss={output['cls_loss'].item():.6f}, "
                   f"con_loss={output['con_loss'].item():.6f} ✓")
    
    logger.info("✓ PASSED: CombinedLoss works with all loss types\n")
    return True


def test_focal_alpha_effect():
    """Test 7: focal_alpha parameter controls positive class emphasis."""
    logger.info("=" * 70)
    logger.info("TEST 7: focal_alpha Parameter Effect")
    logger.info("=" * 70)
    
    logits = torch.randn(100)
    labels = torch.randint(0, 2, (100,)).float()
    
    # Test different alpha values
    alphas = [0.3, 0.5, 0.7]
    losses = []
    
    for alpha in alphas:
        loss_fn = FocalLossWithLogits(alpha=alpha, gamma=2.0)
        loss_val = loss_fn(logits, labels).item()
        losses.append(loss_val)
        logger.info(f"  alpha={alpha}: loss={loss_val:.6f}")
    
    # Loss should vary with alpha (not constant)
    assert not np.allclose(losses, losses[0]), "Loss should vary with alpha parameter"
    
    logger.info("✓ PASSED: focal_alpha parameter affects loss\n")
    return True


def test_focal_gamma_effect():
    """Test 8: focal_gamma parameter controls easy-example down-weighting."""
    logger.info("=" * 70)
    logger.info("TEST 8: focal_gamma Parameter Effect")
    logger.info("=" * 70)
    
    # Very confident correct prediction (easy example)
    logits = torch.tensor([10.0])
    labels = torch.tensor([1.0])
    
    gammas = [0.0, 1.0, 2.0, 4.0]
    losses = []
    
    for gamma in gammas:
        loss_fn = FocalLossWithLogits(alpha=0.5, gamma=gamma)
        loss_val = loss_fn(logits, labels).item()
        losses.append(loss_val)
        logger.info(f"  gamma={gamma}: loss={loss_val:.6f}")
    
    # Higher gamma should produce lower loss for easy examples (more down-weighting)
    for i in range(len(gammas) - 1):
        assert losses[i] > losses[i + 1], \
            f"Higher gamma should produce lower loss for easy examples"
    
    logger.info("✓ PASSED: Higher gamma down-weights easy examples\n")
    return True


def main():
    """Run all verification tests."""
    logger.info("\n" + "=" * 70)
    logger.info("FOCAL LOSS FOR ASYMMETRIC MISCLASSIFICATION COST - VERIFICATION TESTS")
    logger.info("=" * 70 + "\n")
    
    tests = [
        ("Focal γ=0 Reduces to BCE", test_focal_gamma_0_equals_bce),
        ("Focal Down-Weights Easy Examples", test_focal_downweights_easy_examples),
        ("Asymmetric Focal Emphasizes FN over FP", test_asymmetric_emphasizes_fn_over_fp),
        ("No NaN or Inf in Output", test_no_nan_or_inf),
        ("Gradients Flow Correctly", test_gradients_flow),
        ("CombinedLoss All Types", test_combined_loss_all_types),
        ("focal_alpha Parameter Effect", test_focal_alpha_effect),
        ("focal_gamma Parameter Effect", test_focal_gamma_effect),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ FAILED: {test_name}")
            logger.error(f"  Error: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Focal Loss is working correctly.\n")
        return 0
    else:
        logger.error(f"\n⚠ {total - passed} test(s) failed. Please review the errors above.\n")
        return 1


if __name__ == "__main__":
    exit(main())
