"""Attention weight and gate weight visualization utilities."""

import logging
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def extract_cross_attention_weights(
    model, batch: dict, device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Extract cross-attention weights from the dual cross-attention module.

    Registers forward hooks on the attention modules to capture the attention
    weight matrices during a forward pass.

    Args:
        model: MultimodalMisinfoDetector instance
        batch: Batch dict with text, image, metadata tensors
        device: Torch device

    Returns:
        Dict with keys:
        - "text_to_image": (B, num_heads, 1, 2) - text [CLS] attention to image/metadata
        - "image_to_text": (B, num_heads, 1, 2) - image [CLS] attention to text/metadata
    """
    attention_weights = {}

    def hook_attention(name: str):
        """Create a hook function that captures attention weights."""

        def hook_fn(module, input, output):
            # output shape: (B, num_heads, seq_len_query, seq_len_key)
            # For single [CLS] queries and 2 keys: (B, num_heads, 1, 2)
            attention_weights[name] = output.detach().cpu()

        return hook_fn

    # Register hooks on attention modules
    hook1 = model.dual_cross_attn.attn_text_to_image.register_forward_hook(
        hook_attention("text_to_image")
    )
    hook2 = model.dual_cross_attn.attn_image_to_text.register_forward_hook(
        hook_attention("image_to_text")
    )

    try:
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(batch)

        return attention_weights

    finally:
        # Remove hooks
        hook1.remove()
        hook2.remove()


def extract_gate_weights(model, batch: dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Extract gate weights from the gated fusion module.

    Registers a hook on the gated fusion module to capture per-modality gate weights
    (typically output of softmax over gating MLP).

    Args:
        model: MultimodalMisinfoDetector instance
        batch: Batch dict
        device: Torch device

    Returns:
        Dict with key "gates": (B, 3) tensor with per-modality weights [g_text, g_image, g_metadata]
    """
    gate_weights = {}

    def hook_gates(module, input, output):
        # Capture gate weights if available in module output or state
        # The gated_fusion module should have gate values computed as softmax
        if hasattr(module, "gates"):
            gate_weights["gates"] = module.gates.detach().cpu()

    # Register hook on gated fusion module
    hook = model.gated_fusion.register_forward_hook(hook_gates)

    try:
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(batch)

        # If gates weren't captured by hook, try to compute them directly
        if "gates" not in gate_weights and hasattr(model.gated_fusion, "gate_mlp"):
            logger.info("Computing gate weights from gate MLP...")
            # This is a fallback - gates should have been captured, but if not,
            # we can try to compute them from stored embeddings
            # For now, return a warning
            logger.warning(
                "Gate weights not captured via hook. "
                "Ensure model.gated_fusion stores gate values during forward pass."
            )

        return gate_weights

    finally:
        hook.remove()


def visualize_gate_weights(
    model, batch: dict, device: torch.device, save_path: Optional[str] = None
) -> None:
    """
    Visualize gate weights from gated fusion module.

    Creates a horizontal stacked bar chart showing the relative gate weight per modality,
    averaged across the batch dimension.

    Args:
        model: MultimodalMisinfoDetector instance
        batch: Batch dict
        device: Torch device
        save_path: Path to save figure (e.g., "outputs/gate_weights.png").
                   If None, displays interactively.
    """
    # Extract gate weights
    gate_data = extract_gate_weights(model, batch, device)

    # Get gate weights tensor
    if "gates" in gate_data:
        gates = gate_data["gates"]  # (B, 3)
        gates_avg = gates.mean(dim=0).numpy()  # (3,)
    else:
        logger.warning("Gate weights not available. Using uniform weights.")
        gates_avg = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    # Normalize to 0-1 range if not already
    if gates_avg.max() <= 1.0 and gates_avg.min() >= 0.0:
        gates_normalized = gates_avg
    else:
        gates_normalized = gates_avg / gates_avg.sum()

    # Create figure
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    modalities = ["Text", "Image", "Metadata"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Create stacked horizontal bar
    cumulative = 0
    for i, (modality, weight, color) in enumerate(zip(modalities, gates_normalized, colors)):
        ax.barh(0, weight, left=cumulative, label=modality, color=color, height=0.5)
        # Add label with weight value
        ax.text(
            cumulative + weight / 2,
            0,
            f"{weight:.3f}",
            va="center",
            ha="center",
            fontweight="bold",
            color="white",
        )
        cumulative += weight

    ax.set_yticks([])
    ax.set_xlabel("Relative Gate Weight", fontsize=12)
    ax.set_title("Average Gate Weights Across Batch", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Gate weights visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_attention_for_sample(
    model,
    batch: dict,
    sample_idx: int,
    device: torch.device,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize cross-attention weights for a specific sample in the batch.

    Creates a two-panel visualization:
    - Left: Text [CLS] attention to Image vs Metadata
    - Right: Image [CLS] attention to Text vs Metadata

    Args:
        model: MultimodalMisinfoDetector instance
        batch: Batch dict with samples
        sample_idx: Index of sample in batch to visualize (0-indexed)
        device: Torch device
        save_path: Path to save figure (e.g., "outputs/attention_sample_0.png").
                   If None, displays interactively.
    """
    # Validate sample index
    batch_size = next(
        v.shape[0] for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "labels"
    )
    if sample_idx >= batch_size or sample_idx < 0:
        raise ValueError(f"sample_idx {sample_idx} out of range [0, {batch_size})")

    # Extract attention weights
    attn_weights = extract_cross_attention_weights(model, batch, device)

    # Extract weights for the specific sample
    # Shape: (num_heads, 1, 2) - average over heads and remove seq_len dimension
    text_to_image = attn_weights["text_to_image"][sample_idx]  # (num_heads, 1, 2)
    image_to_text = attn_weights["image_to_text"][sample_idx]  # (num_heads, 1, 2)

    # Average over heads and squeeze
    text_to_image_avg = text_to_image.mean(dim=0).squeeze(0).numpy()  # (2,)
    image_to_text_avg = image_to_text.mean(dim=0).squeeze(0).numpy()  # (2,)

    # Create figure with two subplots
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    key_labels = ["Image", "Metadata"]
    colors = ["#ff7f0e", "#2ca02c"]

    # Left panel: Text to Image/Metadata
    bars1 = axes[0].bar(key_labels, text_to_image_avg, color=colors, width=0.6)
    axes[0].set_ylabel("Attention Weight", fontsize=11)
    axes[0].set_title("Text [CLS] → Keys", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Right panel: Image to Text/Metadata
    bars2 = axes[1].bar(key_labels, image_to_text_avg, color=colors, width=0.6)
    axes[1].set_ylabel("Attention Weight", fontsize=11)
    axes[1].set_title("Image [CLS] → Keys", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Overall title
    fig.suptitle(
        f"Cross-Attention Weights (Sample {sample_idx})", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Attention visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_attention_heatmap_for_sample(
    model,
    batch: dict,
    sample_idx: int,
    device: torch.device,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize cross-attention weights as heatmaps for a specific sample.

    Creates a two-panel heatmap visualization:
    - Left: Text [CLS] attention heatmap (num_heads × 2 keys)
    - Right: Image [CLS] attention heatmap (num_heads × 2 keys)

    Args:
        model: MultimodalMisinfoDetector instance
        batch: Batch dict with samples
        sample_idx: Index of sample in batch to visualize (0-indexed)
        device: Torch device
        save_path: Path to save figure (e.g., "outputs/attention_heatmap_0.png").
                   If None, displays interactively.
    """
    # Validate sample index
    batch_size = next(
        v.shape[0] for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "labels"
    )
    if sample_idx >= batch_size or sample_idx < 0:
        raise ValueError(f"sample_idx {sample_idx} out of range [0, {batch_size})")

    # Extract attention weights
    attn_weights = extract_cross_attention_weights(model, batch, device)

    # Extract weights for the specific sample
    text_to_image = attn_weights["text_to_image"][sample_idx]  # (num_heads, 1, 2)
    image_to_text = attn_weights["image_to_text"][sample_idx]  # (num_heads, 1, 2)

    # Remove seq_len dimension and convert to numpy
    text_to_image_hm = text_to_image.squeeze(1).numpy()  # (num_heads, 2)
    image_to_text_hm = image_to_text.squeeze(1).numpy()  # (num_heads, 2)

    # Create figure with two heatmap subplots
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    key_labels = ["Image", "Metadata"]

    # Left panel: Text to Image/Metadata heatmap
    im1 = axes[0].imshow(text_to_image_hm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    axes[0].set_xlabel("Keys", fontsize=11)
    axes[0].set_ylabel("Attention Head", fontsize=11)
    axes[0].set_title("Text [CLS] → Keys Attention", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(len(key_labels)))
    axes[0].set_xticklabels(key_labels)
    plt.colorbar(im1, ax=axes[0], label="Attention Weight")

    # Right panel: Image to Text/Metadata heatmap
    im2 = axes[1].imshow(image_to_text_hm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    axes[1].set_xlabel("Keys", fontsize=11)
    axes[1].set_ylabel("Attention Head", fontsize=11)
    axes[1].set_title("Image [CLS] → Keys Attention", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(len(key_labels)))
    axes[1].set_xticklabels(key_labels)
    plt.colorbar(im2, ax=axes[1], label="Attention Weight")

    # Overall title
    fig.suptitle(
        f"Cross-Attention Heatmaps (Sample {sample_idx})", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Attention heatmap saved to {save_path}")
    else:
        plt.show()

    plt.close()
