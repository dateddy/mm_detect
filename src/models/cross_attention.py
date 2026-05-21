"""
Fixed Dual Bidirectional Cross-Attention Module.

Key architectural fixes:
1. Self-reference: query modality is included in key/value context
2. Strong residual: output = LayerNorm(input + gate * attention_output)
3. Learnable per-dimension gates: model controls attention's influence per dimension
4. Small-init output projection: cross-attention starts as near-identity
5. Diagnostic outputs: gate stats and attention weights exposed for logging
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class DualBidirectionalCrossAttention(nn.Module):
    """
    Cross-attention where each modality attends to the full context including itself.

    Args:
        embed_dim: dimensionality of input projections, e.g. 256
        num_heads: number of attention heads
        dropout: attention dropout probability
        gate_init_logit: initial value for gate logits. Default -2.0 gives
            sigmoid(-2.0) ~= 0.12, biasing toward query preservation.
        attn_init_scale: scale factor for output projection initialization.
            Default 0.1 makes attention near-zero at init.
        use_residual: whether to add residual connection.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_init_logit: float = -2.0,
        attn_init_scale: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_residual = use_residual

        self.text_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_text_to_image = self.text_attn
        self.attn_image_to_text = self.image_attn

        self.text_gate_logits = nn.Parameter(torch.full((embed_dim,), gate_init_logit))
        self.image_gate_logits = nn.Parameter(torch.full((embed_dim,), gate_init_logit))

        self.text_norm = nn.LayerNorm(embed_dim)
        self.image_norm = nn.LayerNorm(embed_dim)

        self._initialize_attention_output(attn_init_scale)

        self.register_buffer("_gate_stats_text_mean", torch.tensor(0.0))
        self.register_buffer("_gate_stats_image_mean", torch.tensor(0.0))

    def _initialize_attention_output(self, scale: float) -> None:
        """Scale down output projections so attention contribution starts small."""
        with torch.no_grad():
            self.text_attn.out_proj.weight.mul_(scale)
            self.text_attn.out_proj.bias.zero_()
            self.image_attn.out_proj.weight.mul_(scale)
            self.image_attn.out_proj.bias.zero_()

    @property
    def text_gate(self) -> torch.Tensor:
        """Effective text gate values after sigmoid."""
        return torch.sigmoid(self.text_gate_logits)

    @property
    def image_gate(self) -> torch.Tensor:
        """Effective image gate values after sigmoid."""
        return torch.sigmoid(self.image_gate_logits)

    def forward(
        self,
        t_proj: torch.Tensor,
        i_proj: torch.Tensor,
        m_proj: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_proj: text projection [B, embed_dim]
            i_proj: image projection [B, embed_dim]
            m_proj: metadata projection [B, embed_dim] or None
            return_attention_weights: if True, also return attention weight tensors

        Returns:
            text_enriched and image_enriched. If return_attention_weights is True,
            also returns a dict containing attention weights.
        """
        t_q = t_proj.unsqueeze(1)
        i_q = i_proj.unsqueeze(1)

        if m_proj is not None:
            context_for_text = torch.stack([t_proj, i_proj, m_proj], dim=1)
            context_for_image = torch.stack([i_proj, t_proj, m_proj], dim=1)
        else:
            context_for_text = torch.stack([t_proj, i_proj], dim=1)
            context_for_image = torch.stack([i_proj, t_proj], dim=1)

        text_attn_out, text_attn_weights = self.text_attn(
            query=t_q,
            key=context_for_text,
            value=context_for_text,
            need_weights=return_attention_weights,
            average_attn_weights=False,
        )
        image_attn_out, image_attn_weights = self.image_attn(
            query=i_q,
            key=context_for_image,
            value=context_for_image,
            need_weights=return_attention_weights,
            average_attn_weights=False,
        )

        text_attn_out = text_attn_out.squeeze(1)
        image_attn_out = image_attn_out.squeeze(1)

        g_t = self.text_gate
        g_i = self.image_gate

        text_combined = (1.0 - g_t) * t_proj + g_t * text_attn_out
        image_combined = (1.0 - g_i) * i_proj + g_i * image_attn_out

        if self.use_residual:
            text_enriched = self.text_norm(text_combined + t_proj)
            image_enriched = self.image_norm(image_combined + i_proj)
        else:
            text_enriched = text_combined
            image_enriched = image_combined

        with torch.no_grad():
            self._gate_stats_text_mean.copy_(g_t.mean())
            self._gate_stats_image_mean.copy_(g_i.mean())

        if return_attention_weights:
            return text_enriched, image_enriched, {
                "text_attn_weights": text_attn_weights,
                "image_attn_weights": image_attn_weights,
            }

        return text_enriched, image_enriched

    def get_gate_stats(self) -> Dict[str, float]:
        """Return diagnostic gate statistics for logging."""
        with torch.no_grad():
            g_t = self.text_gate
            g_i = self.image_gate
            return {
                "text_gate_mean": g_t.mean().item(),
                "text_gate_min": g_t.min().item(),
                "text_gate_max": g_t.max().item(),
                "text_gate_std": g_t.std().item(),
                "image_gate_mean": g_i.mean().item(),
                "image_gate_min": g_i.min().item(),
                "image_gate_max": g_i.max().item(),
                "image_gate_std": g_i.std().item(),
            }

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"use_residual={self.use_residual}"
        )


class SelectiveCrossAttention(nn.Module):
    """
    Cross-attention with per-sample learnable gating.

    The wrapped DualBidirectionalCrossAttention still handles per-dimension
    gates. This module adds a scalar per-sample gate predicted from
    concat(t_proj, i_proj, m_proj), allowing attention to be bypassed when a
    sample does not benefit from multimodal context.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_init_logit: float = -2.0,
        attn_init_scale: float = 0.1,
        use_residual: bool = True,
        sample_gate_hidden: int = 64,
        sample_gate_init_bias: float = -1.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sample_gate_hidden = sample_gate_hidden

        self.cross_attn = DualBidirectionalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            gate_init_logit=gate_init_logit,
            attn_init_scale=attn_init_scale,
            use_residual=use_residual,
        )

        # Compatibility aliases for existing visualization hooks.
        self.attn_text_to_image = self.cross_attn.attn_text_to_image
        self.attn_image_to_text = self.cross_attn.attn_image_to_text

        self.sample_gate_predictor = nn.Sequential(
            nn.Linear(3 * embed_dim, sample_gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sample_gate_hidden, 1),
        )

        with torch.no_grad():
            self.sample_gate_predictor[-1].bias.fill_(sample_gate_init_bias)
            self.sample_gate_predictor[-1].weight.mul_(0.1)

        self.register_buffer("_last_sample_gate_mean", torch.tensor(0.0))
        self.register_buffer("_last_sample_gate_std", torch.tensor(0.0))
        self.register_buffer("_last_sample_gate_min", torch.tensor(0.0))
        self.register_buffer("_last_sample_gate_max", torch.tensor(0.0))
        self._last_sample_gates = None

    def forward(
        self,
        t_proj: torch.Tensor,
        i_proj: torch.Tensor,
        m_proj: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if m_proj is None:
            gate_m = torch.zeros_like(t_proj)
        else:
            gate_m = m_proj
        gate_input = torch.cat([t_proj, i_proj, gate_m], dim=-1)

        sample_gate = torch.sigmoid(self.sample_gate_predictor(gate_input))

        if return_attention_weights:
            text_attn, image_attn, attn_info = self.cross_attn(
                t_proj,
                i_proj,
                m_proj,
                return_attention_weights=True,
            )
        else:
            text_attn, image_attn = self.cross_attn(t_proj, i_proj, m_proj)

        text_enriched = sample_gate * text_attn + (1.0 - sample_gate) * t_proj
        image_enriched = sample_gate * image_attn + (1.0 - sample_gate) * i_proj

        with torch.no_grad():
            self._last_sample_gates = sample_gate.detach().squeeze(-1)
            self._last_sample_gate_mean.copy_(sample_gate.mean())
            self._last_sample_gate_std.copy_(sample_gate.std(unbiased=False))
            self._last_sample_gate_min.copy_(sample_gate.min())
            self._last_sample_gate_max.copy_(sample_gate.max())

        if return_attention_weights:
            return text_enriched, image_enriched, attn_info
        return text_enriched, image_enriched

    def get_sample_gate_stats(self) -> Dict[str, float]:
        """Return statistics about the last batch's per-sample gates."""
        return {
            "sample_gate_mean": self._last_sample_gate_mean.item(),
            "sample_gate_std": self._last_sample_gate_std.item(),
            "sample_gate_min": self._last_sample_gate_min.item(),
            "sample_gate_max": self._last_sample_gate_max.item(),
        }

    def get_gate_stats(self) -> Dict[str, float]:
        """Compatibility delegate for callers expecting per-dimension stats."""
        return self.cross_attn.get_gate_stats()

    def get_combined_stats(self) -> Dict[str, float]:
        """Return both per-sample and per-dimension gate statistics."""
        return {
            **self.get_sample_gate_stats(),
            **self.cross_attn.get_gate_stats(),
        }

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, sample_gate_hidden={self.sample_gate_hidden}, "
            "per-sample gating + per-dim gating"
        )


DualCrossAttention = DualBidirectionalCrossAttention
