"""FFN (MLP) expansion-ratio variants of the reference GPT — the architecture
change this ablation studies.

Subclasses the reference GPT and swaps in MLP blocks with a configurable
expansion ratio. The standard transformer uses 4x:

    c_fc:   n_embd -> 4 * n_embd
    c_proj: 4 * n_embd -> n_embd

This module provides trunks that change that ratio — nothing else changes.
The attention, RoPE, RMSNorm, no-bias convention, and the rest of the trunk
contract are all inherited from the core GPT.

Each ratio is a separate class so `model.trunk_class` selects it without any
core edit.  To add a new ratio, add a class like:

    class GPT_MLP3x(MLPRatioGPT):
        mlp_ratio = 3.0

…and add "3.0" to MLP_RATIOS in spec.py.
"""
import torch.nn as nn

from core.model.gpt import GPT, GPTConfig, Block


# ---------------------------------------------------------------------------
# Custom MLP with configurable ratio
# ---------------------------------------------------------------------------
class RatioMLP(nn.Module):
    """MLP whose hidden dimension = mlp_ratio × n_embd (standard transformer = 4x).

    Uses the same ReLU² activation and no-bias convention as the core GPT,
    so the only difference from the baseline is the hidden dimension.
    """

    def __init__(self, config, mlp_ratio):
        super().__init__()
        hidden_dim = int(config.n_embd * mlp_ratio)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = nn.functional.relu(x).square()   # ReLU² — same activation as core GPT
        x = self.c_proj(x)
        return x


class RatioBlock(Block):
    """A transformer block whose MLP uses a custom expansion ratio.

    Inherits attention, residual structure, and the block signature from
    the core Block — only the MLP is replaced.
    """

    def __init__(self, config, layer_idx, mlp_ratio):
        super().__init__(config, layer_idx)
        self.mlp = RatioMLP(config, mlp_ratio)   # swap in the ratio-MLP


# ---------------------------------------------------------------------------
# Base trunk — rebuilds every block with a given MLP ratio
# ---------------------------------------------------------------------------
class MLPRatioGPT(GPT):
    """Base class: rebuilds every block's MLP with a configurable ratio.

    Subclasses pin `mlp_ratio` to a specific value, so each is a complete
    trunk selectable via `model.trunk_class`.  Everything else (attention,
    RoPE, RMSNorm, weight init, FLOPs estimate, the trunk contract) is
    inherited from GPT unchanged.
    """

    Config = GPTConfig
    mlp_ratio: float = 4.0   # overridden by subclasses

    def __init__(self, config):
        super().__init__(config)
        # Rebuild all blocks with the custom MLP ratio
        self.transformer.h = nn.ModuleList(
            [RatioBlock(config, i, self.mlp_ratio) for i in range(config.n_layer)]
        )

    def init_weights(self):
        """Standard init + zero c_proj (residual-era trick).
        Mirrors GPT.init_weights exactly — the only difference is the MLP
        hidden dimension, which the init handles generically by fan_in/fan_out."""
        super().init_weights()


# ---------------------------------------------------------------------------
# Concrete ratio classes — one per arm.  Selected via model.trunk_class.
#
# To add a new ratio, add a class below AND add the ratio to MLP_RATIOS in
# spec.py.  The ARMS list and MLP_RATIO_MAP are built automatically from that
# list — no other wiring needed.
# ---------------------------------------------------------------------------
class GPT_MLP1x(MLPRatioGPT):
    """MLP hidden = 1 × n_embd — no expansion (the FFN is a square matrix)."""
    mlp_ratio = 1.0


class GPT_MLP2x(MLPRatioGPT):
    """MLP hidden = 2 × n_embd — half the standard expansion."""
    mlp_ratio = 2.0


class GPT_MLP4x(MLPRatioGPT):
    """MLP hidden = 4 × n_embd — the standard transformer ratio (baseline)."""
    mlp_ratio = 4.0


class GPT_MLP6x(MLPRatioGPT):
    """MLP hidden = 6 × n_embd — 50% wider than standard."""
    mlp_ratio = 6.0


class GPT_MLP8x(MLPRatioGPT):
    """MLP hidden = 8 × n_embd — double the standard expansion."""
    mlp_ratio = 8.0
