"""FFN expansion ratio variants — the architecture change this ablation studies.

Subclasses the reference GPT and swaps in MLPs with different hidden-dimension
ratios. Everything else (attention, RoPE, RMSNorm, QK-norm, residual path, the
trunk contract, the FLOPs estimate) is inherited. The ONLY difference from the
baseline is `MLP.c_fc` / `MLP.c_proj` shape.

Why a factory instead of separate classes: each variant differs only in one
integer (the expansion multiplier), so a single parameterised builder keeps the
file short and the differences explicit. Each generated class is a module-level
name, importable via `model.trunk_class`.
"""
import torch.nn as nn

from core.model.gpt import GPT, GPTConfig


def _make_ffn_expansion_gpt(expansion: int):
    """Return a GPT subclass whose MLP hidden dim = expansion * n_embd."""

    class _FFNExpansionGPT(GPT):
        Config = GPTConfig

        def __init__(self, config):
            super().__init__(config)
            # Swap in MLPs with the target expansion ratio
            for block in self.transformer.h:
                block.mlp.c_fc = nn.Linear(config.n_embd, expansion * config.n_embd, bias=False)
                block.mlp.c_proj = nn.Linear(expansion * config.n_embd, config.n_embd, bias=False)

        def init_weights(self):
            super().init_weights()
            # Re-init the swapped MLP layers (super().init_weights only touched
            # the ORIGINAL layers).  c_fc gets normal init; c_proj stays zero-init
            # (residual-era trick — the block starts as identity x + 0 = x).
            for block in self.transformer.h:
                self._init_weights(block.mlp.c_fc)
                self._init_weights(block.mlp.c_proj)
                nn.init.zeros_(block.mlp.c_proj.weight)

    # Give the class a stable, human-readable name (the import path includes the
    # module name, so this is just for logging / tracebacks).
    _FFNExpansionGPT.__name__ = f"FFN{expansion}xGPT"
    _FFNExpansionGPT.__qualname__ = f"FFN{expansion}xGPT"
    return _FFNExpansionGPT


FFN2xGPT = _make_ffn_expansion_gpt(2)
FFN6xGPT = _make_ffn_expansion_gpt(6)
FFN8xGPT = _make_ffn_expansion_gpt(8)
