"""
MoE (Mixture of Experts) GPT trunk — a drop-in replacement for the standard GPT.

Replaces the dense MLP in every transformer block with a sparse Mixture of
Experts layer.  A load-balancing auxiliary loss encourages uniform expert
utilisation, and a custom MoESystem wraps trunk+head to add that loss to the
standard cross-entropy.

Design for FAIR comparison with the baseline (same param count):
  - 4 experts per layer, each at expansion factor 1 (768→768→768)
  - This MATCHES the baseline MLP params per layer:
      baseline: 768→3072→768  = 2 × 768 × 3072 = 4,718,592
      MoE:      4 × (768→768→768) = 4 × 2 × 768² = 4,718,592  (+3,072 router)
  - Top-2 routing: each token sees 2 experts → C(4,2)=6 possible expert pairs
  - Per-token FFN compute: ~50% of baseline (two thin experts vs one wide MLP)

The trunk contract (Config, blocks, estimate_flops, forward) is fully satisfied,
so this drives through the SAME orchestrator via:
    model.trunk_class=projects.example_moe.trunk.MoEGPT
    model.system_class=projects.example_moe.trunk.MoESystem

References:
  - Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
    Mixture-of-Experts Layer" (2017)
  - Fedus et al., "Switch Transformers" (2021) — load-balancing aux loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.gpt import GPT, GPTConfig, Block, CausalSelfAttention, norm
from core.model.system import LMSystem


# ═════════════════════════════════════════════════════════════════════════════
# Router
# ═════════════════════════════════════════════════════════════════════════════

class Router(nn.Module):
    """Top-k gating router.  Projects token hidden states to expert logits,
    selects top-k experts, returns normalised gate weights and a load-balancing
    auxiliary loss (Switch Transformer eq. 3–4)."""

    def __init__(self, n_embd, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, n_experts, bias=False)
        self.aux_loss = 0.0          # accumulated during forward

    def forward(self, x):
        """x: [B×T, n_embd] → (expert_indices [B×T, top_k], gates [B×T, top_k])"""
        logits = self.gate(x)                              # [B×T, n_experts]
        probs = F.softmax(logits, dim=-1)                  # [B×T, n_experts]

        # top-k selection
        gates, indices = torch.topk(probs, self.top_k, dim=-1)
        gates = gates / gates.sum(dim=-1, keepdim=True)   # re-normalise

        # load-balancing aux loss (Switch Transformer style):
        #   L_aux = n_experts · Σ_i (f_i · P_i)
        #   f_i = fraction of tokens dispatched to expert i
        #   P_i = average router probability for expert i
        with torch.no_grad():
            oh = F.one_hot(indices[:, 0], self.n_experts).float()
            f = oh.mean(dim=0)                             # [n_experts]
        P = probs.mean(dim=0)                              # [n_experts]
        self.aux_loss = self.n_experts * (f * P).sum()

        return indices, gates

    def get_aux_loss(self) -> float:
        loss = self.aux_loss
        self.aux_loss = 0.0
        return loss


# ═════════════════════════════════════════════════════════════════════════════
# Expert
# ═════════════════════════════════════════════════════════════════════════════

class ExpertMLP(nn.Module):
    """A single MoE expert — thin FFN with ReLU², no hidden expansion.

    For the E=4, fexp=1 design:  n_embd → n_embd (ReLU²) → n_embd.
    Param count: 2 × n_embd².  At n_embd=768 that is 1,179,648 ≈ 1.18M.
    4 experts = 4.72M — exactly the baseline MLP (768→3072→768)."""

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


# ═════════════════════════════════════════════════════════════════════════════
# MoE Layer  (drop-in replacement for the dense MLP inside a Block)
# ═════════════════════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    """Sparse Mixture of Experts layer — same interface as the dense MLP:
    forward(x [B,T,D]) → out [B,T,D]."""

    def __init__(self, config, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.n_embd    = config.n_embd
        self.router    = Router(config.n_embd, n_experts, top_k)
        self.experts   = nn.ModuleList(
            [ExpertMLP(config.n_embd) for _ in range(n_experts)])

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)                          # [B×T, D]

        # route every token
        expert_indices, gates = self.router(x_flat)        # [B×T, K], [B×T, K]

        out = torch.zeros_like(x_flat)

        # dispatch: each expert processes the tokens it was assigned to
        for e_idx in range(self.n_experts):
            expert_mask = (expert_indices == e_idx)         # [B×T, K]
            token_mask  = expert_mask.any(dim=-1)           # [B×T]

            if token_mask.any():
                # gate weight for this expert (summed across top-k positions)
                gate_vals = torch.where(
                    expert_mask, gates,
                    torch.zeros_like(gates)
                ).sum(dim=-1)                               # [B×T]

                token_x    = x_flat[token_mask]             # [n_routed, D]
                token_out  = self.experts[e_idx](token_x)   # [n_routed, D]
                token_gate = gate_vals[token_mask].unsqueeze(-1)
                out[token_mask] += token_gate * token_out

        return out.reshape(B, T, D)

    def get_aux_loss(self) -> float:
        return self.router.get_aux_loss()


# ═════════════════════════════════════════════════════════════════════════════
# MoE Block  (replaces the standard Block — only the MLP changes)
# ═════════════════════════════════════════════════════════════════════════════

class MoEBlock(Block):
    """Transformer block with MoE FFN instead of dense MLP.  Attention, norm,
    and residual structure are inherited from the standard Block."""

    def __init__(self, config, layer_idx, n_experts=4, top_k=2):
        nn.Module.__init__(self)                            # skip Block.__init__
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MoELayer(config, n_experts=n_experts, top_k=top_k)


# ═════════════════════════════════════════════════════════════════════════════
# MoE GPT trunk
# ═════════════════════════════════════════════════════════════════════════════

class MoEGPT(GPT):
    """GPT trunk with MoE layers in every block.  Config-compatible with GPT
    (same GPTConfig) — drives through the same orchestrator."""

    Config = GPTConfig

    def __init__(self, config, n_experts=4, top_k=2):
        nn.Module.__init__(self)
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([
                MoEBlock(config, layer_idx=i, n_experts=n_experts, top_k=top_k)
                for i in range(config.n_layer)
            ]),
        })
        self.type_emb = nn.Embedding(config.n_token_types, config.n_embd)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # ---- init_weights (override) -------------------------------------------
    def init_weights(self):
        self.apply(self._init_weights)
        for block in self.transformer.h:
            # zero-init attention output projection (residual-era trick)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            # zero-init every expert's output projection
            for expert in block.mlp.experts:
                torch.nn.init.zeros_(expert.c_proj.weight)
            # small init for the router gate (all experts start equally likely)
            torch.nn.init.normal_(block.mlp.router.gate.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.type_emb.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            self.type_emb.to(dtype=torch.bfloat16)

    # ---- FLOPs estimate (override) -----------------------------------------
    def estimate_flops(self):
        """FLOPs/token for MFU.  Uses the SAME formula as the baseline
        (6·(nparams - wte) + attention term) for CONSISTENT compute accounting
        across the comparison.  With matched param counts the two models report
        identical FLOPs/token here, though the MoE actually performs ~50% fewer
        FFN matmuls per token (only top-k experts activate).  The MFU metric
        will therefore read HIGHER for MoE — this is expected: same GPU time,
        fewer real FLOPs."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h = self.config.n_layer, self.config.n_head
        q, t = self.config.n_embd // self.config.n_head, self.config.sequence_len
        return 6 * (nparams - nparams_embedding) + 12 * l * h * q * t


# ═════════════════════════════════════════════════════════════════════════════
# MoE System  (adds aux loss to the standard CE loss)
# ═════════════════════════════════════════════════════════════════════════════

class MoESystem(LMSystem):
    """LMSystem subclass that collects load-balancing auxiliary loss from all
    MoE blocks and adds it to the cross-entropy training loss.

    The aux-loss coefficient (default 0.01) follows Switch Transformer practice.
    Tune it via `AUX_LOSS_COEF` — larger values push toward uniform expert usage
    at the cost of model quality; smaller values let experts specialise more freely
    but risk collapse (all tokens routed to one expert)."""

    AUX_LOSS_COEF = 0.01

    def loss(self, batch):
        hidden = self._run_trunk(batch["idx"],
                                 token_types=batch.get("token_types"))
        ce_loss = self.head.loss(hidden, batch["targets"])

        # collect load-balancing loss from every MoE block
        aux_loss = sum(
            block.mlp.get_aux_loss()
            for block in self.trunk.blocks
        )
        return ce_loss + self.AUX_LOSS_COEF * aux_loss
