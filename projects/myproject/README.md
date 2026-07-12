# FFN (MLP) expansion ratio ablation

A **parameter-efficiency ablation** of the transformer's feed-forward network.
Same model depth/width, same data, same budget, same recipe — only the MLP
hidden dimension (expansion ratio) changes. The question: *how much do we gain
by spending parameters on a wider FFN, and where does it stop paying off?*

This is the `projects/` tier: a fork of an exemplar with one thing swept.
Point a newcomer here to see *"how do I ablate a hyperparameter and measure
parameter efficiency?"*

## The ablation

A standard transformer MLP expands from `n_embd` to `4 × n_embd` and projects
back (the "4x ratio"). This project tests five ratios:

| ratio | MLP hidden dim | trunk class |
|-------|---------------|-------------|
| 1× | `n_embd` (no expansion) | `GPT_MLP1x` |
| 2× | `2 × n_embd` | `GPT_MLP2x` |
| **4×** | **`4 × n_embd` (standard)** | **core GPT** (baseline) |
| 6× | `6 × n_embd` | `GPT_MLP6x` |
| 8× | `8 × n_embd` | `GPT_MLP8x` |

The 4× arm uses the reference GPT directly (`None` in ARMS); all other arms
use thin subclasses of GPT that only replace the MLP — attention, RoPE,
RMSNorm, no-bias convention, and weight init are all inherited unchanged.

## How it's wired

No core edit, no forked training loop. Each ratio gets a trunk class in
`trunk.py` that subclasses the reference GPT and swaps in a `RatioMLP` with
the target hidden dimension. The blessed orchestrator selects it via one
config knob, `model.trunk_class`. That is the framework's pluggable-trunk
seam: to ablate an architectural parameter you provide a trunk, you do not
patch core.

| file | what |
|------|------|
| `trunk.py` | `MLPRatioGPT` and concrete ratio classes — the architecture change |
| `spec.py`  | the recipe (depth, budget, the arm list) — **THE one knob** |
| `run.py`   | trains all arms through the orchestrator, collects the val curves |
| `plot.py`  | dual-panel figure → `ff_ablation.png` |

## Changing the ratios

Edit the `MLP_RATIOS` list in `spec.py` — everything else updates
automatically:

```python
# spec.py — THE one knob
MLP_RATIOS = [1.0, 2.0, 4.0, 6.0, 8.0]   # add/remove as needed
```

To add a new ratio class in `trunk.py`:

```python
class GPT_MLP3x(MLPRatioGPT):
    mlp_ratio = 3.0
```

Then add `3.0` to `MLP_RATIOS` in `spec.py`. The `_build_arms()` function
generates the ARMS list and MLP_RATIO_MAP automatically — no other wiring
needed.

## Run it

```bash
# once: fetch a FineWeb shard (shared with the text exemplar)
python exemplars/text_pretrain/data/download_shards.py

python projects/myproject/run.py     # trains all arms (d6, minutes on one GPU)
python projects/myproject/plot.py    # -> ff_ablation.png
```

Defaults to a tiny **d6** smoke scale (a couple of minutes on one GPU, ~20M
tokens). Raise `DEPTH` in `spec.py` for a full study — wider ratios cost more
FLOPs/step, so at deeper scales the param-efficiency trade-off becomes sharper.

## What to expect

At d6 the differences are modest — a few hundredths of a CE — because
parameter-count differences are small in absolute terms at this scale. The
trend to watch for:

- **1× and 2×** typically underperform the baseline — the FFN is too
  narrow to learn useful features.
- **6× and 8×** may match or slightly beat the baseline, but the
  *parameter efficiency* (ΔCE per Δparam) deteriorates — you pay more
  params for diminishing returns.
- **The right panel** (CE vs params) is the real answer: the best ratio
  is the one on the Pareto frontier for your compute budget.

At deeper scales (d12+), the gap between narrow and wide FFNs typically
widens — deeper stacks benefit more from wider MLPs because the residual
stream carries richer representations that need more capacity to process.

## Extension ideas

- **Sweep depth × ratio** — does the optimal ratio depend on depth?
- **Fix total params** — compare a deeper narrow model vs a shallower wide
  one at iso-param.
- **Different activations** — does the optimal ratio change with GELU,
  SwiGLU, or plain ReLU?
