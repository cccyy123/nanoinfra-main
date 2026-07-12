# example: FFN expansion ratio ablation

A **positive architecture ablation** — vary the FFN hidden-dimension ratio and
read the loss curve. Hold everything fixed — model depth, data, token budget,
optimizer — and sweep ONE thing: the FFN expansion multiplier
(`hidden_dim = expansion * n_embd`). The modern reference uses 4×. This project
sweeps **2× / 4× / 6× / 8×** to map the compute-vs-capacity trade-off.

This is the `projects/` tier: a fork of an exemplar with something changed.
Point a newcomer here to see *"how much does the FFN width actually matter?"*

## The four arms

All are pre-norm decoder-only transformers sharing the SAME width/depth and the
SAME (untied) `LMHead`; only the MLP hidden dimension differs:

| arm | expansion | MLP hidden dim | params (d6) | note |
|-----|-----------|---------------|-------------|------|
| ffn_2x | 2 | 2 × 384 = 768 | smallest | capacity floor — saturates fastest |
| **ffn_4x** | **4** | **4 × 384 = 1536** | **baseline** | **the modern reference** |
| ffn_6x | 6 | 6 × 384 = 2304 | larger | more capacity, more compute |
| ffn_8x | 8 | 8 × 384 = 3072 | largest | most capacity — does it pay off? |

## How it's wired

No forked training loop: `FFN{2,6,8}xGPT` ([trunk.py](trunk.py)) subclasses the
reference `GPT` and swaps in a wider `MLP.c_fc` / `MLP.c_proj`. The factory
`_make_ffn_expansion_gpt(expansion)` generates the three variant classes from
one parameterised body — every variant differs only in one integer, so the
differences are explicit and the file stays short.

All arms drive through the **same** orchestrator (`modalities.text.train_text`)
via `model.trunk_class`. Everything else — RoPE, RMSNorm, ReLU², QK-norm,
residual path, FLOPs estimate — is inherited unchanged.

| file | what |
|------|------|
| `spec.py`  | the recipe + four-arm sweep — the one knob |
| `trunk.py` | `FFN{2,6,8}xGPT` — modern GPT with a different MLP width |
| `run.py`   | trains all four arms through the orchestrator, collects the val curves |
| `plot.py`    | the four curves → `ffn_expansion.png` |
| `run_all.sh` | one-command pipeline: download → train → plot → git push |

## Run it

**One command:**
```bash
bash projects/yuze_example_ffn_expansion/run_all.sh
```

**Or step by step:**
```bash
# once: fetch a FineWeb shard (shared with the text exemplar)
python exemplars/text_pretrain/data/download_shards.py

python projects/yuze_example_ffn_expansion/run.py     # trains 2x / 4x / 6x / 8x (d6, minutes)
python projects/yuze_example_ffn_expansion/plot.py    # -> ffn_expansion.png
```

## What to look for

1. **The capacity ceiling.** 2× should saturate first and highest — the model
   simply doesn't have enough MLP capacity to use the transformer depth.
2. **The 4×→6× jump.** Is 6× worth the extra compute, or is the gain marginal?
3. **The 6×→8× gap.** At d6 the extra width may barely move the curve — the
   depth bottleneck may dominate before the width bottleneck does.
4. **Crossover.** At very early steps smaller MLPs sometimes descend a hair
   faster (fewer params to wrangle); read the curve out to where it flattens.

## Extension ideas

- **Depth × expansion grid.** Run this same sweep at d3, d6, d12 — does the
  optimal expansion ratio depend on depth?
- **Tie it to the scaling law.** Pick the winner at each depth and re-run the
  exemplar's `scaling.py` with that trunk — does the compute-optimal exponent
  a ≈ 0.52 hold, or does a tuned FFN width shift it?
- **Measure MFU.** Wider MLPs change the matmul shape — at what expansion does
  GPU utilisation drop?
