"""spec.py — FFN (MLP) expansion ratio ablation. THE one knob.

A parameter-efficiency ablation: hold everything fixed — the model depth/width,
the data, the token budget, the optimizer — and vary ONE thing, the MLP hidden
dimension (expansion ratio relative to n_embd).  Standard transformer = 4x.

  * baseline   — the reference GPT (4x MLP by default)     (core/model/gpt.py)
  * ratio_1x   — MLP hidden = 1 × n_embd  (no expansion)
  * ratio_2x   — MLP hidden = 2 × n_embd
  * ratio_6x   — MLP hidden = 6 × n_embd
  * ratio_8x   — MLP hidden = 8 × n_embd

All arms drive through the SAME orchestrator (modalities.text.train_text) via
its `model.trunk_class` knob — no core edit, no forked training loop.

To change which ratios are tested, edit MLP_RATIOS below.  Each ratio gets its
own trunk class registered in ARMS automatically.
"""
DEPTH = 6                  # smoke scale (a couple of minutes on one GPU)
LR_MAX = "3e-4"
SEED = 42

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS = 20_000_000
WARMUP_STEPS = 100
N_EVALS = 30               # log-spaced val evals
EVAL_TOKENS = 131072       # 128K val tokens/eval — cheap; only relative CE matters

# Set to True when network is unavailable (e.g. AutoDL without HF access).
# Uses random synthetic tokens — proves the pipeline, but CE is meaningless.
# Switch to False with real FineWeb parquet data for actual results.
USE_SYNTHETIC = True

ORCHESTRATOR = "modalities.text.train_text"

# ---------------------------------------------------------------------------
# THE KNOB: which MLP expansion ratios to compare.
# Edit this list to add/remove arms. Each value creates an arm automatically.
#   4.0 is the standard transformer default (baseline, uses None = core GPT).
#   All other values get a custom trunk from trunk.py.
# ---------------------------------------------------------------------------
MLP_RATIOS = [1.0, 2.0, 4.0, 6.0, 8.0]

# Build ARMS from MLP_RATIOS: 4x → baseline (None), others → custom trunk
def _build_arms():
    arms = []
    for r in MLP_RATIOS:
        if r == 4.0:
            arms.append(("ratio_4x", None))
        else:
            # e.g. ratio_1x → projects.myproject.trunk.GPT_MLP1x
            label = f"ratio_{r:.0f}x" if r == int(r) else f"ratio_{r}x"
            cls_name = f"GPT_MLP{int(r)}x" if r == int(r) else f"GPT_MLP{r}x"
            trunk_path = f"projects.myproject.trunk.{cls_name}"
            arms.append((label, trunk_path))
    return arms

ARMS = _build_arms()

# Expose MLP_RATIO_MAP for run.py to compute param counts
MLP_RATIO_MAP = {f"ratio_{int(r) if r == int(r) else r}x": float(r) for r in MLP_RATIOS}


def train_overrides(trunk_class, max_steps, eval_at):
    """Hydra CLI overrides — this project's recipe on the orchestrator's defaults.
    Constant LR (no warmdown) so each curve is genuine loss-vs-step; an explicit
    log-spaced eval schedule; no checkpoints. The ONLY thing that changes between
    arms is `model.trunk_class`."""
    ov = {
        "model.depth": DEPTH,
        "optimizer.lr_max": LR_MAX,
        "seed": SEED,
        "sequence_len": SEQ_LEN,
        "device_batch_size": DBS,
        "total_batch_size": TBS,
        "max_steps": max_steps,
        "optimizer.scheduler.warmup_steps": WARMUP_STEPS,
        "optimizer.scheduler.warmdown_ratio": 0.0,   # constant LR after warmup
        "optimizer.scheduler.final_lr_frac": 1.0,
        "checkpoint.enabled": "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, eval_at)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 100,
    }
    if trunk_class:
        ov["model.trunk_class"] = trunk_class
    if USE_SYNTHETIC:
        ov["data.source"] = "synthetic"
    return [f"{k}={v}" for k, v in ov.items()]
