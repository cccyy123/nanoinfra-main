"""
spec.py — the MoE-vs-baseline comparison this project runs.  THE one knob.

A positive architecture ablation: hold everything fixed — model geometry (depth →
dim), data, token budget, optimizer — and replace the dense MLP with a sparse
Mixture of Experts.  Two arms differ ONLY in trunk + system class:

  * baseline — the reference GPT               (core/model/gpt.py)
  * moe      — MoEGPT (trunk.py): 4 experts per layer, top-2 routing

Both are driven through the blessed text Orchestrator (modalities.text.train_text)
via `model.trunk_class` + `model.system_class` knobs — zero core edit beyond the
`system_cls` hook already added to `build_system`.

PARAMETER COUNT: matched by construction (see trunk.py module docstring for the
arithmetic).  The MoE has ~37K extra router-gate params (0.1%).

SCALE: the defaults below are a SMOKE TEST (depth=6 → ~25M params, 20M tokens,
~10 min/arm on one GPU).  To run the FULL comparison at 135M scale, change
DEPTH to 12 and MAX_TOKENS to ~2.7B (or set max_steps=-1 for Chinchilla
auto-sizing).
"""

# --- the model + recipe  (edit here to re-target the whole pipeline) ---------
DEPTH       = 6             # 6=smoke (~25M);  12=full (~135M)
LR_MAX      = "3e-4"
SEED        = 42
N_EXPERTS   = 4             # experts per MoE layer
TOP_K       = 2             # selected per token

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS  = 20_000_000    # tiny on purpose — smoke test; bump for a real study
WARMUP_STEPS = 100
N_EVALS     = 30            # log-spaced val evals
EVAL_TOKENS = 131072        # 128K val tokens per eval

ORCHESTRATOR = "modalities.text.train_text"

# Fully-qualified class paths for the orchestrator's dynamic import.
# DEFINED FIRST — ARMS references them below.
_TRUNK_CLS  = "projects.example_moe.trunk.MoEGPT"
_SYSTEM_CLS = "projects.example_moe.trunk.MoESystem"

# The two arms: (label, trunk_class, system_class).
# None → the orchestrator's default (GPT / LMSystem).
ARMS = [
    ("baseline", None,        None),
    ("moe",      _TRUNK_CLS,  _SYSTEM_CLS),
]


def train_overrides(trunk_class, system_class, max_steps, eval_at,
                    n_experts=N_EXPERTS, top_k=TOP_K):
    """Hydra CLI overrides — this project's recipe on the orchestrator's
    defaults.  Constant LR (no warmdown); explicit log-spaced eval schedule;
    no checkpoints.  `trunk_class` and `system_class` distinguish the two arms."""
    ov = {
        "model.depth":            DEPTH,
        "optimizer.lr_max":       LR_MAX,
        "seed":                   SEED,
        "sequence_len":           SEQ_LEN,
        "device_batch_size":      DBS,
        "total_batch_size":       TBS,
        "max_steps":              max_steps,
        "optimizer.scheduler.warmup_steps":   WARMUP_STEPS,
        "optimizer.scheduler.warmdown_ratio": 0.0,   # constant LR after warmup
        "optimizer.scheduler.final_lr_frac":  1.0,
        "checkpoint.enabled":     "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, eval_at)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every":      100,
        # MoE: compile may cause graph breaks around dynamic routing;
        # leave it off for reliability.  Enable if you've verified it works.
        "use_compile":            "false",
    }
    if trunk_class:
        ov["model.trunk_class"]  = trunk_class
    if system_class:
        ov["model.system_class"] = system_class
    return [f"{k}={v}" for k, v in ov.items()]
