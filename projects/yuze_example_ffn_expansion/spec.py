"""spec.py — FFN expansion ratio ablation. THE one knob.

A positive architecture ablation: hold everything fixed — the model geometry,
the data, the token budget, the optimizer — and vary ONE thing, the FFN
expansion ratio (hidden_dim = expansion * n_embd). The modern reference uses 4x.
This sweeps 2x / 4x / 6x / 8x to map the compute-vs-capacity trade-off.

All arms are driven through the blessed text Orchestrator (modalities.text.train_text)
via its `model.trunk_class` knob — no core edit, no forked training loop.
"""
DEPTH = 6                  # smoke scale (minutes on one GPU)
LR_MAX = "3e-4"
SEED = 42

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS = 200_000_000    # tiny on purpose — a smoke test / worked example (minutes on one GPU)
WARMUP_STEPS = 100
N_EVALS = 30               # log-spaced val evals
EVAL_TOKENS = 131072       # 128K val tokens/eval — cheap; only relative CE matters here

ORCHESTRATOR = "modalities.text.train_text"

# None => the reference GPT (4x expansion, the default).
ARMS = [
    ("ffn_2x", "projects.yuze_example_ffn_expansion.trunk.FFN2xGPT"),
    ("ffn_4x", None),                                                    # baseline
    ("ffn_6x", "projects.yuze_example_ffn_expansion.trunk.FFN6xGPT"),
    ("ffn_8x", "projects.yuze_example_ffn_expansion.trunk.FFN8xGPT"),
]


def train_overrides(trunk_class, max_steps, eval_at):
    """Hydra CLI overrides — this project's recipe on the orchestrator's defaults.
    Constant LR (no warmdown) so each curve is genuine loss-vs-step; an explicit
    log-spaced eval schedule; no checkpoints. The ONLY thing that changes between
    the arms is `model.trunk_class`."""
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
    return [f"{k}={v}" for k, v in ov.items()]
