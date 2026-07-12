"""run.py — train all MLP-ratio arms and collect their val-loss trajectories.

Same model depth/width, same data, same budget, same recipe; the ONLY
difference is model.trunk_class — each trunk pins a different MLP expansion
ratio. Drives the blessed orchestrator via subprocess.

Needs a FineWeb shard on disk (outputs/base_data/) — fetch one first:
    python exemplars/text_pretrain/data/download_shards.py

    python projects/myproject/run.py      # trains all arms
    python projects/myproject/plot.py     # -> ff_ablation.png
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import spec

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent           # repo root — the orchestrator subprocess runs here
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")
# Training loss: "Step 00100/01220 (  8.2%) | loss: X.XXXXXX | ..."
TRAIN_LOSS_RE = re.compile(r"Step\s+(\d+)/\d+\s+\(.*?\)\s+\|\s+loss:\s+([\d.]+)")


def eval_schedule(max_steps, n=spec.N_EVALS, first=5):
    """~n log-spaced integer steps in [first, max_steps] (deduped, sorted)."""
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def count_params(depth, dim, mlp_ratio, vocab_size=50304):
    """Estimate non-embedding parameters for given geometry + mlp_ratio."""
    attn_per_layer = 4 * dim * dim                  # c_q, c_k, c_v, c_proj
    mlp_per_layer = 2 * dim * int(dim * mlp_ratio)  # c_fc + c_proj
    n_layer_params = depth * (attn_per_layer + mlp_per_layer)
    return n_layer_params + 3 * dim                 # final norms (~3*dim)


def run_arm(label, trunk_class, mlp_ratio, max_steps, steps):
    ov = spec.train_overrides(trunk_class, max_steps, steps)
    print(f"[run ] {label} (ratio={mlp_ratio}x, trunk={trunk_class or 'GPT'}): "
          f"d{spec.DEPTH} -> {max_steps} steps ...", flush=True)
    out = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                         cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                         capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    val_traj = [{"step": int(s), "val": float(v)} for s, v in EVAL_RE.findall(text)]
    train_traj = [{"step": int(s), "loss": float(v)} for s, v in TRAIN_LOSS_RE.findall(text)]
    if out.returncode != 0 or len(val_traj) < 3:
        raise SystemExit(
            f"arm {label} FAILED (rc={out.returncode}, {len(val_traj)} evals):\n{text[-3000:]}")
    print(f"[done] {label}: {len(val_traj)} val evals, {len(train_traj)} train logs, "
          f"val {val_traj[0]['val']:.3f} -> {val_traj[-1]['val']:.3f}", flush=True)
    return {"arm": label, "mlp_ratio": mlp_ratio, "trunk_class": trunk_class,
            "trajectory": val_traj, "train_loss": train_traj}


def main():
    dim = spec.DEPTH * 64                         # n_embd for this depth
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)

    arms_data = []
    for label, trunk_class in spec.ARMS:
        ratio = spec.MLP_RATIO_MAP[label]
        arm = run_arm(label, trunk_class, ratio, max_steps, steps)
        N = count_params(spec.DEPTH, dim, ratio)
        arm["N"] = N
        arm["dim"] = dim
        arms_data.append(arm)

    out = {"depth": spec.DEPTH, "dim": dim,
           "max_steps": max_steps, "arms": arms_data}
    (RESULTS / "curves.json").write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {RESULTS / 'curves.json'}")


if __name__ == "__main__":
    main()
