"""run.py — train all mlp_ratio arms and collect val-CE trajectories.

Drives the blessed orchestrator (modalities.text.train_text) via subprocess.
Same model / data / budget; the ONLY knob that changes is model.mlp_ratio.

Usage:
    python projects/mlp_ratio_ablation/run.py      # trains all arms
    python projects/mlp_ratio_ablation/plot.py     # -> mlp_ratio_ablation.png
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
REPO = HERE.parent.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")


def eval_schedule(max_steps, n=spec.N_EVALS, first=5):
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def count_params(depth, dim, mlp_ratio, vocab_size=50304):
    """Estimate non-embedding params for given geometry + mlp_ratio.
    From the 6N formula: 12 * depth * dim^2 per layer, MLP uses ratio*dim*2
    additional params per layer vs baseline."""
    # attention: 4 * dim*dim (q,k,v,proj) per layer
    attn = 4 * depth * dim * dim
    # mlp: fc(dim->ratio*dim) + proj(ratio*dim->dim) per layer
    mlp = 2 * depth * dim * int(dim * mlp_ratio)
    return attn + mlp + 3 * dim  # final norm


def run_arm(label, mlp_ratio, max_steps, steps):
    ov = spec.train_overrides(mlp_ratio, max_steps, steps)
    print(f"[run ] {label} (mlp_ratio={mlp_ratio}): d{spec.DEPTH} -> "
          f"{max_steps} steps ...", flush=True)
    out = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                         cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                         capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    traj = [{"step": int(s), "val": float(v)} for s, v in EVAL_RE.findall(text)]
    if out.returncode != 0 or len(traj) < 3:
        raise SystemExit(
            f"arm {label} FAILED (rc={out.returncode}, {len(traj)} evals):\n{text[-3000:]}")
    print(f"[done] {label}: {len(traj)} evals, "
          f"val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}", flush=True)
    return {"arm": label, "mlp_ratio": mlp_ratio, "trajectory": traj}


def main():
    dim = spec.DEPTH * 64
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)

    arms_data = []
    for label, ratio in spec.ARMS:
        arm = run_arm(label, ratio, max_steps, steps)
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
