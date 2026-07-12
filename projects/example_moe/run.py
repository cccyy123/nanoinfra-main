"""
run.py — train both arms (baseline GPT vs MoE) and collect their val-loss
trajectories.  Same model geometry, data, and token budget; the ONLY differences
are the trunk (GPT → MoEGPT) and system (LMSystem → MoESystem).

Mirrors the example-residual-ablation driving pattern: subprocess the blessed
orchestrator, parse the scheduled val evals straight from its log.

Needs a FineWeb shard on disk (outputs/base_data/) — fetch a couple first:
    python exemplars/text_pretrain/data/download_shards.py

    python projects/example_moe/run.py       # trains both arms sequentially
    python projects/example_moe/plot.py      # → moe_vs_baseline.png
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import spec

HERE    = Path(__file__).resolve().parent
REPO    = HERE.parent.parent           # repo root — the orchestrator subprocess runs here
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")


def n_params(depth):
    """Non-embedding parameter count for this GPT geometry.
    Mirrors the family rule: dim = 64*depth."""
    dim = depth * 64
    return 12 * depth * dim * dim + 3 * dim


def eval_schedule(max_steps, n=spec.N_EVALS, first=5):
    """~n log-spaced integer steps in [first, max_steps] (deduped, sorted)."""
    s = np.unique(np.round(
        np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def run_arm(label, trunk_cls, system_cls, max_steps, steps):
    ov = spec.train_overrides(trunk_cls, system_cls, max_steps, steps)
    print(f"\n{'='*60}")
    print(f"[run] {label}: d{spec.DEPTH} → {max_steps} steps "
          f"({max_steps * spec.TBS / 1e6:.0f}M tokens)", flush=True)
    print(f"{'='*60}")
    out = subprocess.run(
        [sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
        cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True, text=True,
    )
    text = out.stdout + "\n" + out.stderr
    traj = [{"step": int(s), "val": float(v)}
            for s, v in EVAL_RE.findall(text)]
    if out.returncode != 0 or len(traj) < 3:
        print(f"\n[FAIL] {label}: rc={out.returncode}, {len(traj)} evals")
        print(text[-3000:])
        raise SystemExit(
            f"arm '{label}' FAILED (rc={out.returncode}, {len(traj)} evals)")
    print(f"\n[done] {label}: {len(traj)} evals, "
          f"val {traj[0]['val']:.3f} → {traj[-1]['val']:.3f}", flush=True)
    return {"arm": label, "trajectory": traj,
            "trunk_class": trunk_cls, "system_class": system_cls}


def main():
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps    = eval_schedule(max_steps)

    N = n_params(spec.DEPTH)
    print(f"MoE vs baseline — d{spec.DEPTH} (N≈{N/1e6:.1f}M non-embedding params)")
    print(f"Token budget: {spec.MAX_TOKENS/1e6:.0f}M  ({max_steps} steps)")
    print(f"Eval schedule: {len(steps)} log-spaced points, "
          f"first at step {steps[0]}, last at step {steps[-1]}")
    print(f"MoE config: {spec.N_EXPERTS} experts, top-{spec.TOP_K} routing\n")

    arms = []
    for label, trunk_cls, system_cls in spec.ARMS:
        arms.append(run_arm(label, trunk_cls, system_cls, max_steps, steps))

    out = {
        "depth":      spec.DEPTH,
        "max_steps":  max_steps,
        "max_tokens": spec.MAX_TOKENS,
        "n_experts":  spec.N_EXPERTS,
        "top_k":      spec.TOP_K,
        "n_params":   N,
        "arms":       arms,
    }
    out_path = RESULTS / "curves.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
