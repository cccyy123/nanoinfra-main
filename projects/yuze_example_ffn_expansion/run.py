"""run.py — train all FFN expansion arms (2x / 4x / 6x / 8x) and collect their
val-loss trajectories. Same model / data / budget / recipe; the ONLY difference
is the FFN hidden-dimension ratio. Mirrors the exemplar's scaling.py driving
pattern: subprocess the blessed orchestrator, parse the scheduled val evals
straight from its log.

Needs a FineWeb shard on disk (outputs/base_data/) — fetch a couple first:
    python exemplars/text_pretrain/data/download_shards.py

    python projects/yuze_example_ffn_expansion/run.py      # trains all four arms
    python projects/yuze_example_ffn_expansion/plot.py     # -> ffn_expansion.png

Set RESULTS_DIR env var to write into a specific directory (default: results/).
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import spec

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent           # repo root — the orchestrator subprocess runs here
RESULTS = Path(os.environ.get("RESULTS_DIR", HERE / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)

EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")


def eval_schedule(max_steps, n=spec.N_EVALS, first=5):
    """~n log-spaced integer steps in [first, max_steps] (deduped, sorted)."""
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def run_arm(label, trunk_class, max_steps, steps):
    ov = spec.train_overrides(trunk_class, max_steps, steps)
    print(f"[run ] {label}: d{spec.DEPTH} -> {max_steps} steps ...", flush=True)
    out = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                         cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                         capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    traj = [{"step": int(s), "tokens": int(s) * spec.TBS,
             "val": float(v)} for s, v in EVAL_RE.findall(text)]
    if out.returncode != 0 or len(traj) < 3:
        raise SystemExit(f"arm {label} FAILED (rc={out.returncode}, {len(traj)} evals):\n{text[-3000:]}")
    print(f"[done] {label}: {len(traj)} evals, val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}",
          flush=True)
    return {"arm": label, "trajectory": traj}


def write_experiment_yaml(arms_out):
    """Write a human-readable experiment summary."""
    lines = [
        f"# FFN Expansion Ablation — experiment summary",
        f"timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"",
        f"model:",
        f"  family: GPT (decoder-only transformer)",
        f"  depth: {spec.DEPTH}",
        f"  position: RoPE",
        f"  normalization: RMSNorm (no learnable params)",
        f"  activation: ReLU²",
        f"  biases: none",
        f"  attention: causal flash-attention + QK-norm",
        f"  head: untied LMHead",
        f"",
        f"recipe:",
        f"  lr_max: {spec.LR_MAX}",
        f"  seed: {spec.SEED}",
        f"  seq_len: {spec.SEQ_LEN}",
        f"  device_batch_size: {spec.DBS}",
        f"  total_batch_size: {spec.TBS}",
        f"  max_tokens: {spec.MAX_TOKENS // 1_000_000}M",
        f"  lr_schedule: constant (warmup {spec.WARMUP_STEPS} steps, no warmdown)",
        f"",
        f"experiment:",
        f"  type: FFN expansion ratio sweep",
        f"  arms:",
    ]
    for a in arms_out:
        lines.append(f"    - {a['arm']}")
    lines.extend([
        f"",
        f"data:",
        f"  source: FineWeb sample-10BT (parquet streaming)",
        f"  tokenizer: custom BPE (FineWeb-trained, 50274 vocab)",
        f"  eval_tokens_per_point: {spec.EVAL_TOKENS // 1024}K",
        f"",
    ])
    (RESULTS / "experiment.yaml").write_text("\n".join(lines))


def main():
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)
    arms = [run_arm(label, tc, max_steps, steps) for label, tc in spec.ARMS]
    out = {"depth": spec.DEPTH, "max_steps": max_steps, "arms": arms}
    (RESULTS / "curves.json").write_text(json.dumps(out, indent=2))
    write_experiment_yaml(arms)
    print(f"WROTE {RESULTS / 'curves.json'}")
    print(f"WROTE {RESULTS / 'experiment.yaml'}")


if __name__ == "__main__":
    main()
