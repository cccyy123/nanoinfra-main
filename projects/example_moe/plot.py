"""
plot.py — val-loss curves (baseline GPT vs MoE) from run.py's
results/curves.json.  Writes moe_vs_baseline.png.

Produces TWO views:
  1. val CE vs step  (both models on the same axes)
  2. val CE vs compute (6·N·tokens) if param counts differ meaningfully

The first view is the primary comparison — "at the same data budget,
which model achieves lower validation loss?"
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


STYLE = {
    "baseline": ("#1f77b4", "baseline (dense MLP)"),
    "moe":      ("#2ca02c", "MoE (4 experts, top-2)"),
}


def plot_trajectories(data, out_path):
    """Single-panel: val CE vs step for each arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    for arm in data["arms"]:
        tr = arm["trajectory"]
        color, label = STYLE.get(arm["arm"], ("gray", arm["arm"]))
        ax.plot([p["step"] for p in tr], [p["val"] for p in tr],
                "-o", color=color, lw=1.9, ms=3.5, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation cross-entropy")
    ax.set_title(
        f"MoE vs dense baseline  (d{data['depth']}, "
        f"{data['n_params']/1e6:.1f}M non-embedding params)\n"
        f"{data['n_experts']} experts, top-{data['top_k']} routing — "
        f"same params, same data, same budget"
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=10)

    # annotate final val CE difference
    finals = []
    for arm in data["arms"]:
        tr = arm["trajectory"]
        if tr:
            finals.append((arm["arm"], tr[-1]["val"]))
    if len(finals) == 2:
        (la, va), (lb, vb) = finals
        delta = vb - va
        winner = la if va < vb else lb
        ax.text(0.97, 0.12,
                f"Δ final CE = {delta:+.4f}  ({winner} lower)",
                transform=ax.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_loss_gap(data, out_path):
    """Optional: val CE difference (baseline - MoE) vs step, with a zero
    reference line.  Positive = MoE better."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = {a["arm"]: a["trajectory"] for a in data["arms"]}
    if "baseline" not in arms or "moe" not in arms:
        print("  skip gap plot — need both 'baseline' and 'moe' arms")
        return

    bl_steps = {p["step"]: p["val"] for p in arms["baseline"]}
    moe_steps = {p["step"]: p["val"] for p in arms["moe"]}
    common = sorted(set(bl_steps) & set(moe_steps))

    gaps = [bl_steps[s] - moe_steps[s] for s in common]  # + → MoE lower CE

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.plot(common, gaps, "-o", color="#2ca02c", lw=1.9, ms=3.5,
            label="baseline CE − MoE CE  (positive = MoE better)")
    ax.axhline(y=0, color="0.5", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("Δ val CE  (baseline − MoE)")
    ax.set_title(f"MoE advantage over training  (d{data['depth']}, "
                 f"{data['n_experts']} experts, top-{data['top_k']})")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    data = json.loads((RESULTS / "curves.json").read_text())

    plot_trajectories(data, HERE / "moe_vs_baseline.png")
    plot_loss_gap(data,     HERE / "moe_loss_gap.png")

    # quick summary
    print("\n--- summary ---")
    for arm in data["arms"]:
        tr = arm["trajectory"]
        print(f"  {arm['arm']:10s}  steps {tr[0]['step']:5d}→{tr[-1]['step']:5d}"
              f"  val CE {tr[0]['val']:.4f} → {tr[-1]['val']:.4f}")

    finals = [(a["arm"], a["trajectory"][-1]["val"]) for a in data["arms"]]
    if len(finals) == 2:
        (la, va), (lb, vb) = finals
        delta = vb - va
        winner = la if va < vb else lb
        print(f"  Δ final CE: {delta:+.4f}  ({winner} lower)")
    print(f"  plots → {HERE}/moe_vs_baseline.png  +  moe_loss_gap.png")


if __name__ == "__main__":
    main()
