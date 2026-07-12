"""plot.py — FFN (MLP) ratio ablation: training loss + val CE + param efficiency.

Produces ff_ablation.png with three panels:
  top left:  train loss vs training tokens
  top right: val CE vs training tokens
  bottom:    final val CE vs non-embedding parameters (parameter efficiency)

Usage:
    python projects/myproject/plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Color palette — one colour per ratio, consistent across all panels
COLORS = {
    "ratio_1x": "#e41a1c",   # red
    "ratio_2x": "#ff7f00",   # orange
    "ratio_4x": "#377eb8",   # blue (baseline)
    "ratio_6x": "#4daf4a",   # green
    "ratio_8x": "#984ea3",   # purple
}

TBS = 16384   # must match spec.py.TBS


def load():
    return json.loads((RESULTS / "curves.json").read_text())


def plot(curves_data):
    arms = curves_data["arms"]
    total_tokens_M = curves_data["max_steps"] * TBS / 1e6

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax_train = fig.add_subplot(gs[0, 0])
    ax_val = fig.add_subplot(gs[0, 1])
    ax_params = fig.add_subplot(gs[1, :])

    # ---- top-left: train loss vs training tokens ----
    for arm in arms:
        label = arm["arm"]
        train = arm.get("train_loss", [])
        if not train:
            continue
        xs = [p["step"] * TBS / 1e6 for p in train]   # tokens (millions)
        ys = [p["loss"] for p in train]
        ax_train.plot(xs, ys, "-", color=COLORS.get(label, "#888"),
                      lw=1.2, alpha=0.8,
                      label=f'{label} ({arm["mlp_ratio"]}×)')

    ax_train.set_xlabel("Training Tokens (millions)")
    ax_train.set_ylabel("Training Loss (EMA smoothed)")
    ax_train.set_title(f"Training Loss — d{curves_data['depth']}, "
                       f"dim={curves_data['dim']}")
    ax_train.legend(fontsize=7, ncol=2)
    ax_train.grid(True, ls=":", alpha=0.4)

    # ---- top-right: val CE vs training tokens ----
    for arm in arms:
        label = arm["arm"]
        traj = arm["trajectory"]
        xs = [p["step"] * TBS / 1e6 for p in traj]   # tokens (millions)
        ys = [p["val"] for p in traj]
        ax_val.plot(xs, ys, "-o", color=COLORS.get(label, "#888"),
                    lw=1.5, ms=3,
                    label=f'{label} ({arm["mlp_ratio"]}×)')

    ax_val.set_xlabel("Training Tokens (millions)")
    ax_val.set_ylabel("Validation Cross-Entropy")
    ax_val.set_title(f"Validation CE — d{curves_data['depth']}, "
                     f"dim={curves_data['dim']}\n"
                     f"fixed budget: {total_tokens_M:.0f}M tokens, constant LR")
    ax_val.legend(fontsize=7, ncol=2)
    ax_val.grid(True, ls=":", alpha=0.4)

    # ---- bottom: final CE vs non-embedding params ----
    final_ce = [arm["trajectory"][-1]["val"] for arm in arms]
    params_M = [arm["N"] / 1e6 for arm in arms]

    for i, arm in enumerate(arms):
        ax_params.plot(params_M[i], final_ce[i], "o",
                       color=COLORS.get(arm["arm"], "#888"), ms=14,
                       markeredgewidth=1.5, markeredgecolor="white")
        ax_params.annotate(f'{arm["mlp_ratio"]}×', (params_M[i], final_ce[i]),
                           textcoords="offset points", xytext=(0, 14),
                           fontsize=10, fontweight="bold", ha="center",
                           color=COLORS.get(arm["arm"], "#888"))

    ax_params.set_xlabel("Non-embedding Parameters (millions)")
    ax_params.set_ylabel("Final Validation Cross-Entropy")
    ax_params.set_title(f"Parameter Efficiency — d{curves_data['depth']}, "
                        f"dim={curves_data['dim']}")
    ax_params.grid(True, ls=":", alpha=0.4)
    ax_params.invert_yaxis()

    # ---- summary box ----
    baseline_idx = next(i for i, a in enumerate(arms) if a["arm"] == "ratio_4x")
    baseline_ce = final_ce[baseline_idx]
    baseline_params = params_M[baseline_idx]

    lines = [
        f"Baseline (4×): {baseline_params:.2f}M params, CE={baseline_ce:.4f}",
    ]
    for i, arm in enumerate(arms):
        if arm["arm"] == "ratio_4x":
            continue
        d_ce = final_ce[i] - baseline_ce
        d_p = (params_M[i] - baseline_params) / baseline_params * 100
        lines.append(f'{arm["arm"]}: ΔCE={d_ce:+.4f}  Δparams={d_p:+.0f}%')

    best_idx = int(np.argmin(final_ce))
    best_arm = arms[best_idx]
    lines.insert(0, f'Best: {best_arm["arm"]} ({best_arm["mlp_ratio"]}×) — '
                     f'CE={final_ce[best_idx]:.4f}')

    ax_params.text(0.98, 0.02, "\n".join(lines), transform=ax_params.transAxes,
                   fontsize=8, fontfamily="monospace", va="bottom", ha="right",
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.6", alpha=0.92))

    fig.suptitle("FFN (MLP) Expansion Ratio Ablation", fontsize=14, fontweight="bold", y=0.98)
    outpath = HERE / "ff_ablation.png"
    fig.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    data = load()
    plot(data)
