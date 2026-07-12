"""plot.py — the four val-loss curves, one per FFN expansion ratio, from run.py's
results/curves.json. Writes ffn_expansion.png. X-axis is used tokens (step × TBS).

Set RESULTS_DIR env var to read/write from a specific directory.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("RESULTS_DIR", HERE / "results"))

# Color-blind-friendly palette, one colour per arm
COLORS = {
    "ffn_2x": "#1f77b4",
    "ffn_4x": "#2ca02c",   # baseline gets green
    "ffn_6x": "#ff7f0e",
    "ffn_8x": "#d62728",
}


def main():
    data = json.loads((RESULTS / "curves.json").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for arm in data["arms"]:
        tr = arm["trajectory"]
        color = COLORS.get(arm["arm"], "gray")
        label = arm["arm"].replace("_", " ")
        ax.plot([p["tokens"] for p in tr], [p["val"] for p in tr], "-o",
                color=color, lw=1.9, ms=4, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("used tokens")
    ax.set_ylabel("validation cross-entropy")
    ax.set_title(f"FFN expansion ratio ablation  (d{data['depth']})\n"
                 "same data, budget, and recipe — only the FFN hidden dim differs")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "ffn_expansion.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
