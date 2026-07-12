#!/usr/bin/env bash
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PROJECT="projects/yuze_example_ffn_expansion"
LOG_DIR="$PROJECT/logs"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log

NOW=$(date '+%Y-%m-%d %H:%M:%S')
echo "========================================"
echo "FFN Expansion Ablation Pipeline"
echo "Start : $NOW"
echo "Project: $PROJECT"
echo "========================================"

# ── 1. Setup venv (auto-create if missing) ──────────────────────
if [ ! -f .venv/bin/activate ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    source .venv/bin/activate
fi

# HuggingFace mirror (for downloading FineWeb shards)
export HF_ENDPOINT="https://hf-mirror.com"
export HUGGINGFACE_HUB_ENDPOINT="https://hf-mirror.com"

# ── 2. Download data (skip if already exists) ───────────────────
echo ""
echo "[1/3] Checking FineWeb shards..."
python exemplars/text_pretrain/data/download_shards.py 2>&1 | tee "$LOG_DIR/download.log"

# ── 3. Run experiment ───────────────────────────────────────────
echo ""
echo "[2/3] Running FFN expansion ablation..."
python "$PROJECT/run.py" 2>&1 | tee "$LOG_DIR/run.log"

# ── 4. Plot results ─────────────────────────────────────────────
echo ""
echo "[3/3] Plotting results..."
python "$PROJECT/plot.py" 2>&1 | tee "$LOG_DIR/plot.log"

# ── 5. Git commit & push ────────────────────────────────────────
echo ""
echo "========================================"
echo "Committing & pushing..."
echo "========================================"

git add "$PROJECT/"

# Only commit if there are staged changes
if git diff --cached --quiet; then
    echo "Nothing to commit."
else
    git commit -m "yuze: FFN expansion ablation results (2x/4x/6x/8x)"
    git push origin yuze/test
fi

echo ""
echo "Done! Results:"
echo "  JSON : $PROJECT/results/curves.json"
echo "  PNG  : $PROJECT/ffn_expansion.png"
echo "  Logs : $LOG_DIR/"
