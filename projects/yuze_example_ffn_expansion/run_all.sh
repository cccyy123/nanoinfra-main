#!/usr/bin/env bash
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PROJECT="projects/yuze_example_ffn_expansion"
RUN_TS=$(date '+%Y%m%d_%H%M%S')
RUN_DIR="$PROJECT/results/$RUN_TS"
LOG_DIR="$RUN_DIR/logs"
export RESULTS_DIR="$RUN_DIR"

mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log

NOW=$(date '+%Y-%m-%d %H:%M:%S')
echo "========================================"
echo "FFN Expansion Ablation Pipeline"
echo "Start : $NOW"
echo "Project: $PROJECT"
echo "Run    : $RUN_TS"
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

# ── 1b. Install extra deps not in pyproject.toml ────────────────
echo ""
echo "Checking extra dependencies..."
pip install -q pyarrow tiktoken rustbpe matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tee "$LOG_DIR/deps.log"

# ── 2. Download data (skip if already exists) ───────────────────
echo ""
echo "[1/5] Checking FineWeb shards..."
python exemplars/text_pretrain/data/download_shards.py 2>&1 | tee "$LOG_DIR/download.log"

# ── 3. Train tokenizer (skip if already exists) ──────────────────
echo ""
echo "[2/5] Checking tokenizer..."
if [ ! -f outputs/tokenizer/tokenizer.pkl ]; then
    echo "Training tokenizer (50274 vocab, FineWeb data)..."
    python -c "
from pathlib import Path
import pyarrow.parquet as pq
from modalities.control import CONTROL_TOKENS, display_form
from modalities.text.tokenizer import RustBPETokenizer

specials = [display_form(n) for n in CONTROL_TOKENS]
shards = sorted(Path('outputs/base_data').glob('shard_*_00000.parquet'))
print(f'Shards: {len(shards)}, specials: {len(specials)}')

def stream():
    for s in shards:
        for row in pq.read_table(s, columns=['text']).to_pylist():
            yield row['text']

tok = RustBPETokenizer.train_from_iterator(stream(), 50274, specials)
tok.save('outputs/tokenizer')
print(f'Done — vocab={tok.get_vocab_size()}')
" 2>&1 | tee "$LOG_DIR/tokenizer.log"
else
    echo "Tokenizer already exists, skip."
fi

# ── 4. Run experiment ───────────────────────────────────────────
echo ""
echo "[3/5] Running FFN expansion ablation..."
python "$PROJECT/run.py" 2>&1 | tee "$LOG_DIR/run.log"

# ── 5. Plot results ─────────────────────────────────────────────
echo ""
echo "[4/5] Plotting results..."
python "$PROJECT/plot.py" 2>&1 | tee "$LOG_DIR/plot.log"

# ── 6. Git commit & push ────────────────────────────────────────
echo ""
echo "========================================"
echo "[5/5] Committing & pushing..."
echo "========================================"

# ensure git identity
git config user.email "cccyy123@users.noreply.github.com" 2>/dev/null || true
git config user.name "cccyy123" 2>/dev/null || true

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
echo "  Dir  : $RUN_DIR"
echo "  JSON : $RUN_DIR/curves.json"
echo "  PNG  : $RUN_DIR/ffn_expansion.png"
echo "  YAML : $RUN_DIR/experiment.yaml"
echo "  Logs : $LOG_DIR/"
