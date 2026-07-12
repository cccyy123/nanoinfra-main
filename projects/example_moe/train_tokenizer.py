"""
Train a BPE tokenizer — same algorithm as GPT-2 / GPT-4.

Requires: pip install rustbpe tiktoken

Usage:
  # From FineWeb parquet shards in outputs/base_data/
  python projects/example_moe/train_tokenizer.py

  # From any plain text file (one document per line)
  python projects/example_moe/train_tokenizer.py --text-file corpus.txt
"""

import os, sys, argparse
from modalities.text.tokenizer import RustBPETokenizer
from modalities.control import CONTROL_TOKENS, display_form


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--num-docs", type=int, default=1_000_000)
    args = parser.parse_args()

    # ---- 1. control tokens ------------------------------------------------
    special_tokens = [display_form(n) for n in CONTROL_TOKENS]

    # ---- 2. text iterator ------------------------------------------------
    if args.text_file:
        def text_iter():
            with open(args.text_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        yield line
                    if args.num_docs and i >= args.num_docs:
                        break
        print(f"Reading from: {args.text_file}")
    else:
        data_dir = "outputs/base_data"
        if not os.path.isdir(data_dir):
            sys.exit(f"✗ {data_dir}/ not found. Download FineWeb shards or pass --text-file.")

        from modalities.text.fineweb import _document_batches
        def text_iter():
            n = 0
            for batch, _ in _document_batches("train", data_dir, {}, 256, 0, 1):
                for doc in batch:
                    yield doc
                    n += 1
                    if n >= args.num_docs:
                        return
        print(f"Streaming from: {data_dir}/")

    print(f"Vocab: {args.vocab_size} (content={args.vocab_size - len(special_tokens)} + control={len(special_tokens)})")
    print(f"Training (this takes a few minutes)...")

    # ---- 3. train ---------------------------------------------------------
    tok = RustBPETokenizer.train_from_iterator(
        text_iter(),
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )

    # ---- 4. save ----------------------------------------------------------
    out = "outputs/tokenizer"
    os.makedirs(out, exist_ok=True)
    tok.save(out)

    # ---- 5. verify --------------------------------------------------------
    test = "Hello world, this is a BPE tokenizer test."
    ids = tok.encode(test)
    print(f"\n✓ saved to {out}/")
    print(f"  vocab_size: {tok.get_vocab_size()}")
    print(f"  encode('{test}') → {ids}")
    print(f"  decode → '{tok.decode(ids)}'")


if __name__ == "__main__":
    main()
