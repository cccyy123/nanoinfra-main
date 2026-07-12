"""Train the text BPE tokenizer used by nanoinfra.

The resulting ``tokenizer.pkl`` contains both the learned text vocabulary and
the canonical control-token band required by ``modalities.text.train_text``.

Run from the repository root::

    python -u scripts/train_tokenizer.py

By default, parquet files are read from ``outputs/base_data`` and the last
sorted shard is reserved for validation, matching the FineWeb loader's split
convention.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from modalities.control import CONTROL_TOKENS, display_form
from modalities.text.tokenizer import RustBPETokenizer


DEFAULT_VOCAB_SIZE = 50_304
DEFAULT_MAX_DOCUMENTS = 1_000_000


def default_data_dir() -> Path:
    """Resolve the same default data location used by the training pipeline."""
    explicit = os.environ.get("NANOINFRA_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser()

    base_dir = Path(os.environ.get("NANOINFRA_BASE_DIR", "./outputs")).expanduser()
    return base_dir / "base_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train nanoinfra's RustBPE/tiktoken tokenizer from parquet text."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Directory containing FineWeb parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tokenizer"),
        help="Directory in which tokenizer.pkl will be saved.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Total vocabulary size, including control tokens.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=DEFAULT_MAX_DOCUMENTS,
        help="Maximum training documents; use 0 to consume all train shards.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Number of parquet rows read at a time.",
    )
    return parser.parse_args()


def find_train_shards(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {data_dir.resolve()}. "
            "Download FineWeb shards before training the tokenizer."
        )

    # FineWeb's project convention is: all but the last shard are train, and
    # the last shard is validation. A single shard is accepted for development.
    return files[:-1] if len(files) > 1 else files


def iter_texts(
    train_files: list[Path],
    max_documents: int,
    batch_size: int,
) -> Iterator[str]:
    yielded = 0

    for path in train_files:
        print(f"[data] reading {path}", flush=True)
        parquet = pq.ParquetFile(path)

        if "text" not in parquet.schema.names:
            raise ValueError(f"Parquet shard has no 'text' column: {path}")

        for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
            # A RecordBatch.column call uses a numeric index across pyarrow
            # versions; only the requested text column is present here.
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str) or not text:
                    continue

                yield text
                yielded += 1

                if max_documents > 0 and yielded >= max_documents:
                    print(f"[data] reached {yielded:,} documents", flush=True)
                    return

    print(f"[data] consumed {yielded:,} documents", flush=True)


def validate_tokenizer(tokenizer: RustBPETokenizer, special_tokens: list[str]) -> None:
    actual_vocab_size = tokenizer.get_vocab_size()
    actual_specials = tokenizer.get_special_tokens()
    if len(actual_specials) != len(special_tokens):
        raise RuntimeError(
            f"Expected {len(special_tokens)} special tokens, got {len(actual_specials)}"
        )

    control_offset = actual_vocab_size - len(special_tokens)
    for local_id, token in enumerate(special_tokens):
        expected_id = control_offset + local_id
        actual_id = tokenizer.encode_special(token)
        if actual_id != expected_id:
            raise RuntimeError(
                f"Control-token layout mismatch for {token}: "
                f"expected {expected_id}, got {actual_id}"
            )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    if args.vocab_size <= len(CONTROL_TOKENS) + 256:
        raise ValueError(
            "vocab-size must leave at least 256 byte tokens after reserving "
            "the control-token band"
        )
    if args.max_documents < 0:
        raise ValueError("max-documents must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    train_files = find_train_shards(data_dir)
    special_tokens = [display_form(name) for name in CONTROL_TOKENS]

    print(f"[config] data directory: {data_dir.resolve()}")
    print(f"[config] training shards: {len(train_files)}")
    print(f"[config] output directory: {output_dir.resolve()}")
    print(f"[config] total vocabulary: {args.vocab_size:,}")
    print(f"[config] control tokens: {len(special_tokens)}")
    limit = "all" if args.max_documents == 0 else f"{args.max_documents:,}"
    print(f"[config] document limit: {limit}")

    tokenizer = RustBPETokenizer.train_from_iterator(
        text_iterator=iter_texts(train_files, args.max_documents, args.batch_size),
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )
    validate_tokenizer(tokenizer, special_tokens)
    tokenizer.save(output_dir)

    artifact = output_dir / "tokenizer.pkl"
    print(f"[done] saved {artifact.resolve()}")
    print(f"[done] vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"[done] special tokens: {len(tokenizer.get_special_tokens())}")


if __name__ == "__main__":
    main()
