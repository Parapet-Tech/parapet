"""Build a stratified L2 latency-bench corpus from curated train/val sources.

Thin CLI wrapper around ``parapet_runner.l2_corpus``. The script's job is
argument shape; all sampling, hashing, and manifest assembly lives in the
package so it stays unit-tested.

Typical use (matches the v8 layout on disk):

    python scripts/build_l2_latency_corpus.py \\
        --source ../parapet-data/curated/v8_jsonl/train.yaml \\
        --source ../parapet-data/curated/v8_jsonl/val.yaml \\
        --output runs/l2_latency_v8/l2_latency_v8_train_val_stratified.jsonl \\
        --target-rows 5000 \\
        --seed 42

Output is a strict JSONL file plus a sidecar ``<output>.manifest.json``.
The output sits under ``parapet-runner/runs/`` per AGENTS.md and is
gitignored — only the script + module are committed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from parapet_runner.l2_corpus import (
    StratifySpec,
    build_l2_latency_corpus,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_l2_latency_corpus.py",
        description=(
            "Build a stratified L2 latency-bench corpus from one or more "
            "curated source files (train/val). Output is JSONL with a "
            "sidecar manifest."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Source corpus file (.yaml/.yml/.jsonl). Pass multiple times.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--axes",
        nargs="+",
        default=["language", "length_bucket"],
        help=(
            "Stratification axes (subset of: language, label, reason, source, "
            "length_bucket). Default: language length_bucket."
        ),
    )
    parser.add_argument(
        "--length-buckets",
        nargs="+",
        type=int,
        default=[128, 512, 2048],
        help="Character-length bucket edges. Default: 128 512 2048.",
    )
    parser.add_argument("--text-field", type=str, default="content")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = StratifySpec(
        target_rows=args.target_rows,
        seed=args.seed,
        axes=tuple(args.axes),
        length_buckets=tuple(args.length_buckets),
        text_field=args.text_field,
    )
    manifest = build_l2_latency_corpus(args.source, spec, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    print(
        f"wrote {manifest.n_output_rows} rows from {manifest.n_input_rows} input rows\n"
        f"  output:        {args.output}\n"
        f"  manifest:      {manifest_path}\n"
        f"  output_sha256: {manifest.output_sha256}\n"
        f"  axes:          {manifest.stratify_axes}\n"
        f"  buckets:       {manifest.length_buckets}\n"
        f"  seed:          {manifest.seed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
