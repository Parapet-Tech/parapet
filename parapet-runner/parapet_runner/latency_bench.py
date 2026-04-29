"""Argparse entry-point for the L2 latency bench (direction.md Phase 0.6).

Workflow contract (remote-first, local-confirm-last):

* Primary use is on Kaggle/Colab. ``--model-path``, ``--tokenizer-path``, and
  ``--corpus`` typically come from ``/kaggle/input/...``; ``--output`` lands
  in ``/kaggle/working/...``. ONNX export is performed by the notebook
  upstream (``optimum-cli export onnx ...``); this entry-point only loads,
  measures, and writes a result manifest.
* Local sanity: ``--local-smoke`` switches to a length-stratified synthetic
  corpus and shrinks measure_calls from 500 → 200 so a final local run is
  small enough to be a sanity check rather than a 30-minute experiment.

Lazy ML import: argument parsing, config construction, and corpus loading
do NOT require the ``[bench]`` dep group. Adapters in ``latency_onnx`` are
imported only inside ``main()``, which is the call path that actually runs
inference. This keeps fast tests in the default ``dev`` environment.

No-network discipline is enforced by the adapters in ``latency_onnx``;
this entry-point passes through filesystem paths and never resolves model
ids against the HuggingFace Hub.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from .latency import (
    InferenceSession,
    LatencyConfig,
    LatencyManifest,
    LatencyResult,
    Tokenizer,
    measure_latency,
)
from .latency_corpus import (
    compute_corpus_sha256,
    load_corpus,
    synthetic_length_stratified,
)


# A length-stratified histogram covering short/medium/long bins. Total rows:
# 80 + 150 + 80 = 310. Enough for default warmup (50) + smoke measure (200)
# = 250 without cycling. Lengths are character lengths, not token counts —
# see synthetic_length_stratified docstring for the approximation caveat.
_DEFAULT_SMOKE_HISTOGRAM: dict[int, int] = {
    32: 80,
    128: 150,
    512: 80,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m parapet_runner.latency_bench",
        description="L2 latency feasibility bench (direction.md Phase 0.6).",
    )

    # Model + tokenizer (required, local paths)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--model-revision", type=str, required=True)
    parser.add_argument("--onnx-sha256", type=str, required=True)
    parser.add_argument("--quant", choices=["fp32", "int8"], required=True)

    # Output (required)
    parser.add_argument("--output", type=Path, required=True)

    # Corpus selection (one of: --corpus or --local-smoke)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to a strict .jsonl/.yaml/.yml corpus file. Required unless --local-smoke is set.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="content",
        help="Field name for text in JSONL/YAML object rows (default: content).",
    )
    parser.add_argument(
        "--corpus-kind",
        choices=["real", "synthetic", "fixture"],
        default=None,
        help="Override the corpus_kind label written to the manifest. "
        "Default: 'real' for --corpus, 'synthetic' for --local-smoke without --corpus.",
    )
    parser.add_argument(
        "--corpus-path-recorded",
        type=str,
        default=None,
        help="Override the corpus_path_recorded label. Default: the --corpus CLI "
        "string (preserving relative paths, NOT resolving to absolute), or "
        "'synthetic:length_stratified:seed=<N>' for synthetic corpora.",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=42,
        help="Seed for synthetic length-stratified corpus (only used with --local-smoke without --corpus).",
    )
    parser.add_argument("--local-smoke", action="store_true")

    # Bench knobs
    parser.add_argument(
        "--provider", type=str, default="CPUExecutionProvider"
    )
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--measure", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument("--allow-cycle", action="store_true")
    parser.add_argument(
        "--environment",
        choices=["kaggle", "colab", "local", "ci", "unknown"],
        default=None,
        help="Override environment auto-detection.",
    )

    args = parser.parse_args(argv)

    if args.corpus is None and not args.local_smoke:
        parser.error("Either --corpus PATH or --local-smoke must be provided.")

    # Smoke-mode default overrides (only when user didn't set them explicitly).
    if args.warmup is None:
        args.warmup = 50  # same default smoke vs full
    if args.measure is None:
        args.measure = 200 if args.local_smoke else 500

    return args


# ---------------------------------------------------------------------------
# Config + corpus assembly (no ML deps)
# ---------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> LatencyConfig:
    return LatencyConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        model_revision=args.model_revision,
        onnx_sha256=args.onnx_sha256,
        quant=args.quant,
        provider=args.provider,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
        warmup_calls=args.warmup,
        measure_calls=args.measure,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        allow_cycle=args.allow_cycle,
    )


def build_corpus(
    args: argparse.Namespace,
) -> tuple[list[str], Literal["real", "synthetic", "fixture"], str]:
    """Materialize the corpus, decide its kind label, and resolve the path
    recorded into the manifest."""

    if args.corpus is not None:
        rows = list(load_corpus(args.corpus, text_field=args.text_field))
        kind: Literal["real", "synthetic", "fixture"] = args.corpus_kind or "real"
        # Record the user-supplied CLI string, not an absolute resolution —
        # absolute paths leak machine/user details into the manifest.
        recorded = args.corpus_path_recorded or str(args.corpus)
        return rows, kind, recorded

    rows = list(synthetic_length_stratified(_DEFAULT_SMOKE_HISTOGRAM, seed=args.synthetic_seed))
    kind = args.corpus_kind or "synthetic"
    recorded = (
        args.corpus_path_recorded
        or f"synthetic:length_stratified:seed={args.synthetic_seed}"
    )
    return rows, kind, recorded


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def serialize_result(result: LatencyResult) -> dict[str, Any]:
    """Convert a LatencyResult to a JSON-serializable dict."""

    return {
        "tokenize": asdict(result.tokenize),
        "infer": asdict(result.infer),
        "end_to_end": asdict(result.end_to_end),
        "n_calls": result.n_calls,
        "token_length_histogram": result.token_length_histogram,
        "config": _config_to_dict(result.config),
        "manifest": result.manifest.model_dump(mode="json"),
    }


def _config_to_dict(config: LatencyConfig) -> dict[str, Any]:
    return {
        "model_path": str(config.model_path),
        "tokenizer_path": str(config.tokenizer_path),
        "model_revision": config.model_revision,
        "onnx_sha256": config.onnx_sha256,
        "quant": config.quant,
        "provider": config.provider,
        "intra_op_threads": config.intra_op_threads,
        "inter_op_threads": config.inter_op_threads,
        "warmup_calls": config.warmup_calls,
        "measure_calls": config.measure_calls,
        "max_seq_len": config.max_seq_len,
        "batch_size": config.batch_size,
        "allow_cycle": config.allow_cycle,
        "histogram_bucket_edges": list(config.histogram_bucket_edges),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_bench(
    *,
    config: LatencyConfig,
    manifest: LatencyManifest,
    corpus: Iterable[str],
    session: InferenceSession,
    tokenizer: Tokenizer,
    output_path: Path,
) -> LatencyResult:
    """Run the bench and write the serialized result to ``output_path``.

    No ML deps here — adapters are injected. This is the integration point
    tests can exercise with fakes.
    """

    result = measure_latency(
        session=session,
        tokenizer=tokenizer,
        corpus=corpus,
        config=config,
        manifest=manifest,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serialize_result(result), indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point. Lazy-imports ``latency_onnx`` to keep arg parsing
    available without the ``[bench]`` dep group."""

    args = _parse_args(argv)

    rows, corpus_kind, corpus_path_recorded = build_corpus(args)
    corpus_sha = compute_corpus_sha256(rows)
    config = build_config(args)

    # Lazy: only here do we require [bench] deps.
    from .latency_onnx import HfTokenizerAdapter, OrtInferenceSession, build_manifest

    tokenizer = HfTokenizerAdapter(args.tokenizer_path)
    session = OrtInferenceSession(
        args.model_path,
        expected_sha256=args.onnx_sha256,
        provider=args.provider,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
    )
    manifest = build_manifest(
        config=config,
        tokenizer_path=args.tokenizer_path,
        corpus_sha256=corpus_sha,
        corpus_kind=corpus_kind,
        corpus_path_recorded=corpus_path_recorded,
        environment=args.environment,
    )

    result = run_bench(
        config=config,
        manifest=manifest,
        corpus=rows,
        session=session,
        tokenizer=tokenizer,
        output_path=args.output,
    )

    print(
        f"end_to_end p50={result.end_to_end.p50_ms:.2f}ms "
        f"p95={result.end_to_end.p95_ms:.2f}ms "
        f"p99={result.end_to_end.p99_ms:.2f}ms "
        f"(n_calls={result.n_calls}, env={result.manifest.environment}, "
        f"hw={result.manifest.hardware_string})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
