"""L2 latency feasibility bench: types and pure measurement loop.

Implements direction.md Phase 0.6 — measure CPU p50/p99 latency for an ONNX
classifier candidate (mDeBERTa, MiniLM, ...) under single-thread, no-batching
conditions, against representative residual-shaped text.

This module is hardware- and ML-framework-agnostic. It contains zero direct
dependencies on torch, onnxruntime, or transformers. Real adapters live in
``latency_onnx.py`` behind the optional ``[bench]`` dep group; this module is
fully exercisable with fakes (see ``tests/test_latency.py``).

Threading model: callers must construct adapters with intra_op=1, inter_op=1
to honor the no-batching, single-thread gate. The bench loop itself is purely
sequential.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, field_validator

_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


# ---------------------------------------------------------------------------
# Injection boundaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodedInput:
    """Tokenized text ready for an ONNX session.

    ``inputs`` maps ONNX input names (``input_ids``, ``attention_mask``,
    optionally ``token_type_ids``) to their tensors. This indirection lets the
    same bench drive mDeBERTa (3 inputs), MiniLM (2 inputs), or any future
    encoder without core changes.

    ``token_count`` is the post-truncation token count, used to bucket the
    histogram in the result.
    """

    inputs: dict[str, np.ndarray]
    token_count: int


class Tokenizer(Protocol):
    """Boundary between text and token tensors."""

    def encode(self, text: str, max_len: int) -> EncodedInput: ...


class InferenceSession(Protocol):
    """Boundary between token tensors and raw model output.

    Output type is ``Any`` because the bench does not interpret logits — it
    only times the call. Real adapters return whatever onnxruntime hands back.
    """

    def infer(self, encoded: EncodedInput) -> Any: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# Default histogram edges in tokens, matching common short/medium/long bins.
_DEFAULT_HIST_EDGES: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048)


@dataclass(frozen=True)
class LatencyConfig:
    """Bench parameters, validated at construction.

    File-existence and on-disk hash verification are the adapter's job
    (``latency_onnx.py``); this dataclass enforces shape only — the contract
    that any downstream consumer (corpus loader, ONNX adapter, result
    serializer) is allowed to assume.
    """

    model_path: Path
    tokenizer_path: Path
    model_revision: str
    onnx_sha256: str
    quant: Literal["fp32", "int8"]
    provider: str = "CPUExecutionProvider"
    intra_op_threads: int = 1
    inter_op_threads: int = 1
    warmup_calls: int = 50
    measure_calls: int = 500
    max_seq_len: int = 512
    batch_size: int = 1
    allow_cycle: bool = False
    histogram_bucket_edges: tuple[int, ...] = _DEFAULT_HIST_EDGES

    def __post_init__(self) -> None:
        if self.warmup_calls < 0:
            raise ValueError(f"warmup_calls must be >= 0, got {self.warmup_calls}")
        if self.measure_calls <= 0:
            raise ValueError(f"measure_calls must be > 0, got {self.measure_calls}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")
        if self.intra_op_threads < 1:
            raise ValueError(f"intra_op_threads must be >= 1, got {self.intra_op_threads}")
        if self.inter_op_threads < 1:
            raise ValueError(f"inter_op_threads must be >= 1, got {self.inter_op_threads}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.model_revision.strip():
            raise ValueError("model_revision must be non-empty")
        if not _SHA256_HEX.match(self.onnx_sha256):
            raise ValueError(
                f"onnx_sha256 must be 64-char hex, got {self.onnx_sha256!r}"
            )
        if not self.histogram_bucket_edges:
            raise ValueError("histogram_bucket_edges must be non-empty")
        if any(e < 0 for e in self.histogram_bucket_edges):
            raise ValueError(
                f"histogram_bucket_edges must all be non-negative, got {self.histogram_bucket_edges}"
            )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyPercentiles:
    p50_ms: float
    p95_ms: float
    p99_ms: float


class LatencyManifest(BaseModel):
    """Reproducibility metadata. The adapter assembles this; the core embeds
    it verbatim into the result so a manifest fully identifies a bench run.

    Field groupings:
      * Toolchain: git_sha, python_version, numpy_version, ort_version
      * Environment + hardware: environment, hardware_string, provider
      * Run shape: batch_size
      * Artifact identities: model_revision, onnx_sha256, tokenizer_files_sha256,
        corpus_sha256, corpus_path_recorded

    ``environment`` is observational metadata only — ``measure_latency`` does
    not enforce it. ``provider`` and ``batch_size`` are enforced via
    consistency check against ``LatencyConfig``.
    """

    # Toolchain
    git_sha: str
    python_version: str
    numpy_version: str
    ort_version: str | None = None

    # Environment + hardware
    environment: Literal["kaggle", "colab", "local", "ci", "unknown"]
    hardware_string: str
    provider: str

    # Run shape
    batch_size: int

    # Artifact identities
    model_revision: str
    onnx_sha256: str
    tokenizer_files_sha256: str
    corpus_sha256: str
    corpus_kind: Literal["real", "synthetic", "fixture"]
    corpus_path_recorded: str

    @field_validator("onnx_sha256", "corpus_sha256", "tokenizer_files_sha256")
    @classmethod
    def _validate_sha256_hex(cls, v: str) -> str:
        if not _SHA256_HEX.match(v):
            raise ValueError(f"must be 64-char hex, got {v!r}")
        return v

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"batch_size must be >= 1, got {v}")
        return v

    @field_validator("provider", "hardware_string", "model_revision", "corpus_path_recorded")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v


@dataclass(frozen=True)
class LatencyResult:
    tokenize: LatencyPercentiles
    infer: LatencyPercentiles
    end_to_end: LatencyPercentiles
    n_calls: int
    token_length_histogram: dict[str, int]
    config: LatencyConfig
    manifest: LatencyManifest


class CorpusError(ValueError):
    """Raised when the corpus is too small to satisfy the requested call count
    and ``allow_cycle`` is False."""


# ---------------------------------------------------------------------------
# Measurement loop
# ---------------------------------------------------------------------------


def measure_latency(
    session: InferenceSession,
    tokenizer: Tokenizer,
    corpus: Iterable[str],
    config: LatencyConfig,
    manifest: LatencyManifest,
    clock: Callable[[], float] | None = None,
) -> LatencyResult:
    """Run the bench loop and return percentile results.

    Caller's responsibilities (enforced by the adapter, not here):
      - ``session`` was constructed with the threading from ``config``.
      - ``manifest.onnx_sha256`` was verified at session construction.
      - ``corpus`` text rows are non-eval (Phase 0.6 corpus policy).

    The loop tokenizes-then-infers for each call, separately timing both
    halves. Warmup calls are run but excluded from the percentile samples.
    """

    if config.batch_size != 1:
        raise NotImplementedError(
            f"batch_size={config.batch_size}: only batch_size=1 is implemented "
            "(Phase 0.6 gate is no-batching). Batched mode is a follow-up — "
            "do not silently fall back to sequential."
        )

    if clock is None:
        clock = time.perf_counter

    if config.model_revision != manifest.model_revision:
        raise ValueError(
            "Config/manifest mismatch on model_revision: "
            f"config={config.model_revision!r} vs manifest={manifest.model_revision!r}"
        )
    if config.onnx_sha256.lower() != manifest.onnx_sha256.lower():
        raise ValueError(
            "Config/manifest mismatch on onnx_sha256 — refusing to produce a "
            "result whose reproducibility metadata disagrees with its config"
        )
    if config.provider != manifest.provider:
        raise ValueError(
            "Config/manifest mismatch on provider: "
            f"config={config.provider!r} vs manifest={manifest.provider!r}"
        )
    if config.batch_size != manifest.batch_size:
        raise ValueError(
            "Config/manifest mismatch on batch_size: "
            f"config={config.batch_size} vs manifest={manifest.batch_size}"
        )

    needed = config.warmup_calls + config.measure_calls
    items = list(corpus)

    if not items:
        raise CorpusError("Empty corpus: cannot run latency bench")
    if len(items) < needed and not config.allow_cycle:
        raise CorpusError(
            f"Corpus has {len(items)} rows; need {needed} "
            f"(warmup_calls={config.warmup_calls} + measure_calls={config.measure_calls}). "
            "Pass LatencyConfig(allow_cycle=True) for development smoke checks."
        )

    tokenize_ms: list[float] = []
    infer_ms: list[float] = []
    end_to_end_ms: list[float] = []
    token_counts: list[int] = []

    for i in range(needed):
        text = items[i % len(items)]

        t0 = clock()
        encoded = tokenizer.encode(text, config.max_seq_len)
        t1 = clock()
        session.infer(encoded)
        t2 = clock()

        if i < config.warmup_calls:
            continue

        tokenize_ms.append((t1 - t0) * 1000.0)
        infer_ms.append((t2 - t1) * 1000.0)
        end_to_end_ms.append((t2 - t0) * 1000.0)
        token_counts.append(encoded.token_count)

    return LatencyResult(
        tokenize=_percentiles(tokenize_ms),
        infer=_percentiles(infer_ms),
        end_to_end=_percentiles(end_to_end_ms),
        n_calls=config.measure_calls,
        token_length_histogram=_histogram(token_counts, config.histogram_bucket_edges),
        config=config,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Percentiles + histogram (pure helpers)
# ---------------------------------------------------------------------------


def _percentiles(times_ms: list[float]) -> LatencyPercentiles:
    if not times_ms:
        raise ValueError("Cannot compute percentiles on empty sample")
    sorted_times = sorted(times_ms)
    return LatencyPercentiles(
        p50_ms=_quantile(sorted_times, 0.50),
        p95_ms=_quantile(sorted_times, 0.95),
        p99_ms=_quantile(sorted_times, 0.99),
    )


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between adjacent sorted values (NumPy 'linear')."""
    if not sorted_values:
        raise ValueError("Empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0,1], got {q}")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _histogram(token_counts: list[int], bucket_edges: tuple[int, ...]) -> dict[str, int]:
    """Bucket token counts into labeled ranges.

    Edges (32, 64, 128) produce buckets:
      "0-32", "33-64", "65-128", "129+"
    """
    if not bucket_edges:
        raise ValueError("histogram_bucket_edges must be non-empty")
    edges = sorted(set(bucket_edges))
    if any(e < 0 for e in edges):
        raise ValueError(f"histogram_bucket_edges must be non-negative, got {bucket_edges}")

    labels: list[tuple[str, int, int | None]] = []
    prev = 0
    for edge in edges:
        labels.append((f"{prev}-{edge}", prev, edge))
        prev = edge + 1
    labels.append((f"{prev}+", prev, None))

    histogram: dict[str, int] = {label: 0 for label, _, _ in labels}
    for count in token_counts:
        if count < 0:
            raise ValueError(f"Token count must be non-negative, got {count}")
        for label, low, high in labels:
            if high is None:
                if count >= low:
                    histogram[label] += 1
                    break
            elif low <= count <= high:
                histogram[label] += 1
                break
    return histogram
