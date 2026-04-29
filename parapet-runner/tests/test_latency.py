from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from parapet_runner.latency import (
    CorpusError,
    EncodedInput,
    LatencyConfig,
    LatencyManifest,
    _histogram,
    _percentiles,
    _quantile,
    measure_latency,
)


# ---------------------------------------------------------------------------
# Fakes (no torch, no onnxruntime, no transformers)
# ---------------------------------------------------------------------------


class FakeClock:
    """Returns prescribed timestamps in order. One call = one tick."""

    def __init__(self, values: Iterable[float]) -> None:
        self._values = list(values)
        self._idx = 0

    def __call__(self) -> float:
        if self._idx >= len(self._values):
            raise AssertionError(
                f"FakeClock exhausted after {self._idx} calls; "
                f"only {len(self._values)} values were prescribed"
            )
        value = self._values[self._idx]
        self._idx += 1
        return value


class FakeTokenizer:
    """Returns deterministic EncodedInput with prescribed token counts."""

    def __init__(self, token_counts: Iterable[int]) -> None:
        self._counts = list(token_counts)
        self._idx = 0

    def encode(self, text: str, max_len: int) -> EncodedInput:
        if self._idx >= len(self._counts):
            count = self._counts[-1]  # cycle on last value if caller goes long
        else:
            count = self._counts[self._idx]
            self._idx += 1
        truncated = min(count, max_len)
        return EncodedInput(
            inputs={
                "input_ids": np.zeros((1, truncated), dtype=np.int64),
                "attention_mask": np.ones((1, truncated), dtype=np.int64),
            },
            token_count=truncated,
        )


class NoOpSession:
    def infer(self, encoded: EncodedInput) -> Any:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_config(
    *,
    warmup: int = 2,
    measure: int = 4,
    allow_cycle: bool = False,
    histogram_bucket_edges: tuple[int, ...] | None = None,
) -> LatencyConfig:
    edges = histogram_bucket_edges if histogram_bucket_edges is not None else (32, 64, 128)
    return LatencyConfig(
        model_path=Path("/fake/model.onnx"),
        tokenizer_path=Path("/fake/tokenizer"),
        model_revision="fake@deadbeef",
        onnx_sha256="0" * 64,
        quant="int8",
        warmup_calls=warmup,
        measure_calls=measure,
        max_seq_len=512,
        allow_cycle=allow_cycle,
        histogram_bucket_edges=edges,
    )


def make_manifest(**overrides: Any) -> LatencyManifest:
    base: dict[str, Any] = dict(
        git_sha="deadbeef",
        python_version="3.11.0",
        numpy_version="1.26.0",
        ort_version=None,
        environment="local",
        hardware_string="fake-cpu / 1 core",
        provider="CPUExecutionProvider",
        batch_size=1,
        model_revision="fake@deadbeef",
        onnx_sha256="0" * 64,
        tokenizer_files_sha256="1" * 64,
        corpus_sha256="2" * 64,
        corpus_kind="real",
        corpus_path_recorded="fixtures/corpus.jsonl",
    )
    base.update(overrides)
    return LatencyManifest(**base)


def synthetic_clock_values(
    n_iterations: int,
    *,
    tokenize_ms: float,
    infer_ms: float,
) -> list[float]:
    """Build clock ticks producing exactly ``tokenize_ms`` per encode and
    ``infer_ms`` per infer, for ``n_iterations`` iterations.

    Each iteration consumes 3 ticks: t0, t1=t0+tokenize, t2=t1+infer.
    """
    values: list[float] = []
    cursor = 0.0
    tokenize_s = tokenize_ms / 1000.0
    infer_s = infer_ms / 1000.0
    for _ in range(n_iterations):
        values.append(cursor)
        cursor += tokenize_s
        values.append(cursor)
        cursor += infer_s
        values.append(cursor)
    return values


# ---------------------------------------------------------------------------
# measure_latency: behavior
# ---------------------------------------------------------------------------


def test_warmup_calls_excluded_from_measurement() -> None:
    """Tokenize fast during warmup, slow during measure → percentiles see only the slow times."""
    config = make_config(warmup=2, measure=3)
    n_iter = config.warmup_calls + config.measure_calls

    # Warmup: 1ms tokenize / 5ms infer. Measure: 10ms tokenize / 50ms infer.
    warmup_ticks = synthetic_clock_values(config.warmup_calls, tokenize_ms=1.0, infer_ms=5.0)
    measure_ticks_relative = synthetic_clock_values(
        config.measure_calls, tokenize_ms=10.0, infer_ms=50.0
    )
    # Stitch — measure ticks continue from end of warmup.
    measure_offset = warmup_ticks[-1] if warmup_ticks else 0.0
    measure_ticks = [t + measure_offset for t in measure_ticks_relative]
    ticks = warmup_ticks + measure_ticks

    result = measure_latency(
        session=NoOpSession(),
        tokenizer=FakeTokenizer([10] * n_iter),
        corpus=["x"] * n_iter,
        config=config,
        manifest=make_manifest(),
        clock=FakeClock(ticks),
    )

    # Only measure-phase timings should appear.
    assert result.tokenize.p50_ms == pytest.approx(10.0, abs=0.01)
    assert result.infer.p50_ms == pytest.approx(50.0, abs=0.01)
    assert result.end_to_end.p50_ms == pytest.approx(60.0, abs=0.01)
    assert result.n_calls == config.measure_calls


def test_n_calls_equals_measure_calls_not_total_iterations() -> None:
    config = make_config(warmup=5, measure=10)
    n_iter = config.warmup_calls + config.measure_calls
    ticks = synthetic_clock_values(n_iter, tokenize_ms=1.0, infer_ms=2.0)

    result = measure_latency(
        session=NoOpSession(),
        tokenizer=FakeTokenizer([5] * n_iter),
        corpus=["x"] * n_iter,
        config=config,
        manifest=make_manifest(),
        clock=FakeClock(ticks),
    )
    assert result.n_calls == 10


def test_end_to_end_equals_tokenize_plus_infer_per_call() -> None:
    config = make_config(warmup=0, measure=4)
    ticks = synthetic_clock_values(4, tokenize_ms=2.0, infer_ms=8.0)

    result = measure_latency(
        session=NoOpSession(),
        tokenizer=FakeTokenizer([5] * 4),
        corpus=["x"] * 4,
        config=config,
        manifest=make_manifest(),
        clock=FakeClock(ticks),
    )

    # All four samples are identical → percentiles equal the per-call totals.
    assert result.tokenize.p50_ms == pytest.approx(2.0, abs=0.01)
    assert result.infer.p50_ms == pytest.approx(8.0, abs=0.01)
    assert result.end_to_end.p50_ms == pytest.approx(10.0, abs=0.01)


def test_token_length_histogram_buckets_correctly() -> None:
    config = make_config(warmup=0, measure=6, histogram_bucket_edges=(32, 64, 128))
    ticks = synthetic_clock_values(6, tokenize_ms=1.0, infer_ms=1.0)

    # Lengths chosen to hit each bucket.
    counts = [10, 32, 33, 64, 100, 500]  # → "0-32":2, "33-64":2, "65-128":1, "129+":1
    result = measure_latency(
        session=NoOpSession(),
        tokenizer=FakeTokenizer(counts),
        corpus=["x"] * 6,
        config=config,
        manifest=make_manifest(),
        clock=FakeClock(ticks),
    )

    assert result.token_length_histogram == {
        "0-32": 2,
        "33-64": 2,
        "65-128": 1,
        "129+": 1,
    }


# ---------------------------------------------------------------------------
# measure_latency: corpus shortfall
# ---------------------------------------------------------------------------


def test_corpus_shortfall_raises_when_allow_cycle_false() -> None:
    config = make_config(warmup=2, measure=4, allow_cycle=False)
    # Need 6 rows; provide 3.
    with pytest.raises(CorpusError, match="3 rows; need 6"):
        measure_latency(
            session=NoOpSession(),
            tokenizer=FakeTokenizer([5] * 6),
            corpus=["a", "b", "c"],
            config=config,
            manifest=make_manifest(),
            clock=FakeClock(synthetic_clock_values(6, tokenize_ms=1.0, infer_ms=1.0)),
        )


def test_corpus_shortfall_cycles_when_allow_cycle_true() -> None:
    config = make_config(warmup=0, measure=6, allow_cycle=True)
    # 3 rows cycled twice gives 6 calls.
    ticks = synthetic_clock_values(6, tokenize_ms=1.0, infer_ms=1.0)

    captured: list[str] = []

    class CapturingTokenizer:
        def encode(self, text: str, max_len: int) -> EncodedInput:
            captured.append(text)
            return EncodedInput(
                inputs={"input_ids": np.zeros((1, 5), dtype=np.int64)},
                token_count=5,
            )

    measure_latency(
        session=NoOpSession(),
        tokenizer=CapturingTokenizer(),
        corpus=["a", "b", "c"],
        config=config,
        manifest=make_manifest(),
        clock=FakeClock(ticks),
    )

    assert captured == ["a", "b", "c", "a", "b", "c"]


def test_empty_corpus_raises() -> None:
    config = make_config(warmup=0, measure=1, allow_cycle=True)
    with pytest.raises(CorpusError, match="Empty corpus"):
        measure_latency(
            session=NoOpSession(),
            tokenizer=FakeTokenizer([5]),
            corpus=[],
            config=config,
            manifest=make_manifest(),
            clock=FakeClock([0.0, 0.001, 0.002]),
        )


# ---------------------------------------------------------------------------
# LatencyConfig: construction-time validation
# ---------------------------------------------------------------------------


def _kwargs_for_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        model_path=Path("/fake/model.onnx"),
        tokenizer_path=Path("/fake/tokenizer"),
        model_revision="fake@deadbeef",
        onnx_sha256="0" * 64,
        quant="int8",
    )
    base.update(overrides)
    return base


def test_config_rejects_zero_measure_calls() -> None:
    with pytest.raises(ValueError, match="measure_calls must be > 0"):
        LatencyConfig(**_kwargs_for_config(measure_calls=0))


def test_config_rejects_negative_warmup() -> None:
    with pytest.raises(ValueError, match="warmup_calls must be >= 0"):
        LatencyConfig(**_kwargs_for_config(warmup_calls=-1))


def test_config_rejects_zero_max_seq_len() -> None:
    with pytest.raises(ValueError, match="max_seq_len must be > 0"):
        LatencyConfig(**_kwargs_for_config(max_seq_len=0))


def test_config_rejects_zero_intra_op_threads() -> None:
    with pytest.raises(ValueError, match="intra_op_threads must be >= 1"):
        LatencyConfig(**_kwargs_for_config(intra_op_threads=0))


def test_config_rejects_zero_inter_op_threads() -> None:
    with pytest.raises(ValueError, match="inter_op_threads must be >= 1"):
        LatencyConfig(**_kwargs_for_config(inter_op_threads=0))


def test_config_rejects_blank_model_revision() -> None:
    with pytest.raises(ValueError, match="model_revision must be non-empty"):
        LatencyConfig(**_kwargs_for_config(model_revision="   "))


def test_config_rejects_non_hex_sha() -> None:
    with pytest.raises(ValueError, match="64-char hex"):
        LatencyConfig(**_kwargs_for_config(onnx_sha256="not-a-hash"))


def test_config_rejects_short_sha() -> None:
    with pytest.raises(ValueError, match="64-char hex"):
        LatencyConfig(**_kwargs_for_config(onnx_sha256="abc"))


def test_config_rejects_empty_histogram_edges() -> None:
    with pytest.raises(ValueError, match="histogram_bucket_edges must be non-empty"):
        LatencyConfig(**_kwargs_for_config(histogram_bucket_edges=()))


def test_config_rejects_negative_histogram_edges() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LatencyConfig(**_kwargs_for_config(histogram_bucket_edges=(-1, 32)))


def test_config_accepts_uppercase_hex_sha() -> None:
    LatencyConfig(**_kwargs_for_config(onnx_sha256="A" * 64))


# ---------------------------------------------------------------------------
# measure_latency: config/manifest consistency
# ---------------------------------------------------------------------------


def test_measure_latency_rejects_revision_mismatch() -> None:
    config = make_config(warmup=0, measure=2)
    manifest = make_manifest().model_copy(update={"model_revision": "different@cafef00d"})
    with pytest.raises(ValueError, match="model_revision"):
        measure_latency(
            session=NoOpSession(),
            tokenizer=FakeTokenizer([5, 5]),
            corpus=["a", "b"],
            config=config,
            manifest=manifest,
            clock=FakeClock(synthetic_clock_values(2, tokenize_ms=1.0, infer_ms=1.0)),
        )


def test_measure_latency_rejects_sha_mismatch() -> None:
    config = make_config(warmup=0, measure=2)
    manifest = make_manifest().model_copy(update={"onnx_sha256": "1" * 64})
    with pytest.raises(ValueError, match="onnx_sha256"):
        measure_latency(
            session=NoOpSession(),
            tokenizer=FakeTokenizer([5, 5]),
            corpus=["a", "b"],
            config=config,
            manifest=manifest,
            clock=FakeClock(synthetic_clock_values(2, tokenize_ms=1.0, infer_ms=1.0)),
        )


def test_measure_latency_sha_compare_is_case_insensitive() -> None:
    # Config has lowercase, manifest has uppercase — same hash, should pass.
    config = make_config(warmup=0, measure=2)
    manifest = make_manifest().model_copy(update={"onnx_sha256": "0" * 64})  # both all-zero, same case
    # Now flip manifest to uppercase form of the same digest.
    manifest = manifest.model_copy(update={"onnx_sha256": manifest.onnx_sha256.upper()})
    measure_latency(
        session=NoOpSession(),
        tokenizer=FakeTokenizer([5, 5]),
        corpus=["a", "b"],
        config=config,
        manifest=manifest,
        clock=FakeClock(synthetic_clock_values(2, tokenize_ms=1.0, infer_ms=1.0)),
    )


def test_measure_latency_rejects_provider_mismatch() -> None:
    config = make_config(warmup=0, measure=2)
    manifest = make_manifest(provider="CUDAExecutionProvider")
    with pytest.raises(ValueError, match="provider"):
        measure_latency(
            session=NoOpSession(),
            tokenizer=FakeTokenizer([5, 5]),
            corpus=["a", "b"],
            config=config,
            manifest=manifest,
            clock=FakeClock(synthetic_clock_values(2, tokenize_ms=1.0, infer_ms=1.0)),
        )


def test_measure_latency_rejects_batch_size_mismatch() -> None:
    # config.batch_size=1 (default, supported) but manifest.batch_size=2 → reject.
    config = make_config(warmup=0, measure=2)
    manifest = make_manifest(batch_size=2)
    with pytest.raises(ValueError, match="batch_size"):
        measure_latency(
            session=NoOpSession(),
            tokenizer=FakeTokenizer([5, 5]),
            corpus=["a", "b"],
            config=config,
            manifest=manifest,
            clock=FakeClock(synthetic_clock_values(2, tokenize_ms=1.0, infer_ms=1.0)),
        )


# ---------------------------------------------------------------------------
# measure_latency: batch_size > 1 is not yet implemented
# ---------------------------------------------------------------------------


def test_measure_latency_raises_not_implemented_for_batched_mode() -> None:
    config = LatencyConfig(**_kwargs_for_config(batch_size=2))
    with pytest.raises(NotImplementedError, match="batch_size=2"):
        measure_latency(
            session=NoOpSession(),
            tokenizer=FakeTokenizer([5]),
            corpus=["a"],
            config=config,
            manifest=make_manifest(batch_size=2),
            # If NotImplementedError fires before clock use, an exhausted clock
            # would still raise — but we put the check first, so this empty
            # clock proves the early-fail.
            clock=FakeClock([]),
        )


def test_batched_check_runs_before_consistency_check() -> None:
    # Even if the manifest is also wrong, batch_size != 1 must fire first
    # so the user sees the real blocker, not a downstream symptom.
    config = LatencyConfig(**_kwargs_for_config(batch_size=4))
    manifest = make_manifest(batch_size=1, model_revision="wrong@feedface")
    with pytest.raises(NotImplementedError, match="batch_size=4"):
        measure_latency(
            session=NoOpSession(),
            tokenizer=FakeTokenizer([5]),
            corpus=["a"],
            config=config,
            manifest=manifest,
            clock=FakeClock([]),
        )


# ---------------------------------------------------------------------------
# LatencyConfig batch_size validation
# ---------------------------------------------------------------------------


def test_config_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        LatencyConfig(**_kwargs_for_config(batch_size=0))


def test_config_rejects_negative_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        LatencyConfig(**_kwargs_for_config(batch_size=-1))


def test_config_default_batch_size_is_1() -> None:
    cfg = LatencyConfig(**_kwargs_for_config())
    assert cfg.batch_size == 1


def test_config_rejects_blank_provider() -> None:
    with pytest.raises(ValueError, match="provider must be non-empty"):
        LatencyConfig(**_kwargs_for_config(provider="   "))


# ---------------------------------------------------------------------------
# LatencyManifest validation (pydantic field validators)
# ---------------------------------------------------------------------------


def test_manifest_rejects_invalid_corpus_sha() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="64-char hex"):
        make_manifest(corpus_sha256="not-a-hash")


def test_manifest_rejects_invalid_onnx_sha() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="64-char hex"):
        make_manifest(onnx_sha256="abc")


def test_manifest_rejects_invalid_tokenizer_sha() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="64-char hex"):
        make_manifest(tokenizer_files_sha256="bad")


def test_manifest_rejects_unknown_environment() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        make_manifest(environment="staging")  # type: ignore[arg-type]


def test_manifest_rejects_zero_batch_size() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="batch_size must be >= 1"):
        make_manifest(batch_size=0)


def test_manifest_rejects_blank_provider() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="non-empty"):
        make_manifest(provider="   ")


def test_manifest_rejects_blank_hardware_string() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="non-empty"):
        make_manifest(hardware_string="")


def test_manifest_rejects_blank_corpus_path() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="non-empty"):
        make_manifest(corpus_path_recorded="")


def test_manifest_rejects_unknown_corpus_kind() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        make_manifest(corpus_kind="garbage")  # type: ignore[arg-type]


def test_manifest_accepts_real_synthetic_fixture_kinds() -> None:
    for kind in ("real", "synthetic", "fixture"):
        make_manifest(corpus_kind=kind)


# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------


def test_quantile_linear_interpolation_matches_numpy() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0]
    sorted_sample = sorted(sample)
    for q in (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
        expected = float(np.quantile(np.array(sample), q, method="linear"))
        assert _quantile(sorted_sample, q) == pytest.approx(expected, abs=1e-9), f"q={q}"


def test_quantile_single_value() -> None:
    assert _quantile([42.0], 0.5) == 42.0
    assert _quantile([42.0], 0.99) == 42.0


def test_quantile_rejects_out_of_range_q() -> None:
    with pytest.raises(ValueError, match="q must be in"):
        _quantile([1.0, 2.0], 1.5)
    with pytest.raises(ValueError, match="q must be in"):
        _quantile([1.0, 2.0], -0.1)


def test_quantile_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="Empty sample"):
        _quantile([], 0.5)


def test_percentiles_on_known_sample() -> None:
    # 1..100 ms; numpy percentiles for reference.
    sample = [float(x) for x in range(1, 101)]
    pct = _percentiles(sample)
    assert pct.p50_ms == pytest.approx(50.5, abs=1e-6)
    assert pct.p95_ms == pytest.approx(95.05, abs=1e-6)
    assert pct.p99_ms == pytest.approx(99.01, abs=1e-6)


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------


def test_histogram_top_bucket_is_unbounded() -> None:
    h = _histogram([0, 32, 33, 64, 65, 128, 129, 10_000], (32, 64, 128))
    assert h == {"0-32": 2, "33-64": 2, "65-128": 2, "129+": 2}


def test_histogram_empty_input_yields_zeros() -> None:
    h = _histogram([], (32, 64))
    assert h == {"0-32": 0, "33-64": 0, "65+": 0}


def test_histogram_dedupes_and_sorts_edges() -> None:
    h = _histogram([10, 50], (64, 32, 32))  # duplicates + unsorted
    assert h == {"0-32": 1, "33-64": 1, "65+": 0}


def test_histogram_rejects_negative_edges() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _histogram([10], (-1, 32))


def test_histogram_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _histogram([-1], (32,))


def test_histogram_rejects_empty_edges() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _histogram([10], ())
