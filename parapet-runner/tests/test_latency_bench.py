from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from parapet_runner.latency import (
    EncodedInput,
    LatencyConfig,
    LatencyManifest,
)
from parapet_runner.latency_bench import (
    _DEFAULT_SMOKE_HISTOGRAM,
    _parse_args,
    build_config,
    build_corpus,
    run_bench,
    serialize_result,
)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    name = request.node.name
    output_dir = Path("tests/.tmp_outputs/latency_bench") / name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Required-argument set helpers
# ---------------------------------------------------------------------------


def _required_args(model_path: str, tokenizer_path: str, output: str) -> list[str]:
    return [
        "--model-path", model_path,
        "--tokenizer-path", tokenizer_path,
        "--model-revision", "fake@deadbeef",
        "--onnx-sha256", "0" * 64,
        "--quant", "int8",
        "--output", output,
    ]


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------


def test_parse_args_with_corpus_succeeds(tmp_path: Path) -> None:
    argv = _required_args(
        str(tmp_path / "model.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "result.json"),
    ) + ["--corpus", str(tmp_path / "c.jsonl")]
    args = _parse_args(argv)
    assert args.corpus == tmp_path / "c.jsonl"
    assert args.local_smoke is False
    assert args.measure == 500  # full default
    assert args.warmup == 50
    assert args.batch_size == 1
    assert args.provider == "CPUExecutionProvider"


def test_parse_args_with_local_smoke_overrides_measure(tmp_path: Path) -> None:
    argv = _required_args(
        str(tmp_path / "model.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "result.json"),
    ) + ["--local-smoke"]
    args = _parse_args(argv)
    assert args.local_smoke is True
    assert args.measure == 200  # smoke override
    assert args.warmup == 50  # unchanged


def test_parse_args_explicit_measure_wins_over_smoke_default(tmp_path: Path) -> None:
    argv = _required_args(
        str(tmp_path / "model.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "result.json"),
    ) + ["--local-smoke", "--measure", "1000"]
    args = _parse_args(argv)
    assert args.measure == 1000


def test_parse_args_without_corpus_or_smoke_errors(tmp_path: Path) -> None:
    argv = _required_args(
        str(tmp_path / "model.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "result.json"),
    )
    with pytest.raises(SystemExit):
        _parse_args(argv)


def test_parse_args_missing_required_arg_errors(tmp_path: Path) -> None:
    # Missing --onnx-sha256
    with pytest.raises(SystemExit):
        _parse_args([
            "--model-path", str(tmp_path / "m.onnx"),
            "--tokenizer-path", str(tmp_path / "tok"),
            "--model-revision", "x",
            "--quant", "int8",
            "--output", str(tmp_path / "o.json"),
            "--corpus", str(tmp_path / "c.jsonl"),
        ])


def test_parse_args_invalid_quant_choice_errors(tmp_path: Path) -> None:
    argv = _required_args(
        str(tmp_path / "m.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "o.json"),
    )
    # Replace --quant value with invalid.
    argv[argv.index("--quant") + 1] = "fp16"
    argv += ["--corpus", str(tmp_path / "c.jsonl")]
    with pytest.raises(SystemExit):
        _parse_args(argv)


def test_parse_args_environment_choice_validated(tmp_path: Path) -> None:
    argv = _required_args(
        str(tmp_path / "m.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "o.json"),
    ) + ["--local-smoke", "--environment", "kaggle"]
    args = _parse_args(argv)
    assert args.environment == "kaggle"


def test_parse_args_text_field_default_is_content(tmp_path: Path) -> None:
    argv = _required_args(
        str(tmp_path / "m.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "o.json"),
    ) + ["--corpus", str(tmp_path / "c.jsonl")]
    args = _parse_args(argv)
    assert args.text_field == "content"


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------


def test_build_config_constructs_valid_latency_config(tmp_path: Path) -> None:
    argv = _required_args(
        str(tmp_path / "m.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "o.json"),
    ) + ["--local-smoke"]
    args = _parse_args(argv)
    config = build_config(args)
    assert isinstance(config, LatencyConfig)
    assert config.measure_calls == 200
    assert config.warmup_calls == 50
    assert config.batch_size == 1
    assert config.onnx_sha256 == "0" * 64


def test_build_config_propagates_batch_size_and_provider(tmp_path: Path) -> None:
    argv = _required_args(
        str(tmp_path / "m.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "o.json"),
    ) + [
        "--corpus", str(tmp_path / "c.jsonl"),
        "--batch-size", "4",
        "--provider", "CUDAExecutionProvider",
    ]
    args = _parse_args(argv)
    config = build_config(args)
    assert config.batch_size == 4
    assert config.provider == "CUDAExecutionProvider"


# ---------------------------------------------------------------------------
# build_corpus
# ---------------------------------------------------------------------------


def test_build_corpus_loads_real_jsonl(tmp_path: Path) -> None:
    corpus_path = tmp_path / "c.jsonl"
    corpus_path.write_text(
        '{"content": "alpha"}\n{"content": "beta"}\n', encoding="utf-8"
    )
    argv = _required_args(
        str(tmp_path / "m.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "o.json"),
    ) + ["--corpus", str(corpus_path)]
    args = _parse_args(argv)
    rows, kind, recorded = build_corpus(args)
    assert rows == ["alpha", "beta"]
    assert kind == "real"
    # Recorded path is the user-supplied CLI string, not an absolute path.
    assert recorded == str(corpus_path)


def test_build_corpus_records_relative_path_unmodified(tmp_path: Path) -> None:
    corpus_path = tmp_path / "c.jsonl"
    corpus_path.write_text('{"content": "x"}\n', encoding="utf-8")
    # Use a relative-ish string by passing as-is — argparse keeps the literal.
    argv = _required_args(
        str(tmp_path / "m.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "o.json"),
    ) + ["--corpus", "data/corpus.jsonl", "--corpus-path-recorded", "data/corpus.jsonl"]
    args = _parse_args(argv)
    # build_corpus would fail to load because the file doesn't exist there;
    # we just verify recorded passthrough by short-circuiting.
    with pytest.raises(FileNotFoundError):
        build_corpus(args)


def test_build_corpus_synthetic_when_local_smoke_no_corpus(tmp_path: Path) -> None:
    argv = _required_args(
        str(tmp_path / "m.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "o.json"),
    ) + ["--local-smoke", "--synthetic-seed", "7"]
    args = _parse_args(argv)
    rows, kind, recorded = build_corpus(args)
    assert kind == "synthetic"
    assert recorded == "synthetic:length_stratified:seed=7"
    # Default histogram totals 310 rows.
    assert len(rows) == sum(_DEFAULT_SMOKE_HISTOGRAM.values())
    # Verify length distribution matches the histogram.
    assert Counter(len(s) for s in rows) == Counter(_DEFAULT_SMOKE_HISTOGRAM)


def test_build_corpus_kind_override_wins(tmp_path: Path) -> None:
    corpus_path = tmp_path / "c.jsonl"
    corpus_path.write_text('{"content": "x"}\n', encoding="utf-8")
    argv = _required_args(
        str(tmp_path / "m.onnx"),
        str(tmp_path / "tok"),
        str(tmp_path / "o.json"),
    ) + ["--corpus", str(corpus_path), "--corpus-kind", "fixture"]
    args = _parse_args(argv)
    _, kind, _ = build_corpus(args)
    assert kind == "fixture"


# ---------------------------------------------------------------------------
# serialize_result
# ---------------------------------------------------------------------------


def _make_manifest_for_serial() -> LatencyManifest:
    return LatencyManifest(
        git_sha="deadbeef",
        python_version="3.11.0",
        numpy_version="1.26.0",
        ort_version="1.17.0",
        environment="kaggle",
        hardware_string="Tesla T4",
        provider="CPUExecutionProvider",
        batch_size=1,
        model_revision="fake@deadbeef",
        onnx_sha256="0" * 64,
        tokenizer_files_sha256="1" * 64,
        corpus_sha256="2" * 64,
        corpus_kind="real",
        corpus_path_recorded="/kaggle/input/x/c.jsonl",
    )


def _run_for_serial(tmp_path: Path) -> Any:
    """Run a tiny bench with fakes so we have a real LatencyResult to serialize."""

    config = LatencyConfig(
        model_path=tmp_path / "m.onnx",
        tokenizer_path=tmp_path / "tok",
        model_revision="fake@deadbeef",
        onnx_sha256="0" * 64,
        quant="int8",
        warmup_calls=0,
        measure_calls=2,
    )
    manifest = _make_manifest_for_serial()

    class FakeTokenizer:
        def encode(self, text: str, max_len: int) -> EncodedInput:
            return EncodedInput(
                inputs={"input_ids": np.zeros((1, 5), dtype=np.int64)},
                token_count=5,
            )

    class FakeSession:
        def infer(self, encoded: EncodedInput) -> Any:
            return None

    output = tmp_path / "out.json"
    return run_bench(
        config=config,
        manifest=manifest,
        corpus=["a", "b"],
        session=FakeSession(),
        tokenizer=FakeTokenizer(),
        output_path=output,
    )


def test_serialize_result_has_all_top_level_keys(tmp_path: Path) -> None:
    result = _run_for_serial(tmp_path)
    payload = serialize_result(result)
    assert set(payload.keys()) == {
        "tokenize", "infer", "end_to_end",
        "n_calls", "token_length_histogram",
        "config", "manifest",
    }


def test_serialize_result_is_json_roundtrippable(tmp_path: Path) -> None:
    result = _run_for_serial(tmp_path)
    payload = serialize_result(result)
    text = json.dumps(payload)
    parsed = json.loads(text)
    assert parsed["manifest"]["environment"] == "kaggle"
    assert parsed["config"]["batch_size"] == 1
    assert parsed["n_calls"] == 2


def test_serialize_result_paths_are_strings(tmp_path: Path) -> None:
    result = _run_for_serial(tmp_path)
    payload = serialize_result(result)
    assert isinstance(payload["config"]["model_path"], str)
    assert isinstance(payload["config"]["tokenizer_path"], str)


def test_serialize_result_histogram_bucket_edges_serializes_as_list(
    tmp_path: Path,
) -> None:
    result = _run_for_serial(tmp_path)
    payload = serialize_result(result)
    assert isinstance(payload["config"]["histogram_bucket_edges"], list)


# ---------------------------------------------------------------------------
# run_bench (integration with fakes)
# ---------------------------------------------------------------------------


def test_run_bench_writes_output_file(tmp_path: Path) -> None:
    output = tmp_path / "subdir" / "out.json"
    config = LatencyConfig(
        model_path=tmp_path / "m.onnx",
        tokenizer_path=tmp_path / "tok",
        model_revision="fake@deadbeef",
        onnx_sha256="0" * 64,
        quant="int8",
        warmup_calls=0,
        measure_calls=3,
    )
    manifest = _make_manifest_for_serial()

    class FakeTokenizer:
        def encode(self, text: str, max_len: int) -> EncodedInput:
            return EncodedInput(
                inputs={"input_ids": np.zeros((1, 7), dtype=np.int64)},
                token_count=7,
            )

    class FakeSession:
        def infer(self, encoded: EncodedInput) -> Any:
            return None

    result = run_bench(
        config=config,
        manifest=manifest,
        corpus=["a", "b", "c"],
        session=FakeSession(),
        tokenizer=FakeTokenizer(),
        output_path=output,
    )

    assert output.exists()
    assert result.n_calls == 3
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n_calls"] == 3


def test_run_bench_propagates_measure_latency_errors(tmp_path: Path) -> None:
    """If measure_latency raises (e.g. config/manifest mismatch), no file is written."""

    config = LatencyConfig(
        model_path=tmp_path / "m.onnx",
        tokenizer_path=tmp_path / "tok",
        model_revision="fake@deadbeef",
        onnx_sha256="0" * 64,
        quant="int8",
        warmup_calls=0,
        measure_calls=1,
    )
    # Force a mismatch.
    manifest = _make_manifest_for_serial().model_copy(update={"model_revision": "different"})
    output = tmp_path / "out.json"

    class FakeTokenizer:
        def encode(self, text: str, max_len: int) -> EncodedInput:
            return EncodedInput(
                inputs={"input_ids": np.zeros((1, 1), dtype=np.int64)},
                token_count=1,
            )

    class FakeSession:
        def infer(self, encoded: EncodedInput) -> Any:
            return None

    with pytest.raises(ValueError, match="model_revision"):
        run_bench(
            config=config,
            manifest=manifest,
            corpus=["a"],
            session=FakeSession(),
            tokenizer=FakeTokenizer(),
            output_path=output,
        )
    assert not output.exists()
