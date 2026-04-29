from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# These tests require the [bench] optional dep group. If those packages are
# missing, the whole module imports fail and the tests are skipped.
ort = pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")

from parapet_runner.latency import EncodedInput, LatencyConfig
from parapet_runner.latency_onnx import (
    HfTokenizerAdapter,
    OrtInferenceSession,
    build_manifest,
)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    name = request.node.name
    output_dir = Path("tests/.tmp_outputs/latency_onnx") / name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# HfTokenizerAdapter
# ---------------------------------------------------------------------------


class _FakeAutoTokenizer:
    """Drop-in replacement for transformers.AutoTokenizer."""

    captured: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "_FakeTokenizerInstance":
        cls.captured = {"args": args, "kwargs": kwargs}
        return _FakeTokenizerInstance()


class _FakeTokenizerInstance:
    def __call__(self, text: str, **kwargs: Any) -> dict[str, np.ndarray]:
        max_len = kwargs.get("max_length", 512)
        # Simulate truncation: token count = min(char_len, max_len).
        n = min(len(text), max_len)
        return {
            "input_ids": np.zeros((1, n), dtype=np.int64),
            "attention_mask": np.ones((1, n), dtype=np.int64),
        }


def test_hf_tokenizer_invokes_local_files_only_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("parapet_runner.latency_onnx.AutoTokenizer", _FakeAutoTokenizer)
    _FakeAutoTokenizer.captured = {}

    HfTokenizerAdapter(tmp_path)

    captured = _FakeAutoTokenizer.captured
    assert captured["kwargs"].get("local_files_only") is True, (
        "Adapter must call AutoTokenizer.from_pretrained with local_files_only=True "
        "to refuse silent network fetches"
    )


def test_hf_tokenizer_passes_str_path_not_pathobj(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("parapet_runner.latency_onnx.AutoTokenizer", _FakeAutoTokenizer)
    _FakeAutoTokenizer.captured = {}

    HfTokenizerAdapter(tmp_path)

    args = _FakeAutoTokenizer.captured["args"]
    assert len(args) == 1 and isinstance(args[0], str), (
        f"Tokenizer adapter should pass a string path, got args={args!r}"
    )


def test_hf_tokenizer_raises_on_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        HfTokenizerAdapter(tmp_path / "does_not_exist")


def test_hf_tokenizer_raises_on_file_path_not_directory(tmp_path: Path) -> None:
    f = tmp_path / "tokenizer.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        HfTokenizerAdapter(f)


def test_hf_tokenizer_encode_truncates_to_max_len(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("parapet_runner.latency_onnx.AutoTokenizer", _FakeAutoTokenizer)

    adapter = HfTokenizerAdapter(tmp_path)
    encoded = adapter.encode("a" * 100, max_len=32)
    assert encoded.token_count == 32
    assert encoded.inputs["input_ids"].shape == (1, 32)


def test_hf_tokenizer_encode_uses_attention_mask_for_token_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("parapet_runner.latency_onnx.AutoTokenizer", _FakeAutoTokenizer)

    adapter = HfTokenizerAdapter(tmp_path)
    encoded = adapter.encode("hello", max_len=512)
    assert encoded.token_count == 5  # len("hello"), per fake


# ---------------------------------------------------------------------------
# OrtInferenceSession
# ---------------------------------------------------------------------------


@dataclass
class _FakeOrtSessionTracker:
    constructed: bool = False
    last_path: str | None = None
    last_options: Any = None
    last_providers: list[str] | None = None


def _patch_inference_session(
    monkeypatch: pytest.MonkeyPatch,
    tracker: _FakeOrtSessionTracker,
    *,
    simulated_active_providers: list[str] | None = None,
) -> None:
    """Patch ort.InferenceSession with a fake.

    ``simulated_active_providers`` controls what ``get_providers()`` returns
    after construction — useful for simulating ORT silently falling back to
    CPU when a GPU provider was requested.
    """

    class _FakeInferenceSession:
        def __init__(
            self, path: str, sess_options: Any = None, providers: Any = None
        ) -> None:
            tracker.constructed = True
            tracker.last_path = path
            tracker.last_options = sess_options
            tracker.last_providers = list(providers) if providers else None
            self._active = (
                simulated_active_providers
                if simulated_active_providers is not None
                else (list(providers) if providers else [])
            )

        def get_providers(self) -> list[str]:
            return list(self._active)

        def get_inputs(self) -> list[Any]:
            class _Input:
                name = "input_ids"

            class _Mask:
                name = "attention_mask"

            return [_Input(), _Mask()]

        def run(self, _outputs: Any, feed: dict[str, np.ndarray]) -> Any:
            return [np.zeros((1, 2))]

    monkeypatch.setattr(
        "parapet_runner.latency_onnx.ort.InferenceSession", _FakeInferenceSession
    )


def _write_fake_model(tmp_path: Path, payload: bytes = b"\x08\x01") -> tuple[Path, str]:
    path = tmp_path / "model.onnx"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return path, digest


def test_ort_session_raises_on_hash_mismatch_before_constructing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _write_fake_model(tmp_path)
    tracker = _FakeOrtSessionTracker()
    _patch_inference_session(monkeypatch, tracker)

    with pytest.raises(ValueError, match="hash mismatch"):
        OrtInferenceSession(
            path,
            expected_sha256="0" * 64,  # known-wrong
            provider="CPUExecutionProvider",
            intra_op_threads=1,
            inter_op_threads=1,
        )

    assert not tracker.constructed, (
        "InferenceSession must not be created when the on-disk hash mismatches"
    )


def test_ort_session_passes_str_path_to_inference_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_fake_model(tmp_path)
    tracker = _FakeOrtSessionTracker()
    _patch_inference_session(monkeypatch, tracker)

    OrtInferenceSession(
        path,
        expected_sha256=digest,
        provider="CPUExecutionProvider",
        intra_op_threads=1,
        inter_op_threads=1,
    )

    assert isinstance(tracker.last_path, str), (
        f"InferenceSession must receive a string path, got {type(tracker.last_path).__name__}"
    )
    assert tracker.last_providers == ["CPUExecutionProvider"]


def test_ort_session_sets_threading_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_fake_model(tmp_path)
    tracker = _FakeOrtSessionTracker()
    _patch_inference_session(monkeypatch, tracker)

    OrtInferenceSession(
        path,
        expected_sha256=digest,
        provider="CPUExecutionProvider",
        intra_op_threads=1,
        inter_op_threads=1,
    )
    opts = tracker.last_options
    assert opts.intra_op_num_threads == 1
    assert opts.inter_op_num_threads == 1


def test_ort_session_raises_on_unavailable_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_fake_model(tmp_path)
    tracker = _FakeOrtSessionTracker()
    _patch_inference_session(monkeypatch, tracker)

    with pytest.raises(ValueError, match="not available"):
        OrtInferenceSession(
            path,
            expected_sha256=digest,
            provider="MadeUpExecutionProvider",
            intra_op_threads=1,
            inter_op_threads=1,
        )

    assert not tracker.constructed


def test_ort_session_raises_on_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OrtInferenceSession(
            tmp_path / "nope.onnx",
            expected_sha256="0" * 64,
            provider="CPUExecutionProvider",
            intra_op_threads=1,
            inter_op_threads=1,
        )


def test_ort_session_infer_filters_to_required_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_fake_model(tmp_path)
    tracker = _FakeOrtSessionTracker()
    _patch_inference_session(monkeypatch, tracker)

    session = OrtInferenceSession(
        path,
        expected_sha256=digest,
        provider="CPUExecutionProvider",
        intra_op_threads=1,
        inter_op_threads=1,
    )

    encoded = EncodedInput(
        inputs={
            "input_ids": np.zeros((1, 5), dtype=np.int64),
            "attention_mask": np.ones((1, 5), dtype=np.int64),
            "token_type_ids": np.zeros((1, 5), dtype=np.int64),  # extra; should be filtered
        },
        token_count=5,
    )
    session.infer(encoded)  # does not raise


def test_ort_session_infer_raises_on_missing_required_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_fake_model(tmp_path)
    tracker = _FakeOrtSessionTracker()
    _patch_inference_session(monkeypatch, tracker)

    session = OrtInferenceSession(
        path,
        expected_sha256=digest,
        provider="CPUExecutionProvider",
        intra_op_threads=1,
        inter_op_threads=1,
    )

    encoded = EncodedInput(
        inputs={"input_ids": np.zeros((1, 5), dtype=np.int64)},  # attention_mask missing
        token_count=5,
    )
    with pytest.raises(ValueError, match="required ONNX input 'attention_mask'"):
        session.infer(encoded)


def test_ort_session_input_names_property_exposes_required_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_fake_model(tmp_path)
    tracker = _FakeOrtSessionTracker()
    _patch_inference_session(monkeypatch, tracker)

    session = OrtInferenceSession(
        path,
        expected_sha256=digest,
        provider="CPUExecutionProvider",
        intra_op_threads=1,
        inter_op_threads=1,
    )
    assert set(session.input_names) == {"input_ids", "attention_mask"}


def test_ort_session_rejects_silent_provider_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORT may activate CPU when CUDA was requested but GPU is unavailable.
    The session must refuse rather than letting the manifest overclaim."""

    path, digest = _write_fake_model(tmp_path)
    tracker = _FakeOrtSessionTracker()

    # Simulate ORT silently activating CPU instead of the requested CUDA.
    # We tell ort.get_available_providers that CUDA is "available" (compiled
    # in) but the actual session lands on CPU.
    monkeypatch.setattr(
        "parapet_runner.latency_onnx.ort.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _patch_inference_session(
        monkeypatch,
        tracker,
        simulated_active_providers=["CPUExecutionProvider"],
    )

    with pytest.raises(ValueError, match="Provider mismatch"):
        OrtInferenceSession(
            path,
            expected_sha256=digest,
            provider="CUDAExecutionProvider",
            intra_op_threads=1,
            inter_op_threads=1,
        )


def test_ort_session_active_providers_property_exposes_runtime_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_fake_model(tmp_path)
    tracker = _FakeOrtSessionTracker()
    # Simulate ORT activating CPU as primary plus one fallback.
    _patch_inference_session(
        monkeypatch,
        tracker,
        simulated_active_providers=["CPUExecutionProvider", "AzureExecutionProvider"],
    )

    session = OrtInferenceSession(
        path,
        expected_sha256=digest,
        provider="CPUExecutionProvider",
        intra_op_threads=1,
        inter_op_threads=1,
    )
    assert session.active_providers == (
        "CPUExecutionProvider",
        "AzureExecutionProvider",
    )


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------


def _make_config(model_path: Path, tokenizer_path: Path, sha: str) -> LatencyConfig:
    return LatencyConfig(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        model_revision="microsoft/mdeberta-v3-base@deadbeef",
        onnx_sha256=sha,
        quant="int8",
        provider="CPUExecutionProvider",
    )


def test_build_manifest_populates_all_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tokenizer dir with one fake file so sha256_directory works.
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    sha = hashlib.sha256(b"x").hexdigest()

    monkeypatch.setattr("parapet_runner.latency_onnx.git_sha", lambda: "fakehead")
    monkeypatch.setattr(
        "parapet_runner.latency_onnx.detect_environment", lambda: "local"
    )
    monkeypatch.setattr(
        "parapet_runner.latency_onnx.detect_hardware_string",
        lambda provider: f"fake-cpu/{provider}",
    )

    manifest = build_manifest(
        config=_make_config(model_path, tok_dir, sha),
        tokenizer_path=tok_dir,
        corpus_sha256="a" * 64,
        corpus_kind="real",
        corpus_path_recorded="/kaggle/input/curated/train.jsonl",
    )

    assert manifest.git_sha == "fakehead"
    assert manifest.environment == "local"
    assert manifest.hardware_string == "fake-cpu/CPUExecutionProvider"
    assert manifest.provider == "CPUExecutionProvider"
    assert manifest.batch_size == 1
    assert manifest.model_revision == "microsoft/mdeberta-v3-base@deadbeef"
    assert manifest.onnx_sha256 == sha
    assert manifest.corpus_sha256 == "a" * 64
    assert manifest.corpus_kind == "real"
    assert manifest.corpus_path_recorded == "/kaggle/input/curated/train.jsonl"
    assert manifest.python_version  # populated from sys
    assert manifest.numpy_version  # populated from numpy
    assert manifest.ort_version  # populated from onnxruntime


def test_build_manifest_tokenizer_hash_excludes_model_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tokenizer hash must not change when an unrelated ONNX model
    appears in the same directory (Kaggle's optimum-cli export pattern)."""

    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tok_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    sha = hashlib.sha256(b"x").hexdigest()

    monkeypatch.setattr("parapet_runner.latency_onnx.git_sha", lambda: "head1")
    monkeypatch.setattr(
        "parapet_runner.latency_onnx.detect_environment", lambda: "local"
    )
    monkeypatch.setattr(
        "parapet_runner.latency_onnx.detect_hardware_string", lambda p: "cpu"
    )

    manifest_before = build_manifest(
        config=_make_config(model_path, tok_dir, sha),
        tokenizer_path=tok_dir,
        corpus_sha256="a" * 64,
        corpus_kind="real",
        corpus_path_recorded="c.jsonl",
    )

    # Drop a model.onnx into the tokenizer dir (the Kaggle pattern).
    (tok_dir / "model.onnx").write_bytes(b"\x00\xff" * 1000)

    manifest_after = build_manifest(
        config=_make_config(model_path, tok_dir, sha),
        tokenizer_path=tok_dir,
        corpus_sha256="a" * 64,
        corpus_kind="real",
        corpus_path_recorded="c.jsonl",
    )

    assert manifest_before.tokenizer_files_sha256 == manifest_after.tokenizer_files_sha256


def test_build_manifest_environment_override_wins_over_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    sha = hashlib.sha256(b"x").hexdigest()

    # Detection would say "local" but caller overrides to "kaggle".
    monkeypatch.setattr("parapet_runner.latency_onnx.git_sha", lambda: "deadbeef")
    monkeypatch.setattr(
        "parapet_runner.latency_onnx.detect_environment", lambda: "local"
    )
    monkeypatch.setattr(
        "parapet_runner.latency_onnx.detect_hardware_string", lambda provider: "cpu"
    )

    manifest = build_manifest(
        config=_make_config(model_path, tok_dir, sha),
        tokenizer_path=tok_dir,
        corpus_sha256="a" * 64,
        corpus_kind="synthetic",
        corpus_path_recorded="synthetic:length_stratified:seed=42",
        environment="kaggle",
    )
    assert manifest.environment == "kaggle"
    assert manifest.corpus_kind == "synthetic"
