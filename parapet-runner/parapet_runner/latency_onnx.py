"""Real ONNX runtime + HuggingFace tokenizer adapters.

This module is the boundary between the bench contract (``latency.py``) and
the ML toolchain. It is gated behind the ``[bench]`` optional dep group:

    pip install -e ".[bench]"

Importing this module from an environment that lacks ``onnxruntime`` or
``transformers`` will raise ``ImportError`` immediately. That is intentional
— if you cannot import this module, you cannot run a real benchmark. Tests
that don't need the adapters live in ``test_latency_runtime.py`` /
``test_latency.py`` and run in any environment.

No-network discipline:

* ``HfTokenizerAdapter`` calls ``AutoTokenizer.from_pretrained`` with
  ``local_files_only=True`` so a missing local file fails closed instead of
  silently fetching from the HuggingFace Hub.
* ``OrtInferenceSession`` accepts only a filesystem path and verifies its
  SHA-256 against the value declared in the bench config before constructing
  the underlying ``onnxruntime.InferenceSession``. A hash mismatch refuses
  to load.

The provider check is best-effort: ``ort.get_available_providers()`` reflects
what the ORT build was compiled with, not whether GPU hardware is actually
present. A CUDAExecutionProvider session can still silently land on CPU if
the GPU is unavailable; the manifest's ``hardware_string`` (populated via
``nvidia-smi``) is the correctness witness for that.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .latency import EncodedInput, LatencyConfig, LatencyManifest
from .latency_runtime import (
    detect_environment,
    detect_hardware_string,
    git_sha,
    sha256_file,
    sha256_tokenizer_files,
)


# ---------------------------------------------------------------------------
# HuggingFace tokenizer adapter
# ---------------------------------------------------------------------------


class HfTokenizerAdapter:
    """Implements the ``Tokenizer`` protocol via a local HuggingFace tokenizer."""

    def __init__(self, tokenizer_path: Path) -> None:
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")
        if not tokenizer_path.is_dir():
            raise ValueError(
                f"Tokenizer path must be a directory containing tokenizer files, "
                f"got file: {tokenizer_path}"
            )
        self._path = tokenizer_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
            use_fast=True,
        )

    def encode(self, text: str, max_len: int) -> EncodedInput:
        encoded = self._tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding=False,
            return_tensors="np",
        )
        inputs: dict[str, np.ndarray] = {key: np.asarray(encoded[key]) for key in encoded}
        if "attention_mask" in inputs:
            token_count = int(inputs["attention_mask"].sum())
        elif "input_ids" in inputs:
            token_count = int(inputs["input_ids"].shape[-1])
        else:
            raise ValueError(
                f"Tokenizer at {self._path} produced no input_ids or attention_mask"
            )
        return EncodedInput(inputs=inputs, token_count=token_count)


# ---------------------------------------------------------------------------
# ONNX runtime session adapter
# ---------------------------------------------------------------------------


class OrtInferenceSession:
    """Implements the ``InferenceSession`` protocol via ``onnxruntime``.

    Verifies the on-disk model's SHA-256 against ``expected_sha256`` BEFORE
    constructing the underlying session. A mismatch raises and the session
    is never created.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        expected_sha256: str,
        provider: str,
        intra_op_threads: int,
        inter_op_threads: int,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        actual = sha256_file(model_path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"ONNX model hash mismatch at {model_path}: "
                f"expected {expected_sha256.lower()!r}, got {actual!r}"
            )
        available = ort.get_available_providers()
        if provider not in available:
            raise ValueError(
                f"Provider {provider!r} not available in this onnxruntime build. "
                f"Available: {available}"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[provider],
        )

        # Runtime confirmation: ORT may silently fall back to CPU if a
        # CUDAExecutionProvider was requested but the GPU is unavailable.
        # The first entry of get_providers() is the priority/active provider
        # for ops it supports. Refuse if it doesn't match the request — the
        # manifest must reflect the actual runtime, not the asked-for one.
        active = list(self._session.get_providers())
        if not active or active[0] != provider:
            raise ValueError(
                f"Provider mismatch: requested {provider!r} but ORT activated "
                f"{active!r}. Refusing to construct session — the bench manifest "
                "must reflect the actual runtime, not the requested provider."
            )
        self._active_providers: tuple[str, ...] = tuple(active)
        self._input_names: tuple[str, ...] = tuple(i.name for i in self._session.get_inputs())

    @property
    def active_providers(self) -> tuple[str, ...]:
        """Providers ORT actually selected, in priority order. The first is
        the one used for ops it supports; subsequent entries are fallbacks."""
        return self._active_providers

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    def infer(self, encoded: EncodedInput) -> Any:
        feed: dict[str, np.ndarray] = {}
        for name in self._input_names:
            if name not in encoded.inputs:
                raise ValueError(
                    f"Tokenizer did not produce required ONNX input {name!r}; "
                    f"available: {sorted(encoded.inputs.keys())}"
                )
            feed[name] = encoded.inputs[name]
        return self._session.run(None, feed)


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


CorpusKind = Literal["real", "synthetic", "fixture"]


def build_manifest(
    *,
    config: LatencyConfig,
    tokenizer_path: Path,
    corpus_sha256: str,
    corpus_kind: CorpusKind,
    corpus_path_recorded: str,
    environment: Literal["kaggle", "colab", "local", "ci", "unknown"] | None = None,
) -> LatencyManifest:
    """Assemble a LatencyManifest from the live runtime.

    ``environment`` may be passed to override auto-detection (e.g. when the
    bench is launched from a script that knows it is on Kaggle but the env
    vars haven't been set yet).
    """

    return LatencyManifest(
        git_sha=git_sha(),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        ort_version=ort.__version__,
        environment=environment or detect_environment(),
        hardware_string=detect_hardware_string(config.provider),
        provider=config.provider,
        batch_size=config.batch_size,
        model_revision=config.model_revision,
        onnx_sha256=config.onnx_sha256,
        tokenizer_files_sha256=sha256_tokenizer_files(tokenizer_path),
        corpus_sha256=corpus_sha256,
        corpus_kind=corpus_kind,
        corpus_path_recorded=corpus_path_recorded,
    )
