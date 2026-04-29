"""Runtime introspection helpers for the latency bench.

Hash artifacts deterministically, detect the host environment (Kaggle / Colab
/ CI / local), describe the hardware, and read the parapet git SHA. None of
this requires onnxruntime or transformers — it deliberately runs in CI and
in environments where the ``[bench]`` dep group is not installed.

The ``build_manifest`` helper that assembles a ``LatencyManifest`` lives in
``latency_onnx.py`` (where ``onnxruntime.__version__`` is available); this
module provides the inputs that ``build_manifest`` composes.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Literal

EnvironmentLiteral = Literal["kaggle", "colab", "local", "ci", "unknown"]


# ---------------------------------------------------------------------------
# Artifact hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path, *, chunk_size: int = 65536) -> str:
    """SHA-256 of a single file's bytes, streamed in chunks."""

    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# HuggingFace tokenizer file allowlist. Anything else in the directory
# (e.g. an ONNX model exported to the same dir by ``optimum-cli``) is
# excluded from the hash so the tokenizer identity is not contaminated by
# co-located model artifacts.
_HF_TOKENIZER_FILES: frozenset[str] = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        # SentencePiece variants. mDeBERTa uses spm.model; DeBERTa-v3 / XLM-R
        # use spiece.model; some XLM-R checkpoints use sentencepiece.bpe.model.
        "spm.model",
        "spiece.model",
        "sentencepiece.bpe.model",
    }
)


def sha256_tokenizer_files(path: Path, *, chunk_size: int = 65536) -> str:
    """Deterministic SHA-256 over canonical HuggingFace tokenizer files.

    Only top-level files matching the canonical HF tokenizer file allowlist
    are included. This is deliberately narrow so hashing the same directory
    that ``optimum-cli`` exported (which also contains ``model.onnx``) does
    NOT let the model bytes leak into the tokenizer identity.

    Each entry contributes:
      * 2-byte big-endian filename length (UTF-8 bytes)
      * filename bytes
      * 8-byte big-endian content size
      * file content
    so that filename/content boundaries cannot alias.

    Raises ``FileNotFoundError`` if no allowlisted files are present.
    """

    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")
    files = sorted(
        f for f in path.iterdir() if f.is_file() and f.name in _HF_TOKENIZER_FILES
    )
    if not files:
        raise FileNotFoundError(
            f"No HuggingFace tokenizer files found under {path}. "
            f"Looked for: {sorted(_HF_TOKENIZER_FILES)}"
        )
    h = hashlib.sha256()
    for f in files:
        name_bytes = f.name.encode("utf-8")
        h.update(len(name_bytes).to_bytes(2, "big"))
        h.update(name_bytes)
        h.update(f.stat().st_size.to_bytes(8, "big"))
        with f.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Environment + hardware detection
# ---------------------------------------------------------------------------


def detect_environment(env: dict[str, str] | None = None) -> EnvironmentLiteral:
    """Best-effort detection of the host runtime."""

    e = env if env is not None else dict(os.environ)
    if "KAGGLE_KERNEL_RUN_TYPE" in e or "KAGGLE_URL_BASE" in e:
        return "kaggle"
    if "COLAB_GPU" in e or "COLAB_RELEASE_TAG" in e:
        return "colab"
    if "GITHUB_ACTIONS" in e or e.get("CI", "").lower() in ("true", "1"):
        return "ci"
    return "local"


def detect_hardware_string(
    provider: str,
    *,
    nvidia_smi_runner: callable | None = None,  # type: ignore[type-arg]
    cpu_describer: callable | None = None,  # type: ignore[type-arg]
) -> str:
    """Human-readable hardware identifier.

    For GPU providers, queries ``nvidia-smi`` for the GPU name. For CPU
    providers (or if nvidia-smi fails), returns the CPU model string from
    ``platform.processor()`` plus the visible thread count. The two
    injectable hooks make this deterministic in tests.
    """

    if provider == "CUDAExecutionProvider":
        gpu = (nvidia_smi_runner or _run_nvidia_smi)()
        if gpu:
            return gpu

    cpu = (cpu_describer or _describe_cpu)()
    return cpu


def _run_nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return line or None


def _describe_cpu() -> str:
    cpu = (platform.processor() or "unknown-cpu").strip() or "unknown-cpu"
    threads = os.cpu_count() or 0
    return f"{cpu} / {threads} threads"


# ---------------------------------------------------------------------------
# Git SHA
# ---------------------------------------------------------------------------


def git_sha(repo_root: Path | None = None) -> str:
    """Read HEAD SHA from git, or 'unknown' if unavailable.

    Defaults to the parapet repo root inferred from this file's location.
    """

    root = repo_root if repo_root is not None else _default_repo_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            cwd=str(root),
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"
    sha = result.stdout.strip()
    return sha or "unknown"


def _default_repo_root() -> Path:
    # parapet/parapet-runner/parapet_runner/latency_runtime.py
    # → parents[2] = parapet-runner → parents[3] = parapet (the cargo root)
    return Path(__file__).resolve().parents[2]
