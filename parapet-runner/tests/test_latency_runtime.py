from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from parapet_runner.latency_runtime import (
    detect_environment,
    detect_hardware_string,
    git_sha,
    sha256_file,
    sha256_tokenizer_files,
)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    name = request.node.name
    output_dir = Path("tests/.tmp_outputs/latency_runtime") / name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashlib_reference(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    payload = b"hello\x00world"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_handles_chunked_reads(tmp_path: Path) -> None:
    path = tmp_path / "big.bin"
    payload = b"abc" * 100_000
    path.write_bytes(payload)
    assert sha256_file(path, chunk_size=4096) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing.bin")


def test_sha256_file_raises_on_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path)


# ---------------------------------------------------------------------------
# sha256_tokenizer_files
# ---------------------------------------------------------------------------


def test_sha256_tokenizer_files_hashes_only_allowlisted_files(tmp_path: Path) -> None:
    # Tokenizer-relevant files
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    h1 = sha256_tokenizer_files(tmp_path)

    # Adding an ONNX model file MUST NOT change the hash.
    (tmp_path / "model.onnx").write_bytes(b"\x08\x01\x02\x03")
    h2 = sha256_tokenizer_files(tmp_path)
    assert h1 == h2, "Tokenizer hash must not depend on co-located model artifacts"


def test_sha256_tokenizer_files_changes_when_tokenizer_content_changes(
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer.json").write_text('{"v": 1}', encoding="utf-8")
    h1 = sha256_tokenizer_files(tmp_path)
    (tmp_path / "tokenizer.json").write_text('{"v": 2}', encoding="utf-8")
    h2 = sha256_tokenizer_files(tmp_path)
    assert h1 != h2


def test_sha256_tokenizer_files_independent_of_filesystem_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer_config.json").write_text("a", encoding="utf-8")
    (tmp_path / "special_tokens_map.json").write_text("b", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("c", encoding="utf-8")
    h1 = sha256_tokenizer_files(tmp_path)

    other = tmp_path.parent / "other_tok_dir"
    shutil.rmtree(other, ignore_errors=True)
    other.mkdir()
    # Different creation order.
    (other / "tokenizer.json").write_text("c", encoding="utf-8")
    (other / "special_tokens_map.json").write_text("b", encoding="utf-8")
    (other / "tokenizer_config.json").write_text("a", encoding="utf-8")
    h2 = sha256_tokenizer_files(other)

    assert h1 == h2


def test_sha256_tokenizer_files_supports_sentencepiece(tmp_path: Path) -> None:
    (tmp_path / "spiece.model").write_bytes(b"sentencepiece-bytes")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    # Should not raise.
    sha256_tokenizer_files(tmp_path)


def test_sha256_tokenizer_files_includes_spm_model_for_mdeberta(tmp_path: Path) -> None:
    """Regression: mDeBERTa-v3-base ships its SentencePiece model as
    ``spm.model`` (not ``spiece.model``). If the allowlist misses it, the
    tokenizer hash silently degrades to hashing only the JSON config files
    and would not change when the actual SentencePiece weights change.
    """

    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "spm.model").write_bytes(b"sentencepiece-v1")
    h1 = sha256_tokenizer_files(tmp_path)

    (tmp_path / "spm.model").write_bytes(b"sentencepiece-v2")
    h2 = sha256_tokenizer_files(tmp_path)

    assert h1 != h2, (
        "Changing spm.model must change the tokenizer hash; otherwise the "
        "hash is silently degenerate for mDeBERTa-shaped tokenizers."
    )


def test_sha256_tokenizer_files_supports_sentencepiece_bpe_for_xlmr(
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sentencepiece.bpe.model").write_bytes(b"v1")
    h1 = sha256_tokenizer_files(tmp_path)
    (tmp_path / "sentencepiece.bpe.model").write_bytes(b"v2")
    h2 = sha256_tokenizer_files(tmp_path)
    assert h1 != h2


def test_sha256_tokenizer_files_raises_when_no_canonical_files(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "random.bin").write_bytes(b"y")
    with pytest.raises(FileNotFoundError, match="No HuggingFace tokenizer files"):
        sha256_tokenizer_files(tmp_path)


def test_sha256_tokenizer_files_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_tokenizer_files(tmp_path / "missing")


def test_sha256_tokenizer_files_raises_on_file_path(tmp_path: Path) -> None:
    f = tmp_path / "tokenizer.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        sha256_tokenizer_files(f)


def test_sha256_tokenizer_files_ignores_subdirectory_contents(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "tokenizer.json").write_text('{"nested": true}', encoding="utf-8")
    h1 = sha256_tokenizer_files(tmp_path)

    # Modify the subdirectory file — top-level hash should not change.
    (sub / "tokenizer.json").write_text('{"nested": "different"}', encoding="utf-8")
    h2 = sha256_tokenizer_files(tmp_path)
    assert h1 == h2


# ---------------------------------------------------------------------------
# detect_environment
# ---------------------------------------------------------------------------


def test_detect_environment_kaggle_via_kernel_var() -> None:
    assert detect_environment({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}) == "kaggle"


def test_detect_environment_kaggle_via_url_var() -> None:
    assert detect_environment({"KAGGLE_URL_BASE": "https://www.kaggle.com"}) == "kaggle"


def test_detect_environment_colab_via_gpu_var() -> None:
    assert detect_environment({"COLAB_GPU": "1"}) == "colab"


def test_detect_environment_colab_via_release_var() -> None:
    assert detect_environment({"COLAB_RELEASE_TAG": "release-2024-01"}) == "colab"


def test_detect_environment_ci_github_actions() -> None:
    assert detect_environment({"GITHUB_ACTIONS": "true"}) == "ci"


def test_detect_environment_ci_via_ci_var() -> None:
    assert detect_environment({"CI": "true"}) == "ci"


def test_detect_environment_local_default() -> None:
    assert detect_environment({}) == "local"


def test_detect_environment_kaggle_takes_priority_over_ci() -> None:
    # Kaggle kernel may have CI=true set; kaggle wins.
    assert detect_environment(
        {"KAGGLE_KERNEL_RUN_TYPE": "Interactive", "CI": "true"}
    ) == "kaggle"


# ---------------------------------------------------------------------------
# detect_hardware_string
# ---------------------------------------------------------------------------


def test_hardware_string_uses_nvidia_smi_for_cuda() -> None:
    out = detect_hardware_string(
        "CUDAExecutionProvider",
        nvidia_smi_runner=lambda: "Tesla T4",
    )
    assert out == "Tesla T4"


def test_hardware_string_falls_back_to_cpu_on_nvidia_smi_failure() -> None:
    out = detect_hardware_string(
        "CUDAExecutionProvider",
        nvidia_smi_runner=lambda: None,
        cpu_describer=lambda: "AMD EPYC / 16 threads",
    )
    assert out == "AMD EPYC / 16 threads"


def test_hardware_string_uses_cpu_describer_for_cpu_provider() -> None:
    out = detect_hardware_string(
        "CPUExecutionProvider",
        cpu_describer=lambda: "Intel Xeon / 8 threads",
    )
    assert out == "Intel Xeon / 8 threads"


def test_hardware_string_does_not_call_nvidia_smi_for_cpu() -> None:
    called = []

    def nvidia_smi_should_not_be_called() -> str | None:
        called.append(True)
        return "GPU"

    detect_hardware_string(
        "CPUExecutionProvider",
        nvidia_smi_runner=nvidia_smi_should_not_be_called,
        cpu_describer=lambda: "cpu",
    )
    assert called == []


# ---------------------------------------------------------------------------
# git_sha
# ---------------------------------------------------------------------------


def test_git_sha_reads_head_when_repo_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sha = "abcdef1234567890" * 2 + "abcd1234"

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        class Result:
            stdout = fake_sha + "\n"
            stderr = ""
            returncode = 0

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_sha(repo_root=tmp_path) == fake_sha


def test_git_sha_returns_unknown_on_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(returncode=128, cmd=args[0])

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_sha(repo_root=tmp_path) == "unknown"


def test_git_sha_returns_unknown_when_git_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("git not on PATH")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_sha(repo_root=tmp_path) == "unknown"


def test_git_sha_returns_unknown_on_blank_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> Any:
        class Result:
            stdout = "\n"
            stderr = ""
            returncode = 0

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_sha(repo_root=tmp_path) == "unknown"
