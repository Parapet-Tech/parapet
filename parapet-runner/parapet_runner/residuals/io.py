"""I/O helpers for residual analysis artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class InputBundle:
    """Loaded residual-analysis inputs."""

    residuals_path: Path
    baseline_correct_path: Path
    export_manifest_path: Path
    residuals: list[dict[str, Any]]
    baseline_correct: list[dict[str, Any]]
    export_manifest: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest for a file."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object, got {type(data).__name__}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from ``path``."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, data: Any) -> None:
    """Write stable, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write JSONL rows and return row count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def load_inputs(
    residuals_path: Path,
    baseline_correct_path: Path,
    export_manifest_path: Path,
) -> InputBundle:
    """Load residuals, required sidecar, and export manifest."""

    for path in (residuals_path, baseline_correct_path, export_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    return InputBundle(
        residuals_path=residuals_path,
        baseline_correct_path=baseline_correct_path,
        export_manifest_path=export_manifest_path,
        residuals=load_jsonl(residuals_path),
        baseline_correct=load_jsonl(baseline_correct_path),
        export_manifest=load_json(export_manifest_path),
    )


def input_receipts(bundle: InputBundle) -> dict[str, dict[str, Any]]:
    """Return manifest-ready receipts for loaded inputs."""

    return {
        "residuals": {
            "path": str(bundle.residuals_path),
            "sha256": sha256_file(bundle.residuals_path),
            "rows": len(bundle.residuals),
        },
        "baseline_correct": {
            "path": str(bundle.baseline_correct_path),
            "sha256": sha256_file(bundle.baseline_correct_path),
            "rows": len(bundle.baseline_correct),
        },
        "export_manifest": {
            "path": str(bundle.export_manifest_path),
            "sha256": sha256_file(bundle.export_manifest_path),
        },
    }


def output_receipts(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Return manifest-ready receipts for output files."""

    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }
