"""Shared loaders for curated artifact directories and files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import yaml


def resolve_curated_path(path: Path) -> Path:
    """Resolve a curated artifact directory or file to a concrete artifact file.

    Preference order is:
    1. explicit file path as provided
    2. ``curated.jsonl`` inside a directory
    3. ``curated.yaml`` inside a directory
    4. ``curated.yml`` inside a directory
    """
    if path.is_file():
        return path

    candidates = (
        path / "curated.jsonl",
        path / "curated.yaml",
        path / "curated.yml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find curated artifact under {path} "
        f"(looked for curated.jsonl, curated.yaml, curated.yml)"
    )


def iter_curated_entries(path: Path) -> Iterator[dict[str, Any]]:
    """Yield curated entries from a curated file or curated artifact directory."""
    resolved = resolve_curated_path(path)
    suffix = resolved.suffix.lower()

    if suffix == ".jsonl":
        with open(resolved, encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{resolved}:{line_number}: expected JSON object row"
                    )
                yield row
        return

    if suffix in {".yaml", ".yml"}:
        with open(resolved, encoding="utf-8") as handle:
            rows = yaml.safe_load(handle)
        if rows is None:
            return
        if not isinstance(rows, list):
            raise ValueError(f"{resolved}: expected top-level YAML list")
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"{resolved}:{index}: expected mapping row")
            yield row
        return

    raise ValueError(f"{resolved}: unsupported curated artifact suffix {suffix}")


def load_curated_entries(path: Path) -> list[dict[str, Any]]:
    """Materialize all curated entries from a curated file or artifact directory."""
    return list(iter_curated_entries(path))
