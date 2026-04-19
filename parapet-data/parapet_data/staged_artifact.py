"""Shared reader/writer for staged artifact files (per-source staged YAML/JSONL).

Staged artifacts live one-per-source under the staging directory (e.g.
``en_attacks_merged_attacks_staged.yaml``). They are intermediate outputs
of the staging pipeline and inputs to verified-sync, sampling, and
several reporting scripts.

Callers should prefer :func:`iter_staged_rows` / :func:`write_staged_rows`
and only fall back to :func:`load_staged_rows` when a concrete list is
genuinely required — this keeps the door open for bounded-retention
streaming on the JSONL path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import yaml

try:
    _YAML_LOADER = yaml.CSafeLoader  # type: ignore[attr-defined]
except AttributeError:
    _YAML_LOADER = yaml.SafeLoader  # type: ignore[assignment]

StagedFormat = Literal["yaml", "jsonl"]

STAGED_FORMATS: tuple[StagedFormat, ...] = ("yaml", "jsonl")


def staged_extension(fmt: StagedFormat) -> str:
    """Return the filename extension (without leading dot) for a staged format."""
    if fmt == "jsonl":
        return "jsonl"
    if fmt == "yaml":
        return "yaml"
    raise ValueError(f"unsupported staged artifact format: {fmt!r}")


def iter_staged_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield staged rows from a single staged artifact file.

    Supports ``.jsonl`` (true streaming) and ``.yaml``/``.yml`` (currently
    full-load — PyYAML cannot stream a top-level list without a custom
    event-driven parser). The iterator interface is the contract; the
    backing implementation can swap to streaming when staged outputs
    migrate to JSONL.
    """
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{path}:{line_number}: expected JSON object row"
                    )
                yield row
        return

    if suffix in {".yaml", ".yml"}:
        with open(path, encoding="utf-8") as handle:
            rows = yaml.load(handle, Loader=_YAML_LOADER)
        if rows is None:
            return
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected top-level YAML list")
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{index}: expected mapping row")
            yield row
        return

    raise ValueError(f"{path}: unsupported staged artifact suffix {suffix}")


def load_staged_rows(path: Path) -> list[dict[str, Any]]:
    """Materialize all staged rows from a staged artifact file."""
    return list(iter_staged_rows(path))


def write_staged_rows(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fmt: StagedFormat = "yaml",
) -> str:
    """Write staged rows in ``fmt`` and return the sha256 of the file bytes.

    ``jsonl`` streams row-by-row — the writer never materializes the full
    list, and the returned hash is computed incrementally over the exact
    bytes written to disk.

    ``yaml`` materializes internally because PyYAML cannot stream a top-level
    list; byte-level formatting matches the prior staging writer so existing
    hashes remain stable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        hasher = hashlib.sha256()
        with open(path, "wb") as handle:
            for row in rows:
                if not isinstance(row, dict):
                    raise TypeError(
                        f"{path}: expected mapping row, got {type(row).__name__}"
                    )
                line_bytes = (
                    json.dumps(row, ensure_ascii=False) + "\n"
                ).encode("utf-8")
                handle.write(line_bytes)
                hasher.update(line_bytes)
        return hasher.hexdigest()

    if fmt == "yaml":
        rows_list = list(rows)
        for index, row in enumerate(rows_list, start=1):
            if not isinstance(row, dict):
                raise TypeError(
                    f"{path}:{index}: expected mapping row, got {type(row).__name__}"
                )
        with open(path, "w", encoding="utf-8") as handle:
            yaml.dump(
                rows_list,
                handle,
                allow_unicode=True,
                default_flow_style=False,
                width=2000,
            )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    raise ValueError(f"unsupported staged artifact format: {fmt!r}")
