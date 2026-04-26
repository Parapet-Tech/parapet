"""Shared reader/writer for staged artifact files.

Staged artifacts live one-per-source under the staging directory (e.g.
``en_attacks_merged_attacks_staged.jsonl``). They are intermediate outputs
of the staging pipeline and inputs to verified-sync, sampling, and several
reporting scripts.

Active staging is JSONL-only. YAML read support is retained for historical
artifacts — frozen run snapshots, review bundles, and hand-named source
files — but the writer never produces YAML. Callers should prefer
:func:`iter_staged_rows` / :func:`write_staged_rows` and only fall back to
:func:`load_staged_rows` when a concrete list is genuinely required.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

try:
    _YAML_LOADER = yaml.CSafeLoader  # type: ignore[attr-defined]
except AttributeError:
    _YAML_LOADER = yaml.SafeLoader  # type: ignore[assignment]


_STAGED_SUFFIXES: tuple[str, ...] = (".yaml", ".yml", ".jsonl")

# Known sidecar files that stage_all writes into the same directory as
# staged artifacts. These must NEVER be treated as staged-row sources by
# dir-mode loaders. Add to this list when staging.py grows new sidecars.
_SIDECAR_EXACT_NAMES: frozenset[str] = frozenset({"staging_rejected.jsonl"})
_SIDECAR_NAME_SUFFIXES: tuple[str, ...] = (
    "_quarantine.jsonl",   # per-dataset quarantine
    ".partial.jsonl",      # in-flight checkpoint files
)


def is_staged_artifact_path(path: Path) -> bool:
    """True if ``path`` is a staged artifact, not a control-plane sidecar.

    Filters by extension AND by sidecar exclusion. Files in a staging
    directory that share an extension with staged artifacts but are
    structurally different — quarantine logs, rejection logs, in-flight
    checkpoint shards — are explicitly excluded so dir-mode loaders never
    slurp them as sample rows.
    """
    if not path.is_file():
        return False
    if path.suffix.lower() not in _STAGED_SUFFIXES:
        return False
    name = path.name.lower()
    if name in _SIDECAR_EXACT_NAMES:
        return False
    if any(name.endswith(suffix) for suffix in _SIDECAR_NAME_SUFFIXES):
        return False
    return True


def iter_staged_artifact_paths(directory: Path) -> list[Path]:
    """Return staged artifact paths from ``directory``, sorted by path."""
    return sorted(p for p in directory.iterdir() if is_staged_artifact_path(p))


def iter_staged_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield staged rows from a single staged artifact file.

    Supports ``.jsonl`` (true streaming) and ``.yaml``/``.yml`` (full-load —
    PyYAML cannot stream a top-level list without a custom event-driven
    parser). YAML support is retained for historical artifacts only; the
    writer never produces YAML.
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


def write_staged_rows(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    """Stream staged rows as JSONL and return the sha256 of the file bytes.

    Rows are written one-by-one — the writer never materializes the full
    list, and the returned hash is computed incrementally over the exact
    bytes written to disk.

    Only ``.jsonl`` output paths are accepted. Active staging is JSONL-only;
    historical YAML artifacts are read-only via :func:`iter_staged_rows`.
    """
    if path.suffix.lower() != ".jsonl":
        raise ValueError(
            f"{path}: write_staged_rows only emits JSONL "
            f"(got suffix {path.suffix!r})"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

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
