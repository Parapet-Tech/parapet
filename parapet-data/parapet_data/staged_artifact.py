"""Shared loader for staged artifact files (per-source staged YAML/JSONL).

Staged artifacts live one-per-source under the staging directory (e.g.
``en_attacks_merged_attacks_staged.yaml``). They are intermediate outputs
of the staging pipeline and inputs to verified-sync, sampling, and
several reporting scripts.

Callers should prefer :func:`iter_staged_rows` and only fall back to
:func:`load_staged_rows` when a concrete list is genuinely required —
this keeps the door open for a future migration to streaming JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import yaml

try:
    _YAML_LOADER = yaml.CSafeLoader  # type: ignore[attr-defined]
except AttributeError:
    _YAML_LOADER = yaml.SafeLoader  # type: ignore[assignment]


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
