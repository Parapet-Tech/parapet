"""
Verified sync: ledger-filtered projection of staging.

Reads staged YAML files, applies adjudication ledger actions, and writes
surviving rows to the verified output directory. This is a mechanical
transform — no new judgment, no heuristic decisions. All judgment comes
from the ledger.

Usage:
    staging/ -> verified/sync -> verified/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import sys
import time

import yaml

try:
    _YAML_DUMPER = yaml.CSafeDumper
except AttributeError:
    _YAML_DUMPER = yaml.SafeDumper  # type: ignore[assignment]

from parapet_data.ledger import Ledger, apply_ledger_to_row
from parapet_data.staged_artifact import (
    StagedFormat,
    iter_staged_artifact_paths,
    iter_staged_rows,
    load_staged_rows,
    write_staged_rows,
)

log = logging.getLogger(__name__)


@dataclass
class SyncStats:
    """Counters emitted by a verified sync run."""

    files_processed: int = 0
    total_input: int = 0
    passed: int = 0
    dropped: int = 0
    quarantined: int = 0
    rerouted: int = 0
    relabeled: int = 0

def _sync_rows(
    row_stream: Iterable[dict],
    ledger: Ledger,
    stats: SyncStats,
) -> Iterator[dict]:
    """Apply ledger actions to each input row, yielding surviving rows.

    Streaming core for the JSONL verified-sync path: one row at a time, no
    intermediate list. The YAML path materializes by wrapping the caller in
    ``list(_sync_rows(...))`` — PyYAML cannot stream a top-level sequence.
    """
    for row in row_stream:
        stats.total_input += 1
        result = apply_ledger_to_row(row, ledger)
        stats.passed += result.passed
        stats.dropped += result.dropped
        stats.quarantined += result.quarantined
        stats.rerouted += result.rerouted
        stats.relabeled += result.relabeled
        if result.row is not None:
            yield result.row


def _sync_file(rows: list[dict], ledger: Ledger, stats: SyncStats) -> list[dict]:
    """Backwards-compatible list-based sync — thin adapter over ``_sync_rows``.

    Retained so that callers wanting materialized output (and tests from
    earlier phases) continue to work unchanged. New code paths should prefer
    ``_sync_rows`` so JSONL stays bounded-retention.
    """
    return list(_sync_rows(iter(rows), ledger, stats))


def _staged_format_from_suffix(path: Path) -> StagedFormat:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return "yaml"
    if suffix == ".jsonl":
        return "jsonl"
    raise ValueError(f"{path}: unsupported staged artifact suffix {suffix}")


def _write_verified(path: Path, rows: list[dict], fmt: StagedFormat) -> None:
    """Write verified output in the same format as the input.

    JSONL uses the shared streaming writer. YAML preserves the prior verified
    dumper settings (CSafeDumper, sort_keys=False) so existing consumers see
    byte-identical output on that path.
    """
    if fmt == "jsonl":
        write_staged_rows(path, rows, fmt="jsonl")
        return
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            rows,
            f,
            Dumper=_YAML_DUMPER,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=2000,
        )


def sync_verified(
    staging_dir: Path,
    verified_dir: Path,
    ledger: Ledger,
) -> SyncStats:
    """Sync staged artifacts from staging_dir to verified_dir through the ledger.

    - staging_dir is not mutated
    - verified_dir is created if missing
    - both .yaml/.yml and .jsonl staged artifacts are processed; each verified
      output is written in the same format as its input
    - the JSONL path is end-to-end streaming: one row read, ledger applied,
      surviving row written, row discarded — no list[dict] is ever built
    - the YAML path materializes (PyYAML cannot stream a top-level sequence)
    """
    verified_dir.mkdir(parents=True, exist_ok=True)
    stats = SyncStats()

    staged_files = iter_staged_artifact_paths(staging_dir)
    for i, staged_file in enumerate(staged_files, 1):
        size_mb = staged_file.stat().st_size / (1024 * 1024)
        print(
            f"  [{i}/{len(staged_files)}] {staged_file.name} ({size_mb:.1f} MB)...",
            file=sys.stderr,
            flush=True,
        )

        out_path = verified_dir / staged_file.name
        fmt = _staged_format_from_suffix(staged_file)
        before_input = stats.total_input
        before_passed = stats.passed

        t0 = time.perf_counter()
        if fmt == "jsonl":
            # Bounded-retention path: read, ledger, write, discard per row.
            write_staged_rows(
                out_path,
                _sync_rows(iter_staged_rows(staged_file), ledger, stats),
                fmt="jsonl",
            )
            t_load = t_sync = t_dump = time.perf_counter()
        else:
            raw = load_staged_rows(staged_file)
            t_load = time.perf_counter()
            output = list(_sync_rows(iter(raw), ledger, stats))
            t_sync = time.perf_counter()
            _write_verified(out_path, output, fmt)
            t_dump = time.perf_counter()

        stats.files_processed += 1
        n_in = stats.total_input - before_input
        n_out = stats.passed - before_passed
        n_dropped = n_in - n_out

        if n_dropped:
            print(
                f"           -> {n_in} in, {n_out} out ({n_dropped} removed)",
                file=sys.stderr,
            )
        print(
            "           "
            f"load={t_load - t0:.2f}s "
            f"sync={t_sync - t_load:.2f}s "
            f"dump={t_dump - t_sync:.2f}s",
            file=sys.stderr,
        )

    log.info(
        "Verified sync: %d files, %d input -> %d passed "
        "(%d dropped, %d quarantined, %d rerouted, %d relabeled)",
        stats.files_processed,
        stats.total_input,
        stats.passed,
        stats.dropped,
        stats.quarantined,
        stats.rerouted,
        stats.relabeled,
    )
    return stats
