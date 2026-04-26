"""
Verified sync: ledger-filtered projection of staging.

Reads staged JSONL artifacts, applies adjudication ledger actions, and
writes surviving rows to the verified output directory as JSONL. This is
a mechanical transform — no new judgment, no heuristic decisions. All
judgment comes from the ledger.

Active staging is JSONL-only (post-Phase-5). Legacy ``*_staged.yaml``
inputs are rejected with an actionable error; the canonical recovery is
to re-run staging.

Usage:
    staging/ -> verified/sync -> verified/
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from parapet_data.ledger import Ledger, apply_ledger_to_row
from parapet_data.staged_artifact import (
    iter_staged_artifact_paths,
    iter_staged_rows,
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


def _reject_legacy_yaml_staged_input(path: Path) -> None:
    """Raise if ``path`` is a legacy ``*_staged.{yaml,yml}`` artifact.

    Active verified-sync is JSONL-only. The error directs the user to the
    canonical recovery path — re-running staging produces JSONL artifacts.
    """
    if path.suffix.lower() in (".yaml", ".yml"):
        raise ValueError(
            f"{path}: verified-sync only accepts JSONL staged artifacts. "
            "Re-run `python -m parapet_data stage` to produce JSONL "
            "artifacts in the active staging directory."
        )


def sync_verified(
    staging_dir: Path,
    verified_dir: Path,
    ledger: Ledger,
) -> SyncStats:
    """Sync staged JSONL artifacts from staging_dir to verified_dir.

    - staging_dir is not mutated
    - verified_dir is created if missing
    - only ``.jsonl`` staged artifacts are accepted; legacy ``*_staged.yaml``
      input is rejected with an actionable error before any output is written
    - the sync path is end-to-end streaming: one row read, ledger applied,
      surviving row written, row discarded — no list[dict] is ever built
    """
    verified_dir.mkdir(parents=True, exist_ok=True)
    stats = SyncStats()

    staged_files = iter_staged_artifact_paths(staging_dir)

    # Validate every input up front so a legacy YAML 13 files in doesn't
    # leave a half-populated verified_dir behind.
    for staged_file in staged_files:
        _reject_legacy_yaml_staged_input(staged_file)

    for i, staged_file in enumerate(staged_files, 1):
        size_mb = staged_file.stat().st_size / (1024 * 1024)
        print(
            f"  [{i}/{len(staged_files)}] {staged_file.name} ({size_mb:.1f} MB)...",
            file=sys.stderr,
            flush=True,
        )

        out_path = verified_dir / staged_file.name
        before_input = stats.total_input
        before_passed = stats.passed

        t0 = time.perf_counter()
        write_staged_rows(
            out_path,
            _sync_rows(iter_staged_rows(staged_file), ledger, stats),
        )
        t_done = time.perf_counter()

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
            f"           sync={t_done - t0:.2f}s",
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
