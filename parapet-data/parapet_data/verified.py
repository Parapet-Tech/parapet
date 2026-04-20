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

def _sync_file(rows: list[dict], ledger: Ledger, stats: SyncStats) -> list[dict]:
    """Apply ledger actions to a list of staged rows. Returns surviving rows."""
    output: list[dict] = []
    for row in rows:
        stats.total_input += 1
        result = apply_ledger_to_row(row, ledger)
        stats.passed += result.passed
        stats.dropped += result.dropped
        stats.quarantined += result.quarantined
        stats.rerouted += result.rerouted
        stats.relabeled += result.relabeled
        if result.row is not None:
            output.append(result.row)

    return output


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
      output is written in the same format as its input (Phase 3 will make the
      JSONL path streaming — today it is correct but not yet bounded-retention)
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

        t0 = time.perf_counter()
        raw = load_staged_rows(staged_file)
        t_load = time.perf_counter()

        before = stats.total_input
        output = _sync_file(raw, ledger, stats)
        t_sync = time.perf_counter()
        stats.files_processed += 1
        n_in = stats.total_input - before
        n_dropped = n_in - len(output)

        out_path = verified_dir / staged_file.name
        fmt = _staged_format_from_suffix(staged_file)
        _write_verified(out_path, output, fmt)
        t_dump = time.perf_counter()

        if n_dropped:
            print(
                f"           -> {n_in} in, {len(output)} out ({n_dropped} removed)",
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
