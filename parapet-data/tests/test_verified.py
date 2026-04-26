"""Tests for verified sync — ledger-filtered projection of staging.

Active staging is JSONL-only (post-Phase-5). Verified-sync accepts only
JSONL staged inputs; legacy YAML inputs are rejected up front with an
actionable error. These tests cover:

- Core sync behavior (passthrough, drop, quarantine, reroute, relabel,
  keep_hard_case) on JSONL inputs.
- Multi-file processing.
- Sidecar filtering (quarantine/rejection JSONL must be ignored).
- Stats accumulation and content_hash computation on rows missing it.
- Streaming proof: ``_sync_rows`` produces a generator and only consumes
  on iteration.
- Fail-closed: ``*_staged.yaml`` / ``*_staged.yml`` inputs raise a clear
  ValueError BEFORE any verified output is written.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from parapet_data.filters import content_hash
from parapet_data.ledger import (
    AdjudicationReason,
    Ledger,
    LedgerAction,
    LedgerEntry,
)
from parapet_data.verified import SyncStats, sync_verified


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir():
    d = Path(tempfile.mkdtemp(prefix="parapet_verified_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _staged_row(
    text: str,
    label: str = "malicious",
    reason: str = "instruction_override",
    source: str = "test_src",
    language: str = "EN",
) -> dict:
    return {
        "content": text,
        "content_hash": content_hash(text),
        "label": label,
        "reason": reason,
        "source": source,
        "language": language,
    }


def _write_staged(path: Path, rows: list[dict]) -> Path:
    """Write a JSONL staged artifact. Active staging is JSONL-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_ledger(entries: list[LedgerEntry]) -> Ledger:
    return Ledger(entries)


# ---------------------------------------------------------------------------
# Core sync behavior
# ---------------------------------------------------------------------------


class TestSyncPassthrough:
    """With an empty ledger, all rows pass through unchanged."""

    def test_all_rows_survive(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        _write_staged(
            staging / "en_attacks_staged.jsonl",
            [_staged_row("attack one"), _staged_row("attack two")],
        )
        stats = sync_verified(staging, verified, _make_ledger([]))
        out = _read_jsonl(verified / "en_attacks_staged.jsonl")
        assert len(out) == 2
        assert stats.passed == 2

    def test_output_preserves_fields(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        row = _staged_row("test content", reason="exfiltration", language="RU")
        _write_staged(staging / "test_staged.jsonl", [row])
        sync_verified(staging, verified, _make_ledger([]))
        out = _read_jsonl(verified / "test_staged.jsonl")
        assert out[0]["reason"] == "exfiltration"
        assert out[0]["language"] == "RU"
        assert out[0]["content_hash"] == content_hash("test content")


class TestSyncDrop:
    """Rows with drop action are excluded from verified output."""

    def test_dropped_row_excluded(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text_drop = "bad mislabeled row"
        text_keep = "good attack row"
        _write_staged(
            staging / "attacks_staged.jsonl",
            [_staged_row(text_drop), _staged_row(text_keep)],
        )
        ledger = _make_ledger(
            [
                LedgerEntry(
                    content_hash=content_hash(text_drop),
                    source="test_src",
                    action=LedgerAction.DROP,
                    adjudication=AdjudicationReason.MISLABEL,
                ),
            ]
        )
        stats = sync_verified(staging, verified, ledger)
        out = _read_jsonl(verified / "attacks_staged.jsonl")
        assert len(out) == 1
        assert out[0]["content"] == text_keep
        assert stats.dropped == 1
        assert stats.passed == 1

    def test_drop_all_rows_writes_empty(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text = "only row"
        _write_staged(staging / "attacks_staged.jsonl", [_staged_row(text)])
        ledger = _make_ledger(
            [
                LedgerEntry(
                    content_hash=content_hash(text),
                    source="test_src",
                    action=LedgerAction.DROP,
                    adjudication=AdjudicationReason.MISLABEL,
                ),
            ]
        )
        stats = sync_verified(staging, verified, ledger)
        assert _read_jsonl(verified / "attacks_staged.jsonl") == []
        assert stats.dropped == 1
        assert stats.passed == 0


class TestSyncQuarantine:
    """Quarantined rows are excluded like drops."""

    def test_quarantined_row_excluded(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text = "disputed content"
        _write_staged(
            staging / "attacks_staged.jsonl",
            [_staged_row(text), _staged_row("clean row")],
        )
        ledger = _make_ledger(
            [
                LedgerEntry(
                    content_hash=content_hash(text),
                    source="test_src",
                    action=LedgerAction.QUARANTINE,
                    adjudication=AdjudicationReason.OUT_OF_SCOPE,
                ),
            ]
        )
        stats = sync_verified(staging, verified, ledger)
        out = _read_jsonl(verified / "attacks_staged.jsonl")
        assert len(out) == 1
        assert stats.quarantined == 1


class TestSyncReroute:
    """Rerouted rows pass through with corrected reason."""

    def test_reason_corrected(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text = "misrouted attack"
        _write_staged(
            staging / "attacks_staged.jsonl",
            [_staged_row(text, reason="adversarial_suffix")],
        )
        ledger = _make_ledger(
            [
                LedgerEntry(
                    content_hash=content_hash(text),
                    source="test_src",
                    action=LedgerAction.REROUTE_REASON,
                    adjudication=AdjudicationReason.ROUTING_DEFECT,
                    reroute_to="instruction_override",
                ),
            ]
        )
        stats = sync_verified(staging, verified, ledger)
        out = _read_jsonl(verified / "attacks_staged.jsonl")
        assert len(out) == 1
        assert out[0]["reason"] == "instruction_override"
        assert stats.rerouted == 1
        assert stats.passed == 1


class TestSyncRelabel:
    """Relabeled rows pass through with corrected class and optional reason."""

    def test_class_relabeled_row_survives(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text = "misclassified benign row"
        _write_staged(
            staging / "rows_staged.jsonl",
            [_staged_row(text, label="malicious", reason="instruction_override")],
        )
        ledger = _make_ledger(
            [
                LedgerEntry(
                    content_hash=content_hash(text),
                    source="test_src",
                    action=LedgerAction.RELABEL_CLASS,
                    adjudication=AdjudicationReason.MISLABEL,
                    relabel_to="benign",
                    reroute_to="meta_probe",
                ),
            ]
        )
        stats = sync_verified(staging, verified, ledger)
        out = _read_jsonl(verified / "rows_staged.jsonl")
        assert len(out) == 1
        assert out[0]["label"] == "benign"
        assert out[0]["reason"] == "meta_probe"
        assert stats.relabeled == 1
        assert stats.rerouted == 1
        assert stats.passed == 1


class TestSyncKeepHardCase:
    """keep_hard_case is a no-op — row passes unchanged."""

    def test_hard_case_passes_unchanged(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text = "difficult but valid"
        _write_staged(
            staging / "attacks_staged.jsonl",
            [_staged_row(text, reason="obfuscation")],
        )
        ledger = _make_ledger(
            [
                LedgerEntry(
                    content_hash=content_hash(text),
                    source="test_src",
                    action=LedgerAction.KEEP_HARD_CASE,
                    adjudication=AdjudicationReason.MISLABEL,
                    rationale="reviewed, actually valid",
                ),
            ]
        )
        stats = sync_verified(staging, verified, ledger)
        out = _read_jsonl(verified / "attacks_staged.jsonl")
        assert len(out) == 1
        assert out[0]["reason"] == "obfuscation"
        assert stats.passed == 1


# ---------------------------------------------------------------------------
# Multi-file sync + sidecar filtering
# ---------------------------------------------------------------------------


class TestSyncMultiFile:
    """Sync processes all JSONL staged files in staging dir."""

    def test_processes_all_jsonl_files(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        _write_staged(
            staging / "en_attacks_staged.jsonl", [_staged_row("en attack")]
        )
        _write_staged(
            staging / "ru_attacks_staged.jsonl",
            [_staged_row("ru attack", language="RU")],
        )
        stats = sync_verified(staging, verified, _make_ledger([]))
        assert (verified / "en_attacks_staged.jsonl").exists()
        assert (verified / "ru_attacks_staged.jsonl").exists()
        assert stats.passed == 2
        assert stats.files_processed == 2

    def test_ignores_non_staged_files(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        _write_staged(
            staging / "attacks_staged.jsonl", [_staged_row("attack")]
        )
        (staging / "readme.txt").write_text("not staged", encoding="utf-8")
        stats = sync_verified(staging, verified, _make_ledger([]))
        assert stats.files_processed == 1
        assert not (verified / "readme.txt").exists()

    def test_ignores_quarantine_and_rejection_sidecars(self, tmp_dir: Path) -> None:
        """sync_verified must skip stage_all's sidecar JSONL files."""
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        _write_staged(
            staging / "en_ds_attacks_staged.jsonl",
            [_staged_row("real attack")],
        )
        # Sidecars that share the .jsonl extension but are not staged artifacts:
        (staging / "ds_quarantine.jsonl").write_text(
            '{"content":"QUARANTINE_LEAK","source":"x","reason":"x"}\n',
            encoding="utf-8",
        )
        (staging / "staging_rejected.jsonl").write_text(
            '{"source":"x","gate":"x","detail":"x","preview":"REJECTED_LEAK"}\n',
            encoding="utf-8",
        )

        stats = sync_verified(staging, verified, _make_ledger([]))

        assert stats.files_processed == 1
        assert stats.total_input == 1
        assert (verified / "en_ds_attacks_staged.jsonl").exists()
        assert not (verified / "ds_quarantine.jsonl").exists()
        assert not (verified / "staging_rejected.jsonl").exists()


# ---------------------------------------------------------------------------
# SyncStats
# ---------------------------------------------------------------------------


class TestSyncStats:
    def test_stats_accumulate_across_files(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text_drop = "drop me"
        _write_staged(
            staging / "file1_staged.jsonl",
            [_staged_row("keep1"), _staged_row(text_drop)],
        )
        _write_staged(
            staging / "file2_staged.jsonl",
            [_staged_row("keep2")],
        )
        ledger = _make_ledger(
            [
                LedgerEntry(
                    content_hash=content_hash(text_drop),
                    source="test_src",
                    action=LedgerAction.DROP,
                    adjudication=AdjudicationReason.MISLABEL,
                ),
            ]
        )
        stats = sync_verified(staging, verified, ledger)
        assert stats.total_input == 3
        assert stats.passed == 2
        assert stats.dropped == 1
        assert stats.quarantined == 0
        assert stats.rerouted == 0
        assert stats.files_processed == 2

    def test_stats_returns_syncstats_instance(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        _write_staged(
            staging / "en_attacks_staged.jsonl",
            [_staged_row(f"sample {i}") for i in range(5)],
        )
        stats = sync_verified(staging, verified, _make_ledger([]))
        assert isinstance(stats, SyncStats)
        assert stats.files_processed == 1
        assert stats.total_input == 5
        assert stats.passed == 5
        assert stats.dropped == 0


# ---------------------------------------------------------------------------
# Hash computation on rows missing content_hash
# ---------------------------------------------------------------------------


class TestSyncHashComputation:
    """Verified sync should compute content_hash if missing from staged row."""

    def test_computes_hash_when_missing(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        row = {
            "content": "no hash here",
            "label": "malicious",
            "reason": "instruction_override",
            "source": "test",
            "language": "EN",
        }
        _write_staged(staging / "attacks_staged.jsonl", [row])
        text_hash = content_hash("no hash here")
        ledger = _make_ledger(
            [
                LedgerEntry(
                    content_hash=text_hash,
                    source="test",
                    action=LedgerAction.DROP,
                    adjudication=AdjudicationReason.MISLABEL,
                ),
            ]
        )
        stats = sync_verified(staging, verified, ledger)
        assert stats.dropped == 1

    def test_output_always_has_content_hash(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        row = {
            "content": "missing hash field",
            "label": "benign",
            "reason": "meta_probe",
            "source": "test",
            "language": "EN",
        }
        _write_staged(staging / "benign_staged.jsonl", [row])
        sync_verified(staging, verified, _make_ledger([]))
        out = _read_jsonl(verified / "benign_staged.jsonl")
        assert "content_hash" in out[0]
        assert out[0]["content_hash"] == content_hash("missing hash field")


# ---------------------------------------------------------------------------
# Streaming contract
# ---------------------------------------------------------------------------


class TestJsonlStreaming:
    """The sync path is end-to-end streaming. A row is read, ledger is
    applied, surviving row is written, then the row is dropped — no
    list[dict] is ever materialized."""

    def test_jsonl_sync_accepts_pure_generator_input(self, tmp_dir: Path) -> None:
        """``_sync_rows`` produces a generator and only consumes on iteration."""
        from parapet_data.verified import _sync_rows

        stats = SyncStats()
        consumed: list[int] = []

        def source():
            for i, content in enumerate(("a", "b", "c")):
                consumed.append(i)
                yield _staged_row(content)

        gen = _sync_rows(source(), _make_ledger([]), stats)

        # Not a list — the input hasn't been touched until we iterate.
        assert iter(gen) is gen
        assert consumed == []

        out = list(gen)
        assert consumed == [0, 1, 2]
        assert len(out) == 3
        assert stats.total_input == 3
        assert stats.passed == 3

    def test_multi_file_jsonl_sync_processes_all(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        for idx, prefix in enumerate(("en_one", "en_two", "en_three"), start=1):
            _write_staged(
                staging / f"{prefix}_attacks_staged.jsonl",
                [_staged_row(f"{prefix} sample {i}") for i in range(idx)],
            )

        stats = sync_verified(staging, verified, _make_ledger([]))

        assert stats.files_processed == 3
        assert stats.total_input == 1 + 2 + 3
        assert stats.passed == stats.total_input
        for prefix, count in zip(("en_one", "en_two", "en_three"), (1, 2, 3)):
            rows = _read_jsonl(verified / f"{prefix}_attacks_staged.jsonl")
            assert len(rows) == count


# ---------------------------------------------------------------------------
# Fail-closed: legacy YAML staged input must be rejected
# ---------------------------------------------------------------------------


class TestRejectsLegacyYamlInput:
    """Active verified-sync is JSONL-only. Legacy ``*_staged.yaml`` /
    ``*_staged.yml`` inputs must raise an actionable error before any
    output is written."""

    def _write_legacy_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "- content: legacy row\n  label: malicious\n",
            encoding="utf-8",
        )

    def test_yaml_input_raises_value_error(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        self._write_legacy_yaml(staging / "en_attacks_staged.yaml")

        with pytest.raises(ValueError, match="only accepts JSONL"):
            sync_verified(staging, verified, _make_ledger([]))

    def test_yml_input_raises_value_error(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        self._write_legacy_yaml(staging / "en_attacks_staged.yml")

        with pytest.raises(ValueError, match="only accepts JSONL"):
            sync_verified(staging, verified, _make_ledger([]))

    def test_error_directs_user_to_restage(self, tmp_dir: Path) -> None:
        """Error message must point at the canonical recovery path."""
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        self._write_legacy_yaml(staging / "en_attacks_staged.yaml")

        with pytest.raises(ValueError) as exc_info:
            sync_verified(staging, verified, _make_ledger([]))
        # The error must NOT reference the deleted Phase 4 converter.
        assert "convert_staged_yaml_to_jsonl" not in str(exc_info.value)
        assert "convert_staging_dir" not in str(exc_info.value)
        # It MUST direct the user to re-run staging.
        assert "stage" in str(exc_info.value).lower()

    def test_yaml_input_fails_before_any_output_written(
        self, tmp_dir: Path
    ) -> None:
        """Validation runs up-front so a YAML file 13-deep doesn't leave
        a half-populated verified_dir behind."""
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        # Mix: many JSONL files first, then one legacy YAML.
        for i in range(3):
            _write_staged(
                staging / f"en_clean_{i}_staged.jsonl",
                [_staged_row(f"clean {i}")],
            )
        self._write_legacy_yaml(staging / "en_attacks_staged.yaml")

        with pytest.raises(ValueError, match="only accepts JSONL"):
            sync_verified(staging, verified, _make_ledger([]))

        # No JSONL outputs should have been written either — the up-front
        # validation rejects the whole batch before any I/O.
        assert not any(verified.glob("*_staged.jsonl"))
