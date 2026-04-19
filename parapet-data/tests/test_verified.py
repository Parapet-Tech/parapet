"""Tests for verified sync — ledger-filtered projection of staging."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from parapet_data.filters import content_hash
from parapet_data.ledger import Ledger, LedgerEntry, LedgerAction, AdjudicationReason
from parapet_data.verified import sync_verified, SyncStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir():
    d = Path(tempfile.mkdtemp(prefix="parapet_verified_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _staged_row(text: str, label: str = "malicious", reason: str = "instruction_override",
                source: str = "test_src", language: str = "EN") -> dict:
    return {
        "content": text,
        "content_hash": content_hash(text),
        "label": label,
        "reason": reason,
        "source": source,
        "language": language,
    }


def _write_staged(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(rows, allow_unicode=True), encoding="utf-8")
    return path


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
        _write_staged(staging / "en_attacks_staged.yaml", [
            _staged_row("attack one"),
            _staged_row("attack two"),
        ])
        stats = sync_verified(staging, verified, _make_ledger([]))
        out = yaml.safe_load((verified / "en_attacks_staged.yaml").read_text(encoding="utf-8"))
        assert len(out) == 2
        assert stats.passed == 2

    def test_output_preserves_fields(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        row = _staged_row("test content", reason="exfiltration", language="RU")
        _write_staged(staging / "test.yaml", [row])
        sync_verified(staging, verified, _make_ledger([]))
        out = yaml.safe_load((verified / "test.yaml").read_text(encoding="utf-8"))
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
        _write_staged(staging / "attacks.yaml", [
            _staged_row(text_drop),
            _staged_row(text_keep),
        ])
        ledger = _make_ledger([
            LedgerEntry(
                content_hash=content_hash(text_drop),
                source="test_src",
                action=LedgerAction.DROP,
                adjudication=AdjudicationReason.MISLABEL,
            ),
        ])
        stats = sync_verified(staging, verified, ledger)
        out = yaml.safe_load((verified / "attacks.yaml").read_text(encoding="utf-8"))
        assert len(out) == 1
        assert out[0]["content"] == text_keep
        assert stats.dropped == 1
        assert stats.passed == 1

    def test_drop_all_rows_writes_empty(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text = "only row"
        _write_staged(staging / "attacks.yaml", [_staged_row(text)])
        ledger = _make_ledger([
            LedgerEntry(
                content_hash=content_hash(text),
                source="test_src",
                action=LedgerAction.DROP,
                adjudication=AdjudicationReason.MISLABEL,
            ),
        ])
        stats = sync_verified(staging, verified, ledger)
        out = yaml.safe_load((verified / "attacks.yaml").read_text(encoding="utf-8"))
        assert out == [] or out is None
        assert stats.dropped == 1
        assert stats.passed == 0


class TestSyncQuarantine:
    """Quarantined rows are excluded like drops."""

    def test_quarantined_row_excluded(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text = "disputed content"
        _write_staged(staging / "attacks.yaml", [
            _staged_row(text),
            _staged_row("clean row"),
        ])
        ledger = _make_ledger([
            LedgerEntry(
                content_hash=content_hash(text),
                source="test_src",
                action=LedgerAction.QUARANTINE,
                adjudication=AdjudicationReason.OUT_OF_SCOPE,
            ),
        ])
        stats = sync_verified(staging, verified, ledger)
        out = yaml.safe_load((verified / "attacks.yaml").read_text(encoding="utf-8"))
        assert len(out) == 1
        assert stats.quarantined == 1


class TestSyncReroute:
    """Rerouted rows pass through with corrected reason."""

    def test_reason_corrected(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text = "misrouted attack"
        _write_staged(staging / "attacks.yaml", [
            _staged_row(text, reason="adversarial_suffix"),
        ])
        ledger = _make_ledger([
            LedgerEntry(
                content_hash=content_hash(text),
                source="test_src",
                action=LedgerAction.REROUTE_REASON,
                adjudication=AdjudicationReason.ROUTING_DEFECT,
                reroute_to="instruction_override",
            ),
        ])
        stats = sync_verified(staging, verified, ledger)
        out = yaml.safe_load((verified / "attacks.yaml").read_text(encoding="utf-8"))
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
        _write_staged(staging / "rows.yaml", [
            _staged_row(text, label="malicious", reason="instruction_override"),
        ])
        ledger = _make_ledger([
            LedgerEntry(
                content_hash=content_hash(text),
                source="test_src",
                action=LedgerAction.RELABEL_CLASS,
                adjudication=AdjudicationReason.MISLABEL,
                relabel_to="benign",
                reroute_to="meta_probe",
            ),
        ])
        stats = sync_verified(staging, verified, ledger)
        out = yaml.safe_load((verified / "rows.yaml").read_text(encoding="utf-8"))
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
        _write_staged(staging / "attacks.yaml", [
            _staged_row(text, reason="obfuscation"),
        ])
        ledger = _make_ledger([
            LedgerEntry(
                content_hash=content_hash(text),
                source="test_src",
                action=LedgerAction.KEEP_HARD_CASE,
                adjudication=AdjudicationReason.MISLABEL,
                rationale="reviewed, actually valid",
            ),
        ])
        stats = sync_verified(staging, verified, ledger)
        out = yaml.safe_load((verified / "attacks.yaml").read_text(encoding="utf-8"))
        assert len(out) == 1
        assert out[0]["reason"] == "obfuscation"
        assert stats.passed == 1


# ---------------------------------------------------------------------------
# Multi-file sync
# ---------------------------------------------------------------------------


class TestSyncMultiFile:
    """Sync processes all YAML files in staging dir."""

    def test_processes_all_yaml_files(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        _write_staged(staging / "en_attacks.yaml", [_staged_row("en attack")])
        _write_staged(staging / "ru_attacks.yaml", [
            _staged_row("ru attack", language="RU"),
        ])
        stats = sync_verified(staging, verified, _make_ledger([]))
        assert (verified / "en_attacks.yaml").exists()
        assert (verified / "ru_attacks.yaml").exists()
        assert stats.passed == 2
        assert stats.files_processed == 2

    def test_ignores_non_yaml_files(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        _write_staged(staging / "attacks.yaml", [_staged_row("attack")])
        (staging / "readme.txt").write_text("not yaml", encoding="utf-8")
        stats = sync_verified(staging, verified, _make_ledger([]))
        assert stats.files_processed == 1
        assert not (verified / "readme.txt").exists()


# ---------------------------------------------------------------------------
# SyncStats
# ---------------------------------------------------------------------------


class TestSyncStats:
    def test_stats_accumulate_across_files(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        text_drop = "drop me"
        _write_staged(staging / "file1.yaml", [
            _staged_row("keep1"),
            _staged_row(text_drop),
        ])
        _write_staged(staging / "file2.yaml", [
            _staged_row("keep2"),
        ])
        ledger = _make_ledger([
            LedgerEntry(
                content_hash=content_hash(text_drop),
                source="test_src",
                action=LedgerAction.DROP,
                adjudication=AdjudicationReason.MISLABEL,
            ),
        ])
        stats = sync_verified(staging, verified, ledger)
        assert stats.total_input == 3
        assert stats.passed == 2
        assert stats.dropped == 1
        assert stats.quarantined == 0
        assert stats.rerouted == 0
        assert stats.files_processed == 2


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
        _write_staged(staging / "attacks.yaml", [row])
        text_hash = content_hash("no hash here")
        ledger = _make_ledger([
            LedgerEntry(
                content_hash=text_hash,
                source="test",
                action=LedgerAction.DROP,
                adjudication=AdjudicationReason.MISLABEL,
            ),
        ])
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
        _write_staged(staging / "benign.yaml", [row])
        sync_verified(staging, verified, _make_ledger([]))
        out = yaml.safe_load((verified / "benign.yaml").read_text(encoding="utf-8"))
        assert "content_hash" in out[0]
        assert out[0]["content_hash"] == content_hash("missing hash field")


# ---------------------------------------------------------------------------
# Guardrail: JSONL staged inputs are explicitly rejected (Phase 1 boundary)
# ---------------------------------------------------------------------------


class TestJsonlGuardrail:
    """verified-sync must refuse .jsonl staged inputs until Phase 3 lands."""

    def test_raises_when_staging_contains_jsonl(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        staging.mkdir(parents=True)
        (staging / "en_ds_attacks_staged.jsonl").write_text(
            '{"content":"x","label":"malicious","content_hash":"h","source":"s","reason":"r","language":"EN"}\n',
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="does not yet support .jsonl"):
            sync_verified(staging, verified, _make_ledger([]))

    def test_error_message_points_to_phase_3(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        staging.mkdir(parents=True)
        (staging / "en_x.jsonl").write_text('{"content":"x"}\n', encoding="utf-8")

        with pytest.raises(ValueError, match="Phase 3"):
            sync_verified(staging, verified, _make_ledger([]))

    def test_raises_even_when_yaml_also_present(self, tmp_dir: Path) -> None:
        """Mixed staging dirs still fail — prevents partial silent processing."""
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        _write_staged(staging / "en_yaml_staged.yaml", [_staged_row("yaml survivor")])
        (staging / "en_jsonl_staged.jsonl").write_text('{"content":"x"}\n', encoding="utf-8")

        with pytest.raises(ValueError, match="does not yet support .jsonl"):
            sync_verified(staging, verified, _make_ledger([]))

        # Verified dir must not have been written to partially.
        assert not (verified / "en_yaml_staged.yaml").exists()
