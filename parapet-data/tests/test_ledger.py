"""Tests for the adjudication ledger model and loader."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from parapet_data.ledger import (
    AdjudicationReason,
    Ledger,
    LedgerAction,
    LedgerEntry,
    apply_ledger_to_row,
)
from parapet_data.filters import content_hash


@pytest.fixture()
def tmp_dir():
    d = Path(tempfile.mkdtemp(prefix="parapet_ledger_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# LedgerEntry
# ---------------------------------------------------------------------------


class TestLedgerEntry:
    def test_minimal_valid_entry(self) -> None:
        entry = LedgerEntry(
            content_hash="abc123",
            source="ru_staged_attacks",
            action=LedgerAction.DROP,
            adjudication=AdjudicationReason.MISLABEL,
        )
        assert entry.content_hash == "abc123"
        assert entry.action == LedgerAction.DROP
        assert entry.adjudication == AdjudicationReason.MISLABEL

    def test_full_entry(self) -> None:
        entry = LedgerEntry(
            content_hash="7d90c7c6a33970f2",
            source="ru_staged_attacks_adversarial_suffix",
            label_at_time="malicious",
            reason_at_time="adversarial_suffix",
            action=LedgerAction.REROUTE_REASON,
            adjudication=AdjudicationReason.ROUTING_DEFECT,
            reroute_to="instruction_override",
            rationale="attack is an instruction override, not adversarial suffix",
            first_seen_run="mirror_v3_19k_clean",
            reviewer="human",
            updated_at="2026-03-06",
        )
        assert entry.reroute_to == "instruction_override"
        assert entry.rationale is not None

    def test_reroute_requires_reroute_to(self) -> None:
        with pytest.raises(ValueError, match="reroute_to"):
            LedgerEntry(
                content_hash="abc",
                source="test",
                action=LedgerAction.REROUTE_REASON,
                adjudication=AdjudicationReason.ROUTING_DEFECT,
                # missing reroute_to
            )

    def test_relabel_requires_relabel_to(self) -> None:
        with pytest.raises(ValueError, match="relabel_to"):
            LedgerEntry(
                content_hash="abc",
                source="test",
                action=LedgerAction.RELABEL_CLASS,
                adjudication=AdjudicationReason.MISLABEL,
            )

    def test_relabel_accepts_reason_override(self) -> None:
        entry = LedgerEntry(
            content_hash="abc",
            source="test",
            action=LedgerAction.RELABEL_CLASS,
            adjudication=AdjudicationReason.MISLABEL,
            relabel_to="benign",
            reroute_to="meta_probe",
        )
        assert entry.relabel_to == "benign"
        assert entry.reroute_to == "meta_probe"

    def test_non_reroute_ignores_reroute_to(self) -> None:
        entry = LedgerEntry(
            content_hash="abc",
            source="test",
            action=LedgerAction.DROP,
            adjudication=AdjudicationReason.MISLABEL,
        )
        assert entry.reroute_to is None


# ---------------------------------------------------------------------------
# LedgerAction enum
# ---------------------------------------------------------------------------


class TestLedgerAction:
    def test_all_actions_present(self) -> None:
        actions = {a.value for a in LedgerAction}
        assert actions == {
            "drop",
            "quarantine",
            "reroute_reason",
            "relabel_class",
            "keep_hard_case",
        }

    def test_excludes_from_training(self) -> None:
        assert LedgerAction.DROP.excludes_from_training
        assert LedgerAction.QUARANTINE.excludes_from_training
        assert not LedgerAction.REROUTE_REASON.excludes_from_training
        assert not LedgerAction.RELABEL_CLASS.excludes_from_training
        assert not LedgerAction.KEEP_HARD_CASE.excludes_from_training


# ---------------------------------------------------------------------------
# AdjudicationReason enum
# ---------------------------------------------------------------------------


class TestAdjudicationReason:
    def test_all_reasons_present(self) -> None:
        reasons = {r.value for r in AdjudicationReason}
        assert reasons == {
            "mislabel",
            "non_attack_in_attack_set",
            "benign_contamination",
            "routing_defect",
            "extraction_defect",
            "malformed_text",
            "duplicate_leakage",
            "holdout_leakage",
            "out_of_scope",
        }


# ---------------------------------------------------------------------------
# Ledger loader
# ---------------------------------------------------------------------------


class TestLedger:
    def _write_ledger(self, path: Path, entries: list[dict]) -> Path:
        ledger_path = path / "ledger.yaml"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(yaml.dump(entries, allow_unicode=True), encoding="utf-8")
        return ledger_path

    def test_load_from_yaml(self, tmp_dir: Path) -> None:
        ledger_path = self._write_ledger(tmp_dir, [
            {
                "content_hash": "aaa111",
                "source": "test_source",
                "action": "drop",
                "adjudication": "mislabel",
            },
            {
                "content_hash": "bbb222",
                "source": "test_source_2",
                "action": "quarantine",
                "adjudication": "out_of_scope",
                "rationale": "not a prompt injection sample",
            },
        ])
        ledger = Ledger.load(ledger_path)
        assert len(ledger) == 2

    def test_lookup_by_hash(self, tmp_dir: Path) -> None:
        ledger_path = self._write_ledger(tmp_dir, [
            {
                "content_hash": "aaa111",
                "source": "src",
                "action": "drop",
                "adjudication": "mislabel",
            },
        ])
        ledger = Ledger.load(ledger_path)
        entry = ledger.lookup("aaa111")
        assert entry is not None
        assert entry.action == LedgerAction.DROP

    def test_lookup_miss_returns_none(self, tmp_dir: Path) -> None:
        ledger_path = self._write_ledger(tmp_dir, [
            {
                "content_hash": "aaa111",
                "source": "src",
                "action": "drop",
                "adjudication": "mislabel",
            },
        ])
        ledger = Ledger.load(ledger_path)
        assert ledger.lookup("zzz999") is None

    def test_empty_ledger(self, tmp_dir: Path) -> None:
        ledger_path = self._write_ledger(tmp_dir, [])
        ledger = Ledger.load(ledger_path)
        assert len(ledger) == 0
        assert ledger.lookup("anything") is None

    def test_missing_file_returns_empty(self, tmp_dir: Path) -> None:
        ledger = Ledger.load(tmp_dir / "nonexistent.yaml")
        assert len(ledger) == 0

    def test_duplicate_hash_raises(self, tmp_dir: Path) -> None:
        ledger_path = self._write_ledger(tmp_dir, [
            {
                "content_hash": "aaa111",
                "source": "src1",
                "action": "drop",
                "adjudication": "mislabel",
            },
            {
                "content_hash": "aaa111",
                "source": "src2",
                "action": "quarantine",
                "adjudication": "out_of_scope",
            },
        ])
        with pytest.raises(ValueError, match="duplicate.*aaa111"):
            Ledger.load(ledger_path)

    def test_reroute_entry_loads(self, tmp_dir: Path) -> None:
        ledger_path = self._write_ledger(tmp_dir, [
            {
                "content_hash": "ccc333",
                "source": "src",
                "action": "reroute_reason",
                "adjudication": "routing_defect",
                "reroute_to": "instruction_override",
            },
        ])
        ledger = Ledger.load(ledger_path)
        entry = ledger.lookup("ccc333")
        assert entry is not None
        assert entry.action == LedgerAction.REROUTE_REASON
        assert entry.reroute_to == "instruction_override"

    def test_keep_hard_case_loads(self, tmp_dir: Path) -> None:
        ledger_path = self._write_ledger(tmp_dir, [
            {
                "content_hash": "ddd444",
                "source": "src",
                "action": "keep_hard_case",
                "adjudication": "mislabel",
                "rationale": "reviewed, actually valid attack",
            },
        ])
        ledger = Ledger.load(ledger_path)
        entry = ledger.lookup("ddd444")
        assert entry is not None
        assert entry.action == LedgerAction.KEEP_HARD_CASE

    def test_relabel_entry_loads(self, tmp_dir: Path) -> None:
        ledger_path = self._write_ledger(tmp_dir, [
            {
                "content_hash": "eee555",
                "source": "src",
                "action": "relabel_class",
                "adjudication": "mislabel",
                "relabel_to": "benign",
                "reroute_to": "meta_probe",
            },
        ])
        ledger = Ledger.load(ledger_path)
        entry = ledger.lookup("eee555")
        assert entry is not None
        assert entry.action == LedgerAction.RELABEL_CLASS
        assert entry.relabel_to == "benign"
        assert entry.reroute_to == "meta_probe"


# ---------------------------------------------------------------------------
# Shared row application
# ---------------------------------------------------------------------------


class TestApplyLedgerToRow:
    def _row(
        self,
        text: str,
        *,
        reason: str = "instruction_override",
        label: str = "malicious",
    ) -> dict:
        return {
            "content": text,
            "label": label,
            "reason": reason,
            "source": "test_source",
            "language": "EN",
        }

    def test_passthrough_adds_content_hash(self) -> None:
        row = self._row("keep me")
        result = apply_ledger_to_row(row, Ledger([]))
        assert result.row is not None
        assert result.row["content_hash"] == content_hash("keep me")
        assert result.passed == 1
        assert result.action is None
        assert "content_hash" not in row

    def test_drop_returns_none_row(self) -> None:
        row = self._row("drop me")
        ledger = Ledger([
            LedgerEntry(
                content_hash=content_hash("drop me"),
                source="test_source",
                action=LedgerAction.DROP,
                adjudication=AdjudicationReason.MISLABEL,
            ),
        ])
        result = apply_ledger_to_row(row, ledger)
        assert result.row is None
        assert result.dropped == 1
        assert result.action == LedgerAction.DROP

    def test_stale_row_content_hash_does_not_override_content(self) -> None:
        row = self._row("drop me")
        row["content_hash"] = content_hash("stale metadata")
        ledger = Ledger([
            LedgerEntry(
                content_hash=content_hash("drop me"),
                source="test_source",
                action=LedgerAction.DROP,
                adjudication=AdjudicationReason.MISLABEL,
            ),
        ])
        result = apply_ledger_to_row(row, ledger)
        assert result.row is None
        assert result.dropped == 1
        assert result.content_hash == content_hash("drop me")

    def test_quarantine_returns_none_row(self) -> None:
        row = self._row("quarantine me")
        ledger = Ledger([
            LedgerEntry(
                content_hash=content_hash("quarantine me"),
                source="test_source",
                action=LedgerAction.QUARANTINE,
                adjudication=AdjudicationReason.OUT_OF_SCOPE,
            ),
        ])
        result = apply_ledger_to_row(row, ledger)
        assert result.row is None
        assert result.quarantined == 1
        assert result.action == LedgerAction.QUARANTINE

    def test_reroute_rewrites_reason_on_copy(self) -> None:
        row = self._row("reroute me", reason="adversarial_suffix")
        ledger = Ledger([
            LedgerEntry(
                content_hash=content_hash("reroute me"),
                source="test_source",
                action=LedgerAction.REROUTE_REASON,
                adjudication=AdjudicationReason.ROUTING_DEFECT,
                reroute_to="instruction_override",
            ),
        ])
        result = apply_ledger_to_row(row, ledger)
        assert result.row is not None
        assert result.row["reason"] == "instruction_override"
        assert row["reason"] == "adversarial_suffix"
        assert result.rerouted == 1
        assert result.passed == 1

    def test_keep_hard_case_passes_through_with_action(self) -> None:
        row = self._row("hard case", reason="obfuscation")
        ledger = Ledger([
            LedgerEntry(
                content_hash=content_hash("hard case"),
                source="test_source",
                action=LedgerAction.KEEP_HARD_CASE,
                adjudication=AdjudicationReason.MISLABEL,
            ),
        ])
        result = apply_ledger_to_row(row, ledger)
        assert result.row is not None
        assert result.row["reason"] == "obfuscation"
        assert result.action == LedgerAction.KEEP_HARD_CASE
        assert result.passed == 1

    def test_relabel_changes_class_and_optionally_reason(self) -> None:
        row = self._row("relabel me", reason="instruction_override", label="malicious")
        ledger = Ledger([
            LedgerEntry(
                content_hash=content_hash("relabel me"),
                source="test_source",
                action=LedgerAction.RELABEL_CLASS,
                adjudication=AdjudicationReason.MISLABEL,
                relabel_to="benign",
                reroute_to="meta_probe",
            ),
        ])
        result = apply_ledger_to_row(row, ledger)
        assert result.row is not None
        assert result.row["label"] == "benign"
        assert result.row["reason"] == "meta_probe"
        assert row["label"] == "malicious"
        assert row["reason"] == "instruction_override"
        assert result.relabeled == 1
        assert result.rerouted == 1
