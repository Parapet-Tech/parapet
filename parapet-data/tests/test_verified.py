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
# Format compatibility: sync_verified handles both .yaml and .jsonl inputs
# and writes output in the matching format per file.
# ---------------------------------------------------------------------------


def _write_staged_jsonl(path: Path, rows: list[dict]) -> Path:
    import json as _json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict]:
    import json as _json
    return [
        _json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestJsonlInputSupport:
    """Phase 2: sync_verified accepts .jsonl staged inputs."""

    def test_jsonl_input_produces_jsonl_output(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        rows = [_staged_row("attack one"), _staged_row("attack two")]
        _write_staged_jsonl(staging / "en_attacks_staged.jsonl", rows)

        stats = sync_verified(staging, verified, _make_ledger([]))

        out_path = verified / "en_attacks_staged.jsonl"
        assert out_path.exists()
        assert not (verified / "en_attacks_staged.yaml").exists()
        out_rows = _read_jsonl(out_path)
        assert len(out_rows) == 2
        assert stats.passed == 2

    def test_mixed_yaml_and_jsonl_preserve_format_per_file(self, tmp_dir: Path) -> None:
        """Each verified output matches its input extension, side by side."""
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        yaml_rows = [_staged_row("yaml row a"), _staged_row("yaml row b")]
        jsonl_rows = [_staged_row("jsonl row a"), _staged_row("jsonl row b")]
        _write_staged(staging / "en_yaml_staged.yaml", yaml_rows)
        _write_staged_jsonl(staging / "en_jsonl_staged.jsonl", jsonl_rows)

        stats = sync_verified(staging, verified, _make_ledger([]))

        yaml_out = verified / "en_yaml_staged.yaml"
        jsonl_out = verified / "en_jsonl_staged.jsonl"
        assert yaml_out.exists() and jsonl_out.exists()
        assert yaml.safe_load(yaml_out.read_text(encoding="utf-8")) == yaml_rows
        assert _read_jsonl(jsonl_out) == jsonl_rows
        # No cross-contamination: yaml input doesn't produce a jsonl output and vice-versa.
        assert not (verified / "en_yaml_staged.jsonl").exists()
        assert not (verified / "en_jsonl_staged.yaml").exists()
        assert stats.passed == 4
        assert stats.files_processed == 2

    def test_ledger_actions_identical_across_formats(self, tmp_dir: Path) -> None:
        """Same rows + same ledger = same passed/dropped counts regardless of input format."""
        rows = [
            _staged_row("keep me"),
            _staged_row("drop me"),
            _staged_row("also keep"),
        ]
        drop_entry = LedgerEntry(
            content_hash=content_hash("drop me"),
            source="test_src",
            action=LedgerAction.DROP,
            adjudication=AdjudicationReason.MISLABEL,
        )
        ledger = _make_ledger([drop_entry])

        yaml_staging = tmp_dir / "yaml_staging"
        yaml_verified = tmp_dir / "yaml_verified"
        _write_staged(yaml_staging / "en_attacks_staged.yaml", rows)
        yaml_stats = sync_verified(yaml_staging, yaml_verified, ledger)

        jsonl_staging = tmp_dir / "jsonl_staging"
        jsonl_verified = tmp_dir / "jsonl_verified"
        _write_staged_jsonl(jsonl_staging / "en_attacks_staged.jsonl", rows)
        jsonl_stats = sync_verified(jsonl_staging, jsonl_verified, _make_ledger([drop_entry]))

        assert yaml_stats.total_input == jsonl_stats.total_input
        assert yaml_stats.passed == jsonl_stats.passed
        assert yaml_stats.dropped == jsonl_stats.dropped
        assert yaml_stats.quarantined == jsonl_stats.quarantined
        assert yaml_stats.rerouted == jsonl_stats.rerouted
        assert yaml_stats.relabeled == jsonl_stats.relabeled
        assert yaml_stats.files_processed == jsonl_stats.files_processed

        yaml_out = yaml.safe_load(
            (yaml_verified / "en_attacks_staged.yaml").read_text(encoding="utf-8")
        )
        jsonl_out = _read_jsonl(jsonl_verified / "en_attacks_staged.jsonl")
        assert yaml_out == jsonl_out

    def test_ignores_quarantine_and_rejection_sidecars(self, tmp_dir: Path) -> None:
        """Regression: sync_verified must skip stage_all's sidecar JSONL files."""
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        _write_staged_jsonl(
            staging / "en_ds_attacks_staged.jsonl",
            [_staged_row("real attack")],
        )
        # Sidecars that share the .jsonl extension but are not staged artifacts:
        (staging.parent / "_init_marker").mkdir(parents=True, exist_ok=True)
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
        # Sidecars must not be copied into verified output dir.
        assert not (verified / "ds_quarantine.jsonl").exists()
        assert not (verified / "staging_rejected.jsonl").exists()

    def test_stats_unchanged_except_filenames(self, tmp_dir: Path) -> None:
        """Stats contract unchanged regardless of input format."""
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        _write_staged_jsonl(
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
# Phase 3: JSONL sync is end-to-end streaming (never materializes a list)
# ---------------------------------------------------------------------------


class TestJsonlStreaming:
    """The JSONL path must not call ``load_staged_rows`` or otherwise
    materialize the input/output as a list. All row survival decisions plus
    writes happen one row at a time."""

    def test_jsonl_sync_does_not_call_load_staged_rows(
        self, tmp_dir: Path, monkeypatch
    ) -> None:
        """Structural proof of streaming: monkeypatch the materializing
        loader to raise. If the JSONL path is truly row-streaming, the sync
        still succeeds because it uses iter_staged_rows, not load_staged_rows."""
        from parapet_data import verified as verified_mod

        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        rows = [_staged_row(f"sample {i}") for i in range(10)]
        _write_staged_jsonl(staging / "en_attacks_staged.jsonl", rows)

        def exploding_load(_path):
            raise AssertionError(
                "load_staged_rows must not be called on the JSONL sync path"
            )

        monkeypatch.setattr(verified_mod, "load_staged_rows", exploding_load)

        stats = sync_verified(staging, verified, _make_ledger([]))

        assert stats.total_input == 10
        assert stats.passed == 10
        out_rows = _read_jsonl(verified / "en_attacks_staged.jsonl")
        assert len(out_rows) == 10

    def test_yaml_sync_still_uses_materializing_loader(
        self, tmp_dir: Path, monkeypatch
    ) -> None:
        """YAML path must still call load_staged_rows — PyYAML cannot stream
        a top-level sequence. This locks the contract so a future refactor
        can't accidentally regress YAML behavior."""
        from parapet_data import verified as verified_mod

        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        rows = [_staged_row("only row")]
        _write_staged(staging / "en_attacks_staged.yaml", rows)

        original = verified_mod.load_staged_rows
        calls: list[Path] = []

        def tracked_load(path):
            calls.append(path)
            return original(path)

        monkeypatch.setattr(verified_mod, "load_staged_rows", tracked_load)

        sync_verified(staging, verified, _make_ledger([]))

        assert len(calls) == 1
        assert calls[0].name == "en_attacks_staged.yaml"

    def test_multi_file_jsonl_sync_processes_all(self, tmp_dir: Path) -> None:
        staging = tmp_dir / "staging"
        verified = tmp_dir / "verified"
        for idx, prefix in enumerate(("en_one", "en_two", "en_three"), start=1):
            _write_staged_jsonl(
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

    def test_all_ledger_actions_parity_across_formats(self, tmp_dir: Path) -> None:
        """Same inputs + same ledger on YAML vs JSONL produce identical
        surviving rows and identical stats deltas across every action."""
        from parapet_data.ledger import LedgerEntry

        rows = [
            _staged_row("pass through me", source="src"),
            _staged_row("drop me", source="src"),
            _staged_row("quarantine me", source="src"),
            _staged_row(
                "reroute me", source="src", reason="instruction_override"
            ),
            _staged_row("relabel me", source="src"),
        ]
        entries = [
            LedgerEntry(
                content_hash=content_hash("drop me"),
                source="src",
                action=LedgerAction.DROP,
                adjudication=AdjudicationReason.MISLABEL,
            ),
            LedgerEntry(
                content_hash=content_hash("quarantine me"),
                source="src",
                action=LedgerAction.QUARANTINE,
                adjudication=AdjudicationReason.MALFORMED_TEXT,
            ),
            LedgerEntry(
                content_hash=content_hash("reroute me"),
                source="src",
                action=LedgerAction.REROUTE_REASON,
                adjudication=AdjudicationReason.ROUTING_DEFECT,
                reroute_to="obfuscation",
            ),
            LedgerEntry(
                content_hash=content_hash("relabel me"),
                source="src",
                action=LedgerAction.RELABEL_CLASS,
                adjudication=AdjudicationReason.MISLABEL,
                relabel_to="benign",
            ),
        ]

        yaml_staging = tmp_dir / "yaml_s"
        yaml_verified = tmp_dir / "yaml_v"
        _write_staged(yaml_staging / "en_attacks_staged.yaml", rows)
        yaml_stats = sync_verified(yaml_staging, yaml_verified, _make_ledger(entries))

        jsonl_staging = tmp_dir / "jsonl_s"
        jsonl_verified = tmp_dir / "jsonl_v"
        _write_staged_jsonl(jsonl_staging / "en_attacks_staged.jsonl", rows)
        jsonl_stats = sync_verified(
            jsonl_staging, jsonl_verified, _make_ledger(entries)
        )

        # All five counter fields should match exactly across formats.
        assert yaml_stats.total_input == jsonl_stats.total_input == 5
        assert yaml_stats.passed == jsonl_stats.passed
        assert yaml_stats.dropped == jsonl_stats.dropped
        assert yaml_stats.quarantined == jsonl_stats.quarantined
        assert yaml_stats.rerouted == jsonl_stats.rerouted
        assert yaml_stats.relabeled == jsonl_stats.relabeled

        # Row-level output parity.
        yaml_out = yaml.safe_load(
            (yaml_verified / "en_attacks_staged.yaml").read_text(encoding="utf-8")
        )
        jsonl_out = _read_jsonl(jsonl_verified / "en_attacks_staged.jsonl")
        assert yaml_out == jsonl_out

        # Specific action semantics survive through streaming:
        out_by_content = {r["content"]: r for r in jsonl_out}
        assert "pass through me" in out_by_content
        assert "drop me" not in out_by_content
        assert "quarantine me" not in out_by_content
        assert out_by_content["reroute me"]["reason"] == "obfuscation"
        assert out_by_content["relabel me"]["label"] == "benign"

    def test_jsonl_sync_accepts_pure_generator_input(
        self, tmp_dir: Path
    ) -> None:
        """Unit test for the streaming core: _sync_rows on a generator
        produces a generator, not a list, and only consumes on iteration."""
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
