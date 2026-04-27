"""Tests for heuristic reason preclassification."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest
import yaml

from parapet_data.filters import content_hash
from parapet_data.models import (
    AttackReason,
    FormatBin,
    Language,
    LengthBin,
    MirrorCell,
    MirrorSpec,
    ReasonProvenance,
    SourceRef,
    SourceGroundingMode,
    SourceRoutePolicy,
)
from parapet_data.reason_preclassify import (
    heuristic_attack_sources,
    preclassified_source_filename,
    preclassify_reason_source,
    preclassify_reason_sources_from_spec,
)


@pytest.fixture()
def tmp_dir():
    root = Path(__file__).resolve().parents[1] / "test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    d = root / f"parapet_reason_preclassify_test_{uuid.uuid4().hex}"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _write_yaml(path: Path, data: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _heuristic_source(path: Path, *, name: str = "heuristic_src") -> SourceRef:
    return SourceRef(
        name=name,
        path=path,
        language=Language.EN,
        extractor="col_content",
        grounding_mode=SourceGroundingMode.REASON_GROUNDED,
        route_policy=SourceRoutePolicy.MIRROR,
        reason_provenance=ReasonProvenance.HEURISTIC,
        applicability_scope="in_domain",
    )


def test_preclassify_reason_source_keeps_unclassified_rows(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unclassified rows must be preserved as ``reason: uncategorized``.

    Dropping them silently loses training signal — they're still
    known-malicious data, the classifier just couldn't sub-route them.
    """
    from parapet_data import reason_preclassify as module

    input_path = _write_yaml(
        tmp_dir / "source.yaml",
        [
            {"content": "keep this classified row"},
            {"content": "keep this unclassified row too"},
            {"content": ""},
        ],
    )
    source = _heuristic_source(input_path)
    output_path = tmp_dir / "out" / "source.jsonl"

    class _Classification:
        reason = AttackReason.INSTRUCTION_OVERRIDE
        confidence = 0.9
        signals = ("unit_test_signal",)

    def fake_classifier(text: str):
        if text.startswith("keep this classified"):
            return _Classification()
        return None

    monkeypatch.setattr(module, "classify_reason", fake_classifier)

    report = preclassify_reason_source(source, output_path)

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert report.rows_read == 3
    assert report.rows_with_text == 2
    assert report.rows_written == 2
    assert report.empty_text == 1
    assert report.classified == 1
    assert report.uncategorized == 1
    assert report.by_reason == {
        AttackReason.INSTRUCTION_OVERRIDE.value: 1,
        AttackReason.UNCATEGORIZED.value: 1,
    }
    assert report.classified_rate == 0.5
    assert report.keep_rate == 1.0  # every text-bearing row is preserved
    assert report.output_sha256
    assert rows == [
        {
            "content": "keep this classified row",
            "label": "malicious",
            "language": "EN",
            "source": "heuristic_src",
            "reason": AttackReason.INSTRUCTION_OVERRIDE.value,
            "reason_confidence": 0.9,
            "reason_signals": ["unit_test_signal"],
            "content_hash": content_hash("keep this classified row"),
        },
        {
            "content": "keep this unclassified row too",
            "label": "malicious",
            "language": "EN",
            "source": "heuristic_src",
            "reason": AttackReason.UNCATEGORIZED.value,
            "reason_confidence": None,
            "reason_signals": [],
            "content_hash": content_hash("keep this unclassified row too"),
        },
    ]


def test_preclassify_reason_source_uncategorized_only(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source whose rows never classify must still produce output —
    every label-passing row lands as ``reason: uncategorized``."""
    from parapet_data import reason_preclassify as module

    input_path = _write_yaml(
        tmp_dir / "source.yaml",
        [{"content": f"row {i} here with text"} for i in range(5)],
    )
    source = _heuristic_source(input_path)
    output_path = tmp_dir / "out" / "all_uncategorized.jsonl"

    monkeypatch.setattr(module, "classify_reason", lambda text: None)

    report = preclassify_reason_source(source, output_path)

    assert report.rows_read == 5
    assert report.rows_written == 5
    assert report.classified == 0
    assert report.uncategorized == 5
    assert report.classified_rate == 0.0
    assert report.keep_rate == 1.0
    assert report.by_reason == {AttackReason.UNCATEGORIZED.value: 5}

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["reason"] == AttackReason.UNCATEGORIZED.value for row in rows)
    assert all(row["reason_confidence"] is None for row in rows)
    assert all(row["reason_signals"] == [] for row in rows)


def test_heuristic_attack_sources_dedupes_shared_source(tmp_dir: Path) -> None:
    path = _write_yaml(tmp_dir / "source.yaml", [{"content": "row"}])
    source = _heuristic_source(path)
    benign = SourceRef(
        name="benign",
        path=path,
        language=Language.EN,
        extractor="col_content",
    )
    spec = MirrorSpec(
        name="preclassify_spec",
        version="0.1.0",
        cells=[
            MirrorCell(
                reason=AttackReason.INSTRUCTION_OVERRIDE,
                attack_sources=[source],
                benign_sources=[benign],
                teaching_goal="instruction",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 1.0},
                length_distribution={LengthBin.SHORT: 1.0},
            ),
            MirrorCell(
                reason=AttackReason.META_PROBE,
                attack_sources=[source],
                benign_sources=[benign],
                teaching_goal="meta",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 1.0},
                length_distribution={LengthBin.SHORT: 1.0},
            ),
        ],
        allow_partial_mirror=True,
        allow_heuristic_mirror_attacks=True,
    )

    assert heuristic_attack_sources(spec) == [source]


def test_preclassify_reason_sources_from_spec_writes_stable_filenames(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from parapet_data import reason_preclassify as module

    input_path = _write_yaml(
        tmp_dir / "source.yaml",
        [{"content": "keep this classified row"}],
    )
    source = _heuristic_source(input_path, name="heuristic/source")
    benign = SourceRef(
        name="benign",
        path=input_path,
        language=Language.EN,
        extractor="col_content",
    )
    spec = MirrorSpec(
        name="preclassify_spec",
        version="0.1.0",
        cells=[
            MirrorCell(
                reason=AttackReason.INSTRUCTION_OVERRIDE,
                attack_sources=[source],
                benign_sources=[benign],
                teaching_goal="instruction",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 1.0},
                length_distribution={LengthBin.SHORT: 1.0},
            )
        ],
        allow_partial_mirror=True,
        allow_heuristic_mirror_attacks=True,
    )

    class _Classification:
        reason = AttackReason.META_PROBE
        confidence = 0.8
        signals = ()

    monkeypatch.setattr(module, "classify_reason", lambda text: _Classification())

    reports = preclassify_reason_sources_from_spec(spec, tmp_dir / "out")

    assert len(reports) == 1
    assert reports[0].output_path.name == preclassified_source_filename(source)
    assert reports[0].output_path.exists()


def test_cmd_preclassify_reasons_writes_report(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import argparse

    from parapet_data import reason_preclassify as module
    from parapet_data.__main__ import cmd_preclassify_reasons

    input_path = _write_yaml(
        tmp_dir / "source.yaml",
        [{"content": "keep this classified row"}],
    )
    source = _heuristic_source(input_path)
    benign = SourceRef(
        name="benign",
        path=input_path,
        language=Language.EN,
        extractor="col_content",
    )
    spec = MirrorSpec(
        name="preclassify_cli_spec",
        version="0.1.0",
        cells=[
            MirrorCell(
                reason=AttackReason.INSTRUCTION_OVERRIDE,
                attack_sources=[source],
                benign_sources=[benign],
                teaching_goal="instruction",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 1.0},
                length_distribution={LengthBin.SHORT: 1.0},
            )
        ],
        allow_partial_mirror=True,
        allow_heuristic_mirror_attacks=True,
    )
    spec_path = tmp_dir / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    class _Classification:
        reason = AttackReason.INSTRUCTION_OVERRIDE
        confidence = 0.7
        signals = ("cli_signal",)

    monkeypatch.setattr(module, "classify_reason", lambda text: _Classification())

    cmd_preclassify_reasons(argparse.Namespace(
        spec=str(spec_path),
        output=str(tmp_dir / "out"),
        base_dir=str(tmp_dir),
        sources=None,
    ))

    report_path = tmp_dir / "out" / "preclassify_reasons_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report[0]["source_name"] == source.name
    assert report[0]["rows_written"] == 1
