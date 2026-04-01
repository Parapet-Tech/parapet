"""Tests for dataset compositor — split, write, manifest."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from parapet_data.filters import content_hash
from parapet_data.ledger import (
    AdjudicationReason,
    Ledger,
    LedgerAction,
    LedgerEntry,
)
from parapet_data.models import (
    AttackReason,
    ApplicabilityScope,
    BackgroundLane,
    DiscussionBenignLane,
    CellFillRecord,
    FormatBin,
    Language,
    LengthBin,
    MirrorCell,
    MirrorSpec,
    SourceRef,
    SourceRoutePolicy,
    SourceMetadata,
)
from parapet_data.sampler import Sample, SamplingResult, sample_spec
from parapet_data.compositor import (
    _collect_source_alias_warnings,
    compose,
    composition_report,
    split_samples,
    write_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir():
    d = Path(tempfile.mkdtemp(prefix="parapet_comp_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_samples(n: int, label: str = "benign", reason: str = "test") -> list[Sample]:
    return [
        Sample(
            content=f"Sample content {label} {i} for testing purposes",
            content_hash=f"hash_{label}_{i:04d}",
            label=label,
            reason=reason,
            source_name="test_source",
            language="EN",
            format_bin="prose",
            length_bin="short",
        )
        for i in range(n)
    ]


def _write_yaml(path: Path, data: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    return path


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------


class TestSplitSamples:
    def test_default_ratios(self) -> None:
        samples = _make_samples(100, "benign") + _make_samples(100, "malicious")
        splits = split_samples(samples, seed=42)
        assert set(splits.keys()) == {"train", "val", "holdout"}
        total = sum(len(s) for s in splits.values())
        assert total == 200

    def test_train_is_largest(self) -> None:
        samples = _make_samples(100, "benign") + _make_samples(100, "malicious")
        splits = split_samples(samples, seed=42)
        assert len(splits["train"]) > len(splits["val"])
        assert len(splits["train"]) > len(splits["holdout"])

    def test_holdout_only_reasons(self) -> None:
        normal = _make_samples(80, "benign", reason="instruction_override")
        holdout_only = _make_samples(20, "malicious", reason="adversarial_suffix")
        all_samples = normal + holdout_only

        splits = split_samples(
            all_samples,
            holdout_only_reasons=["adversarial_suffix"],
            seed=42,
        )
        # All adversarial_suffix samples should be in holdout
        holdout_reasons = {s.reason for s in splits["holdout"]}
        train_reasons = {s.reason for s in splits["train"]}
        val_reasons = {s.reason for s in splits["val"]}

        assert "adversarial_suffix" in holdout_reasons
        assert "adversarial_suffix" not in train_reasons
        assert "adversarial_suffix" not in val_reasons

    def test_deterministic(self) -> None:
        samples = _make_samples(100)
        s1 = split_samples(samples, seed=42)
        s2 = split_samples(samples, seed=42)
        assert [s.content_hash for s in s1["train"]] == [s.content_hash for s in s2["train"]]

    def test_different_seeds_differ(self) -> None:
        samples = _make_samples(100)
        s1 = split_samples(samples, seed=42)
        s2 = split_samples(samples, seed=99)
        h1 = [s.content_hash for s in s1["train"]]
        h2 = [s.content_hash for s in s2["train"]]
        assert h1 != h2

    def test_stratified_false_uses_shuffle_slice(self) -> None:
        samples = _make_samples(100, "benign") + _make_samples(100, "malicious")
        splits = split_samples(samples, seed=42, stratified=False)
        assert len(splits["train"]) + len(splits["val"]) + len(splits["holdout"]) == 200

    def test_stratified_preserves_reason_language_presence(self) -> None:
        en = [
            Sample(
                content=f"EN {i}",
                content_hash=f"en_{i}",
                label="benign",
                reason="instruction_override",
                source_name="s",
                language="EN",
                format_bin="prose",
                length_bin="short",
            )
            for i in range(40)
        ]
        ru = [
            Sample(
                content=f"RU {i}",
                content_hash=f"ru_{i}",
                label="benign",
                reason="instruction_override",
                source_name="s",
                language="RU",
                format_bin="prose",
                length_bin="short",
            )
            for i in range(20)
        ]
        splits = split_samples(en + ru, seed=42, stratified=True)
        train_langs = {s.language for s in splits["train"]}
        holdout_langs = {s.language for s in splits["holdout"]}
        assert "EN" in train_langs and "RU" in train_langs
        assert "EN" in holdout_langs and "RU" in holdout_langs

    def test_small_stratum_gets_val_and_holdout(self) -> None:
        small_stratum = [
            Sample(
                content=f"tiny {i}",
                content_hash=f"tiny_{i}",
                label="benign",
                reason="obfuscation",
                source_name="s",
                language="EN",
                format_bin="prose",
                length_bin="short",
            )
            for i in range(5)
        ]
        splits = split_samples(small_stratum, seed=7, stratified=True)
        assert len(splits["val"]) >= 1
        assert len(splits["holdout"]) >= 1


# ---------------------------------------------------------------------------
# JSONL writing
# ---------------------------------------------------------------------------


class TestWriteJSONL:
    def test_writes_all_samples(self, tmp_dir: Path) -> None:
        samples = _make_samples(10)
        path = tmp_dir / "test.jsonl"
        manifest = write_split(samples, path, fmt="jsonl")
        assert path.exists()
        assert manifest.sample_count == 10
        assert len(manifest.content_hashes) == 10

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 10

    def test_jsonl_is_valid(self, tmp_dir: Path) -> None:
        samples = _make_samples(5)
        path = tmp_dir / "test.jsonl"
        write_split(samples, path, fmt="jsonl")
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            row = json.loads(line)
            assert "content" in row
            assert "label" in row
            assert "reason" in row

    def test_content_hashes_sorted(self, tmp_dir: Path) -> None:
        samples = _make_samples(20)
        path = tmp_dir / "test.jsonl"
        manifest = write_split(samples, path, fmt="jsonl")
        assert manifest.content_hashes == sorted(manifest.content_hashes)


# ---------------------------------------------------------------------------
# Compose (end-to-end)
# ---------------------------------------------------------------------------


class TestCompose:
    def _make_spec_with_files(self, tmp_dir: Path) -> MirrorSpec:
        cells = []
        for reason in AttackReason:
            atk_path = tmp_dir / "sources" / f"{reason.value}_atk.yaml"
            ben_path = tmp_dir / "sources" / f"{reason.value}_ben.yaml"
            _write_yaml(atk_path, [
                {"content": f"Attack text for {reason.value} number {i} testing"}
                for i in range(10)
            ])
            _write_yaml(ben_path, [
                {"content": f"Benign text for {reason.value} number {i} testing"}
                for i in range(10)
            ])
            cells.append(MirrorCell(
                reason=reason,
                attack_sources=[SourceRef(
                    name=f"{reason.value}_atk", path=atk_path,
                    language=Language.EN, extractor="col_content",
                )],
                benign_sources=[SourceRef(
                    name=f"{reason.value}_ben", path=ben_path,
                    language=Language.EN, extractor="col_content",
                )],
                teaching_goal=f"test {reason.value}",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 1.0},
                length_distribution={LengthBin.SHORT: 1.0},
            ))
        return MirrorSpec(
            name="compose_test", version="0.1.0", cells=cells, seed=42,
        )

    def test_produces_all_outputs_jsonl(self, tmp_dir: Path) -> None:
        spec = self._make_spec_with_files(tmp_dir)
        result = sample_spec(spec, base_dir=tmp_dir)
        output_dir = tmp_dir / "output"
        manifest = compose(spec, result, output_dir, base_dir=tmp_dir, fmt="jsonl")

        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "val.jsonl").exists()
        assert (output_dir / "holdout.jsonl").exists()
        assert (output_dir / "curated.jsonl").exists()

    def test_produces_all_outputs_yaml(self, tmp_dir: Path) -> None:
        spec = self._make_spec_with_files(tmp_dir)
        result = sample_spec(spec, base_dir=tmp_dir)
        output_dir = tmp_dir / "output"
        manifest = compose(spec, result, output_dir, base_dir=tmp_dir, fmt="yaml")

        assert (output_dir / "train.yaml").exists()
        assert (output_dir / "val.yaml").exists()
        assert (output_dir / "holdout.yaml").exists()
        assert (output_dir / "curated.yaml").exists()

    def test_manifest_has_correct_counts(self, tmp_dir: Path) -> None:
        spec = self._make_spec_with_files(tmp_dir)
        result = sample_spec(spec, base_dir=tmp_dir)
        output_dir = tmp_dir / "output"
        manifest = compose(spec, result, output_dir, base_dir=tmp_dir)

        split_total = sum(s.sample_count for s in manifest.splits.values())
        assert split_total == manifest.total_samples
        assert manifest.attack_samples + manifest.benign_samples == manifest.total_samples

    def test_manifest_has_semantic_hash(self, tmp_dir: Path) -> None:
        spec = self._make_spec_with_files(tmp_dir)
        result = sample_spec(spec, base_dir=tmp_dir)
        output_dir = tmp_dir / "output"
        manifest = compose(spec, result, output_dir, base_dir=tmp_dir)

        assert manifest.semantic_hash
        assert len(manifest.semantic_hash) == 64

    def test_manifest_has_output_hash(self, tmp_dir: Path) -> None:
        spec = self._make_spec_with_files(tmp_dir)
        result = sample_spec(spec, base_dir=tmp_dir)
        output_dir = tmp_dir / "output"
        manifest = compose(spec, result, output_dir, base_dir=tmp_dir)

        assert manifest.output_hash
        assert len(manifest.output_hash) == 64

    def test_manifest_serializes(self, tmp_dir: Path) -> None:
        spec = self._make_spec_with_files(tmp_dir)
        result = sample_spec(spec, base_dir=tmp_dir)
        output_dir = tmp_dir / "output"
        manifest = compose(spec, result, output_dir, base_dir=tmp_dir)

        # Should serialize without error
        json_str = manifest.model_dump_json(indent=2)
        assert "compose_test" in json_str

    def test_manifest_emits_source_metadata_including_background(self, tmp_dir: Path) -> None:
        spec = self._make_spec_with_files(tmp_dir)
        bg_path = tmp_dir / "sources" / "bg.yaml"
        _write_yaml(bg_path, [{"content": "Background benign text for testing"}])
        spec.background = BackgroundLane(
            budget_fraction=0.15,
            sources=[
                SourceRef(
                    name="bg_source",
                    path=bg_path,
                    language=Language.EN,
                    extractor="col_content",
                    route_policy=SourceRoutePolicy.BACKGROUND,
                )
            ],
        )
        result = sample_spec(spec, base_dir=tmp_dir)
        output_dir = tmp_dir / "output"
        manifest = compose(spec, result, output_dir, base_dir=tmp_dir)

        assert "bg_source" in manifest.source_metadata
        bg_meta = manifest.source_metadata["bg_source"]
        assert bg_meta.path == bg_path
        assert bg_meta.route_policy == SourceRoutePolicy.BACKGROUND

    def test_manifest_emits_source_metadata_including_discussion_benign(
        self, tmp_dir: Path
    ) -> None:
        spec = self._make_spec_with_files(tmp_dir)
        discussion_path = tmp_dir / "sources" / "discussion.yaml"
        _write_yaml(
            discussion_path,
            [{"content": "security writeup quoting ignore previous instructions"}],
        )
        spec.discussion_benign = DiscussionBenignLane(
            budget_fraction=0.05,
            sources=[
                SourceRef(
                    name="discussion_source",
                    path=discussion_path,
                    language=Language.EN,
                    extractor="col_content",
                    route_policy=SourceRoutePolicy.DISCUSSION_BENIGN,
                )
            ],
        )
        result = sample_spec(spec, base_dir=tmp_dir)
        output_dir = tmp_dir / "output"
        manifest = compose(spec, result, output_dir, base_dir=tmp_dir)

        assert "discussion_source" in manifest.source_metadata
        discussion_meta = manifest.source_metadata["discussion_source"]
        assert discussion_meta.path == discussion_path
        assert discussion_meta.route_policy == SourceRoutePolicy.DISCUSSION_BENIGN

    def test_manifest_carries_ledger_counters(self, tmp_dir: Path) -> None:
        spec = self._make_spec_with_files(tmp_dir)
        atk_path = tmp_dir / "sources" / f"{AttackReason.INSTRUCTION_OVERRIDE.value}_atk.yaml"
        _write_yaml(atk_path, [
            {"content": "drop from compose manifest"},
            {"content": "reroute from compose manifest"},
            {"content": "relabel from compose manifest"},
            {"content": "keep from compose manifest"},
        ])
        ledger = Ledger([
            LedgerEntry(
                content_hash=content_hash("drop from compose manifest"),
                source=f"{AttackReason.INSTRUCTION_OVERRIDE.value}_atk",
                action=LedgerAction.DROP,
                adjudication=AdjudicationReason.MISLABEL,
            ),
            LedgerEntry(
                content_hash=content_hash("reroute from compose manifest"),
                source=f"{AttackReason.INSTRUCTION_OVERRIDE.value}_atk",
                action=LedgerAction.REROUTE_REASON,
                adjudication=AdjudicationReason.ROUTING_DEFECT,
                reroute_to=AttackReason.META_PROBE.value,
            ),
            LedgerEntry(
                content_hash=content_hash("relabel from compose manifest"),
                source=f"{AttackReason.INSTRUCTION_OVERRIDE.value}_atk",
                action=LedgerAction.RELABEL_CLASS,
                adjudication=AdjudicationReason.MISLABEL,
                relabel_to="benign",
            ),
        ])

        result = sample_spec(spec, base_dir=tmp_dir, ledger=ledger)
        output_dir = tmp_dir / "output"
        manifest = compose(spec, result, output_dir, base_dir=tmp_dir)

        assert manifest.ledger_dropped == 1
        assert manifest.ledger_quarantined == 0
        assert manifest.ledger_rerouted == 1
        assert manifest.ledger_relabeled == 1


class TestSourceAliasWarnings:
    def test_warns_on_same_path_with_different_route_policies(self) -> None:
        warnings = _collect_source_alias_warnings(
            {
                "bg_alias": SourceMetadata(
                    path=Path("schema/eval/staging/en_shared.yaml"),
                    language=Language.EN,
                    extractor="col_content",
                    route_policy=SourceRoutePolicy.BACKGROUND,
                    applicability_scope=ApplicabilityScope.IN_DOMAIN,
                ),
                "residual_alias": SourceMetadata(
                    path=Path("schema/eval/staging/en_shared.yaml"),
                    language=Language.EN,
                    extractor="col_content",
                    route_policy=SourceRoutePolicy.RESIDUAL,
                    applicability_scope=ApplicabilityScope.IN_DOMAIN,
                ),
            }
        )
        assert len(warnings) == 1
        assert "bg_alias" in warnings[0]
        assert "residual_alias" in warnings[0]
        assert "background" in warnings[0]
        assert "residual" in warnings[0]

    def test_ignores_same_path_when_route_policy_matches(self) -> None:
        warnings = _collect_source_alias_warnings(
            {
                "mirror_a": SourceMetadata(
                    path=Path("schema/eval/staging/en_shared.yaml"),
                    language=Language.EN,
                    extractor="col_content",
                    route_policy=SourceRoutePolicy.MIRROR,
                ),
                "mirror_b": SourceMetadata(
                    path=Path("schema/eval/staging/en_shared.yaml"),
                    language=Language.EN,
                    extractor="col_content",
                    route_policy=SourceRoutePolicy.MIRROR,
                ),
            }
        )
        assert warnings == []


# ---------------------------------------------------------------------------
# Composition report
# ---------------------------------------------------------------------------


class TestCompositionReport:
    def test_empty_samples(self) -> None:
        report = composition_report([])
        assert report["total"] == 0

    def test_has_all_dimensions(self) -> None:
        samples = _make_samples(10, "benign") + _make_samples(5, "malicious")
        report = composition_report(samples)
        assert report["total"] == 15
        assert "by_label" in report
        assert "by_reason" in report
        assert "by_language" in report
        assert "by_format" in report
        assert "by_length" in report
        assert "by_source" in report

    def test_label_counts(self) -> None:
        samples = _make_samples(10, "benign") + _make_samples(5, "malicious")
        report = composition_report(samples)
        labels = {r["name"]: r["count"] for r in report["by_label"]}
        assert labels["benign"] == 10
        assert labels["malicious"] == 5
