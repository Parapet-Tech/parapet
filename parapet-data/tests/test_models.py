"""Tests for mirror taxonomy types.

These tests validate the data contract — that MirrorSpec rejects
structurally unsound experiments before any data is sampled.
"""

from pathlib import Path

import pytest

from parapet_data.models import (
    AttackReason,
    BackfillPolicy,
    CellFillRecord,
    FormatBin,
    Language,
    LanguageQuota,
    LanguageQuotaMode,
    LengthBin,
    MirrorCell,
    MirrorSpec,
    SourceRef,
    Supplement,
    compute_semantic_hash,
    compute_source_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DUMMY_PATH = Path("schema/eval/malicious/attacks.yaml")


def _source(name: str = "test", lang: Language = Language.EN) -> SourceRef:
    return SourceRef(
        name=name,
        path=DUMMY_PATH,
        language=lang,
        extractor="passthrough",
    )


def _cell(
    reason: AttackReason = AttackReason.INSTRUCTION_OVERRIDE,
    languages: list[Language] | None = None,
) -> MirrorCell:
    return MirrorCell(
        reason=reason,
        attack_sources=[_source("atk")],
        benign_sources=[_source("ben")],
        teaching_goal="test goal",
        languages=languages or [Language.EN],
        format_distribution={FormatBin.PROSE: 0.7, FormatBin.STRUCTURED: 0.2, FormatBin.CODE: 0.1},
        length_distribution={LengthBin.SHORT: 0.25, LengthBin.MEDIUM: 0.5, LengthBin.LONG: 0.25},
    )


def _full_cells() -> list[MirrorCell]:
    """One cell per AttackReason — complete mirror."""
    return [_cell(reason=r) for r in AttackReason]


# ---------------------------------------------------------------------------
# MirrorCell tests
# ---------------------------------------------------------------------------


class TestMirrorCell:
    def test_valid_cell(self) -> None:
        cell = _cell()
        assert cell.reason == AttackReason.INSTRUCTION_OVERRIDE
        assert len(cell.attack_sources) == 1
        assert len(cell.benign_sources) == 1

    def test_rejects_empty_attack_sources(self) -> None:
        with pytest.raises(ValueError, match="no attack sources"):
            MirrorCell(
                reason=AttackReason.INSTRUCTION_OVERRIDE,
                attack_sources=[],
                benign_sources=[_source()],
                teaching_goal="test",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 1.0},
                length_distribution={LengthBin.MEDIUM: 1.0},
            )

    def test_rejects_empty_benign_sources(self) -> None:
        with pytest.raises(ValueError, match="no benign sources"):
            MirrorCell(
                reason=AttackReason.META_PROBE,
                attack_sources=[_source()],
                benign_sources=[],
                teaching_goal="test",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 1.0},
                length_distribution={LengthBin.MEDIUM: 1.0},
            )

    def test_rejects_format_distribution_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError, match="format_distribution sums to"):
            MirrorCell(
                reason=AttackReason.OBFUSCATION,
                attack_sources=[_source()],
                benign_sources=[_source()],
                teaching_goal="test",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 0.5, FormatBin.CODE: 0.1},
                length_distribution={LengthBin.MEDIUM: 1.0},
            )

    def test_rejects_length_distribution_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError, match="length_distribution sums to"):
            MirrorCell(
                reason=AttackReason.OBFUSCATION,
                attack_sources=[_source()],
                benign_sources=[_source()],
                teaching_goal="test",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 1.0},
                length_distribution={LengthBin.SHORT: 0.1, LengthBin.LONG: 0.1},
            )

    def test_rejects_negative_format_distribution(self) -> None:
        with pytest.raises(ValueError, match="contains negative values"):
            MirrorCell(
                reason=AttackReason.OBFUSCATION,
                attack_sources=[_source()],
                benign_sources=[_source()],
                teaching_goal="test",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 1.5, FormatBin.CODE: -0.5},
                length_distribution={LengthBin.MEDIUM: 1.0},
            )

    def test_rejects_negative_length_distribution(self) -> None:
        with pytest.raises(ValueError, match="contains negative values"):
            MirrorCell(
                reason=AttackReason.OBFUSCATION,
                attack_sources=[_source()],
                benign_sources=[_source()],
                teaching_goal="test",
                languages=[Language.EN],
                format_distribution={FormatBin.PROSE: 1.0},
                length_distribution={LengthBin.SHORT: 1.5, LengthBin.LONG: -0.5},
            )

    def test_rejects_empty_languages(self) -> None:
        with pytest.raises(ValueError, match="languages must not be empty"):
            MirrorCell(
                reason=AttackReason.OBFUSCATION,
                attack_sources=[_source()],
                benign_sources=[_source()],
                teaching_goal="test",
                languages=[],
                format_distribution={FormatBin.PROSE: 1.0},
                length_distribution={LengthBin.MEDIUM: 1.0},
            )

    def test_cell_id_is_deterministic(self) -> None:
        cell_a = _cell(languages=[Language.ZH, Language.EN, Language.RU])
        cell_b = _cell(languages=[Language.RU, Language.ZH, Language.EN])
        assert cell_a.cell_id == cell_b.cell_id
        assert cell_a.cell_id == "instruction_override__EN,RU,ZH"

    def test_cell_id_distinguishes_language_set(self) -> None:
        en_cell = _cell(languages=[Language.EN])
        ru_cell = _cell(languages=[Language.RU])
        assert en_cell.cell_id != ru_cell.cell_id


# ---------------------------------------------------------------------------
# MirrorSpec tests
# ---------------------------------------------------------------------------


class TestMirrorSpec:
    def test_valid_full_spec(self) -> None:
        spec = MirrorSpec(
            name="test_experiment",
            version="0.1.0",
            cells=_full_cells(),
        )
        assert len(spec.cells) == len(AttackReason)

    def test_rejects_incomplete_mirror(self) -> None:
        incomplete = [_cell(reason=r) for r in list(AttackReason)[:3]]
        with pytest.raises(ValueError, match="Mirror incomplete"):
            MirrorSpec(
                name="incomplete",
                version="0.1.0",
                cells=incomplete,
            )

    def test_allows_partial_mirror_for_ablation(self) -> None:
        partial = [_cell(reason=AttackReason.INSTRUCTION_OVERRIDE)]
        spec = MirrorSpec(
            name="ablation",
            version="0.1.0",
            cells=partial,
            allow_partial_mirror=True,
            holdout_only_reasons=[AttackReason.ROLEPLAY_JAILBREAK],
        )
        assert len(spec.cells) == 1

    def test_rejects_supplement_ratio_without_supplements(self) -> None:
        with pytest.raises(ValueError, match="supplement_ratio > 0 but no supplements"):
            MirrorSpec(
                name="bad",
                version="0.1.0",
                cells=_full_cells(),
                supplement_ratio=0.1,
            )

    def test_rejects_supplement_ratio_exceeding_cap(self) -> None:
        supplement = Supplement(
            name="small",
            weakness="test",
            attack_sources=[_source()],
            benign_sources=[_source()],
            max_samples=10,
        )
        with pytest.raises(ValueError, match="supplements cap at"):
            MirrorSpec(
                name="bad",
                version="0.1.0",
                cells=_full_cells(),
                supplements=[supplement],
                supplement_ratio=0.5,
                total_target=1000,
            )

    def test_supplement_ratio_within_cap(self) -> None:
        supplement = Supplement(
            name="large",
            weakness="test",
            attack_sources=[_source()],
            benign_sources=[_source()],
            max_samples=500,
        )
        spec = MirrorSpec(
            name="ok",
            version="0.1.0",
            cells=_full_cells(),
            supplements=[supplement],
            supplement_ratio=0.1,
            total_target=1000,
        )
        assert spec.supplement_ratio == 0.1

    def test_rejects_supplement_ratio_out_of_bounds(self) -> None:
        with pytest.raises(ValueError):
            MirrorSpec(
                name="bad",
                version="0.1.0",
                cells=_full_cells(),
                supplement_ratio=1.5,
            )

    def test_spec_hash_deterministic(self) -> None:
        spec = MirrorSpec(name="det", version="0.1.0", cells=_full_cells())
        assert spec.spec_hash() == spec.spec_hash()

    def test_spec_hash_changes_with_content(self) -> None:
        spec_a = MirrorSpec(name="a", version="0.1.0", cells=_full_cells())
        spec_b = MirrorSpec(name="b", version="0.1.0", cells=_full_cells())
        assert spec_a.spec_hash() != spec_b.spec_hash()

    def test_serialization_roundtrip(self) -> None:
        spec = MirrorSpec(name="roundtrip", version="0.1.0", cells=_full_cells())
        json_str = spec.model_dump_json()
        restored = MirrorSpec.model_validate_json(json_str)
        assert restored.name == spec.name
        assert len(restored.cells) == len(spec.cells)
        assert restored.spec_hash() == spec.spec_hash()

    def test_language_quota_profile_rejects_bad_sum(self) -> None:
        with pytest.raises(ValueError, match="Language profile sums to"):
            LanguageQuota(
                mode=LanguageQuotaMode.BEST_EFFORT,
                profile={Language.EN: 0.8, Language.RU: 0.1},
            )

    def test_language_quota_profile_accepts_valid_sum(self) -> None:
        quota = LanguageQuota(
            mode=LanguageQuotaMode.BEST_EFFORT,
            profile={
                Language.EN: 0.75,
                Language.RU: 0.10,
                Language.ZH: 0.08,
                Language.AR: 0.07,
            },
        )
        spec = MirrorSpec(
            name="quota_ok",
            version="0.1.0",
            cells=_full_cells(),
            language_quota=quota,
        )
        assert spec.language_quota is not None


# ---------------------------------------------------------------------------
# Semantic hash tests
# ---------------------------------------------------------------------------


class TestSemanticHash:
    def test_order_independent(self) -> None:
        fills = {
            "instruction_override": CellFillRecord(target=100, actual=100, backfilled=0),
            "meta_probe": CellFillRecord(target=50, actual=45, backfilled=5),
        }
        hashes_a = ["aaa", "bbb", "ccc"]
        hashes_b = ["ccc", "aaa", "bbb"]  # different order, same content
        assert compute_semantic_hash(hashes_a, fills) == compute_semantic_hash(hashes_b, fills)

    def test_content_sensitive(self) -> None:
        fills = {
            "instruction_override": CellFillRecord(target=100, actual=100, backfilled=0),
        }
        hash_a = compute_semantic_hash(["aaa", "bbb"], fills)
        hash_b = compute_semantic_hash(["aaa", "ccc"], fills)
        assert hash_a != hash_b

    def test_fill_sensitive(self) -> None:
        hashes = ["aaa", "bbb"]
        fills_a = {"instruction_override": CellFillRecord(target=100, actual=100, backfilled=0)}
        fills_b = {"instruction_override": CellFillRecord(target=100, actual=95, backfilled=5)}
        assert compute_semantic_hash(hashes, fills_a) != compute_semantic_hash(hashes, fills_b)

    def test_cross_package_compatibility(self) -> None:
        """Semantic hash must match whether fills are CellFillRecord or plain dict."""
        hashes = ["aaa", "bbb"]
        fills_record = {
            "instruction_override": CellFillRecord(target=100, actual=98, backfilled=2),
        }
        fills_dict = {
            "instruction_override": {"target": 100, "actual": 98, "backfilled": 2},
        }
        assert compute_semantic_hash(hashes, fills_record) == compute_semantic_hash(
            hashes, fills_dict
        )


class TestCellFillRecord:
    def test_new_fields_have_defaults(self) -> None:
        fill = CellFillRecord(target=10, actual=9, backfilled=1)
        assert fill.by_format == {}
        assert fill.by_length == {}
        assert fill.by_language == {}
        assert fill.degraded is False
        assert fill.degraded_mode is None


# ---------------------------------------------------------------------------
# Source hash tests
# ---------------------------------------------------------------------------


class TestSourceHash:
    @pytest.fixture(autouse=True)
    def _make_tmp(self) -> None:
        import shutil
        import tempfile

        self._tmpdir = Path(tempfile.mkdtemp(prefix="parapet_test_"))
        yield  # type: ignore[misc]
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_hashes_file(self) -> None:
        f = self._tmpdir / "data.txt"
        f.write_text("hello", encoding="utf-8")
        src = SourceRef(name="test", path=f, language=Language.EN, extractor="passthrough")
        h = compute_source_hash(src)
        assert isinstance(h, str) and len(h) == 64

    def test_hashes_directory(self) -> None:
        d = self._tmpdir / "dataset"
        d.mkdir()
        (d / "a.txt").write_text("aaa", encoding="utf-8")
        (d / "b.txt").write_text("bbb", encoding="utf-8")
        src = SourceRef(name="test", path=d, language=Language.EN, extractor="passthrough")
        h = compute_source_hash(src)
        assert isinstance(h, str) and len(h) == 64

    def test_relative_path_resolved_with_base_dir(self) -> None:
        f = self._tmpdir / "data.txt"
        f.write_text("content", encoding="utf-8")
        src = SourceRef(
            name="test", path=Path("data.txt"), language=Language.EN, extractor="passthrough"
        )
        h = compute_source_hash(src, base_dir=self._tmpdir)
        assert isinstance(h, str) and len(h) == 64

    def test_deterministic(self) -> None:
        f = self._tmpdir / "data.txt"
        f.write_text("stable", encoding="utf-8")
        src = SourceRef(name="test", path=f, language=Language.EN, extractor="passthrough")
        assert compute_source_hash(src) == compute_source_hash(src)
