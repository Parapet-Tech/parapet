"""Tests for mirror taxonomy types.

These tests validate the data contract — that MirrorSpec rejects
structurally unsound experiments before any data is sampled.
"""

from pathlib import Path

import pytest

from parapet_data.models import (
    ApplicabilityScope,
    AttackReason,
    BackfillPolicy,
    BackgroundLane,
    DiscussionBenignLane,
    CellFillRecord,
    FormatBin,
    Language,
    LanguageQuota,
    LanguageQuotaMode,
    LengthBin,
    MIRROR_ATTACK_REASONS,
    MirrorCell,
    MirrorSpec,
    ReasonProvenance,
    SourceRef,
    SourceGroundingMode,
    SourceMetadata,
    SourceRoutePolicy,
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


def _strict_attack_source(name: str = "atk", lang: Language = Language.EN) -> SourceRef:
    return SourceRef(
        name=name,
        path=DUMMY_PATH,
        language=lang,
        extractor="passthrough",
        grounding_mode=SourceGroundingMode.REASON_GROUNDED,
        route_policy=SourceRoutePolicy.MIRROR,
        reason_provenance=ReasonProvenance.SOURCE_LABEL,
        applicability_scope=ApplicabilityScope.IN_DOMAIN,
    )


def _mirror_benign_source(name: str = "ben", lang: Language = Language.EN) -> SourceRef:
    return SourceRef(
        name=name,
        path=DUMMY_PATH,
        language=lang,
        extractor="passthrough",
        route_policy=SourceRoutePolicy.MIRROR,
    )


def _cell(
    reason: str | AttackReason = AttackReason.INSTRUCTION_OVERRIDE,
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
    """One cell per calibrated mirror reason."""
    return [_cell(reason=r) for r in AttackReason if r in MIRROR_ATTACK_REASONS]


def _strict_cell(reason: AttackReason = AttackReason.INSTRUCTION_OVERRIDE) -> MirrorCell:
    return MirrorCell(
        reason=reason,
        attack_sources=[_strict_attack_source()],
        benign_sources=[_mirror_benign_source()],
        teaching_goal="test goal",
        languages=[Language.EN],
        format_distribution={FormatBin.PROSE: 0.7, FormatBin.STRUCTURED: 0.2, FormatBin.CODE: 0.1},
        length_distribution={LengthBin.SHORT: 0.25, LengthBin.MEDIUM: 0.5, LengthBin.LONG: 0.25},
    )


def _strict_full_cells() -> list[MirrorCell]:
    return [_strict_cell(reason=r) for r in AttackReason]


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
        expected = [
            reason.value for reason in AttackReason if reason in MIRROR_ATTACK_REASONS
        ]
        assert len(spec.cells) == len(MIRROR_ATTACK_REASONS)
        assert spec.mirror_reason_categories == expected

    def test_rejects_incomplete_mirror(self) -> None:
        mirror_reasons = [
            reason for reason in AttackReason if reason in MIRROR_ATTACK_REASONS
        ]
        incomplete = [_cell(reason=r) for r in mirror_reasons[:3]]
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

    def test_reason_categories_omitted_for_legacy_default(self) -> None:
        dumped = MirrorSpec(
            name="legacy_default",
            version="0.1.0",
            cells=_full_cells(),
        ).model_dump()
        assert "reason_categories" not in dumped

    def test_infers_custom_reason_categories_from_cells(self) -> None:
        spec = MirrorSpec(
            name="residual",
            version="0.1.0",
            cells=[
                _cell(reason="use_vs_mention"),
                _cell(reason="multilingual_gap"),
                _cell(reason="semantic_paraphrase"),
            ],
        )
        assert spec.mirror_reason_categories == [
            "use_vs_mention",
            "multilingual_gap",
            "semantic_paraphrase",
        ]

    def test_explicit_reason_categories_allow_canonical_subset(self) -> None:
        spec = MirrorSpec(
            name="subset",
            version="0.1.0",
            cells=[_cell(reason=AttackReason.INSTRUCTION_OVERRIDE)],
            reason_categories=[AttackReason.INSTRUCTION_OVERRIDE],
        )
        assert spec.mirror_reason_categories == ["instruction_override"]

    def test_entity_categories_alias_maps_to_reason_categories(self) -> None:
        spec = MirrorSpec.model_validate(
            {
                "name": "alias",
                "version": "0.1.0",
                "cells": [
                    _cell(reason="use_vs_mention").model_dump(mode="json"),
                ],
                "entity_categories": ["use_vs_mention"],
            }
        )
        assert spec.reason_categories == ["use_vs_mention"]
        assert spec.mirror_reason_categories == ["use_vs_mention"]

    def test_rejects_cell_reason_outside_declared_reason_categories(self) -> None:
        with pytest.raises(ValueError, match="outside reason_categories"):
            MirrorSpec(
                name="bad_categories",
                version="0.1.0",
                cells=[_cell(reason="semantic_paraphrase")],
                reason_categories=["use_vs_mention"],
            )

    def test_rejects_holdout_reason_outside_declared_reason_categories(self) -> None:
        with pytest.raises(ValueError, match="holdout_only_reasons reference undeclared categories"):
            MirrorSpec(
                name="bad_holdout",
                version="0.1.0",
                cells=[_cell(reason="semantic_paraphrase")],
                reason_categories=["semantic_paraphrase"],
                holdout_only_reasons=["multilingual_gap"],
            )

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

    def test_rejects_benign_side_lane_budget_exhaustion(self) -> None:
        with pytest.raises(ValueError, match="background \\+ discussion_benign"):
            MirrorSpec(
                name="lane_overflow",
                version="0.1.0",
                cells=_full_cells(),
                background=BackgroundLane(
                    budget_fraction=0.50,
                    sources=[_source("bg")],
                ),
                discussion_benign=DiscussionBenignLane(
                    budget_fraction=0.50,
                    sources=[_source("discussion")],
                ),
            )

    def test_strict_contract_flag_omitted_when_unset(self) -> None:
        dumped = MirrorSpec(name="default", version="0.1.0", cells=_full_cells()).model_dump()
        assert "enforce_source_contracts" not in dumped

    def test_heuristic_mirror_flag_omitted_when_unset(self) -> None:
        dumped = MirrorSpec(name="default", version="0.1.0", cells=_full_cells()).model_dump()
        assert "allow_heuristic_mirror_attacks" not in dumped

    def test_strict_source_contracts_accept_clean_lane_routing(self) -> None:
        supplement = Supplement(
            name="residual_gap",
            weakness="broad residual attacks",
            attack_sources=[
                SourceRef(
                    name="supp_atk",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    route_policy=SourceRoutePolicy.RESIDUAL,
                )
            ],
            benign_sources=[
                SourceRef(
                    name="supp_ben",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    route_policy=SourceRoutePolicy.RESIDUAL,
                )
            ],
            max_samples=10,
        )
        background = BackgroundLane(
            budget_fraction=0.15,
            sources=[
                SourceRef(
                    name="bg",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    route_policy=SourceRoutePolicy.BACKGROUND,
                )
            ],
        )
        discussion_benign = DiscussionBenignLane(
            budget_fraction=0.05,
            sources=[
                SourceRef(
                    name="discussion",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    route_policy=SourceRoutePolicy.DISCUSSION_BENIGN,
                )
            ],
        )
        spec = MirrorSpec(
            name="strict_ok",
            version="0.1.0",
            cells=_strict_full_cells(),
            supplements=[supplement],
            background=background,
            discussion_benign=discussion_benign,
            enforce_source_contracts=True,
        )
        assert spec.enforce_source_contracts is True

    def test_strict_source_contracts_reject_pooled_mirror_attack(self) -> None:
        dirty_cells = _strict_full_cells()
        dirty_cells[0] = MirrorCell(
            reason=dirty_cells[0].reason,
            attack_sources=[
                SourceRef(
                    name="pooled_atk",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    grounding_mode=SourceGroundingMode.POOLED,
                    route_policy=SourceRoutePolicy.MIRROR,
                    reason_provenance=ReasonProvenance.HEURISTIC,
                    applicability_scope=ApplicabilityScope.MIXED,
                )
            ],
            benign_sources=[_mirror_benign_source()],
            teaching_goal=dirty_cells[0].teaching_goal,
            languages=dirty_cells[0].languages,
            format_distribution=dirty_cells[0].format_distribution,
            length_distribution=dirty_cells[0].length_distribution,
        )
        with pytest.raises(ValueError, match="grounding_mode=reason_grounded") as excinfo:
            MirrorSpec(
                name="strict_bad_attack",
                version="0.1.0",
                cells=dirty_cells,
                enforce_source_contracts=True,
            )
        message = str(excinfo.value)
        assert "non-heuristic reason_provenance" in message
        assert "applicability_scope=in_domain" in message

    def test_strict_source_contracts_reject_heuristic_mirror_attack(self) -> None:
        dirty_cells = _strict_full_cells()
        dirty_cells[0] = MirrorCell(
            reason=dirty_cells[0].reason,
            attack_sources=[
                SourceRef(
                    name="heuristic_atk",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    grounding_mode=SourceGroundingMode.REASON_GROUNDED,
                    route_policy=SourceRoutePolicy.MIRROR,
                    reason_provenance=ReasonProvenance.HEURISTIC,
                    applicability_scope=ApplicabilityScope.IN_DOMAIN,
                )
            ],
            benign_sources=[_mirror_benign_source()],
            teaching_goal=dirty_cells[0].teaching_goal,
            languages=dirty_cells[0].languages,
            format_distribution=dirty_cells[0].format_distribution,
            length_distribution=dirty_cells[0].length_distribution,
        )
        with pytest.raises(ValueError, match="non-heuristic reason_provenance"):
            MirrorSpec(
                name="strict_bad_provenance",
                version="0.1.0",
                cells=dirty_cells,
                enforce_source_contracts=True,
            )

    def test_strict_source_contracts_allow_heuristic_mirror_attack_with_opt_in(self) -> None:
        dirty_cells = _strict_full_cells()
        dirty_cells[0] = MirrorCell(
            reason=dirty_cells[0].reason,
            attack_sources=[
                SourceRef(
                    name="heuristic_atk",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    grounding_mode=SourceGroundingMode.REASON_GROUNDED,
                    route_policy=SourceRoutePolicy.MIRROR,
                    reason_provenance=ReasonProvenance.HEURISTIC,
                    applicability_scope=ApplicabilityScope.IN_DOMAIN,
                )
            ],
            benign_sources=[_mirror_benign_source()],
            teaching_goal=dirty_cells[0].teaching_goal,
            languages=dirty_cells[0].languages,
            format_distribution=dirty_cells[0].format_distribution,
            length_distribution=dirty_cells[0].length_distribution,
        )
        spec = MirrorSpec(
            name="strict_heuristic_ok",
            version="0.1.0",
            cells=dirty_cells,
            enforce_source_contracts=True,
            allow_heuristic_mirror_attacks=True,
        )
        assert spec.allow_heuristic_mirror_attacks is True

    def test_strict_source_contracts_allow_heuristic_staged_mirror_attack_with_opt_in(self) -> None:
        dirty_cells = _strict_full_cells()
        dirty_cells[0] = MirrorCell(
            reason=dirty_cells[0].reason,
            attack_sources=[
                SourceRef(
                    name="heuristic_staged_atk",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    grounding_mode=SourceGroundingMode.REASON_GROUNDED,
                    route_policy=SourceRoutePolicy.MIRROR,
                    reason_provenance=ReasonProvenance.HEURISTIC_STAGED,
                    applicability_scope=ApplicabilityScope.IN_DOMAIN,
                )
            ],
            benign_sources=[_mirror_benign_source()],
            teaching_goal=dirty_cells[0].teaching_goal,
            languages=dirty_cells[0].languages,
            format_distribution=dirty_cells[0].format_distribution,
            length_distribution=dirty_cells[0].length_distribution,
        )
        spec = MirrorSpec(
            name="strict_heuristic_staged_ok",
            version="0.1.0",
            cells=dirty_cells,
            enforce_source_contracts=True,
            allow_heuristic_mirror_attacks=True,
        )
        assert spec.allow_heuristic_mirror_attacks is True

    def test_strict_source_contracts_reject_heuristic_staged_without_opt_in(self) -> None:
        dirty_cells = _strict_full_cells()
        dirty_cells[0] = MirrorCell(
            reason=dirty_cells[0].reason,
            attack_sources=[
                SourceRef(
                    name="heuristic_staged_atk",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    grounding_mode=SourceGroundingMode.REASON_GROUNDED,
                    route_policy=SourceRoutePolicy.MIRROR,
                    reason_provenance=ReasonProvenance.HEURISTIC_STAGED,
                    applicability_scope=ApplicabilityScope.IN_DOMAIN,
                )
            ],
            benign_sources=[_mirror_benign_source()],
            teaching_goal=dirty_cells[0].teaching_goal,
            languages=dirty_cells[0].languages,
            format_distribution=dirty_cells[0].format_distribution,
            length_distribution=dirty_cells[0].length_distribution,
        )
        with pytest.raises(ValueError, match="non-heuristic reason_provenance"):
            MirrorSpec(
                name="strict_heuristic_staged_bad",
                version="0.1.0",
                cells=dirty_cells,
                enforce_source_contracts=True,
            )

    def test_strict_source_contracts_reject_mixed_scope_mirror_attack(self) -> None:
        dirty_cells = _strict_full_cells()
        dirty_cells[0] = MirrorCell(
            reason=dirty_cells[0].reason,
            attack_sources=[
                SourceRef(
                    name="mixed_scope_atk",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    grounding_mode=SourceGroundingMode.REASON_GROUNDED,
                    route_policy=SourceRoutePolicy.MIRROR,
                    reason_provenance=ReasonProvenance.SOURCE_LABEL,
                    applicability_scope=ApplicabilityScope.MIXED,
                )
            ],
            benign_sources=[_mirror_benign_source()],
            teaching_goal=dirty_cells[0].teaching_goal,
            languages=dirty_cells[0].languages,
            format_distribution=dirty_cells[0].format_distribution,
            length_distribution=dirty_cells[0].length_distribution,
        )
        with pytest.raises(ValueError, match="applicability_scope=in_domain"):
            MirrorSpec(
                name="strict_bad_scope",
                version="0.1.0",
                cells=dirty_cells,
                enforce_source_contracts=True,
            )

    def test_strict_source_contracts_reject_non_mirror_benign_source(self) -> None:
        dirty_cells = _strict_full_cells()
        dirty_cells[0] = MirrorCell(
            reason=dirty_cells[0].reason,
            attack_sources=[_strict_attack_source()],
            benign_sources=[
                SourceRef(
                    name="bg_misrouted",
                    path=DUMMY_PATH,
                    language=Language.EN,
                    extractor="passthrough",
                    route_policy=SourceRoutePolicy.BACKGROUND,
                )
            ],
            teaching_goal=dirty_cells[0].teaching_goal,
            languages=dirty_cells[0].languages,
            format_distribution=dirty_cells[0].format_distribution,
            length_distribution=dirty_cells[0].length_distribution,
        )
        with pytest.raises(ValueError, match="route_policy=mirror"):
            MirrorSpec(
                name="strict_bad_benign",
                version="0.1.0",
                cells=dirty_cells,
                enforce_source_contracts=True,
            )

    def test_strict_source_contracts_reject_non_background_background_source(self) -> None:
        with pytest.raises(ValueError, match="route_policy=background"):
            MirrorSpec(
                name="strict_bad_background",
                version="0.1.0",
                cells=_strict_full_cells(),
                background=BackgroundLane(
                    budget_fraction=0.15,
                    sources=[
                        SourceRef(
                            name="wrong_lane",
                            path=DUMMY_PATH,
                            language=Language.EN,
                            extractor="passthrough",
                            route_policy=SourceRoutePolicy.MIRROR,
                        )
                    ],
                ),
                enforce_source_contracts=True,
            )

    def test_strict_source_contracts_reject_non_discussion_discussion_source(self) -> None:
        with pytest.raises(ValueError, match="route_policy=discussion_benign"):
            MirrorSpec(
                name="strict_bad_discussion",
                version="0.1.0",
                cells=_strict_full_cells(),
                discussion_benign=DiscussionBenignLane(
                    budget_fraction=0.05,
                    sources=[
                        SourceRef(
                            name="wrong_discussion_lane",
                            path=DUMMY_PATH,
                            language=Language.EN,
                            extractor="passthrough",
                            route_policy=SourceRoutePolicy.BACKGROUND,
                        )
                    ],
                ),
                enforce_source_contracts=True,
            )

    def test_strict_source_contracts_reject_non_residual_supplement_source(self) -> None:
        with pytest.raises(ValueError, match="route_policy=residual"):
            MirrorSpec(
                name="strict_bad_supplement",
                version="0.1.0",
                cells=_strict_full_cells(),
                supplements=[
                    Supplement(
                        name="wrong_lane_supp",
                        weakness="bad routing",
                        attack_sources=[
                            SourceRef(
                                name="supp_atk_wrong_lane",
                                path=DUMMY_PATH,
                                language=Language.EN,
                                extractor="passthrough",
                                route_policy=SourceRoutePolicy.MIRROR,
                            )
                        ],
                        benign_sources=[
                            SourceRef(
                                name="supp_ben",
                                path=DUMMY_PATH,
                                language=Language.EN,
                                extractor="passthrough",
                                route_policy=SourceRoutePolicy.RESIDUAL,
                            )
                        ],
                        max_samples=5,
                    )
                ],
                enforce_source_contracts=True,
            )


class TestSourceMetadata:
    def test_source_ref_accepts_lane_metadata(self) -> None:
        src = SourceRef(
            name="grounded_source",
            path=DUMMY_PATH,
            language=Language.EN,
            extractor="passthrough",
            grounding_mode=SourceGroundingMode.REASON_GROUNDED,
            route_policy=SourceRoutePolicy.MIRROR,
            reason_provenance=ReasonProvenance.SOURCE_LABEL,
            applicability_scope=ApplicabilityScope.IN_DOMAIN,
        )
        assert src.grounding_mode == SourceGroundingMode.REASON_GROUNDED
        assert src.route_policy == SourceRoutePolicy.MIRROR
        assert src.reason_provenance == ReasonProvenance.SOURCE_LABEL
        assert src.applicability_scope == ApplicabilityScope.IN_DOMAIN

    def test_source_ref_omits_new_fields_when_unset(self) -> None:
        dumped = _source().model_dump()
        assert "grounding_mode" not in dumped
        assert "route_policy" not in dumped
        assert "reason_provenance" not in dumped
        assert "applicability_scope" not in dumped

    def test_source_metadata_projects_from_source_ref(self) -> None:
        src = SourceRef(
            name="pooled_source",
            path=DUMMY_PATH,
            language=Language.RU,
            extractor="col_content",
            grounding_mode=SourceGroundingMode.POOLED,
            route_policy=SourceRoutePolicy.RESIDUAL,
            reason_provenance=ReasonProvenance.HEURISTIC,
            applicability_scope=ApplicabilityScope.MIXED,
        )
        meta = SourceMetadata.from_source_ref(src)
        assert meta.path == src.path
        assert meta.language == Language.RU
        assert meta.route_policy == SourceRoutePolicy.RESIDUAL
        assert meta.reason_provenance == ReasonProvenance.HEURISTIC


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
