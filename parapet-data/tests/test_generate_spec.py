"""Tests for the compact spec generator.

Validates that generate_spec.py correctly expands compact specs into
full MirrorSpec YAMLs that parapet-data can consume.
"""

import copy
from pathlib import Path

import pytest
import yaml

# generate_spec.py lives alongside the package, import its functions directly
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_spec import (
    ALL_REASONS,
    LANGUAGES,
    build_cell,
    build_supplement,
    expand_spec,
    get_reason_categories,
    make_source_ref,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_compact():
    """Smallest valid compact spec — 8 cells, minimal sources."""
    base_atk = [{"name": "en_attacks", "path": "attacks.yaml", "language": "EN"}]
    base_ben = [{"name": "en_benign", "path": "benign.yaml", "language": "EN"}]

    cells = {}
    for reason in ALL_REASONS:
        cells[reason] = {
            "teaching_goal": f"Test goal for {reason}",
            "format": {"prose": 0.7, "structured": 0.2, "code": 0.1},
            "length": {"short": 0.5, "medium": 0.3, "long": 0.2},
        }

    return {
        "name": "test_spec",
        "version": "1.0.0",
        "seed": 42,
        "ratio": 1.0,
        "total_target": 1000,
        "backfill": {"strategy": "ngram_safe", "log_gaps": True},
        "language_quota": {
            "mode": "best_effort",
            "profile": {"EN": 0.75, "RU": 0.10, "ZH": 0.08, "AR": 0.07},
        },
        "base_attack_sources": base_atk,
        "base_benign_sources": base_ben,
        "cells": cells,
    }


def _v3_compact():
    """Load the real v3 compact spec."""
    path = Path(__file__).resolve().parent.parent / "mirror_v3.compact.yaml"
    if not path.exists():
        pytest.skip("mirror_v3.compact.yaml not found")
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# make_source_ref
# ---------------------------------------------------------------------------


class TestMakeSourceRef:
    def test_basic_fields(self):
        ref = make_source_ref("test", "path.yaml", "EN")
        assert ref["name"] == "test"
        assert ref["path"] == "path.yaml"
        assert ref["language"] == "EN"
        assert ref["extractor"] == "col_content"
        assert "label_filter" not in ref

    def test_with_label_filter(self):
        filt = {"column": "label", "allowed": ["malicious"]}
        ref = make_source_ref("test", "path.yaml", "EN", label_filter=filt)
        assert ref["label_filter"] == filt

    def test_custom_extractor(self):
        ref = make_source_ref("test", "path.yaml", "EN", extractor="wildchat")
        assert ref["extractor"] == "wildchat"

    def test_preserves_lane_metadata(self):
        ref = make_source_ref(
            "test",
            "path.yaml",
            "EN",
            grounding_mode="pooled",
            route_policy="residual",
            reason_provenance="heuristic",
            applicability_scope="mixed",
        )
        assert ref["grounding_mode"] == "pooled"
        assert ref["route_policy"] == "residual"
        assert ref["reason_provenance"] == "heuristic"
        assert ref["applicability_scope"] == "mixed"

    def test_preserves_max_samples(self):
        ref = make_source_ref("test", "path.yaml", "EN", max_samples=123)
        assert ref["max_samples"] == 123


# ---------------------------------------------------------------------------
# build_supplement
# ---------------------------------------------------------------------------


class TestBuildSupplement:
    def test_expands_sources_and_preserves_lane_metadata(self):
        supplement = build_supplement(
            {
                "name": "residual_pool",
                "weakness": "pooled attacks need residual routing",
                "max_samples": 120,
                "attack_sources": [
                    {
                        "name": "pooled_atk",
                        "path": "attacks.yaml",
                        "language": "EN",
                        "route_policy": "residual",
                        "grounding_mode": "pooled",
                        "reason_provenance": "none",
                        "applicability_scope": "mixed",
                    }
                ],
                "benign_sources": [
                    {
                        "name": "residual_ben",
                        "path": "benign.yaml",
                        "language": "EN",
                        "route_policy": "residual",
                        "applicability_scope": "in_domain",
                    }
                ],
            }
        )
        assert supplement["name"] == "residual_pool"
        assert supplement["max_samples"] == 120
        assert supplement["attack_sources"][0]["route_policy"] == "residual"
        assert supplement["attack_sources"][0]["grounding_mode"] == "pooled"
        assert supplement["benign_sources"][0]["route_policy"] == "residual"


# ---------------------------------------------------------------------------
# build_cell — base sources
# ---------------------------------------------------------------------------


class TestBuildCellBaseSources:
    def test_includes_base_attack_sources(self):
        compact = _minimal_compact()
        cell = build_cell("instruction_override", compact)
        atk_names = [s["name"] for s in cell["attack_sources"]]
        assert "en_attacks" in atk_names

    def test_includes_base_benign_sources(self):
        compact = _minimal_compact()
        cell = build_cell("instruction_override", compact)
        ben_names = [s["name"] for s in cell["benign_sources"]]
        assert "en_benign" in ben_names

    def test_cell_has_correct_reason(self):
        compact = _minimal_compact()
        cell = build_cell("meta_probe", compact)
        assert cell["reason"] == "meta_probe"

    def test_cell_has_languages(self):
        compact = _minimal_compact()
        cell = build_cell("obfuscation", compact)
        assert cell["languages"] == LANGUAGES

    def test_cell_has_distributions(self):
        compact = _minimal_compact()
        cell = build_cell("exfiltration", compact)
        assert cell["format_distribution"] == {"prose": 0.7, "structured": 0.2, "code": 0.1}
        assert cell["length_distribution"] == {"short": 0.5, "medium": 0.3, "long": 0.2}

    def test_cell_has_teaching_goal(self):
        compact = _minimal_compact()
        cell = build_cell("exfiltration", compact)
        assert cell["teaching_goal"] == "Test goal for exfiltration"


# ---------------------------------------------------------------------------
# build_cell — extra sources
# ---------------------------------------------------------------------------


class TestBuildCellExtraSources:
    def test_extra_attack_sources_added(self):
        compact = _minimal_compact()
        compact["cells"]["meta_probe"]["extra_attack_sources"] = [
            {"name": "extra_atk", "path": "extra.yaml", "language": "EN"}
        ]
        cell = build_cell("meta_probe", compact)
        atk_names = [s["name"] for s in cell["attack_sources"]]
        assert "extra_atk" in atk_names
        assert "en_attacks" in atk_names  # base still present

    def test_extra_benign_sources_added(self):
        compact = _minimal_compact()
        compact["cells"]["adversarial_suffix"]["extra_benign_sources"] = [
            {"name": "extra_ben", "path": "extra.yaml", "language": "EN"}
        ]
        cell = build_cell("adversarial_suffix", compact)
        ben_names = [s["name"] for s in cell["benign_sources"]]
        assert "extra_ben" in ben_names
        assert "en_benign" in ben_names

    def test_no_extra_sources_means_only_base(self):
        compact = _minimal_compact()
        cell = build_cell("indirect_injection", compact)
        assert len(cell["attack_sources"]) == 1  # just en_attacks
        assert len(cell["benign_sources"]) == 1  # just en_benign


# ---------------------------------------------------------------------------
# build_cell — staged attacks
# ---------------------------------------------------------------------------


class TestBuildCellStagedAttacks:
    def test_staged_attacks_added_for_matching_reason(self):
        compact = _minimal_compact()
        compact["staged_attacks"] = {
            "ru_staged": {
                "path": "ru_staged.yaml",
                "language": "RU",
                "reasons": ["obfuscation", "roleplay_jailbreak"],
            }
        }
        cell = build_cell("obfuscation", compact)
        atk_names = [s["name"] for s in cell["attack_sources"]]
        assert "ru_staged_attacks_obfuscation" in atk_names

    def test_staged_attacks_skipped_for_non_matching_reason(self):
        compact = _minimal_compact()
        compact["staged_attacks"] = {
            "ru_staged": {
                "path": "ru_staged.yaml",
                "language": "RU",
                "reasons": ["obfuscation"],
            }
        }
        cell = build_cell("meta_probe", compact)
        atk_names = [s["name"] for s in cell["attack_sources"]]
        staged = [n for n in atk_names if "staged" in n]
        assert staged == []

    def test_staged_attack_has_reason_label_filter(self):
        compact = _minimal_compact()
        compact["staged_attacks"] = {
            "ru_staged": {
                "path": "ru_staged.yaml",
                "language": "RU",
                "reasons": ["adversarial_suffix"],
            }
        }
        cell = build_cell("adversarial_suffix", compact)
        staged_src = [s for s in cell["attack_sources"] if "staged" in s["name"]][0]
        assert staged_src["label_filter"] == {"column": "reason", "allowed": ["adversarial_suffix"]}
        assert staged_src["language"] == "RU"

    def test_staged_attack_carries_lane_metadata(self):
        compact = _minimal_compact()
        compact["staged_attacks"] = {
            "ru_staged": {
                "path": "ru_staged.yaml",
                "language": "RU",
                "reasons": ["meta_probe"],
                "grounding_mode": "reason_grounded",
                "route_policy": "mirror",
                "reason_provenance": "source_label",
            }
        }
        cell = build_cell("meta_probe", compact)
        staged_src = [s for s in cell["attack_sources"] if "staged" in s["name"]][0]
        assert staged_src["grounding_mode"] == "reason_grounded"
        assert staged_src["route_policy"] == "mirror"
        assert staged_src["reason_provenance"] == "source_label"

    def test_staged_attack_preserves_max_samples(self):
        compact = _minimal_compact()
        compact["staged_attacks"] = {
            "ru_staged": {
                "path": "ru_staged.yaml",
                "language": "RU",
                "reasons": ["meta_probe"],
                "max_samples": 200,
            }
        }
        cell = build_cell("meta_probe", compact)
        staged_src = [s for s in cell["attack_sources"] if "staged" in s["name"]][0]
        assert staged_src["max_samples"] == 200


# ---------------------------------------------------------------------------
# build_cell — staged EN benign
# ---------------------------------------------------------------------------


class TestBuildCellStagedBenignEN:
    def _compact_with_en_staged(self, reasons=None):
        compact = _minimal_compact()
        staged = {
            "datasets": {
                "dolly": {"path": "dolly_staged.yaml"},
                "codealpaca": {"path": "codealpaca_staged.yaml"},
            }
        }
        if reasons is not None:
            staged["reasons"] = reasons
        compact["staged_benign_en"] = staged
        return compact

    def test_en_staged_added_when_reason_matches(self):
        compact = self._compact_with_en_staged(reasons=["meta_probe", "exfiltration"])
        cell = build_cell("meta_probe", compact)
        ben_names = [s["name"] for s in cell["benign_sources"]]
        assert "en_staged_dolly_meta_probe" in ben_names
        assert "en_staged_codealpaca_meta_probe" in ben_names

    def test_en_staged_skipped_when_reason_excluded(self):
        compact = self._compact_with_en_staged(reasons=["meta_probe"])
        cell = build_cell("instruction_override", compact)
        ben_names = [s["name"] for s in cell["benign_sources"]]
        staged = [n for n in ben_names if "staged" in n]
        assert staged == []

    def test_en_staged_all_reasons_when_no_filter(self):
        compact = self._compact_with_en_staged(reasons=None)
        cell = build_cell("instruction_override", compact)
        ben_names = [s["name"] for s in cell["benign_sources"]]
        assert "en_staged_dolly_instruction_override" in ben_names

    def test_en_staged_has_reason_label_filter(self):
        compact = self._compact_with_en_staged(reasons=["exfiltration"])
        cell = build_cell("exfiltration", compact)
        staged = [s for s in cell["benign_sources"] if s["name"] == "en_staged_dolly_exfiltration"][0]
        assert staged["label_filter"] == {"column": "reason", "allowed": ["exfiltration"]}
        assert staged["language"] == "EN"

    def test_en_staged_preserves_max_samples(self):
        compact = _minimal_compact()
        compact["staged_benign_en"] = {
            "reasons": ["meta_probe"],
            "datasets": {
                "dolly": {"path": "dolly_staged.yaml", "max_samples": 50},
            },
        }
        cell = build_cell("meta_probe", compact)
        staged = [s for s in cell["benign_sources"] if s["name"] == "en_staged_dolly_meta_probe"][0]
        assert staged["max_samples"] == 50

    def test_en_staged_defaults_to_dynamic_reason_categories(self):
        compact = {
            "name": "residual_probe",
            "version": "1.0.0",
            "seed": 42,
            "ratio": 1.0,
            "total_target": 100,
            "backfill": {"strategy": "ngram_safe", "log_gaps": True},
            "language_quota": {
                "mode": "best_effort",
                "profile": {"EN": 1.0},
            },
            "base_attack_sources": [{"name": "atk", "path": "attacks.yaml", "language": "EN"}],
            "base_benign_sources": [{"name": "ben", "path": "benign.yaml", "language": "EN"}],
            "cells": {
                "use_vs_mention": {
                    "teaching_goal": "separate mention from use",
                    "format": {"prose": 1.0},
                    "length": {"short": 1.0},
                },
                "multilingual_gap": {
                    "teaching_goal": "recover multilingual residuals",
                    "format": {"prose": 1.0},
                    "length": {"short": 1.0},
                },
            },
            "staged_benign_en": {
                "datasets": {
                    "dolly": {"path": "dolly.yaml"},
                }
            },
        }

        use_cell = build_cell("use_vs_mention", compact)
        multi_cell = build_cell("multilingual_gap", compact)
        assert "en_staged_dolly_use_vs_mention" in [s["name"] for s in use_cell["benign_sources"]]
        assert "en_staged_dolly_multilingual_gap" in [
            s["name"] for s in multi_cell["benign_sources"]
        ]


# ---------------------------------------------------------------------------
# build_cell — staged multilingual benign
# ---------------------------------------------------------------------------


class TestBuildCellStagedBenignMultilingual:
    def _compact_with_multilingual(self):
        compact = _minimal_compact()
        compact["staged_benign_multilingual"] = {
            "ru_saiga": {
                "path": "ru_saiga.yaml",
                "language": "RU",
                "reasons": ["instruction_override", "meta_probe"],
            },
            "zh_stem": {
                "path": "zh_stem.yaml",
                "language": "ZH",
                "reasons": ["meta_probe"],
            },
        }
        return compact

    def test_multilingual_added_for_matching_reason(self):
        compact = self._compact_with_multilingual()
        cell = build_cell("meta_probe", compact)
        ben_names = [s["name"] for s in cell["benign_sources"]]
        assert "ru_staged_benign_meta_probe" in ben_names
        assert "zh_staged_benign_meta_probe" in ben_names

    def test_multilingual_partial_match(self):
        compact = self._compact_with_multilingual()
        cell = build_cell("instruction_override", compact)
        ben_names = [s["name"] for s in cell["benign_sources"]]
        assert "ru_staged_benign_instruction_override" in ben_names
        # zh_stem only covers meta_probe, not instruction_override
        assert "zh_staged_benign_instruction_override" not in ben_names

    def test_multilingual_skipped_for_non_matching(self):
        compact = self._compact_with_multilingual()
        cell = build_cell("obfuscation", compact)
        ben_names = [s["name"] for s in cell["benign_sources"]]
        staged = [n for n in ben_names if "staged" in n]
        assert staged == []

    def test_multilingual_preserves_max_samples(self):
        compact = _minimal_compact()
        compact["staged_benign_multilingual"] = {
            "ru_saiga": {
                "path": "ru_saiga.yaml",
                "language": "RU",
                "reasons": ["meta_probe"],
                "max_samples": 75,
            }
        }
        cell = build_cell("meta_probe", compact)
        staged = [s for s in cell["benign_sources"] if s["name"] == "ru_staged_benign_meta_probe"][0]
        assert staged["max_samples"] == 75


# ---------------------------------------------------------------------------
# expand_spec — top-level
# ---------------------------------------------------------------------------


class TestExpandSpec:
    def test_all_8_cells_present(self):
        compact = _minimal_compact()
        spec = expand_spec(compact)
        reasons = [c["reason"] for c in spec["cells"]]
        assert reasons == ALL_REASONS

    def test_top_level_fields_copied(self):
        compact = _minimal_compact()
        spec = expand_spec(compact)
        assert spec["name"] == "test_spec"
        assert spec["version"] == "1.0.0"
        assert spec["seed"] == 42
        assert spec["ratio"] == 1.0
        assert spec["total_target"] == 1000

    def test_overrides_applied(self):
        compact = _minimal_compact()
        spec = expand_spec(compact, {"total_target": 500, "name": "overridden"})
        assert spec["total_target"] == 500
        assert spec["name"] == "overridden"

    def test_backfill_copied(self):
        compact = _minimal_compact()
        spec = expand_spec(compact)
        assert spec["backfill"]["strategy"] == "ngram_safe"

    def test_background_expanded(self):
        compact = _minimal_compact()
        compact["background"] = {
            "budget_fraction": 0.15,
            "sources": [{"name": "bg1", "path": "bg.yaml", "language": "EN"}],
        }
        spec = expand_spec(compact)
        assert spec["background"]["budget_fraction"] == 0.15
        assert len(spec["background"]["sources"]) == 1
        assert spec["background"]["sources"][0]["name"] == "bg1"

    def test_discussion_benign_expanded(self):
        compact = _minimal_compact()
        compact["discussion_benign"] = {
            "budget_fraction": 0.05,
            "sources": [
                {
                    "name": "discussion1",
                    "path": "discussion.yaml",
                    "language": "EN",
                    "route_policy": "discussion_benign",
                }
            ],
        }
        spec = expand_spec(compact)
        assert spec["discussion_benign"]["budget_fraction"] == 0.05
        assert len(spec["discussion_benign"]["sources"]) == 1
        assert spec["discussion_benign"]["sources"][0]["name"] == "discussion1"
        assert spec["discussion_benign"]["sources"][0]["route_policy"] == "discussion_benign"

    def test_strict_source_contracts_copied_when_enabled(self):
        compact = _minimal_compact()
        compact["enforce_source_contracts"] = True
        spec = expand_spec(compact)
        assert spec["enforce_source_contracts"] is True

    def test_allow_heuristic_mirror_attacks_copied_when_enabled(self):
        compact = _minimal_compact()
        compact["allow_heuristic_mirror_attacks"] = True
        spec = expand_spec(compact)
        assert spec["allow_heuristic_mirror_attacks"] is True

    def test_supplements_expanded_when_present(self):
        compact = _minimal_compact()
        compact["supplement_ratio"] = 0.2
        compact["supplements"] = [
            {
                "name": "residual_pool",
                "weakness": "pooled attack coverage",
                "max_samples": 100,
                "attack_sources": [
                    {
                        "name": "pooled_atk",
                        "path": "attacks.yaml",
                        "language": "EN",
                        "route_policy": "residual",
                        "grounding_mode": "pooled",
                    }
                ],
                "benign_sources": [
                    {
                        "name": "residual_ben",
                        "path": "benign.yaml",
                        "language": "EN",
                        "route_policy": "residual",
                    }
                ],
            }
        ]
        spec = expand_spec(compact)
        assert spec["supplement_ratio"] == 0.2
        assert len(spec["supplements"]) == 1
        assert spec["supplements"][0]["attack_sources"][0]["route_policy"] == "residual"

    def test_missing_cell_exits(self):
        compact = _minimal_compact()
        del compact["cells"]["obfuscation"]
        with pytest.raises(SystemExit):
            expand_spec(compact)

    def test_partial_mirror_allows_missing_cell(self):
        compact = _minimal_compact()
        compact["allow_partial_mirror"] = True
        del compact["cells"]["obfuscation"]
        spec = expand_spec(compact)
        reasons = [c["reason"] for c in spec["cells"]]
        assert "obfuscation" not in reasons
        assert len(reasons) == 7

    def test_overrides_do_not_mutate_compact(self):
        compact = _minimal_compact()
        original_target = compact["total_target"]
        expand_spec(compact, {"total_target": 9999})
        assert compact["total_target"] == original_target

    def test_get_reason_categories_defaults_to_custom_cells(self):
        compact = _minimal_compact()
        compact["cells"] = {
            "use_vs_mention": {
                "teaching_goal": "use vs mention",
                "format": {"prose": 1.0},
                "length": {"short": 1.0},
            },
            "multilingual_gap": {
                "teaching_goal": "multilingual gap",
                "format": {"prose": 1.0},
                "length": {"short": 1.0},
            },
        }
        assert get_reason_categories(compact) == ["use_vs_mention", "multilingual_gap"]

    def test_expand_spec_copies_explicit_reason_categories(self):
        compact = _minimal_compact()
        compact["reason_categories"] = ["instruction_override", "meta_probe"]
        compact["cells"] = {
            "instruction_override": compact["cells"]["instruction_override"],
            "meta_probe": compact["cells"]["meta_probe"],
        }
        spec = expand_spec(compact)
        assert spec["reason_categories"] == ["instruction_override", "meta_probe"]
        assert [cell["reason"] for cell in spec["cells"]] == [
            "instruction_override",
            "meta_probe",
        ]

    def test_expand_spec_infers_custom_reason_categories(self):
        compact = _minimal_compact()
        compact["cells"] = {
            "use_vs_mention": {
                "teaching_goal": "use vs mention",
                "format": {"prose": 1.0},
                "length": {"short": 1.0},
            },
            "multilingual_gap": {
                "teaching_goal": "multilingual gap",
                "format": {"prose": 1.0},
                "length": {"short": 1.0},
            },
        }
        spec = expand_spec(compact)
        assert spec["reason_categories"] == ["use_vs_mention", "multilingual_gap"]
        assert [cell["reason"] for cell in spec["cells"]] == [
            "use_vs_mention",
            "multilingual_gap",
        ]

    def test_reason_categories_reject_extra_cells(self):
        compact = _minimal_compact()
        compact["reason_categories"] = ["instruction_override"]
        compact["cells"] = {
            "instruction_override": compact["cells"]["instruction_override"],
            "meta_probe": compact["cells"]["meta_probe"],
        }
        with pytest.raises(SystemExit):
            expand_spec(compact)


# ---------------------------------------------------------------------------
# Roundtrip against real v3 compact spec
# ---------------------------------------------------------------------------


class TestV3Roundtrip:
    """Verify generated output matches hand-written specs."""

    def _load_hand_written(self, filename):
        path = Path(__file__).resolve().parent.parent / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        with open(path, encoding="utf-8-sig") as f:
            return yaml.safe_load(f)

    def _compare_specs(self, generated, hand_written):
        """Assert all cells match by source names."""
        for gc, hc in zip(generated["cells"], hand_written["cells"]):
            reason = gc["reason"]
            g_atk = sorted(s["name"] for s in gc["attack_sources"])
            h_atk = sorted(s["name"] for s in hc["attack_sources"])
            assert g_atk == h_atk, f"{reason} attack sources differ"

            g_ben = sorted(s["name"] for s in gc["benign_sources"])
            h_ben = sorted(s["name"] for s in hc["benign_sources"])
            assert g_ben == h_ben, f"{reason} benign sources differ"

            assert gc["teaching_goal"] == hc["teaching_goal"], f"{reason} teaching_goal differs"
            assert gc["format_distribution"] == hc["format_distribution"], f"{reason} format differs"
            assert gc["length_distribution"] == hc["length_distribution"], f"{reason} length differs"

    def test_19k_matches_hand_written(self):
        compact = _v3_compact()
        hand = self._load_hand_written("mirror_spec_v3_19k.yaml")
        generated = expand_spec(compact, {"total_target": 19200})
        assert generated["name"] == hand["name"]
        assert generated["version"] == hand["version"]
        assert generated["total_target"] == hand["total_target"]
        self._compare_specs(generated, hand)

    def test_96k_matches_hand_written(self):
        compact = _v3_compact()
        hand = self._load_hand_written("mirror_spec_v3.yaml")
        generated = expand_spec(compact)
        assert generated["name"] == hand["name"]
        assert generated["total_target"] == hand["total_target"]
        self._compare_specs(generated, hand)

    def test_scale_ladder_all_valid(self):
        """All scale targets produce 8 cells with sources."""
        compact = _v3_compact()
        for target in [19200, 40000, 60000, 96000]:
            spec = expand_spec(compact, {"total_target": target})
            assert len(spec["cells"]) == 8
            assert spec["total_target"] == target
            for cell in spec["cells"]:
                assert len(cell["attack_sources"]) > 0
                assert len(cell["benign_sources"]) > 0
