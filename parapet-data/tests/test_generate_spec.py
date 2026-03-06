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
from generate_spec import ALL_REASONS, LANGUAGES, build_cell, expand_spec, make_source_ref


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
