"""Tests for parapet-runner/scripts/audit_legacy_eval_leakage.py."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import yaml


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "audit_legacy_eval_leakage.py"
    )
    spec = importlib.util.spec_from_file_location("audit_legacy_eval_leakage", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new_output_dir(case_name: str) -> Path:
    output_dir = Path("tests/.tmp_outputs") / "audit_legacy" / case_name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Description parsing
# ---------------------------------------------------------------------------


def test_parse_description_source_kv():
    m = _load_module()
    out = m.parse_target_description(
        "tough-attack-v1 source=jbb_paraphrase_attacks orig_id=jbb-para-022 overlap=no",
        Path("tough_attack_v1/x.yaml"),
    )
    assert out["target_source"] == "jbb_paraphrase_attacks"
    assert out["parser"] == "source_kv"


def test_parse_description_holdout_prefix():
    m = _load_module()
    out = m.parse_target_description(
        "holdout: no_robots: Generation",
        Path("l1_holdout.yaml"),
    )
    assert out["target_source"] == "no_robots"
    assert out["parser"] == "holdout_prefix"


def test_parse_description_unparsed():
    m = _load_module()
    out = m.parse_target_description("just some prose", Path("anything.yaml"))
    assert out["target_source"] is None
    assert out["parser"] == "unparsed"


def test_parse_description_none_safe():
    m = _load_module()
    out = m.parse_target_description(None, Path("x.yaml"))
    assert out["target_source"] is None
    assert out["parser"] == "none"


# ---------------------------------------------------------------------------
# Pair categorization
# ---------------------------------------------------------------------------


def test_categorize_exact_duplicate_label_source():
    m = _load_module()
    out = m.categorize_pair(
        train_row={"label": "benign", "source": "no_robots"},
        target_row={"label": "benign"},
        target_meta={"target_source": "no_robots"},
    )
    assert out == "exact_duplicate_label_source"


def test_categorize_same_label_diff_source():
    m = _load_module()
    out = m.categorize_pair(
        train_row={"label": "benign", "source": "src_a"},
        target_row={"label": "benign"},
        target_meta={"target_source": "src_b"},
    )
    assert out == "same_label_diff_source"


def test_categorize_same_label_target_unparsed():
    m = _load_module()
    # When target_source can't be parsed, downgrade to same_label_diff_source.
    out = m.categorize_pair(
        train_row={"label": "benign", "source": "src_a"},
        target_row={"label": "benign"},
        target_meta={"target_source": None},
    )
    assert out == "same_label_diff_source"


def test_categorize_cross_label_conflict():
    m = _load_module()
    out = m.categorize_pair(
        train_row={"label": "benign", "source": "src_a"},
        target_row={"label": "malicious"},
        target_meta={"target_source": "src_a"},
    )
    assert out == "cross_label_conflict"


# ---------------------------------------------------------------------------
# Provenance index
# ---------------------------------------------------------------------------


def test_provenance_index_handles_staged_attacks_keyed_dict():
    m = _load_module()
    tmp_path = _new_output_dir("prov_staged_attacks")
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({
        "staged_attacks": {
            "ru_russian_prompt_injections": {
                "path": "x.jsonl",
                "route_policy": "mirror",
                "reason_provenance": "source_label",
            },
        },
    }), encoding="utf-8")
    idx = m.build_provenance_index(spec)
    assert "ru_russian_prompt_injections" in idx
    assert idx["ru_russian_prompt_injections"]["lane"] == "staged_attacks"
    assert idx["ru_russian_prompt_injections"]["reason_provenance"] == "source_label"


def test_provenance_index_handles_base_attack_sources_list():
    m = _load_module()
    tmp_path = _new_output_dir("prov_base_attack_sources")
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({
        "base_attack_sources": [
            {"name": "en_attacks_merged", "route_policy": "mirror",
             "reason_provenance": "heuristic_staged"},
        ],
    }), encoding="utf-8")
    idx = m.build_provenance_index(spec)
    assert idx["en_attacks_merged"]["reason_provenance"] == "heuristic_staged"
    assert idx["en_attacks_merged"]["lane"] == "base_attack_sources"


def test_provenance_index_handles_cells_extras():
    m = _load_module()
    tmp_path = _new_output_dir("prov_cells")
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({
        "cells": {
            "instruction_override": {
                "extra_attack_sources": [
                    {"name": "en_instruction_override", "route_policy": "mirror",
                     "reason_provenance": "manual_map"},
                ],
            },
        },
    }), encoding="utf-8")
    idx = m.build_provenance_index(spec)
    assert idx["en_instruction_override"]["reason_provenance"] == "manual_map"


def test_provenance_for_unknown_source_returns_unknown_lane():
    m = _load_module()
    out = m.provenance_for_source({}, "no_such_source")
    assert out["lane"] == "unknown"


# ---------------------------------------------------------------------------
# End-to-end main()
# ---------------------------------------------------------------------------


def _setup_e2e(tmp_path: Path) -> dict:
    """Build the minimal artifact tree main() consumes (self-contained)."""
    output_dir = tmp_path / "legacy_audit"
    curation_dir = tmp_path / "curated"
    curation_dir.mkdir()
    spec_path = tmp_path / "spec.yaml"
    schema_eval_dir = tmp_path / "schema_eval"
    (schema_eval_dir / "challenges" / "tough_attack_v1").mkdir(parents=True)

    train_rows = [
        {"content": "shared_holdout_row", "label": "benign", "language": "EN",
         "reason": "instruction_override", "source": "no_robots",
         "format_bin": "prose", "length_bin": "short"},
        {"content": "shared_attack_row", "label": "malicious", "language": "EN",
         "reason": "obfuscation", "source": "jbb_paraphrase_attacks",
         "format_bin": "code", "length_bin": "medium"},
        {"content": "cross_label_row", "label": "benign", "language": "RU",
         "reason": "meta_probe", "source": "ru_turbo_saiga",
         "format_bin": "prose", "length_bin": "long"},
        {"content": "train_only", "label": "benign", "language": "ZH",
         "reason": "indirect_injection", "source": "zh_benign_xquad",
         "format_bin": "prose", "length_bin": "short"},
    ]
    (curation_dir / "train.yaml").write_text(
        yaml.safe_dump(train_rows, allow_unicode=True), encoding="utf-8"
    )

    spec = {
        "base_attack_sources": [
            {"name": "jbb_paraphrase_attacks", "route_policy": "mirror",
             "reason_provenance": "heuristic_staged"},
        ],
        "base_benign_sources": [
            {"name": "zh_benign_xquad", "route_policy": "mirror"},
        ],
        "staged_benign_multilingual": {
            "ru_turbo_saiga": {"route_policy": "mirror"},
        },
    }
    spec_path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")

    # Legacy-style YAMLs at autodiscovery locations:
    holdout_path = schema_eval_dir / "l1_holdout.yaml"
    holdout_path.write_text(yaml.safe_dump([
        {"id": "h-1", "layer": "l1", "label": "benign",
         "description": "holdout: no_robots: Generation",
         "content": "shared_holdout_row"},
        {"id": "h-2", "layer": "l1", "label": "malicious",
         "description": "holdout: somewhere: else",
         "content": "cross_label_row"},
        {"id": "h-3", "layer": "l1", "label": "benign",
         "description": "holdout: x: y",
         "content": "holdout_only"},
    ], allow_unicode=True), encoding="utf-8")
    challenge_path = (schema_eval_dir / "challenges" / "tough_attack_v1"
                      / "tough_attack_mirror_v2_novel.yaml")
    challenge_path.write_text(yaml.safe_dump([
        {"id": "c-1", "layer": "l1", "label": "malicious",
         "description": "tough-attack-v1 source=jbb_paraphrase_attacks orig_id=x",
         "content": "shared_attack_row"},
    ], allow_unicode=True), encoding="utf-8")

    # `_excluded` tombstone — must be skipped by autodiscovery.
    excluded_dir = schema_eval_dir / "challenges" / "_excluded"
    excluded_dir.mkdir(parents=True)
    (excluded_dir / "tombstone.yaml").write_text(
        yaml.safe_dump({"removed_from": "x", "reason": "y"}), encoding="utf-8"
    )

    return {
        "output_dir": output_dir,
        "curation_dir": curation_dir,
        "spec_path": spec_path,
        "schema_eval_dir": schema_eval_dir,
        "holdout_path": holdout_path,
        "challenge_path": challenge_path,
    }


def test_discover_legacy_eval_yamls_includes_holdout_and_challenges():
    tmp_path = _new_output_dir("discover_legacy")
    m = _load_module()
    schema_eval = tmp_path / "schema_eval"
    (schema_eval / "challenges" / "tough_attack_v1").mkdir(parents=True)
    (schema_eval / "challenges" / "_excluded").mkdir(parents=True)
    (schema_eval / "l1_holdout.yaml").write_text("[]", encoding="utf-8")
    (schema_eval / "challenges" / "tough_attack_v1" / "x.yaml").write_text("[]", encoding="utf-8")
    (schema_eval / "challenges" / "_excluded" / "tomb.yaml").write_text("[]", encoding="utf-8")
    found = [str(p) for p in m.discover_legacy_eval_yamls(schema_eval)]
    assert any("l1_holdout.yaml" in p for p in found)
    assert any("tough_attack_v1" in p for p in found)
    assert not any("_excluded" in p for p in found)


def test_main_attribution_end_to_end_self_contained():
    tmp_path = _new_output_dir("main_e2e")
    m = _load_module()
    paths = _setup_e2e(tmp_path)

    rc = m.main([
        "--output-dir", str(paths["output_dir"]),
        "--curation-dir", str(paths["curation_dir"]),
        "--spec", str(paths["spec_path"]),
        "--schema-eval-dir", str(paths["schema_eval_dir"]),
    ])
    assert rc == 0

    # Self-contained outputs; no dependency on a parent leakage_check.json.
    attribution = json.loads(
        (paths["output_dir"] / "leakage_attribution.json").read_text(encoding="utf-8")
    )
    leakage = json.loads(
        (paths["output_dir"] / "leakage_check.json").read_text(encoding="utf-8")
    )
    readme = (paths["output_dir"] / "README.md").read_text(encoding="utf-8")

    assert attribution["diagnostic_only"] is True
    assert leakage["diagnostic_only"] is True
    assert "NOT acceptance gates" in readme

    # 3 overlap pairs: (holdout_row, train), (cross_label, train), (attack, train)
    assert attribution["overall"]["total_pairs"] == 3
    assert attribution["overall"]["total_overlapping_hashes"] == 3

    # Categories
    by_cat = attribution["overall"]["by_category"]
    assert by_cat.get("exact_duplicate_label_source") == 2
    assert by_cat.get("cross_label_conflict") == 1

    # Per-target sanity
    by_path = {Path(t["path"]).name: t for t in attribution["per_target"]}
    assert by_path["l1_holdout.yaml"]["overlap_count"] == 2
    assert by_path["tough_attack_mirror_v2_novel.yaml"]["overlap_count"] == 1
    assert by_path["tough_attack_mirror_v2_novel.yaml"]["categories"][
        "exact_duplicate_label_source"
    ] == 1

    # Provenance lane plumbed through
    pairs = attribution["all_pairs"]
    attack_pair = [p for p in pairs if p["train_source"] == "jbb_paraphrase_attacks"][0]
    assert attack_pair["provenance_lane"] == "base_attack_sources"
    assert attack_pair["reason_provenance"] == "heuristic_staged"

    # Markdown header marks the diagnostic-only nature.
    md = (paths["output_dir"] / "leakage_attribution.md").read_text(encoding="utf-8")
    assert "diagnostic only" in md.lower() or "diagnostic_only" in md.lower()
    assert "l1_holdout.yaml" in md
    assert "tough_attack_mirror_v2_novel.yaml" in md


def test_main_explicit_target_overrides_autodiscovery():
    tmp_path = _new_output_dir("main_explicit_target")
    m = _load_module()
    output_dir = tmp_path / "out"
    curation_dir = tmp_path / "curated"
    curation_dir.mkdir()
    (curation_dir / "train.yaml").write_text(
        yaml.safe_dump([{"content": "shared", "label": "benign", "source": "src"}]),
        encoding="utf-8",
    )
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("{}", encoding="utf-8")

    target = tmp_path / "explicit.yaml"
    target.write_text(
        yaml.safe_dump([{"id": "t-1", "layer": "l1", "label": "benign",
                         "description": "x", "content": "shared"}]),
        encoding="utf-8",
    )

    rc = m.main([
        "--output-dir", str(output_dir),
        "--curation-dir", str(curation_dir),
        "--spec", str(spec_path),
        "--schema-eval-dir", str(tmp_path / "no_legacy"),
        "--target", str(target),
    ])
    assert rc == 0
    attr = json.loads((output_dir / "leakage_attribution.json").read_text(encoding="utf-8"))
    assert attr["overall"]["total_pairs"] == 1


def test_main_handles_clean_target_with_zero_overlap():
    tmp_path = _new_output_dir("main_clean_target")
    m = _load_module()
    output_dir = tmp_path / "out"
    curation_dir = tmp_path / "curated"
    curation_dir.mkdir()
    (curation_dir / "train.yaml").write_text(
        yaml.safe_dump([{"content": "a", "label": "benign"}]), encoding="utf-8"
    )
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump({}), encoding="utf-8")
    target_path = tmp_path / "clean.yaml"
    target_path.write_text(
        yaml.safe_dump([{"id": "c-1", "layer": "l1", "label": "benign",
                         "description": "x", "content": "z"}]),
        encoding="utf-8",
    )

    rc = m.main([
        "--output-dir", str(output_dir),
        "--curation-dir", str(curation_dir),
        "--spec", str(spec_path),
        "--schema-eval-dir", str(tmp_path / "no_legacy"),
        "--target", str(target_path),
    ])
    assert rc == 0
    attr = json.loads((output_dir / "leakage_attribution.json").read_text(encoding="utf-8"))
    assert attr["overall"]["total_pairs"] == 0
    assert attr["per_target"][0]["overlap_count"] == 0
